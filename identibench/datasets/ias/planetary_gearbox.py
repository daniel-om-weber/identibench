"""Planetary gearbox IAS estimation dataset (figshare 28992879).

The IAS label is reconstructed from the **zebra tape on the sun shaft**
(``Channel_5_Data``), using the 1-pulse-per-revolution magnetic pickup on the planet
carrier (``Channel_6_Data``) only as an absolute angle reference. The tape's stripes are
irregular and not every stripe registers, so there is no fixed pulses-per-revolution to
assume; a stripe template is estimated from the data instead, pooled across every
recording, and the matched stripe times are then resampled onto a genuinely uniform angle
grid before the order-domain low-pass.
"""

__all__ = [
    "planetary_gearbox_dataset",
    "dl_planetary_gearbox",
    "BenchmarkPlanetaryGearbox_Estimation",
    "BenchmarkPlanetaryGearbox_Simulation",
]

import tempfile
from pathlib import Path

import numpy as np
import scipy.io
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from tqdm import tqdm

from ...benchmark import BenchmarkSpec, Simulation, WindowedEstimation, GridwiseEstimation
from ...dataset import Dataset
from ...metrics import mae
from ._common import (
    DatasetInfo,
    download_and_unpack,
    order_domain_lowpass,
    rising_edge_times,
    save_signals_hdf5,
    ias_test_sets,
    write_disturbed_test_sets,
)

_INFO = DatasetInfo(
    name="Planetary_Gearbox",
    zip_url="https://ndownloader.figshare.com/files/28992879",
    download_headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://figshare.com/articles/dataset/Planetary_gearbox_vibration_data/13476525?file=28992879",
    },
)

# Crack severities forming the out-of-distribution wear set; G1_P4/G2_P1 are the
# basic test recordings and G1_P3/G2_P0 the validation recordings (verbatim split).
_TEST_WEAR_TYPES = ["P5", "P6", "P7"]
_TEST_BASIC_TYPES = ["G1_P4", "G2_P1"]
_VALID_TYPES = ["G1_P3", "G2_P0"]

# Bonfiglioli 300-L: 13 sun teeth, 24 planet, 62 ring. With the ring fixed the sun turns
# (13 + 62) / 13 times per carrier revolution, which is what converts the carrier-mounted
# 1PR reference into sun-shaft angle -- and hence what makes the label sun-referenced.
_SUN_TEETH, _RING_TEETH = 13, 62
_SUN_PER_CARRIER_REV = (_SUN_TEETH + _RING_TEETH) / _SUN_TEETH

# Order-domain cutoff for the reconstructed IAS, in orders of the sun shaft. The pooled
# template resolves 76 stripes/rev, so the order-domain Nyquist is 38.
_CUTOFF_ORDER = 15.0

# Highest frequency the label retains: IAS_max * cutoff_order on the sun shaft. 39.90 Hz is the
# peak across all 15 recordings (medians run 4.5-17.6 Hz). This is by far the widest band of the
# four IAS datasets -- the zebra tape resolves ~76x more per revolution than the 1PR pickup it
# replaced -- which is what drives both the evaluation grid and the model sample-rate floor.
_IAS_BANDWIDTH_HZ = 39.90 * _CUTOFF_ORDER  # 598.5 Hz

# Stripe-template estimation: phase histogram resolution, and how many times the per-file
# histograms are re-aligned against the running pooled reference before the peaks are fitted.
_N_PHASE_BINS = 4000
_N_POOL_ITER = 2

# `drop_zebra_outliers`: a match whose local instantaneous rate departs from the running
# median over this many neighbours by more than this factor is discarded. The window must be odd.
_OUTLIER_WINDOW = 101
_OUTLIER_RATIO = 1.35
_OUTLIER_MAX_ITER = 5


def _parse_fs(mat_data: dict) -> float:
    """Per-file sampling rate from the .MAT header (verbatim nested-index + decimal-comma parse)."""
    return float(
        np.fromstring(mat_data["File_Header"]["SampleFrequency"][0][0][0].replace(",", "."), sep=";").squeeze()
    )


# ───────────────────────── zebra stripe template ─────────────────────────


def _zebra_phase(t_ref_pulses: np.ndarray, t_zebra_pulses: np.ndarray) -> tuple[np.ndarray, float]:
    """Sun-shaft phase (fraction of a revolution) of each zebra edge, plus the revolutions spanned.

    Cumulative 1PR pulse count is a sample of carrier angle at each reference edge; monotone
    PCHIP through it gives carrier angle at arbitrary times, and scaling by
    ``_SUN_PER_CARRIER_REV`` converts that to sun angle. The first and last ten reference
    pulses are excluded so the interpolant is never extrapolated.
    """
    theta_ref = PchipInterpolator(t_ref_pulses, np.arange(len(t_ref_pulses)))
    mask = (t_zebra_pulses > t_ref_pulses[10]) & (t_zebra_pulses < t_ref_pulses[-10])
    theta_zebra = _SUN_PER_CARRIER_REV * theta_ref(t_zebra_pulses[mask])
    return theta_zebra % 1.0, float(theta_zebra[-1] - theta_zebra[0])


def _phase_histogram(phase: np.ndarray, n_bins: int = _N_PHASE_BINS) -> np.ndarray:
    return np.histogram(phase, bins=n_bins, range=(0, 1))[0].astype(float)


def _circular_offset(reference_hist: np.ndarray, hist: np.ndarray) -> int:
    """Bin shift of ``hist`` that best aligns it with ``reference_hist``, circularly."""
    a = reference_hist - reference_hist.mean()
    b = hist - hist.mean()
    return int(np.argmax(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).real))


def _fit_template_from_phase(
    phase: np.ndarray, n_revs: float, n_bins: int = _N_PHASE_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Stripe angles (in revolutions) and their detection counts, from pooled phases.

    Peaks are found on a three-fold tiling of the histogram so a stripe sitting near phase 0
    is not split by the wrap, then each peak is refined to the median phase of the detections
    around it rather than the bin centre.

    Note the resolution limit: peaks are required to be at least half the *mean* stripe
    spacing apart, so two stripes printed closer together than that merge into one. On the
    real tape this does not bite (the pooled fit resolves all 76), but it caps how uneven a
    tape this can characterize.
    """
    hist = _phase_histogram(phase, n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    n_guess = int(round(len(phase) / n_revs))
    min_distance = max(int(n_bins / n_guess * 0.5), 1)

    tiled = np.concatenate([hist, hist, hist])
    peak_idx_tiled, _ = find_peaks(tiled, distance=min_distance, prominence=hist.mean() * 0.3)
    peak_idx = sorted({(p - n_bins) % n_bins for p in peak_idx_tiled if n_bins <= p < 2 * n_bins})

    template, counts = [], []
    half_width = 0.5 / n_guess * 0.6
    for pk in peak_idx:
        offsets = ((phase - bin_centers[pk] + 0.5) % 1.0) - 0.5
        nearby = offsets[np.abs(offsets) < half_width]
        if len(nearby) == 0:
            continue  # a peak with no detections inside the half-width would give a NaN angle
        # Take the median in the peak's local frame so detections on either side of
        # phase zero remain neighbours; wrap back only after estimating the centre.
        template.append((bin_centers[pk] + np.median(nearby)) % 1.0)
        counts.append(len(nearby))
    order = np.argsort(template)
    return np.array(template)[order], np.array(counts)[order]


def pool_zebra_templates(
    file_phases: list[np.ndarray], file_revs: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool phase-aligned zebra detections across files into one stripe template.

    Files are circularly cross-correlated against a running reference histogram and aligned
    before pooling: small rotational offsets between recording sessions (from
    disassembly/reassembly between test phases) would otherwise blur or split real stripes
    and lose stripe count as more files are added. Pooling every recording roughly doubles
    the evidence behind the template versus a single file and measurably improves the weakest
    stripes; single-file templates from the lower-quality recordings are incomplete on their own.

    Returns:
        ``(template, counts, pooled_reference_hist)`` -- stripe angles in revolutions, the
        detections behind each, and the pooled histogram later files are aligned against.
    """
    hists = [_phase_histogram(p) for p in file_phases]
    reference = hists[int(np.argmax(file_revs))]
    pooled = None
    for _ in range(_N_POOL_ITER):
        offsets = [_circular_offset(reference, h) for h in hists]
        aligned = [(p + o / _N_PHASE_BINS) % 1.0 for p, o in zip(file_phases, offsets)]
        pooled = np.concatenate(aligned)
        reference = _phase_histogram(pooled)
    template, counts = _fit_template_from_phase(pooled, sum(file_revs))
    return template, counts, reference


def zebra_phase_offset(phase: np.ndarray, pooled_reference_hist: np.ndarray) -> float:
    """This file's rotational offset (in revolutions) against the pooled template."""
    return _circular_offset(pooled_reference_hist, _phase_histogram(phase)) / _N_PHASE_BINS


# ───────────────────────── matching and reconstruction ─────────────────────────


def _theta(idx: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cumulative shaft angle, in revolutions, of global stripe index ``idx``."""
    n = len(template)
    return (idx // n) + template[idx % n]


def match_zebra_to_template(
    t_zebra_pulses: np.ndarray,
    t_ref_pulses: np.ndarray,
    template: np.ndarray,
    phase_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each zebra pulse a monotonically increasing global stripe index.

    Matches on nearest template angular position -- circularly, so a pulse can match into the
    neighbouring revolution -- using the 1PR/PCHIP absolute-angle estimate for each pulse
    independently. Unlike a sequentially-tracked local-rate estimate, a bad match therefore
    cannot drift into a self-reinforcing wrong streak: every pulse is placed fresh from the
    same trusted reference. Pulses whose match would not increase the running index are
    dropped as ambiguous. Occasional per-pulse misassignments this still produces (when real
    speed varies enough within one 1PR interval that the reference briefly disagrees with the
    true position by a stripe or two) are handled afterwards by :func:`drop_zebra_outliers`.
    """
    n_template = len(template)
    theta_ref = PchipInterpolator(t_ref_pulses, np.arange(len(t_ref_pulses)))
    mask = (t_zebra_pulses > t_ref_pulses[1]) & (t_zebra_pulses < t_ref_pulses[-2])
    t_in = t_zebra_pulses[mask]
    frac_rev = _SUN_PER_CARRIER_REV * theta_ref(t_in) + phase_offset
    rev_index = np.floor(frac_rev).astype(np.int64)
    local_phase = frac_rev - rev_index

    ext_template = np.concatenate([template - 1, template, template + 1])
    ext_rev_offset = np.repeat([-1, 0, 1], n_template)
    ext_k = np.tile(np.arange(n_template), 3)
    pos = np.clip(np.searchsorted(ext_template, local_phase), 1, len(ext_template) - 1)
    left, right = pos - 1, pos
    use_left = np.abs(ext_template[left] - local_phase) < np.abs(ext_template[right] - local_phase)
    best = np.where(use_left, left, right)
    global_index = (rev_index + ext_rev_offset[best]) * n_template + ext_k[best]

    # Keep only indices that exceed every index seen so far. A dropped index never exceeds the
    # running maximum, so the max over *all* previous entries equals the max over kept ones --
    # which makes this equivalent to a sequential "greater than last kept" scan.
    keep = np.ones(len(t_in), dtype=bool)
    keep[1:] = global_index[1:] > np.maximum.accumulate(global_index)[:-1]
    return t_in[keep], global_index[keep]


def drop_zebra_outliers(
    t_matched: np.ndarray, global_index: np.ndarray, template: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove matches whose local instantaneous rate is inconsistent with their neighbours.

    Not every inconsistency is a matching bug: a few recordings have genuinely poor tape or
    sensor contact over patches of the tape, and no matching algorithm can safely resolve a
    stripe identity from such sparse, contradictory data. Rather than guessing, and risking a
    confidently-wrong value, those matches are dropped and left as an ordinary gap for the
    reconstruction to interpolate over, exactly like a plain missed detection.
    """
    t, idx = t_matched, global_index
    for _ in range(_OUTLIER_MAX_ITER):
        if len(t) < _OUTLIER_WINDOW + 1:
            break
        inst_rate = np.diff(_theta(idx, template)) / np.diff(t)
        ratio = inst_rate / median_filter(inst_rate, size=_OUTLIER_WINDOW)
        bad = (ratio > _OUTLIER_RATIO) | (ratio < 1 / _OUTLIER_RATIO)
        if not bad.any():
            break
        keep = np.ones(len(t), dtype=bool)
        keep[1:][bad] = False
        t, idx = t[keep], idx[keep]
    return t, idx


def reconstruct_ias(
    t_clean: np.ndarray, idx_clean: np.ndarray, template: np.ndarray, fs: float, signal_len: int
) -> tuple[np.ndarray, slice]:
    """Continuous sun-shaft IAS (Hz) from matched zebra stripes, low-passed in the order domain.

    The matched stripes give angle at irregular times. Monotone PCHIP through them, evaluated
    at a uniform *angle* grid of ``len(template)`` points per revolution, inverts that into
    the time of each angle step; differencing gives a rate that is genuinely uniform in angle,
    which is what makes the order-domain filter well-posed.

    Because the result is truncated to the matched span rather than extrapolated beyond it,
    no constant-rate hold is needed outside that span -- which is what the earlier
    ~860,000 Hz blowups came from.

    Returns:
        ``(ias, sl)`` -- IAS in Hz on the recording's sample grid over the matched span, and
        the ``slice`` of that grid, to be applied to the vibration channels as well.
    """
    angle = _theta(idx_clean, template)
    n_per_rev = len(template)
    dtheta = 1 / n_per_rev

    theta_grid = np.arange(angle[0], angle[-1], dtheta)
    t_of_theta = PchipInterpolator(angle, t_clean)(theta_grid)
    ias_angle_domain = dtheta / np.diff(t_of_theta)
    t_mid = (t_of_theta[:-1] + t_of_theta[1:]) / 2

    ias_filt = order_domain_lowpass(ias_angle_domain, _CUTOFF_ORDER, n_per_rev)

    t = np.arange(signal_len) / fs
    sl = slice(int(np.searchsorted(t, t_mid[0], side="left")), int(np.searchsorted(t, t_mid[-1], side="right")))
    return np.clip(np.interp(t[sl], t_mid, ias_filt), 0, None), sl


# ───────────────────────── dataset preparation ─────────────────────────


def dl_planetary_gearbox(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # unused; the framework only calls this when the dataset is missing or forced
) -> None:
    """Download, preprocess (zebra tape → sun-shaft IAS), split, and add disturbed test sets.

    Runs in two passes over the archive because the stripe template is pooled across every
    recording, and the span each recording is truncated to in turn depends on that template.
    The first pass reads only the two tacho channels and keeps just their edge times; the
    second reads only the two vibration channels. Each ~2.5 GB .MAT is therefore read once
    per pass and never held in full alongside the others.
    """
    save_path = Path(save_path)
    for split in ("train", "valid", "test", "test_wear"):
        (save_path / split).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        download_and_unpack(_INFO, temp_dir)

        # The discovery glob is exactly `*_crack/*.MAT` -- other .MAT files in the archive are
        # not used. Sorted, because `pool_zebra_templates` seeds its reference from this order.
        mat_files = sorted(temp_dir.rglob("*_crack/*.MAT"), key=lambda p: p.stem)

        edges = {}
        for mat_file in tqdm(mat_files, desc="Reading tacho channels", unit="file"):
            mat_data = scipy.io.loadmat(mat_file, variable_names=["Channel_5_Data", "Channel_6_Data", "File_Header"])
            fs = _parse_fs(mat_data)
            t_ref = rising_edge_times(mat_data["Channel_6_Data"].squeeze(), fs)
            t_zebra = rising_edge_times(mat_data["Channel_5_Data"].squeeze(), fs)
            phase, n_revs = _zebra_phase(t_ref, t_zebra)
            edges[mat_file] = dict(
                fs=fs,
                signal_len=len(mat_data["Channel_6_Data"]),
                t_ref=t_ref,
                t_zebra=t_zebra,
                phase=phase,
                n_revs=n_revs,
            )
            del mat_data

        template, counts, pooled_hist = pool_zebra_templates(
            [e["phase"] for e in edges.values()], [e["n_revs"] for e in edges.values()]
        )
        total_revs = sum(e["n_revs"] for e in edges.values())
        print(
            f"Pooled stripe template across {len(mat_files)} recordings: {len(template)} stripes; "
            f"per-stripe detection rate min={counts.min() / total_revs:.1%} "
            f"median={np.median(counts) / total_revs:.1%} max={counts.max() / total_revs:.1%}"
        )
        if len(template) <= 2 * _CUTOFF_ORDER:
            raise RuntimeError(
                f"pooled template resolved only {len(template)} stripes/rev, giving an order-domain "
                f"Nyquist of {len(template) / 2} -- too low for the {_CUTOFF_ORDER}-order cutoff"
            )

        for mat_file in tqdm(mat_files, desc="Reconstructing IAS", unit="file"):
            e = edges[mat_file]
            t_matched, idx_matched = match_zebra_to_template(
                e["t_zebra"], e["t_ref"], template, phase_offset=zebra_phase_offset(e["phase"], pooled_hist)
            )
            t_clean, idx_clean = drop_zebra_outliers(t_matched, idx_matched, template)
            ias, sl = reconstruct_ias(t_clean, idx_clean, template, e["fs"], e["signal_len"])

            mat_data = scipy.io.loadmat(mat_file, variable_names=["Channel_2_Data", "Channel_3_Data"])
            signals = {
                "IAS": ias,
                "Acc_Carrier": mat_data["Channel_2_Data"].squeeze()[sl] / 9 * 9.81,
                "Acc_Sun": mat_data["Channel_3_Data"].squeeze()[sl] / 9 * 9.81,
            }
            del mat_data
            stem = mat_file.stem
            if any(t in stem for t in _TEST_WEAR_TYPES):
                target_subdir = "test_wear"
            elif any(t in stem for t in _TEST_BASIC_TYPES):
                target_subdir = "test"
            elif any(t in stem for t in _VALID_TYPES):
                target_subdir = "valid"
            else:
                target_subdir = "train"
            # gear ratio: (planet carrier, sun, mesh) relative to the SUN, which is the shaft the
            # zebra tape -- and hence the IAS -- refers to. The mesh entry is unchanged from the
            # carrier-referenced list: 62 * 13 / 75 = 13 * (1 - 13/75) was already the mesh order
            # relative to the sun (relative to the carrier it would be the ring tooth count, 62).
            save_signals_hdf5(
                signals,
                save_path / target_subdir / f"{mat_file.stem}.hdf5",
                fs=e["fs"],
                gear_ratio=[
                    1 / _SUN_PER_CARRIER_REV,
                    1,
                    _RING_TEETH * _SUN_TEETH / (_SUN_TEETH + _RING_TEETH),
                ],
                ias_bandwidth_hz=_IAS_BANDWIDTH_HZ,
            )

    write_disturbed_test_sets(save_path, vib_keys=["Acc_Carrier", "Acc_Sun"])


# version 2: IAS reconstructed from the sun-shaft zebra tape (was the 1PR carrier pickup with a
# savgol + fixed 12.5 Hz time-domain low-pass). The label is now sun-referenced, so it is
# ~5.77x the previously shipped values.
# version 3: corrected circular stripe-template median.
planetary_gearbox_dataset = Dataset("planetary_gearbox", prepare=dl_planetary_gearbox, version="3")

_planetary_gearbox = dict(
    u_cols=["Acc_Carrier", "Acc_Sun"],
    y_cols=["IAS"],
    train=[(planetary_gearbox_dataset, "train/*.hdf5")],
    valid=[(planetary_gearbox_dataset, "valid/*.hdf5")],
    test_sets=ias_test_sets(planetary_gearbox_dataset),
)

BenchmarkPlanetaryGearbox_Estimation = BenchmarkSpec(
    name="BenchmarkPlanetaryGearbox_Estimation",
    # window_sec = largest window any upstream method needed (Ref-FFT-LSTM 2.70 s),
    # rounded to 2.7 s; the per-file fs (this dataset varies it) sizes the window in samples. See ias/__init__.
    task=WindowedEstimation(window_sec=2.7),
    **_planetary_gearbox,
)

BenchmarkPlanetaryGearbox_GridwiseEstimation = BenchmarkSpec(
    name="BenchmarkPlanetaryGearbox_GridwiseEstimation",
    # window_sec=3.0: the largest single window across every upstream method's search space
    # over all four IAS datasets (unlike the per-dataset WindowedEstimation windows above,
    # this one is kept uniform — it's only a context guarantee, not a tuned averaging window).
    # step_sec: the one dataset NOT sampled at Nyquist. _IAS_BANDWIDTH_HZ = 598.5 Hz would need
    # 0.835 ms, i.e. 1.14M query points per file and ~656 MB of diagnostics per run. Capped at
    # 3 ms instead, which fully resolves 15 orders while the sun is below 11.1 Hz -- 60.5% of
    # recorded time. That is a deliberate trade: pooled MAE is an unbiased estimate of the mean
    # absolute error at ANY grid spacing, so a model that fails to track fast content is still
    # penalised at every query point; and 3 ms is already finer than the finest hop any
    # benchmarked method can emit (5 ms for MOPA/ViBES at their smallest window and largest
    # overlap), so the grid never limits what a method can demonstrate. What it does cost is
    # aliasing-free spectral analysis of the stored diagnostics -- hence step_sec is recorded
    # alongside the results so that limitation is visible rather than silent.
    task=GridwiseEstimation(window_sec=3.0, step_sec=0.003),
    **_planetary_gearbox,
)

# Dense free-run sibling (framework Simulation task): the model predicts one IAS estimate
# per sample over the full recording — the window lives in the model (e.g. a sliding window),
# not the benchmark — scored per-sample MAE in Hz. Same data and test sets as the windowed task.
BenchmarkPlanetaryGearbox_Simulation = BenchmarkSpec(
    name="BenchmarkPlanetaryGearbox_Simulation",
    task=Simulation(metric=mae),
    **_planetary_gearbox,
)
