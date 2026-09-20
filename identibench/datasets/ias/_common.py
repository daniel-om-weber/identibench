"""Shared download/prep helpers for the IAS (instantaneous angular speed) datasets."""

import shutil
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.signal
from scipy import stats
from scipy.signal import butter, buttord, sosfiltfilt
from tqdm import tqdm

from ...dataset import Dataset
from ...utils import download_file, extract_archive, hdf_files_from_path

# Synthetic-noise SNR levels of the disturbed test-set copies (dB).
DISTURBANCE_LEVELS = [15, 7.5, 0, -7.5]

# Disturbed-test-set variants: "combined" sums all three components below (noise power
# split evenly across them); the other three isolate one component at 100% of the noise
# power. Order here is also the per-level order used by `ias_test_sets` (combined, the
# per-level headline, first).
DISTURBANCE_VARIANTS: tuple[str, ...] = ("combined", "harmonic", "gaussian", "impulsive")

# What `add_disturbances(disturbance_types=...)` receives for each variant above.
# "combined" is never passed to `add_disturbances` itself -- it is purely an
# orchestration-layer name meaning "all three real component types at once".
_VARIANT_DISTURBANCE_TYPES: dict[str, tuple[str, ...]] = {
    "combined": ("harmonic", "gaussian", "impulsive"),
    "harmonic": ("harmonic",),
    "gaussian": ("gaussian",),
    "impulsive": ("impulsive",),
}

# Harmonic disturbance: models a nearby machine running at a different RPM as a few
# tones at ratio * IAS(t) and its low multiples, amplitude falling off as 1/k. The base
# frequency wanders with a damped copy of the file's own IAS trajectory (scaled by
# HARMONIC_VARIATION_FACTOR) rather than sitting still, so it can't be notch-filtered out.
HARMONIC_IAS_RATIO = 1.5
HARMONIC_COUNT = 3
HARMONIC_VARIATION_FACTOR = 0.1


@dataclass
class DatasetInfo:
    """Source archive of one IAS dataset."""

    name: str
    zip_url: str
    download_headers: dict[str, str] | None = None


def _require_sklearn():
    """The stratified splits need scikit-learn; import lazily with an install hint."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            'scikit-learn is required for the IAS datasets. Install it with: pip install "identibench[ias]"'
        ) from e
    return train_test_split


def download_and_unpack(dataset_info: DatasetInfo, output_dir: Path, nested_zip: bool = False) -> Path:
    """Download the dataset zip and extract it into ``output_dir``.

    Args:
        dataset_info: Source archive description.
        output_dir: Extraction target; created if it does not exist.
        nested_zip: Set for Parallel_Gearbox, whose zip contains another zip
            with the actual data that must be extracted as well. macOS resource
            forks (``__MACOSX``, ``._*``) are ignored when locating inner zips.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "dataset.zip"
        download_file(
            dataset_info.zip_url, archive_path, headers=dataset_info.download_headers or {"User-Agent": "Mozilla/5.0"}
        )
        print("Download completed, extracting .zip ...", end="")
        extract_archive(archive_path, output_dir)
        if nested_zip:
            inner_zips = sorted(
                p for p in output_dir.rglob("*.zip") if "__MACOSX" not in p.parts and not p.name.startswith("._")
            )
            if not inner_zips:
                raise FileNotFoundError(f"nested_zip=True but no inner .zip found under {output_dir}")
            for inner_zip in inner_zips:
                extract_archive(inner_zip, output_dir)
        print("complete.")
    return output_dir


def rising_edge_times(signal: np.ndarray, fs: float) -> np.ndarray:
    """Sub-sample times (s) of the rising edges of an analog pulse train.

    The signal is normalized to [-1, 1] and each upward zero crossing is located to
    sub-sample resolution by linear interpolation between the two bracketing samples.

    Args:
        signal: Raw analog tacho/encoder channel.
        fs: Sampling rate of ``signal`` in Hz.

    Returns:
        Edge times in seconds, ascending.
    """
    signal = np.asarray(signal, dtype=float)
    signal = ((signal - np.min(signal)) / (np.max(signal) - np.min(signal))) * 2 - 1
    # `signal >= 0` rather than `np.sign(signal)`: the sign form steps 0 -> +1 and -1 -> 0
    # separately, so a sample landing exactly on zero registers as two rising edges and the
    # pulse is counted twice.
    peaks = np.argwhere(np.diff((signal >= 0).astype(np.int8)) > 0).ravel()
    peaks = peaks - signal[peaks] / (signal[peaks + 1] - signal[peaks])
    return peaks / fs


def order_domain_lowpass(x: np.ndarray, cutoff_order: float, pulses_per_revolution: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass of an angle-domain sequence, cut off in shaft orders.

    ``x`` must be sampled uniformly in *angle* at ``pulses_per_revolution`` samples per
    revolution. That value is therefore the filter's sampling rate, which lets the cutoff be
    stated directly in orders (events per revolution) and makes the smoothing independent of
    how fast the shaft happens to be turning.

    The stopband is ``min(4 * cutoff_order, 0.45 * ppr)`` at 60 dB, with 1 dB of passband
    ripple. Deriving it from the cutoff rather than from the Nyquist is what keeps the
    datasets comparable: a fixed ``0.25 * ppr`` stopband would hand the ball bearing a
    2nd-order filter (its Nyquist sits 54x above its cutoff) and the parallel gearbox a
    6th-order one, for no physical reason.

    Args:
        x: Angle-domain sequence, uniformly sampled.
        cutoff_order: Cutoff in orders; must be below the order-domain Nyquist.
        pulses_per_revolution: Samples of ``x`` per shaft revolution.

    Returns:
        The filtered sequence, same shape as ``x``.

    Raises:
        ValueError: If the cutoff is outside ``(0, ppr / 2)``, or the stopband derived from
            it collapses onto the passband.
    """
    nyquist = 0.5 * pulses_per_revolution
    if not 0 < cutoff_order < nyquist:
        raise ValueError(
            f"cutoff_order={cutoff_order} must lie strictly between 0 and the order-domain "
            f"Nyquist {nyquist} (pulses_per_revolution={pulses_per_revolution})"
        )
    stopband = min(4 * cutoff_order, 0.45 * pulses_per_revolution)
    if stopband <= cutoff_order:
        raise ValueError(
            f"stopband {stopband} collapsed onto cutoff_order={cutoff_order}; "
            f"pulses_per_revolution={pulses_per_revolution} is too small for this cutoff"
        )
    butter_order = buttord(cutoff_order, stopband, 1, 60, fs=pulses_per_revolution)[0]
    sos = butter(butter_order, cutoff_order, btype="low", output="sos", fs=pulses_per_revolution)
    return sosfiltfilt(sos, x)


def encoder_pulse_to_ias(
    signal: np.ndarray, fs: float, pulses_per_revolution: int, cutoff_order: float
) -> tuple[np.ndarray, slice]:
    """Instantaneous angular speed (Hz) from an analog encoder pulse train.

    Rising edges give per-pulse periods; their reciprocal is the rate. Each period is
    centred between its two edges, i.e. at angle ``(k - 0.5) / ppr``, so the rate sequence
    is *already* uniform in angle and :func:`order_domain_lowpass` applies to it directly,
    with no angle-domain resampling needed. It is then interpolated onto the recording's own
    sample grid.

    Args:
        signal: Raw analog tacho/encoder channel.
        fs: Sampling rate of ``signal`` in Hz.
        pulses_per_revolution: Encoder pulses per revolution of the measured shaft.
        cutoff_order: Order-domain cutoff, in orders of that shaft.

    Returns:
        ``(ias, sl)`` -- the IAS in Hz (rev/s), truncated to the span where it is genuinely
        measured, and the ``slice`` of the original sample grid it was cut to. The caller
        must apply ``sl`` to every other channel of the recording to keep them aligned; the
        IAS is not extrapolated beyond the first and last pulse.
    """
    pulse_times = rising_edge_times(signal, fs)
    periods = np.diff(pulse_times)
    corrected_t = pulse_times[1:] - periods / 2
    ias = order_domain_lowpass(1 / periods / pulses_per_revolution, cutoff_order, pulses_per_revolution)

    t = np.arange(len(signal)) / fs
    sl = slice(
        int(np.searchsorted(t, corrected_t[0], side="left")),
        int(np.searchsorted(t, corrected_t[-1], side="right")),
    )
    return np.clip(np.interp(t[sl], corrected_t, ias), 0, None), sl


def save_signals_hdf5(
    signals: dict[str, np.ndarray], path: Path, fs: float, gear_ratio, ias_bandwidth_hz: float
) -> None:
    """Write the signal channels as float32 datasets with the dataset's metadata attrs.

    Replaces the upstream ``save_to_hdf5`` (attr name normalized
    ``sampling_rate`` → ``fs``; identibench-style float32 storage).
    ``gear_ratio`` may be a scalar or a list (kinematics pass-through metadata).

    ``ias_bandwidth_hz`` is the highest frequency the ``IAS`` label retains,
    ``IAS_max * min(cutoff_order, ppr/2)`` in Hz. It is a property of how the label was
    built, not of the recording, so it is the same for every file of a dataset. It is
    recorded here because two separate decisions depend on it and would otherwise each
    hardcode it: the benchmark's evaluation-grid spacing (``GridwiseEstimation.step_sec``)
    and, downstream, the lowest model sample rate worth searching — a model sampled below
    ``2 * ias_bandwidth_hz`` cannot represent the target it is scored against.
    """
    with h5py.File(path, "w") as f:
        for key, value in signals.items():
            f.create_dataset(key, data=np.asarray(value), dtype="f4")
        f.attrs["fs"] = fs
        f.attrs["gear_ratio"] = gear_ratio
        f.attrs["ias_bandwidth_hz"] = float(ias_bandwidth_hz)


def add_disturbances(
    sig: np.ndarray,
    fs: float,
    target_snr_db: float,
    rng: np.random.Generator,
    ias_hz: np.ndarray,
    disturbance_types: tuple[str, ...] = ("harmonic", "gaussian", "impulsive"),
) -> np.ndarray:
    """Add a mix of noise components at a target SNR, each drawn from ``rng``.

    ``disturbance_types`` selects which of ``"harmonic"`` (a nearby machine running
    at a different, wandering RPM), ``"gaussian"`` (white noise), and ``"impulsive"``
    (Lévy-stable noise) to generate; the noise power implied by ``target_snr_db`` is
    split evenly across however many types are requested, so a single-element tuple
    gives that type 100% of the noise power. ``ias_hz`` is that recording's own
    per-sample IAS trace (same shape as ``sig``), which anchors the harmonic
    component's base frequency to ``HARMONIC_IAS_RATIO`` times this file's own
    (wandering) IAS instead of a fixed absolute Hz value, so its relative spectral
    position is comparable across datasets with very different shaft speeds.

    Seeded rewrite of the upstream generator: every random draw (the harmonic phase
    offsets, the Gaussian noise, and the Lévy component) comes from ``rng``, so the
    output is fully reproducible. This intentionally differs from the upstream
    output, which seeded only the Lévy term (with a constant seed, so every file got
    the *same* Lévy realization) and drew the rest from global state.
    """
    p_signal_db = 10 * np.log10(np.mean(sig**2))
    # SNR = P_signal_db - P_noise_db  =>  P_noise_db = P_signal_db - SNR
    p_total_noise_desired = 10 ** ((p_signal_db - target_snr_db) / 10)  # Convert dB back to linear power
    percentage = 1.0 / len(disturbance_types)

    disturbances = []
    for kind in ("harmonic", "gaussian", "impulsive"):
        if kind not in disturbance_types:
            continue
        if kind == "harmonic":
            # Base frequency wanders with a damped copy of this file's own IAS
            # trajectory (rather than sitting still), so it can't be notch-filtered
            # out; instantaneous frequency is integrated into a phase track.
            ias_hz_f64 = np.asarray(ias_hz, dtype=np.float64)
            mean_ias = np.mean(ias_hz_f64)
            freq_t = HARMONIC_IAS_RATIO * mean_ias + HARMONIC_VARIATION_FACTOR * (ias_hz_f64 - mean_ias)
            # float64 throughout: cumsum over long recordings (e.g. 200 kHz for 10s)
            # accumulates too much rounding error in float32 to stay phase-accurate.
            phase_t = np.cumsum(2 * np.pi * freq_t / fs)
            harmonic = np.zeros_like(sig)
            for k in range(1, HARMONIC_COUNT + 1):
                phi0 = rng.uniform(0, 2 * np.pi)
                harmonic += np.sin(k * phase_t + phi0) / k
            disturbances.append(harmonic)
        elif kind == "gaussian":
            disturbances.append(rng.normal(0, 1, size=sig.shape))
        elif kind == "impulsive":
            # TODO: why alpha=1.2?
            disturbances.append(
                stats.levy_stable.rvs(alpha=1.2, beta=0, loc=0, scale=1, size=sig.shape, random_state=rng)
            )

    # lowpass filter each noise to be within fs/2 to avoid aliasing
    disturbances = [
        scipy.signal.filtfilt(*scipy.signal.butter(4, fs / 2.1, btype="low", fs=fs), disturbance)
        for disturbance in disturbances
    ]

    # scale each noise to meet its (even) share of the total noise power
    disturbances_scaled = [
        disturbance * np.sqrt(p_total_noise_desired * percentage / np.mean(disturbance**2))
        for disturbance in disturbances
    ]

    return sig + np.sum(np.stack(disturbances_scaled, axis=0), axis=0)


def _disturbance_rng(base_seed: int, stem: str, level: float, variant: str) -> np.random.Generator:
    """Per-(file, level, variant) deterministic generator, independent of iteration order.

    The 4 variants for a given (file, level) deliberately do not share noise
    realizations with each other -- each draws independently from its own seed.
    """
    return np.random.default_rng(base_seed ^ zlib.crc32(f"{stem}|{level}|{variant}".encode()))


def write_disturbed_test_sets(
    dataset_path: Path,
    vib_keys: list[str],
    noise_levels: list[float] = DISTURBANCE_LEVELS,
    base_seed: int = 0,
) -> None:
    """Copy the basic test set into disturbed variant dirs with added noise.

    Reads ``dataset_path/test`` and writes, per noise level, one directory per
    entry in ``DISTURBANCE_VARIANTS`` --
    ``test_disturbed_<level>dB_combined/`` (all three noise components, power
    split evenly) plus ``_harmonic/``, ``_gaussian/``, ``_impulsive/`` (each
    isolating one component at the full target SNR) -- corrupting the
    ``vib_keys`` channels in place. ``fs`` is read per file from its attrs (the
    planetary gearbox has per-file rates; upstream reused one rate for the
    whole dataset). Each file's IAS trace is read once from the pristine source
    file and reused across every level/variant/vib_key for that file, since it
    is level- and variant-invariant. Each (file, level, variant) triple gets its
    own deterministic seed, so re-runs are byte-identical regardless of order,
    and the 4 variants for a given (file, level) do not share noise
    realizations with each other.
    """
    test_files = hdf_files_from_path(dataset_path / "test")
    ias_by_stem = {}
    for file in test_files:
        with h5py.File(file, "r") as f:
            ias_by_stem[file.stem] = f["IAS"][:]

    total_operations = len(noise_levels) * len(DISTURBANCE_VARIANTS) * len(test_files)
    with tqdm(total=total_operations, desc="Creating disturbed test sets", unit="file") as pbar:
        for level in noise_levels:
            for variant in DISTURBANCE_VARIANTS:
                dest_dir = dataset_path / f"test_disturbed_{level}dB_{variant}"
                dest_dir.mkdir(parents=True, exist_ok=True)
                for file in test_files:
                    dest_path = dest_dir / file.name
                    shutil.copy(str(file), str(dest_path))
                    rng = _disturbance_rng(base_seed, file.stem, level, variant)
                    with h5py.File(dest_path, "r+") as f:
                        fs = float(f.attrs["fs"])
                        for vib_key in vib_keys:
                            f[vib_key][:] = add_disturbances(
                                f[vib_key][:],
                                fs,
                                target_snr_db=level,
                                rng=rng,
                                ias_hz=ias_by_stem[file.stem],
                                disturbance_types=_VARIANT_DISTURBANCE_TYPES[variant],
                            )
                    pbar.update(1)


def ias_test_sets(dataset: Dataset, wear: bool = True) -> dict[str, list[tuple[Dataset, str]]]:
    """The named test conditions of an IAS spec as patterns, headline (``basic``) first.

    Maps each condition to its directory: ``basic`` → ``test/``, ``wear`` →
    ``test_wear/`` (only where the dataset has a wear condition), and one
    ``disturbed_<level>dB_<variant>`` → ``test_disturbed_<level>dB_<variant>/``
    per level in ``DISTURBANCE_LEVELS`` × variant in ``DISTURBANCE_VARIANTS``
    (``combined`` first among the variants, as the per-level headline).
    """
    dirs = {"basic": "test"}
    if wear:
        dirs["wear"] = "test_wear"
    dirs |= {
        f"disturbed_{level}dB_{variant}": f"test_disturbed_{level}dB_{variant}"
        for level in DISTURBANCE_LEVELS
        for variant in DISTURBANCE_VARIANTS
    }
    return {name: [(dataset, f"{subdir}/*.hdf5")] for name, subdir in dirs.items()}
