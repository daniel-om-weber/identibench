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
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from ...dataset import Dataset
from ...utils import download_file, extract_archive, hdf_files_from_path

# Synthetic-noise SNR levels of the disturbed test-set copies (dB).
DISTURBANCE_LEVELS = [15, 7.5, 0]


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


def analog_pulse_to_ias(signal: np.ndarray, fs: float, pulses_per_revolution: int = 1) -> np.ndarray:
    """Convert an analog encoder-pulse signal to instantaneous angular speed (Hz).

    Verbatim port of the IAS benchmark preprocessing: zero crossings of the
    normalized pulse train are interpolated sub-sample, converted to per-pulse
    speeds, smoothed over one revolution, resampled to the signal rate, and
    low-pass filtered at 12.5 Hz.
    """
    signal = np.array(signal)
    signal = ((signal - np.min(signal)) / (np.max(signal) - np.min(signal))) * 2 - 1  # normalize to -1 to 1

    # find zero crossings and calculate IAS
    peaks = np.argwhere(np.diff(np.sign(signal)) > 0).squeeze()
    # interpolate zerocrossings linearly
    peaks = peaks - signal[peaks] / (signal[peaks + 1] - signal[peaks])

    pulse_times = peaks / fs
    periods = np.diff(pulse_times)
    ias = 1 / periods / pulses_per_revolution

    # move the value to middle of last period to correct for the fact that the zero crossing is at the end of the period
    corrected_t = pulse_times[1:] - periods / 2

    # smooth over 1 rotation
    ias = convolve1d(ias, np.ones(pulses_per_revolution) / pulses_per_revolution, mode="nearest")

    # interpolate IAS to original fs
    interper = interp1d(corrected_t, ias, kind="linear", fill_value="extrapolate")
    ias_full = interper(np.arange(len(signal)) / fs)
    ias_full[ias_full < 0] = 0  # remove any negative values
    # smoothing filter at 12.5 Hz
    a, b = butter(2, 12.5 / (fs / 2), btype="low")
    ias_full = filtfilt(a, b, ias_full)

    return ias_full


def save_signals_hdf5(signals: dict[str, np.ndarray], path: Path, fs: float, gear_ratio) -> None:
    """Write the signal channels as float32 datasets with ``fs`` and ``gear_ratio`` attrs.

    Replaces the upstream ``save_to_hdf5`` (attr name normalized
    ``sampling_rate`` → ``fs``; identibench-style float32 storage).
    ``gear_ratio`` may be a scalar or a list (kinematics pass-through metadata).
    """
    with h5py.File(path, "w") as f:
        for key, value in signals.items():
            f.create_dataset(key, data=np.asarray(value), dtype="f4")
        f.attrs["fs"] = fs
        f.attrs["gear_ratio"] = gear_ratio


def add_disturbances(
    sig: np.ndarray,
    fs: float,
    target_snr_db: float,
    rng: np.random.Generator,
    target_percentages: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> np.ndarray:
    """Add mains-hum, PWM, Gaussian, and Lévy-stable noise at a target SNR.

    Seeded rewrite of the upstream generator: every random draw (the two phase
    offsets, the Gaussian noise, and the Lévy component) comes from ``rng``, so
    the output is fully reproducible. This intentionally differs from the
    upstream output, which seeded only the Lévy term (with a constant seed, so
    every file got the *same* Lévy realization) and drew the rest from global
    state.
    """
    netnoise_freqs = [50 * i for i in range(1, 4)]  # 50Hz and its harmonics
    pwm_freqs = [2.5e3]

    disturbances = []
    p_signal_db = 10 * np.log10(np.mean(sig**2))
    # SNR = P_signal_db - P_noise_db  =>  P_noise_db = P_signal_db - SNR
    p_total_noise_desired = 10 ** ((p_signal_db - target_snr_db) / 10)  # Convert dB back to linear power

    t = np.arange(len(sig)) / fs

    # --- mains hum (50 Hz + harmonics) ---
    mains_hum = np.zeros_like(sig)
    for freq in netnoise_freqs:
        phi = rng.uniform(0, 2 * np.pi)
        mains_hum += (
            np.sin(2 * np.pi * freq * t + phi) * np.sqrt(p_total_noise_desired * target_percentages[0]) * 50 / freq
        )
    disturbances.append(mains_hum)

    # --- PWM ---
    pwm_hum = np.zeros_like(sig)
    for freq in pwm_freqs:
        phi = rng.uniform(0, 2 * np.pi)
        pwm_hum += np.sqrt(p_total_noise_desired * target_percentages[1]) * scipy.signal.square(
            2 * np.pi * freq * t + phi
        )
    disturbances.append(pwm_hum)

    # --- white Gaussian noise ---
    disturbances.append(rng.normal(0, np.sqrt(p_total_noise_desired * target_percentages[2]), size=sig.shape))
    # --- alpha-stable (impulsive) noise ---
    disturbances.append(
        stats.levy_stable.rvs(
            alpha=1.2,
            beta=0,
            loc=0,
            scale=np.sqrt(p_total_noise_desired * target_percentages[3]),
            size=sig.shape,
            random_state=rng,
        )
    )

    # lowpass filter each noise to be within fs/2 to avoid aliasing
    disturbances = [
        scipy.signal.filtfilt(*scipy.signal.butter(4, fs / 2.1, btype="low", fs=fs), disturbance)
        for disturbance in disturbances
    ]

    # scale each noise to meet its percentage of the total noise power
    disturbances_scaled = [
        disturbance * np.sqrt(p_total_noise_desired * percentage / np.mean(disturbance**2))
        for disturbance, percentage in zip(disturbances, target_percentages)
    ]

    return sig + np.sum(np.stack(disturbances_scaled, axis=0), axis=0)


def _disturbance_rng(base_seed: int, stem: str, level: float) -> np.random.Generator:
    """Per-(file, level) deterministic generator, independent of iteration order."""
    return np.random.default_rng(base_seed ^ zlib.crc32(f"{stem}|{level}".encode()))


def write_disturbed_test_sets(
    dataset_path: Path,
    vib_keys: list[str],
    noise_levels: list[float] = DISTURBANCE_LEVELS,
    base_seed: int = 0,
) -> None:
    """Copy the basic test set into ``test_disturbed_<level>dB/`` dirs with added noise.

    Reads ``dataset_path/test`` and writes one disturbed copy per noise level,
    corrupting the ``vib_keys`` channels in place. ``fs`` is read per file from
    its attrs (the planetary gearbox has per-file rates; upstream reused one
    rate for the whole dataset). Each (file, level) pair gets its own
    deterministic seed, so re-runs are byte-identical regardless of order.
    """
    test_files = hdf_files_from_path(dataset_path / "test")
    total_operations = len(noise_levels) * len(test_files)
    with tqdm(total=total_operations, desc="Creating disturbed test sets", unit="file") as pbar:
        for level in noise_levels:
            dest_dir = dataset_path / f"test_disturbed_{level}dB"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file in test_files:
                dest_path = dest_dir / file.name
                shutil.copy(str(file), str(dest_path))
                rng = _disturbance_rng(base_seed, file.stem, level)
                with h5py.File(dest_path, "r+") as f:
                    fs = float(f.attrs["fs"])
                    for vib_key in vib_keys:
                        f[vib_key][:] = add_disturbances(f[vib_key][:], fs, target_snr_db=level, rng=rng)
                pbar.update(1)


def ias_test_sets(dataset: Dataset, wear: bool = True) -> dict[str, list[tuple[Dataset, str]]]:
    """The named test conditions of an IAS spec as patterns, headline (``basic``) first.

    Maps each condition to its directory: ``basic`` → ``test/``, ``wear`` →
    ``test_wear/`` (only where the dataset has a wear condition), and one
    ``disturbed_<level>dB`` → ``test_disturbed_<level>dB/`` per level.
    """
    dirs = {"basic": "test"}
    if wear:
        dirs["wear"] = "test_wear"
    dirs |= {f"disturbed_{level}dB": f"test_disturbed_{level}dB" for level in DISTURBANCE_LEVELS}
    return {name: [(dataset, f"{subdir}/*.hdf5")] for name, subdir in dirs.items()}
