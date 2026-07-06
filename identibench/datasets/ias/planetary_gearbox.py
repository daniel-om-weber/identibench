"""Planetary gearbox IAS estimation dataset (figshare 28992879)."""

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
from tqdm import tqdm

from ...benchmark import BenchmarkSpec, Simulation, WindowedEstimation, GridwiseEstimation
from ...dataset import Dataset
from ...metrics import mae
from ._common import (
    DatasetInfo,
    analog_pulse_to_ias,
    download_and_unpack,
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


def _parse_fs(mat_data: dict) -> float:
    """Per-file sampling rate from the .MAT header (verbatim nested-index + decimal-comma parse)."""
    return float(
        np.fromstring(mat_data["File_Header"]["SampleFrequency"][0][0][0].replace(",", "."), sep=";").squeeze()
    )


def dl_planetary_gearbox(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # unused; the framework only calls this when the dataset is missing or forced
) -> None:
    """Download, preprocess (encoder → IAS), split, and add disturbed test sets."""
    save_path = Path(save_path)
    for split in ("train", "valid", "test", "test_wear"):
        (save_path / split).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        download_and_unpack(_INFO, temp_dir)

        # The discovery glob is exactly `*_crack/*.MAT` — other .MAT files in the archive are not used.
        # The split target is decided by the source stem alone, so every file is
        # written once, directly into its final split directory.
        mat_files = list(temp_dir.rglob("*_crack/*.MAT"))
        for mat_file in tqdm(mat_files, desc="Preprocessing .MAT files", unit="file"):
            if any(t in mat_file.stem for t in _TEST_WEAR_TYPES):
                target_subdir = "test_wear"
            elif any(t in mat_file.stem for t in _TEST_BASIC_TYPES):
                target_subdir = "test"
            elif any(t in mat_file.stem for t in _VALID_TYPES):
                target_subdir = "valid"
            else:
                target_subdir = "train"
            mat_data = scipy.io.loadmat(mat_file)
            fs = _parse_fs(mat_data)
            signals = {
                "IAS": analog_pulse_to_ias(mat_data["Channel_6_Data"].squeeze(), fs, pulses_per_revolution=1),
                "Acc_Carrier": mat_data["Channel_2_Data"].squeeze() / 9 * 9.81,
                "Acc_Sun": mat_data["Channel_3_Data"].squeeze() / 9 * 9.81,
            }
            # gear ratio: (planet carrier, sun, mesh) relative to planet carrier (with mag. pickup)
            save_signals_hdf5(
                signals,
                save_path / target_subdir / f"{mat_file.stem}.hdf5",
                fs=fs,
                gear_ratio=[1, (13 + 62) / 13, 62 * 13 / (13 + 62)],
            )

    write_disturbed_test_sets(save_path, vib_keys=["Acc_Carrier", "Acc_Sun"])


planetary_gearbox_dataset = Dataset("planetary_gearbox", prepare=dl_planetary_gearbox)

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
    task=GridwiseEstimation(window_sec=3.0, step_sec=0.1),
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
