"""Ball bearing IAS estimation dataset (Mendeley v43hmbwxpm)."""

__all__ = [
    "ball_bearing_dataset",
    "dl_ball_bearing",
    "BenchmarkBallBearing_Estimation",
    "BenchmarkBallBearing_Simulation",
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
    name="Ball_Bearing",
    zip_url="https://data.mendeley.com/public-api/zip/v43hmbwxpm/download/2",
)

# Fixed upstream split (verbatim): file stems of the basic test and valid sets;
# C* recordings (worn bearings) form the out-of-distribution wear set.
_BASIC_TEST_FILES = {"H-D-1", "H-C-3", "I-A-3", "I-B-1", "O-B-2", "B-A-2", "B-C-3", "O-D-2", "O-A-1"}
_VALID_FILES = {"H-A-2", "H-C-1", "I-D-1", "O-A-3", "O-D-1", "B-D-1", "B-B-3", "I-C-2", "I-B-2"}


def dl_ball_bearing(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # unused; the framework only calls this when the dataset is missing or forced
) -> None:
    """Download, preprocess (encoder → IAS), split, and add disturbed test sets.

    The vibration channel is renamed ``vibration`` → ``Acc_x`` for consistency
    with the other IAS datasets. The download is large (200 kHz recordings).
    """
    save_path = Path(save_path)
    for split in ("train", "valid", "test", "test_wear"):
        (save_path / split).mkdir(parents=True, exist_ok=True)

    fs = 2e5
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        download_and_unpack(_INFO, temp_dir)

        # The split target is decided by the source stem alone, so every file is
        # written once, directly into its final split directory.
        mat_files = sorted(temp_dir.rglob("*.mat"))
        for file in tqdm(mat_files, desc="Preprocessing .mat files", unit="file"):
            if file.name[0] == "C":
                target_subdir = "test_wear"
            elif file.stem in _BASIC_TEST_FILES:
                target_subdir = "test"
            elif file.stem in _VALID_FILES:
                target_subdir = "valid"
            else:
                target_subdir = "train"
            mat = scipy.io.loadmat(file)
            signals = {
                # recalculated, so the IAS is from the middle shaft (which the vibration sensor is
                # mounted on, whereas the speed sensor is mounted on the input shaft)
                "IAS": np.asarray(
                    analog_pulse_to_ias(np.array(mat.get("Channel_2")).squeeze(), fs, pulses_per_revolution=1024)
                ).squeeze(),
                "Acc_x": np.asarray(np.array(mat.get("Channel_1")).squeeze() * 10 * 9.81).squeeze(),
            }
            save_signals_hdf5(signals, save_path / target_subdir / f"{file.stem}.hdf5", fs=fs, gear_ratio=1)

    write_disturbed_test_sets(save_path, vib_keys=["Acc_x"])


ball_bearing_dataset = Dataset("ball_bearing", prepare=dl_ball_bearing)

_ball_bearing = dict(
    u_cols=["Acc_x"],
    y_cols=["IAS"],
    train=[(ball_bearing_dataset, "train/*.hdf5")],
    valid=[(ball_bearing_dataset, "valid/*.hdf5")],
    test_sets=ias_test_sets(ball_bearing_dataset),
)

BenchmarkBallBearing_Estimation = BenchmarkSpec(
    name="BenchmarkBallBearing_Estimation",
    # window_sec = largest window any upstream method needs (SIG-GRU 1.96 s, ViBES 1.84 s),
    # rounded to 2.0 s so every method has enough samples; smaller ones crop/decimate. See ias/__init__.
    task=WindowedEstimation(window_sec=2.0),
    **_ball_bearing,
)

BenchmarkBallBearing_GridwiseEstimation = BenchmarkSpec(
    name="BenchmarkBallBearing_GridwiseEstimation",
    task=GridwiseEstimation(window_sec=3.0, step_sec=0.1),
    **_ball_bearing,
)

# Dense free-run sibling (framework Simulation task): the model predicts one IAS estimate
# per sample over the full recording — the window lives in the model (e.g. a sliding window),
# not the benchmark — scored per-sample MAE in Hz. Same data and test sets as the windowed task.
BenchmarkBallBearing_Simulation = BenchmarkSpec(
    name="BenchmarkBallBearing_Simulation",
    task=Simulation(metric=mae),
    **_ball_bearing,
)
