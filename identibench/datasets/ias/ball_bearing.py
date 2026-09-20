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
    download_and_unpack,
    encoder_pulse_to_ias,
    save_signals_hdf5,
    ias_test_sets,
    write_disturbed_test_sets,
)

_INFO = DatasetInfo(
    name="Ball_Bearing",
    zip_url="https://data.mendeley.com/public-api/zip/v43hmbwxpm/download/2",
)

# Order-domain cutoff for the encoder-derived IAS, in orders of the measured shaft.
# No per-tooth encoder calibration is applied: no tooth-width pattern is reproducible across
# recordings above the per-period estimation noise, and ~99% of the apparent pattern's power
# sits above this cutoff anyway (the residual in-band is ~0.04% of the period).
_CUTOFF_ORDER = 4.71
_PPR = 1024

# Highest frequency the label retains: IAS_max * cutoff_order, measured over all 60 recordings
# (29.80 Hz on the encoder shaft, which is also the measured shaft -- no gearing on this rig).
_IAS_BANDWIDTH_HZ = 29.80 * _CUTOFF_ORDER  # 140.3 Hz

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

    Recordings are truncated to the span the encoder resolves; see
    :func:`._common.encoder_pulse_to_ias`.
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
            # `sl` truncates the recording to the span the encoder actually measures, rather
            # than extrapolating the IAS past the first/last pulse; every channel gets it.
            # Here that costs ~0.001% of the file (the first pulse lands within ~0.1 ms).
            ias, sl = encoder_pulse_to_ias(
                np.asarray(mat["Channel_2"]).squeeze(), fs, pulses_per_revolution=_PPR, cutoff_order=_CUTOFF_ORDER
            )
            signals = {
                "IAS": ias,
                "Acc_x": np.asarray(mat["Channel_1"]).squeeze()[sl] * 10 * 9.81,
            }
            # No gearing on this rig: the encoder sits on the shaft the accelerometer measures.
            save_signals_hdf5(
                signals,
                save_path / target_subdir / f"{file.stem}.hdf5",
                fs=fs,
                gear_ratio=1,
                ias_bandwidth_hz=_IAS_BANDWIDTH_HZ,
            )

    write_disturbed_test_sets(save_path, vib_keys=["Acc_x"])


# version 2: order-domain IAS filtering (was a savgol + fixed 12.5 Hz time-domain low-pass).
ball_bearing_dataset = Dataset("ball_bearing", prepare=dl_ball_bearing, version="2")

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
    # window_sec=3.0: the largest single window across every upstream method's search space
    # over all four IAS datasets (unlike the per-dataset WindowedEstimation windows above,
    # this one is kept uniform — it's only a context guarantee, not a tuned averaging window).
    # step_sec: Nyquist for the label's retained band, _IAS_BANDWIDTH_HZ = 140.3 Hz
    # -> 3.56 ms, rounded down to 3 ms (1.19x margin).
    task=GridwiseEstimation(window_sec=3.0, step_sec=0.003),
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
