"""Industrial robot forward/inverse benchmark dataset definitions."""

__all__ = [
    "robot_forward_dataset",
    "robot_inverse_dataset",
    "u_forward",
    "y_forward",
    "BenchmarkRobotForward_Simulation",
    "BenchmarkRobotForward_Prediction",
    "u_inverse",
    "y_inverse",
    "BenchmarkRobotInverse_Simulation",
    "BenchmarkRobotInverse_Prediction",
    "dl_robot_forward",
    "dl_robot_inverse",
]

from ..utils import write_dataset, write_array
from ..benchmark import BenchmarkSpec, Prediction, Simulation
from ..dataset import Dataset
import identibench.metrics
from nonlinear_benchmarks.utilities import cashed_download
from pathlib import Path
import os
import h5py
import numpy as np
import scipy.io as sio

# Robot Identification Benchmark archive (shared by the forward and inverse downloaders).
ROBOT_URL = "https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_Without_Raw_Data.rar"


def robot_mat2hdf(
    save_path: Path,  # directory the files are written to, created if it does not exist
    mat_path: Path,  # path of mat file to extract
) -> None:
    "converts .mat file of industrial robot to hdf5 file, used for forward and inverse"

    fs = 10  # Hz
    train_valid_split = 0.8
    os.makedirs(save_path / "test", exist_ok=True)
    os.makedirs(save_path / "train", exist_ok=True)
    os.makedirs(save_path / "valid", exist_ok=True)

    mf = sio.loadmat(mat_path)
    for mode in ["train", "test"]:
        if mode == "test":
            with h5py.File(save_path / "test" / "test.hdf5", "w") as f:
                write_dataset(f, "dt", np.ones_like(mf[f"time_{mode}"][0]) / fs)
                write_array(f, "u", mf[f"u_{mode}"].T)
                write_array(f, "y", mf[f"y_{mode}"].T)
                f.attrs["fs"] = fs

        else:
            with (
                h5py.File(save_path / "train" / "train.hdf5", "w") as train_f,
                h5py.File(save_path / "valid" / "valid.hdf5", "w") as valid_f,
            ):
                dt = np.ones_like(mf[f"time_{mode}"][0]) / fs
                total_entries = len(dt)
                split_index = int(total_entries * train_valid_split)

                write_dataset(train_f, "dt", dt[:split_index])
                write_array(train_f, "u", mf[f"u_{mode}"][:, :split_index].T)
                write_array(train_f, "y", mf[f"y_{mode}"][:, :split_index].T)
                train_f.attrs["fs"] = fs

                write_dataset(valid_f, "dt", dt[split_index:])
                write_array(valid_f, "u", mf[f"u_{mode}"][:, split_index:].T)
                write_array(valid_f, "y", mf[f"y_{mode}"][:, split_index:].T)
                valid_f.attrs["fs"] = fs


def dl_robot_forward(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
) -> None:
    save_path = Path(save_path)

    tmp_dir = cashed_download(ROBOT_URL, "Industrial_robot", force_download=force_download)
    tmp_dir = Path(tmp_dir)

    path_forward = tmp_dir / "forward_identification_without_raw_data.mat"

    robot_mat2hdf(save_path, path_forward)


u_forward = [f"u{i}" for i in range(0, 6)]
y_forward = [f"y{i}" for i in range(0, 6)]

robot_forward_dataset = Dataset("robot_forward", prepare=dl_robot_forward)

_robot_forward = dict(
    u_cols=u_forward,
    y_cols=y_forward,
    train=[(robot_forward_dataset, "train/*.hdf5")],
    valid=[(robot_forward_dataset, "valid/*.hdf5")],
    test_sets={"test": [(robot_forward_dataset, "test/*.hdf5")]},
)

BenchmarkRobotForward_Simulation = BenchmarkSpec(
    name="BenchmarkRobotForward_Simulation",
    task=Simulation(metric=identibench.metrics.rmse, init_window=100),
    **_robot_forward,
)

BenchmarkRobotForward_Prediction = BenchmarkSpec(
    name="BenchmarkRobotForward_Prediction",
    task=Prediction(horizon=100, step=100, metric=identibench.metrics.rmse, init_window=100),
    **_robot_forward,
)


def dl_robot_inverse(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
) -> None:
    save_path = Path(save_path)

    tmp_dir = cashed_download(ROBOT_URL, "Industrial_robot", force_download=force_download)
    tmp_dir = Path(tmp_dir)

    path_inverse = tmp_dir / "inverse_identification_without_raw_data.mat"

    robot_mat2hdf(save_path, path_inverse)


u_inverse = [f"u{i}" for i in range(0, 12)]
y_inverse = [f"y{i}" for i in range(0, 6)]

robot_inverse_dataset = Dataset("robot_inverse", prepare=dl_robot_inverse)

_robot_inverse = dict(
    u_cols=u_inverse,
    y_cols=y_inverse,
    train=[(robot_inverse_dataset, "train/*.hdf5")],
    valid=[(robot_inverse_dataset, "valid/*.hdf5")],
    test_sets={"test": [(robot_inverse_dataset, "test/*.hdf5")]},
)

BenchmarkRobotInverse_Simulation = BenchmarkSpec(
    name="BenchmarkRobotInverse_Simulation",
    task=Simulation(metric=identibench.metrics.rmse, init_window=100),
    **_robot_inverse,
)

BenchmarkRobotInverse_Prediction = BenchmarkSpec(
    name="BenchmarkRobotInverse_Prediction",
    task=Prediction(horizon=100, step=100, metric=identibench.metrics.rmse, init_window=100),
    **_robot_inverse,
)
