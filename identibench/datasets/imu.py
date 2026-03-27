"""IMU orientation benchmark dataset from Weygers & Kok (2020)."""

__all__ = [
    "dl_imu",
    "BenchmarkIMU_Sensor1",
    "BenchmarkIMU_Sensor2",
    "BenchmarkIMU_Relative",
    "imu_split_all_test",
    "imu_split_train_test",
]

from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import requests
import scipy.io

import identibench.benchmark as idb
from identibench.metrics import inclination_rmse_deg, orientation_rmse_deg
from identibench.utils import write_array

ALL_FILES = [
    "data_1D_01",
    "data_1D_02",
    "data_1D_03",
    "data_1D_04",
    "data_1D_05",
    "data_2D_01",
    "data_2D_02",
    "data_2D_03",
    "data_2D_05",
    "data_2D_07",
    "data_3D_01",
    "data_3D_02",
    "data_3D_03",
    "data_3D_04",
    "data_3D_05",
]

ALL_HDF5_FILES = [f"{name}.hdf5" for name in ALL_FILES]

# Inputs: 12 IMU channels (acc1 x3, gyr1 x3, acc2 x3, gyr2 x3)
imu_u_cols = [f"u{i}" for i in range(12)]

# Outputs: quaternion [w, x, y, z] for sensor 1, sensor 2, and relative
imu_y_q1_cols = [f"y_q1_{i}" for i in range(4)]
imu_y_q2_cols = [f"y_q2_{i}" for i in range(4)]
imu_y_rel_cols = [f"y_rel_{i}" for i in range(4)]

# --- Split definitions ---

imu_split_all_test = {
    "test": ALL_HDF5_FILES,
}

imu_split_train_test = {
    "train": [
        f"{n}.hdf5"
        for n in [
            "data_1D_01",
            "data_1D_02",
            "data_1D_03",
            "data_2D_01",
            "data_2D_02",
            "data_2D_03",
            "data_3D_01",
            "data_3D_02",
            "data_3D_03",
        ]
    ],
    "valid": [f"{n}.hdf5" for n in ["data_1D_04", "data_2D_05", "data_3D_04"]],
    "test": [f"{n}.hdf5" for n in ["data_1D_05", "data_2D_07", "data_3D_05"]],
}


def _quat_relative(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute q1 * inv(q2) for unit quaternions. Shape: (N, 4), [w,x,y,z]."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ow = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    ox = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
    oy = -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2
    oz = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2
    return np.stack([ow, ox, oy, oz], axis=-1)


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/daniel-om-weber/dfjimu/main/data"


def dl_imu(
    save_path: Path,
    force_download: bool = False,
) -> None:
    """Download IMU .mat files from GitHub and convert to HDF5 (flat directory)."""
    save_path = Path(save_path)
    if save_path.is_dir() and not force_download:
        if all((save_path / f).exists() for f in ALL_HDF5_FILES):
            return

    save_path.mkdir(parents=True, exist_ok=True)

    for name in ALL_FILES:
        hdf5_path = save_path / f"{name}.hdf5"
        if hdf5_path.exists() and not force_download:
            continue

        url = f"{GITHUB_RAW_BASE}/{name}.mat"
        response = requests.get(url)
        response.raise_for_status()

        mat = scipy.io.loadmat(BytesIO(response.content), squeeze_me=True, struct_as_record=False)
        data = mat["data"]

        sensor_data = data.sensorData.astype(np.float32)  # (N, 12)
        ref = data.ref  # (N, 17)
        q1_ref = ref[:, 0:4].astype(np.float32)  # (N, 4)
        q2_ref = ref[:, 4:8].astype(np.float32)  # (N, 4)
        q_rel = _quat_relative(q1_ref, q2_ref).astype(np.float32)  # (N, 4)

        r_12 = np.atleast_1d(data.r_12).astype(np.float32)
        r_21 = np.atleast_1d(data.r_21).astype(np.float32)
        fs = float(data.rate)

        with h5py.File(hdf5_path, "w") as f:
            write_array(f, "u", sensor_data)
            write_array(f, "y_q1_", q1_ref)
            write_array(f, "y_q2_", q2_ref)
            write_array(f, "y_rel_", q_rel)
            f.attrs["fs"] = fs
            f.attrs["r_12"] = r_12
            f.attrs["r_21"] = r_21


# --- Benchmark specifications ---

BenchmarkIMU_Sensor1 = idb.BenchmarkSpecSimulation(
    name="BenchmarkIMU_Sensor1",
    dataset_id="imu",
    u_cols=imu_u_cols,
    y_cols=imu_y_q1_cols,
    metric_func=inclination_rmse_deg,
    download_func=dl_imu,
    sampling_time=1.0 / 50.0,
    init_window=0,
    split=imu_split_all_test,
)

BenchmarkIMU_Sensor2 = idb.BenchmarkSpecSimulation(
    name="BenchmarkIMU_Sensor2",
    dataset_id="imu",
    u_cols=imu_u_cols,
    y_cols=imu_y_q2_cols,
    metric_func=inclination_rmse_deg,
    download_func=dl_imu,
    sampling_time=1.0 / 50.0,
    init_window=0,
    split=imu_split_all_test,
)

BenchmarkIMU_Relative = idb.BenchmarkSpecSimulation(
    name="BenchmarkIMU_Relative",
    dataset_id="imu",
    u_cols=imu_u_cols,
    y_cols=imu_y_rel_cols,
    metric_func=orientation_rmse_deg,
    download_func=dl_imu,
    sampling_time=1.0 / 50.0,
    init_window=0,
    split=imu_split_all_test,
)
