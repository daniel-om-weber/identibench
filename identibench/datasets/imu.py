"""IMU orientation benchmark dataset from Weygers & Kok (2020)."""

__all__ = [
    "dl_imu",
    "BenchmarkIMU_Inclination",
    "BenchmarkIMU_Relative",
    "imu_split_all_test",
    "imu_split_train_test",
    "imu_split_all_test_persensor",
    "imu_split_train_test_persensor",
]

from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import requests
import scipy.io

import identibench.benchmark as idb
from identibench.metrics import inclination_rmse_deg, orientation_rmse_deg
from identibench.utils import write_dataset

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
ALL_HDF5_FILES_PERSENSOR = [f"{name}_{s}.hdf5" for name in ALL_FILES for s in ("s1", "s2")]

_xyz = ["x", "y", "z"]
_wxyz = ["w", "x", "y", "z"]

imu_u_s1_cols = [f"acc1_{a}" for a in _xyz] + [f"gyr1_{a}" for a in _xyz]
imu_u_s2_cols = [f"acc2_{a}" for a in _xyz] + [f"gyr2_{a}" for a in _xyz]
imu_u_cols = imu_u_s1_cols + imu_u_s2_cols

imu_u_generic = [f"acc_{a}" for a in _xyz] + [f"gyr_{a}" for a in _xyz]

imu_y_q1_cols = [f"q1_{a}" for a in _wxyz]
imu_y_q2_cols = [f"q2_{a}" for a in _wxyz]
imu_y_rel_cols = [f"qrel_{a}" for a in _wxyz]
imu_y_q_generic = [f"q_{a}" for a in _wxyz]

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

_TRAIN_NAMES = [
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
_VALID_NAMES = ["data_1D_04", "data_2D_05", "data_3D_04"]
_TEST_NAMES = ["data_1D_05", "data_2D_07", "data_3D_05"]

imu_split_all_test_persensor = {
    "test": ALL_HDF5_FILES_PERSENSOR,
}

imu_split_train_test_persensor = {
    "train": [f"{n}_{s}.hdf5" for n in _TRAIN_NAMES for s in ("s1", "s2")],
    "valid": [f"{n}_{s}.hdf5" for n in _VALID_NAMES for s in ("s1", "s2")],
    "test": [f"{n}_{s}.hdf5" for n in _TEST_NAMES for s in ("s1", "s2")],
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

_SENSOR_MAPPINGS = [
    ("s1", list(zip(imu_u_generic, imu_u_s1_cols)) + list(zip(imu_y_q_generic, imu_y_q1_cols))),
    ("s2", list(zip(imu_u_generic, imu_u_s2_cols)) + list(zip(imu_y_q_generic, imu_y_q2_cols))),
]


def dl_imu(
    save_path: Path,
    force_download: bool = False,
) -> None:
    """Download IMU .mat files from GitHub and convert to HDF5 (flat directory)."""
    save_path = Path(save_path)
    all_files = ALL_HDF5_FILES + ALL_HDF5_FILES_PERSENSOR
    if save_path.is_dir() and not force_download:
        if all((save_path / f).exists() for f in all_files):
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
            for i, col in enumerate(imu_u_cols):
                write_dataset(f, col, sensor_data[:, i])
            for i, col in enumerate(imu_y_q1_cols):
                write_dataset(f, col, q1_ref[:, i])
            for i, col in enumerate(imu_y_q2_cols):
                write_dataset(f, col, q2_ref[:, i])
            for i, col in enumerate(imu_y_rel_cols):
                write_dataset(f, col, q_rel[:, i])
            f.attrs["fs"] = fs
            f.attrs["r_12"] = r_12
            f.attrs["r_21"] = r_21

        n = sensor_data.shape[0]
        source_fname = f"{name}.hdf5"
        for suffix, col_mapping in _SENSOR_MAPPINGS:
            virt_path = save_path / f"{name}_{suffix}.hdf5"
            with h5py.File(virt_path, "w") as vf:
                for generic_name, source_name in col_mapping:
                    vsource = h5py.VirtualSource(source_fname, source_name, shape=(n,), dtype="f4")
                    layout = h5py.VirtualLayout(shape=(n,), dtype="f4")
                    layout[:] = vsource
                    vf.create_virtual_dataset(generic_name, layout)
                vf.attrs["fs"] = fs
                vf.attrs["r_12"] = r_12
                vf.attrs["r_21"] = r_21


# --- Benchmark specifications ---

BenchmarkIMU_Inclination = idb.BenchmarkSpecSimulation(
    name="BenchmarkIMU_Inclination",
    dataset_id="imu",
    u_cols=imu_u_generic,
    y_cols=imu_y_q_generic,
    metric_func=inclination_rmse_deg,
    download_func=dl_imu,
    sampling_time=1.0 / 50.0,
    init_window=0,
    split=imu_split_all_test_persensor,
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
