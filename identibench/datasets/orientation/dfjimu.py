"""Weygers & Kok (2020) IMU orientation dataset (the *dfjimu* dataset)."""

__all__ = [
    "dfjimu_dataset",
    "dl_dfjimu",
    "BenchmarkDFJIMU_Inclination",
    "BenchmarkDFJIMU_Relative",
]

from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import requests
import scipy.io

from identibench.benchmark import BenchmarkSpec, Simulation
from identibench.dataset import Dataset
from identibench.metrics import _quat_conj, _quat_mul, inclination_rmse_deg, orientation_rmse_deg
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

dfjimu_u_s1_cols = [f"acc1_{a}" for a in _xyz] + [f"gyr1_{a}" for a in _xyz]
dfjimu_u_s2_cols = [f"acc2_{a}" for a in _xyz] + [f"gyr2_{a}" for a in _xyz]
dfjimu_u_cols = dfjimu_u_s1_cols + dfjimu_u_s2_cols

dfjimu_u_generic = [f"acc_{a}" for a in _xyz] + [f"gyr_{a}" for a in _xyz]

dfjimu_y_q1_cols = [f"q1_{a}" for a in _wxyz]
dfjimu_y_q2_cols = [f"q2_{a}" for a in _wxyz]
dfjimu_y_rel_cols = [f"qrel_{a}" for a in _wxyz]
dfjimu_y_q_generic = [f"q_{a}" for a in _wxyz]


def _quat_relative(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Relative orientation q1 ⊗ inv(q2) for unit quaternions. Shape: (N, 4), [w,x,y,z]."""
    return _quat_mul(q1, _quat_conj(q2))


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/daniel-om-weber/dfjimu/main/data"

_SENSOR_MAPPINGS = [
    ("s1", list(zip(dfjimu_u_generic, dfjimu_u_s1_cols)) + list(zip(dfjimu_y_q_generic, dfjimu_y_q1_cols))),
    ("s2", list(zip(dfjimu_u_generic, dfjimu_u_s2_cols)) + list(zip(dfjimu_y_q_generic, dfjimu_y_q2_cols))),
]


def dl_dfjimu(
    save_path: Path,
    force_download: bool = False,
) -> None:
    """Download dfjimu .mat files from GitHub and convert to HDF5 (flat directory)."""
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
            for i, col in enumerate(dfjimu_u_cols):
                write_dataset(f, col, sensor_data[:, i])
            for i, col in enumerate(dfjimu_y_q1_cols):
                write_dataset(f, col, q1_ref[:, i])
            for i, col in enumerate(dfjimu_y_q2_cols):
                write_dataset(f, col, q2_ref[:, i])
            for i, col in enumerate(dfjimu_y_rel_cols):
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


dfjimu_dataset = Dataset("dfjimu", prepare=dl_dfjimu)

# --- Benchmark specifications ---
# Both specs share one dataset but select DISJOINT file groups via explicit
# wildcard-free patterns (the flat directory mixes base and virtual files, so a
# `*.hdf5` glob would conflate them): Inclination evaluates the 30 per-sensor
# virtual files with generic acc/gyr/q_* columns, Relative the 15 combined
# files with the full two-sensor columns and the relative quaternion target.

BenchmarkDFJIMU_Inclination = BenchmarkSpec(
    name="BenchmarkDFJIMU_Inclination",
    u_cols=dfjimu_u_generic,
    y_cols=dfjimu_y_q_generic,
    train=[],
    valid=[],
    test_sets={"persensor": [(dfjimu_dataset, f) for f in ALL_HDF5_FILES_PERSENSOR]},
    task=Simulation(metric=inclination_rmse_deg, init_window=0),
)

BenchmarkDFJIMU_Relative = BenchmarkSpec(
    name="BenchmarkDFJIMU_Relative",
    u_cols=dfjimu_u_cols,
    y_cols=dfjimu_y_rel_cols,
    train=[],
    valid=[],
    test_sets={"combined": [(dfjimu_dataset, f) for f in ALL_HDF5_FILES]},
    task=Simulation(metric=orientation_rmse_deg, init_window=0),
)
