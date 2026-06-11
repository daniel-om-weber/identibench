"""Caruso-Sassari MIMU optical dataset.

Source: https://github.com/marcocaruso/mimu_optical_dataset_caruso_sassari/releases/tag/v5.0
Output: caruso/Marco::{speed}_v4_{sensor}.hdf5  (18 files, flat)

The release contains 3 mat files (slow_v5.mat, medium_v5.mat, fast_v5.mat),
each bundling data for 6 MIMUs × 2 reps.

v5 mat format (flat arrays, not structs):
  Sensor arrays (AP1, AP2, SH1, SH2, XS1, XS2): (N, 14) float64
    Col  0:    timestamp (seconds)
    Cols 1-3:  accelerometer (x, y, z) in m/s²
    Cols 4-6:  gyroscope (x, y, z) in rad/s
    Cols 7-9:  magnetometer (x, y, z)
    Cols 10-13: onboard quaternion (not used)
  Qs: (N, 4) optical quaternion (w, x, y, z)
  indarb, indx, indy, indz: movement index arrays (1-indexed)

The optical quaternion is stored with fix_quaternion_flips applied.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from identibench.dataset import Dataset
from identibench.utils import download_file

from ._common import _prepare_sources, _spec, fix_quaternion_flips, write_hdf5

_RELEASE = "https://github.com/marcocaruso/mimu_optical_dataset_caruso_sassari/releases/download/v5.0"
_RAW_DIR = "caruso"  # cache sub-directory under raw_dir for downloads
_SPEEDS = ["slow", "medium", "fast"]
_SENSOR_KEYS = ["AP1", "AP2", "SH1", "SH2", "XS1", "XS2"]
_INDEX_KEYS = ["indarb", "indx", "indy", "indz"]


def download(raw_dir: Path, force: bool = False) -> None:
    raw_dir = raw_dir / _RAW_DIR
    for speed in _SPEEDS:
        url = f"{_RELEASE}/{speed}_v5.mat"
        download_file(url, raw_dir / f"{speed}_v5.mat", force=force)


def convert(raw_dir: Path, out_dir: Path, force: bool = False) -> None:
    raw_dir = raw_dir / _RAW_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for speed in _SPEEDS:
        mat_path = raw_dir / f"{speed}_v5.mat"
        if not mat_path.exists():
            print(f"  Missing: {mat_path}")
            continue

        print(f"  Loading {mat_path.name} ...")
        data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=True)

        qs = np.asarray(data["Qs"], dtype=np.float64)  # (N, 4) optical quaternion
        qs_flipped = fix_quaternion_flips(qs)

        # Build movement mask from index arrays (MATLAB 1-indexed)
        n_samples = qs.shape[0]
        movement = np.zeros(n_samples, dtype=np.float64)
        for idx_key in _INDEX_KEYS:
            if idx_key in data:
                indices = np.asarray(data[idx_key]).ravel()
                movement[indices - 1] = 1.0

        dt = 0.01  # 100 Hz

        for sensor_key in _SENSOR_KEYS:
            if sensor_key not in data:
                print(f"    Missing sensor key: {sensor_key}")
                continue

            sensor = np.asarray(data[sensor_key], dtype=np.float64)  # (N, 14)
            acc = sensor[:, 1:4]
            gyr = sensor[:, 4:7]
            mag = sensor[:, 7:10]

            out_path = out_dir / f"Marco::{speed}_v4_{sensor_key}.hdf5"
            if out_path.exists() and not force:
                print(f"    Skipping (exists): {out_path.name}")
            else:
                print(f"    Writing {out_path.name}  ({n_samples} samples)")
                write_hdf5(out_path, acc, gyr, qs_flipped, dt, mag=mag, movement_mask=movement)


def dl_caruso(save_path, force_download: bool = False) -> None:
    """Download + convert Caruso-Sassari flat into ``save_path``."""
    _prepare_sources(save_path, [(download, convert)], force_download=force_download)


caruso_dataset = Dataset("caruso", prepare=dl_caruso)

BenchmarkCaruso_Inclination = _spec("BenchmarkCaruso_Inclination", caruso_dataset)
