"""Caruso-Sassari MIMU optical dataset → two output dirs.

Source: https://github.com/marcocaruso/mimu_optical_dataset_caruso_sassari/releases/tag/v5.0
Output: data/Caruso-Sassari/Marco::{speed}_v4_{sensor}.hdf5      (18 files)
        data/Caruso-Sassari_orig/Marco::{speed}_v4_{sensor}.hdf5  (18 files)

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

Caruso-Sassari      uses Qs with fix_quaternion_flips
Caruso-Sassari_orig uses Qs as-is
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ._common import download_file, fix_quaternion_flips, write_hdf5

_RELEASE = "https://github.com/marcocaruso/mimu_optical_dataset_caruso_sassari/releases/download/v5.0"
_SPEEDS = ["slow", "medium", "fast"]
_SENSOR_KEYS = ["AP1", "AP2", "SH1", "SH2", "XS1", "XS2"]
_INDEX_KEYS = ["indarb", "indx", "indy", "indz"]


def download(raw_dir: Path) -> None:
    raw_dir = raw_dir / "caruso"
    for speed in _SPEEDS:
        url = f"{_RELEASE}/{speed}_v5.mat"
        download_file(url, raw_dir / f"{speed}_v5.mat")


def convert(raw_dir: Path, out_dir: Path) -> None:
    raw_dir = raw_dir / "caruso"
    dir_main = out_dir / "Caruso-Sassari"
    dir_orig = out_dir / "Caruso-Sassari_orig"
    dir_main.mkdir(parents=True, exist_ok=True)
    dir_orig.mkdir(parents=True, exist_ok=True)

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

            out_name = f"{speed}_v4_{sensor_key}"

            # Caruso-Sassari_orig: Qs as-is (no flip correction)
            out_orig = dir_orig / f"Marco::{out_name}.hdf5"
            if out_orig.exists():
                print(f"    Skipping (exists): {out_orig.name}")
            else:
                print(f"    Writing {out_orig.name}  ({n_samples} samples)")
                write_hdf5(out_orig, acc, gyr, qs, dt, mag=mag, movement_mask=movement)

            # Caruso-Sassari: Qs with flip correction
            out_main = dir_main / f"Marco::{out_name}.hdf5"
            if out_main.exists():
                print(f"    Skipping (exists): {out_main.name}")
            else:
                print(f"    Writing {out_main.name}  ({n_samples} samples)")
                write_hdf5(out_main, acc, gyr, qs_flipped, dt, mag=mag, movement_mask=movement)
