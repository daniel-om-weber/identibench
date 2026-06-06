"""OxIOD dataset — Oxford Inertial Odometry Dataset.

Source: http://deepio.cs.ox.ac.uk/
Output: data/OxIOD/OxIOD::{category}_data{N}_{M}:fixed.hdf5  (71 files)

Downloaded from Google Drive (the link revealed after the access form).
"""

import re
from pathlib import Path

import numpy as np

from ._common import extract_archive, fix_quaternion_flips, write_hdf5

G = 9.81  # m/s^2

# Expected output files derived from existing data directory listing.
EXPECTED_FILES = [
    "OxIOD::handbag_data1_1:fixed",
    "OxIOD::handbag_data1_2:fixed",
    "OxIOD::handbag_data1_3:fixed",
    "OxIOD::handbag_data1_4:fixed",
    "OxIOD::handbag_data2_1:fixed",
    "OxIOD::handbag_data2_2:fixed",
    "OxIOD::handbag_data2_3:fixed",
    "OxIOD::handbag_data2_4:fixed",
    "OxIOD::handheld_data1_1:fixed",
    "OxIOD::handheld_data1_2:fixed",
    "OxIOD::handheld_data1_3:fixed",
    "OxIOD::handheld_data1_4:fixed",
    "OxIOD::handheld_data1_5:fixed",
    "OxIOD::handheld_data1_6:fixed",
    "OxIOD::handheld_data1_7:fixed",
    "OxIOD::handheld_data2_1:fixed",
    "OxIOD::handheld_data2_2:fixed",
    "OxIOD::handheld_data2_3:fixed",
    "OxIOD::handheld_data3_1:fixed",
    "OxIOD::handheld_data3_2:fixed",
    "OxIOD::handheld_data3_3:fixed",
    "OxIOD::handheld_data3_4:fixed",
    "OxIOD::handheld_data3_5:fixed",
    "OxIOD::handheld_data4_1:fixed",
    "OxIOD::handheld_data4_2:fixed",
    "OxIOD::handheld_data4_3:fixed",
    "OxIOD::handheld_data4_4:fixed",
    "OxIOD::handheld_data4_5:fixed",
    "OxIOD::handheld_data5_1:fixed",
    "OxIOD::handheld_data5_2:fixed",
    "OxIOD::handheld_data5_3:fixed",
    "OxIOD::handheld_data5_4:fixed",
    "OxIOD::pocket_data1_1:fixed",
    "OxIOD::pocket_data1_2:fixed",
    "OxIOD::pocket_data1_3:fixed",
    "OxIOD::pocket_data1_4:fixed",
    "OxIOD::pocket_data1_5:fixed",
    "OxIOD::pocket_data2_1:fixed",
    "OxIOD::pocket_data2_2:fixed",
    "OxIOD::pocket_data2_3:fixed",
    "OxIOD::pocket_data2_4:fixed",
    "OxIOD::pocket_data2_5:fixed",
    "OxIOD::pocket_data2_6:fixed",
    "OxIOD::running_data1_1:fixed",
    "OxIOD::running_data1_2:fixed",
    "OxIOD::running_data1_3:fixed",
    "OxIOD::running_data1_4:fixed",
    "OxIOD::running_data1_5:fixed",
    "OxIOD::running_data1_6:fixed",
    "OxIOD::running_data1_7:fixed",
    "OxIOD::slow_walking_data1_1:fixed",
    "OxIOD::slow_walking_data1_2:fixed",
    "OxIOD::slow_walking_data1_3:fixed",
    "OxIOD::slow_walking_data1_4:fixed",
    "OxIOD::slow_walking_data1_5:fixed",
    "OxIOD::slow_walking_data1_6:fixed",
    "OxIOD::slow_walking_data1_7:fixed",
    "OxIOD::slow_walking_data1_8:fixed",
    "OxIOD::trolley_data1_1:fixed",
    "OxIOD::trolley_data1_2:fixed",
    "OxIOD::trolley_data1_3:fixed",
    "OxIOD::trolley_data1_4:fixed",
    "OxIOD::trolley_data1_5:fixed",
    "OxIOD::trolley_data1_6:fixed",
    "OxIOD::trolley_data1_7:fixed",
    "OxIOD::trolley_data2_1:fixed",
    "OxIOD::trolley_data2_2:fixed",
    "OxIOD::trolley_data2_3:fixed",
    "OxIOD::trolley_data2_4:fixed",
    "OxIOD::trolley_data2_5:fixed",
    "OxIOD::trolley_data2_6:fixed",
]


def _parse_expected(name: str) -> tuple[str, int, int]:
    """Parse 'OxIOD::category_dataN_M:fixed' → (category, N, M)."""
    m = re.match(r"OxIOD::(.+)_data(\d+)_(\d+):fixed", name)
    assert m, f"Cannot parse {name}"
    return m.group(1), int(m.group(2)), int(m.group(3))


_GDRIVE_ID = "1UCHY3ENCybcBNyiC2wx1gQEWSLqzJag0"
_ZIP_NAME = "Oxford_Inertial_Odometry_Dataset_2.0.zip"


def download(raw_dir: Path) -> None:
    """Download OxIOD ZIP from Google Drive and extract."""
    oxiod_dir = raw_dir / "OxIOD"
    oxiod_dir.mkdir(parents=True, exist_ok=True)
    zip_path = oxiod_dir / _ZIP_NAME

    # Check if already extracted
    if any(oxiod_dir.glob("*/data*/syn/imu*.csv")):
        print("  OxIOD raw data already extracted.")
        return

    if not zip_path.exists():
        try:
            import gdown
        except ImportError:
            raise ImportError("gdown is required for OxIOD download: uv pip install gdown")
        url = f"https://drive.google.com/uc?id={_GDRIVE_ID}"
        print(f"  Downloading {_ZIP_NAME} from Google Drive ...")
        gdown.download(url, str(zip_path), quiet=False)

    print(f"  Extracting {_ZIP_NAME} ...")
    extract_archive(zip_path, oxiod_dir)


def _find_imu_vi_pairs(oxiod_dir: Path) -> list[tuple[str, int, int, Path, Path]]:
    """Discover imu/vi CSV pairs in the OxIOD directory tree.

    Returns list of (category, data_num, seq_num, imu_path, vi_path).
    Handles both underscore and space-separated category names, and nested
    extraction layouts (e.g., "Oxford Inertial Odometry Dataset/" prefix).
    """
    pairs = []
    for expected in EXPECTED_FILES:
        cat, dnum, snum = _parse_expected(expected)
        # Try multiple directory name variants
        cat_variants = [cat, cat.replace("_", " ")]
        found = False
        for root in [oxiod_dir] + list(oxiod_dir.iterdir()):
            if not root.is_dir():
                continue
            for cv in cat_variants:
                imu_path = root / cv / f"data{dnum}" / "syn" / f"imu{snum}.csv"
                vi_path = root / cv / f"data{dnum}" / "syn" / f"vi{snum}.csv"
                if imu_path.exists() and vi_path.exists():
                    pairs.append((cat, dnum, snum, imu_path, vi_path))
                    found = True
                    break
            if found:
                break
    return pairs


def convert(raw_dir: Path, out_dir: Path) -> None:
    raw_dir = raw_dir / "OxIOD"
    out_dir = out_dir / "OxIOD"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print("  OxIOD raw data not found. Run with download step first.")
        return

    pairs = _find_imu_vi_pairs(raw_dir)
    if not pairs:
        print("  No imu/vi CSV pairs found in OxIOD raw data.")
        return

    for cat, dnum, snum, imu_path, vi_path in pairs:
        out_name = f"OxIOD::{cat}_data{dnum}_{snum}:fixed.hdf5"
        out_path = out_dir / out_name
        if out_path.exists():
            print(f"  Skipping (exists): {out_path.name}")
            continue

        # ── Load IMU CSV (no header row in v2.0) ──
        # Columns: Time, attitude(roll/pitch/yaw), gyro(x/y/z), gravity(x/y/z),
        #          user_acc(x/y/z), mag(x/y/z)
        imu = np.loadtxt(imu_path, delimiter=",")
        gravity = imu[:, 7:10]      # G units
        user_acc = imu[:, 10:13]     # G units
        acc = (gravity + user_acc) * G  # m/s^2
        gyr = imu[:, 4:7]           # rad/s
        mag = imu[:, 13:16]         # microtesla

        # ── Load VI ground truth CSV (no header row) ──
        # Columns: Time, header, tx, ty, tz, qx, qy, qz, qw
        vi = np.loadtxt(vi_path, delimiter=",")
        quat = vi[:, [8, 5, 6, 7]]  # reorder to wxyz

        quat = fix_quaternion_flips(quat)

        print(f"  Writing {out_path.name}  ({len(acc)} samples)")
        write_hdf5(out_path, acc, gyr, quat, dt=0.01, mag=mag)
