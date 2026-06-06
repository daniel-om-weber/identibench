"""RepoIMU TStick dataset.

Source: https://github.com/agnieszkaszczesna/RepoIMU
Output: data/RepoIMU/RepoIMU::TStick_{XX}_{Y}.hdf5  (21 files)

CSV format: semicolon-separated, 2 header rows, 14 columns:
  time(s), qw, qx, qy, qz, acc_x, acc_y, acc_z,
  gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z
Units: acc in m/s^2, gyr in rad/s, mag normalized, quat WXYZ.
Sampling rate: 100 Hz.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import download_file, extract_archive, fix_quaternion_flips, write_hdf5

_REPO_ZIP = "https://github.com/agnieszkaszczesna/RepoIMU/archive/refs/heads/master.zip"

# Expected output files (derived from existing Myon HDF5 directory listing).
# Maps (test_num, trial_num) → output name.
EXPECTED_TRIALS: set[tuple[int, int]] = {
    (2, 1), (2, 2),
    (3, 1), (3, 2), (3, 3),
    (4, 1), (4, 2), (4, 3),
    (6, 2),
    (7, 1), (7, 2), (7, 3),
    (8, 1), (8, 2), (8, 3),
    (9, 1), (9, 2), (9, 3),
    (11, 1), (11, 2), (11, 3),
}

_COL_NAMES = [
    "time", "qw", "qx", "qy", "qz",
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "mag_x", "mag_y", "mag_z",
]


def download(raw_dir: Path) -> None:
    raw_dir = raw_dir / "repoimu"
    download_file(_REPO_ZIP, raw_dir / "RepoIMU-master.zip")


def convert(raw_dir: Path, out_dir: Path) -> None:
    raw_dir = raw_dir / "repoimu"
    out_dir = out_dir / "RepoIMU"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "RepoIMU-master.zip"
    # GitHub ZIP root can be RepoIMU-master or RepoIMU-main depending on branch
    ext_dir = raw_dir / "RepoIMU-master"
    if not ext_dir.exists():
        ext_dir = raw_dir / "RepoIMU-main"
    if not ext_dir.exists():
        print("  Extracting RepoIMU repo ...")
        extract_archive(zip_path, raw_dir)
        # Detect whichever directory was created
        for name in ("RepoIMU-master", "RepoIMU-main"):
            if (raw_dir / name).exists():
                ext_dir = raw_dir / name
                break

    tstick_dir = ext_dir / "TStick"
    csv_files = sorted(tstick_dir.glob("TStick_Test*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No TStick CSV files found in {tstick_dir}")

    for csv_path in csv_files:
        # Parse test/trial numbers: TStick_Test02_Trial1.csv
        m = re.match(r"TStick_Test(\d+)_Trial(\d+)\.csv", csv_path.name)
        if not m:
            continue
        test_num, trial_num = int(m.group(1)), int(m.group(2))

        if (test_num, trial_num) not in EXPECTED_TRIALS:
            continue

        out_name = f"RepoIMU::TStick_{test_num:02d}_{trial_num}.hdf5"
        out_path = out_dir / out_name
        if out_path.exists():
            print(f"  Skipping (exists): {out_path.name}")
            continue

        df = pd.read_csv(csv_path, sep=";", skiprows=2, header=None, names=_COL_NAMES)

        acc = df[["acc_x", "acc_y", "acc_z"]].values
        gyr = df[["gyr_x", "gyr_y", "gyr_z"]].values
        mag = df[["mag_x", "mag_y", "mag_z"]].values
        quat = df[["qw", "qx", "qy", "qz"]].values

        quat = fix_quaternion_flips(quat)
        dt = 0.01  # 100 Hz

        print(f"  Writing {out_path.name}  ({len(acc)} samples)")
        write_hdf5(out_path, acc, gyr, quat, dt, mag=mag)
