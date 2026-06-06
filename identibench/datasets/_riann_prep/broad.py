"""BROAD dataset → data/Myon/ (39 trials).

Source: https://github.com/dlaidig/broad
Output: data/Myon/{NN_description}.hdf5  (39 files)

The BROAD .mat files are numbered 01-39 but the mapping to Myon output files
is NOT 1:1 — two BROAD trials are re-ordered:
  BROAD 03 (slow_rotation_C)    → Myon 05 (medium_rotation)
  BROAD 04 (rotation_w_breaks_A)→ Myon 03
  BROAD 05 (rotation_w_breaks_B)→ Myon 04
  BROAD 12 (slow_translation_C) → Myon 14 (medium_translation)
  BROAD 13 (translation_w_brk_A)→ Myon 12
  BROAD 14 (translation_w_brk_B)→ Myon 13
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ._common import download_file, extract_archive, fix_quaternion_flips, write_hdf5

_REPO_ZIP = "https://github.com/dlaidig/broad/archive/refs/heads/main.zip"

# Known Myon output filenames keyed by trial number (01-39).
MYON_NAMES = {
    1: "01_slow_rotation_A",
    2: "02_slow_rotation_B",
    3: "05_medium_rotation",
    4: "03_slow_rotation_with_breaks_A",
    5: "04_slow_rotation_with_breaks_B",
    6: "06_fast_rotation_A",
    7: "07_fast_rotation_B",
    8: "08_fast_rotation_with_breaks_A",
    9: "09_fast_rotation_with_breaks_B",
    10: "10_slow_translation_A",
    11: "11_slow_translation_B",
    12: "14_medium_translation",
    13: "12_slow_translation_with_breaks_A",
    14: "13_slow_translation_with_breaks_B",
    15: "15_fast_translation_A",
    16: "16_fast_translation_B",
    17: "17_fast_translation_with_breaks_A",
    18: "18_fast_translation_with_breaks_B",
    19: "19_slow_arbitrary_240s",
    20: "20_slow_arbitrary_360s",
    21: "21_medium_arbitrary",
    22: "22_medium_arbitrary_240s",
    23: "23_medium_arbitrary_360s",
    24: "24_medium_rotation_tapping",
    25: "25_medium_translation_tapping",
    26: "26_medium_rotation_phone_vibration",
    27: "27_medium_translation_phone_vibration",
    28: "28_medium_arbitrary_stationary_magnet_A",
    29: "29_medium_arbitrary_stationary_magnet_B",
    30: "30_medium_arbitrary_with_breaks_stationary_magnet_A",
    31: "31_medium_arbitrary_with_breaks_stationary_magnet_B",
    32: "32_medium_arbitrary_attached_magnet_1cm",
    33: "33_medium_arbitrary_attached_magnet_2cm",
    34: "34_medium_arbitrary_attached_magnet_3cm",
    35: "35_medium_arbitrary_attached_magnet_4cm",
    36: "36_medium_arbitrary_attached_magnet_5cm",
    37: "37_medium_arbitrary_office_A",
    38: "38_medium_arbitrary_office_B",
    39: "39_mixed",
}


def download(raw_dir: Path) -> None:
    raw_dir = raw_dir / "broad"
    download_file(_REPO_ZIP, raw_dir / "broad-main.zip")


def convert(raw_dir: Path, out_dir: Path) -> None:
    raw_dir = raw_dir / "broad"
    out_dir = out_dir / "Myon"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "broad-main.zip"
    ext_dir = raw_dir / "broad-main"
    if not ext_dir.exists():
        print("  Extracting BROAD repo ...")
        extract_archive(zip_path, raw_dir)

    mat_dir = ext_dir / "data_mat"
    mat_files = sorted(mat_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")

    for mat_path in mat_files:
        # Extract trial number from filename (e.g. "05_undisturbed_..." -> 5)
        num = int(mat_path.stem.split("_")[0])
        myon_name = MYON_NAMES.get(num)
        if myon_name is None:
            print(f"  Skipping unknown trial number: {mat_path.name}")
            continue

        out_path = out_dir / f"{myon_name}.hdf5"
        if out_path.exists():
            print(f"  Skipping (exists): {out_path.name}")
            continue

        data = loadmat(str(mat_path), squeeze_me=True)

        acc = np.asarray(data["imu_acc"], dtype=np.float64)      # (N, 3)
        gyr = np.asarray(data["imu_gyr"], dtype=np.float64)      # (N, 3)
        mag = np.asarray(data["imu_mag"], dtype=np.float64)       # (N, 3)
        quat = np.asarray(data["opt_quat"], dtype=np.float64)     # (N, 4) wxyz
        fs = float(data["sampling_rate"])
        movement = np.asarray(data["movement"], dtype=np.float64).ravel()

        quat = fix_quaternion_flips(quat)
        dt = 1.0 / fs

        print(f"  Writing {out_path.name}  ({len(acc)} samples, {fs:.1f} Hz)")
        write_hdf5(out_path, acc, gyr, quat, dt, mag=mag, movement_mask=movement)
