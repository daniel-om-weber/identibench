"""EuRoC-MAV dataset — 6 Vicon Room sequences.

Source: ETH Research Collection (https://doi.org/10.3929/ethz-b-000690084)
Output: data/EuRoC-MAV/EurocMAV::{seq}.hdf5  (6 files)

The original per-sequence downloads at robotics.ethz.ch are no longer available.
The dataset is now hosted as two bundles (Vicon Room 1 + Vicon Room 2) on the
ETH Research Collection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import download_file, extract_archive, fix_quaternion_flips, write_hdf5

SEQUENCES = [
    ("vicon_room1", "V1_01_easy"),
    ("vicon_room1", "V1_02_medium"),
    ("vicon_room1", "V1_03_difficult"),
    ("vicon_room2", "V2_01_easy"),
    ("vicon_room2", "V2_02_medium"),
    ("vicon_room2", "V2_03_difficult"),
]

_BUNDLES = {
    "vicon_room1": "https://www.research-collection.ethz.ch/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/download",
    "vicon_room2": "https://www.research-collection.ethz.ch/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/download",
}


def download(raw_dir: Path) -> None:
    raw_dir = raw_dir / "euroc"
    for room, url in _BUNDLES.items():
        dest = raw_dir / f"{room}.zip"
        download_file(url, dest)


def _find_seq_dir(raw_dir: Path, seq: str) -> Path | None:
    """Locate extracted sequence directory — handles both old and bundle layouts."""
    # Direct layout: raw_dir/{seq}/mav0/...
    candidate = raw_dir / seq
    if (candidate / "mav0" / "imu0" / "data.csv").exists():
        return candidate
    # Search recursively
    for parent in raw_dir.rglob(seq):
        if parent.is_dir() and (parent / "mav0" / "imu0" / "data.csv").exists():
            return parent
    return None


def _ensure_extracted(raw_dir: Path) -> None:
    """Extract room bundles and nested per-sequence ZIPs as needed."""
    for room in _BUNDLES:
        bundle_zip = raw_dir / f"{room}.zip"
        if not bundle_zip.exists():
            continue
        # Extract the outer bundle to get per-sequence ZIPs
        room_dir = raw_dir / room
        if not room_dir.exists():
            print(f"  Extracting {bundle_zip.name} ...")
            extract_archive(bundle_zip, raw_dir)
        # Extract each inner per-sequence ZIP into its own directory
        for _, seq in SEQUENCES:
            if _find_seq_dir(raw_dir, seq):
                continue
            inner_zip = room_dir / seq / f"{seq}.zip"
            if inner_zip.exists():
                seq_dir = raw_dir / seq
                seq_dir.mkdir(exist_ok=True)
                print(f"  Extracting {inner_zip.name} → {seq}/ ...")
                extract_archive(inner_zip, seq_dir)


def convert(raw_dir: Path, out_dir: Path) -> None:
    raw_dir = raw_dir / "euroc"
    out_dir = out_dir / "EuRoC-MAV"
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_extracted(raw_dir)

    for _, seq in SEQUENCES:
        out_path = out_dir / f"EurocMAV::{seq}.hdf5"
        if out_path.exists():
            print(f"  Skipping (exists): {out_path.name}")
            continue

        ext_dir = _find_seq_dir(raw_dir, seq)
        if ext_dir is None:
            print(f"  Skipping (not found): {seq}")
            continue

        # IMU data
        imu_csv = ext_dir / "mav0" / "imu0" / "data.csv"
        imu = pd.read_csv(imu_csv)
        imu.columns = imu.columns.str.strip()
        # Handle both "#timestamp [ns]" and "#timestamp" column names
        ts_col = [c for c in imu.columns if c.startswith("#timestamp")][0]
        imu.rename(columns={ts_col: "ts"}, inplace=True)
        imu.sort_values("ts", inplace=True)

        # Ground truth
        gt_csv = ext_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        gt = pd.read_csv(gt_csv)
        gt.columns = gt.columns.str.strip()
        ts_col = [c for c in gt.columns if c.startswith("#timestamp")][0]
        gt.rename(columns={ts_col: "ts"}, inplace=True)
        # Normalize quaternion column names (may have " []" suffix)
        qcols = {}
        for base in ("q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z"):
            match = [c for c in gt.columns if c.startswith(base)]
            if match:
                qcols[match[0]] = base
        gt.rename(columns=qcols, inplace=True)
        gt.sort_values("ts", inplace=True)

        # Merge IMU readings onto GT timestamps (one row per GT timestamp)
        gt_cols = ["ts", "q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z"]
        # Include gyro bias columns for correction
        bias_cols = [c for c in gt.columns if c.startswith("b_w_RS_S")]
        imu_cols = [c for c in imu.columns if c != "ts"]
        merged = pd.merge_asof(
            gt[gt_cols + bias_cols],
            imu[["ts"] + imu_cols],
            on="ts", direction="nearest",
        )

        acc = merged[["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]].values
        gyr = merged[["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]].values
        # Subtract gyro bias estimated by the filter
        if bias_cols:
            gyr_bias = merged[sorted(bias_cols)].values  # x, y, z order
            gyr = gyr - gyr_bias
        quat = merged[["q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z"]].values

        quat = fix_quaternion_flips(quat)

        print(f"  Writing {out_path.name}  ({len(acc)} samples)")
        write_hdf5(out_path, acc, gyr, quat, dt=1.0 / 200.0)
