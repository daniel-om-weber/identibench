"""TUM-VI dataset — 6 room sequences.

Source: https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
Output: data/TUM-VI/TumVI::room{N}.hdf5  (6 files)

Archives are large (~1.6 GB each, includes camera images).
Only imu.txt and gt_imu.csv are streamed from each tar — the full
archive is never saved to disk.
"""

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from ._common import (
    fix_quaternion_flips,
    write_hdf5,
)

_BASE = "https://cvg.cit.tum.de/tumvi/exported/euroc/512_16"
_ROOMS = list(range(1, 7))
_NEEDED = {"imu.txt", "gt_imu.csv"}


def download(raw_dir: Path) -> None:
    """Stream each tar and extract only imu.txt + gt_imu.csv (skip camera data)."""
    raw_dir = raw_dir / "tumvi"
    for n in _ROOMS:
        seq_name = f"dataset-room{n}_512_16"
        ext_dir = raw_dir / seq_name / "dso"
        # Skip if both files already extracted
        if (ext_dir / "imu.txt").exists() and (ext_dir / "gt_imu.csv").exists():
            print(f"  Already extracted: {seq_name}/dso/imu.txt + gt_imu.csv")
            continue

        url = f"{_BASE}/{seq_name}.tar"
        print(f"  Streaming {seq_name}.tar (extracting IMU+GT only) ...")
        _stream_extract(url, raw_dir)


def _stream_extract(url: str, dest: Path) -> None:
    """Stream a tar from URL, extract only _NEEDED files, then close early."""
    resp = requests.get(url, stream=True, timeout=(30, 300))
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    found = set()
    with tqdm(total=total or None, unit="B", unit_scale=True, desc="streaming") as bar:
        # Wrap response stream with a counting reader for progress
        raw = _ProgressReader(resp.raw, bar)
        with tarfile.open(fileobj=raw, mode="r|") as tf:
            for member in tf:
                basename = Path(member.name).name
                if basename in _NEEDED:
                    tf.extract(member, dest, filter="data")
                    found.add(basename)
                    print(f"    extracted: {member.name}")
                    if found == _NEEDED:
                        break  # got everything, stop downloading
    resp.close()


class _ProgressReader:
    """Wrap a file-like stream to update a tqdm progress bar on reads."""

    def __init__(self, raw, bar):
        self._raw = raw
        self._bar = bar

    def read(self, n=-1):
        data = self._raw.read(n)
        if data:
            self._bar.update(len(data))
        return data

    def readinto(self, b):
        n = self._raw.readinto(b)
        if n:
            self._bar.update(n)
        return n


def convert(raw_dir: Path, out_dir: Path) -> None:
    raw_dir = raw_dir / "tumvi"
    out_dir = out_dir / "TUM-VI"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in _ROOMS:
        out_path = out_dir / f"TumVI::room{n}.hdf5"
        if out_path.exists():
            print(f"  Skipping (exists): {out_path.name}")
            continue

        seq_name = f"dataset-room{n}_512_16"
        ext_dir = raw_dir / seq_name

        # ── Load IMU data ──
        imu_path = ext_dir / "dso" / "imu.txt"
        imu = pd.read_csv(
            imu_path, sep=" ", comment="#", header=None,
            names=["ts", "gyr_x", "gyr_y", "gyr_z", "acc_x", "acc_y", "acc_z"],
        )

        # ── Load ground truth ──
        gt_path = ext_dir / "dso" / "gt_imu.csv"
        gt = pd.read_csv(
            gt_path, sep=",", comment="#", header=None,
            names=["ts", "x", "y", "z", "qw", "qx", "qy", "qz"],
        )

        imu = imu.sort_values("ts").reset_index(drop=True)
        gt = gt.sort_values("ts").reset_index(drop=True)

        # ── De-duplicate burst IMU samples ──
        # IMU has ~200 Hz but includes burst readings at sub-ms intervals.
        # Keep one sample per ≥4 ms to get a clean ~200 Hz signal.
        ts_ns = imu["ts"].values
        keep = [0]
        for i in range(1, len(ts_ns)):
            if ts_ns[i] - ts_ns[keep[-1]] >= 4_000_000:
                keep.append(i)
        imu = imu.iloc[keep].reset_index(drop=True)

        # ── Merge GT quaternions onto IMU timestamps ──
        merged = pd.merge_asof(
            imu,
            gt[["ts", "qw", "qx", "qy", "qz"]],
            on="ts",
            direction="nearest",
            tolerance=int(5e6),  # 5 ms
        )

        # ── NaN-mask quaternions inside GT tracking gaps ──
        imu_ts = merged["ts"].values
        gt_ts = gt["ts"].values
        gt_dt = np.diff(gt_ts)
        gap_starts = gt_ts[:-1][gt_dt > 20e6]  # gaps > 20 ms
        gap_ends = gt_ts[1:][gt_dt > 20e6]
        quat_cols = ["qw", "qx", "qy", "qz"]
        for gs, ge in zip(gap_starts, gap_ends):
            mask = (imu_ts >= gs) & (imu_ts <= ge)
            merged.loc[mask, quat_cols] = np.nan
        merged.loc[imu_ts < gt_ts[0], quat_cols] = np.nan
        merged.loc[imu_ts > gt_ts[-1], quat_cols] = np.nan

        # ── Quaternion flip correction (skip NaN) ──
        quat = merged[quat_cols].values
        valid = np.where(~np.isnan(quat[:, 0]))[0]
        for i in range(1, len(valid)):
            curr, prev = valid[i], valid[i - 1]
            if np.linalg.norm(quat[curr] - quat[prev]) > 1.0:
                quat[curr:] *= -1

        acc = merged[["acc_x", "acc_y", "acc_z"]].values
        gyr = merged[["gyr_x", "gyr_y", "gyr_z"]].values

        print(f"  Writing {out_path.name}  ({len(acc)} samples)")
        write_hdf5(out_path, acc, gyr, quat, dt=0.005)
