"""Shared utilities for data preparation: download, HDF5 write, quaternion correction."""

import tarfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Streaming HTTP download with progress bar, skip if file exists."""
    dest = Path(dest)
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=(30, 300))
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total or None, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def extract_archive(path: Path, dest: Path, members: list[str] | None = None) -> Path:
    """Extract zip, tar, or tar.gz archive to dest directory."""
    path, dest = Path(path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            targets = members or zf.namelist()
            for m in targets:
                zf.extract(m, dest)
    elif path.name.endswith(".tar.gz") or path.name.endswith(".tgz") or path.suffix == ".tar":
        mode = "r:gz" if (path.name.endswith(".tar.gz") or path.name.endswith(".tgz")) else "r"
        with tarfile.open(path, mode) as tf:
            if members:
                for m in members:
                    tf.extract(m, dest, filter="data")
            else:
                tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"Unsupported archive format: {path}")
    return dest


def write_hdf5(
    path: Path,
    acc: np.ndarray,
    gyr: np.ndarray,
    quat: np.ndarray,
    dt: float,
    mag: np.ndarray | None = None,
    movement_mask: np.ndarray | None = None,
) -> None:
    """Write standardized HDF5 with 1D float32 datasets.

    Args:
        acc: (N, 3) accelerometer — columns are x, y, z in m/s^2
        gyr: (N, 3) gyroscope — columns are x, y, z in rad/s
        quat: (N, 4) orientation quaternion — columns are w, x, y, z
        dt: sampling interval in seconds (scalar, broadcast to all samples)
        mag: optional (N, 3) magnetometer
        movement_mask: optional (N,) boolean/float mask (1=moving, 0=static)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = acc.shape[0]
    with h5py.File(path, "w") as f:
        f.create_dataset("acc_x", data=acc[:, 0].astype(np.float32))
        f.create_dataset("acc_y", data=acc[:, 1].astype(np.float32))
        f.create_dataset("acc_z", data=acc[:, 2].astype(np.float32))
        f.create_dataset("gyr_x", data=gyr[:, 0].astype(np.float32))
        f.create_dataset("gyr_y", data=gyr[:, 1].astype(np.float32))
        f.create_dataset("gyr_z", data=gyr[:, 2].astype(np.float32))
        f.create_dataset("opt_a", data=quat[:, 0].astype(np.float32))
        f.create_dataset("opt_b", data=quat[:, 1].astype(np.float32))
        f.create_dataset("opt_c", data=quat[:, 2].astype(np.float32))
        f.create_dataset("opt_d", data=quat[:, 3].astype(np.float32))
        f.create_dataset("dt", data=np.full(n, dt, dtype=np.float32))
        if mag is not None:
            f.create_dataset("mag_x", data=mag[:, 0].astype(np.float32))
            f.create_dataset("mag_y", data=mag[:, 1].astype(np.float32))
            f.create_dataset("mag_z", data=mag[:, 2].astype(np.float32))
        if movement_mask is not None:
            f.create_dataset("movement_mask", data=movement_mask.astype(np.float32))
        else:
            f.create_dataset("movement_mask", data=np.ones(n, dtype=np.float32))


def fix_quaternion_flips(quat: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Detect and correct quaternion sign flips in a timeseries.

    When the Euclidean distance between consecutive quaternions exceeds the
    threshold, flip the sign from that point onward.  q and -q represent the
    same rotation, but sign flips cause problems for learning.
    """
    quat = quat.copy()
    for i in range(1, len(quat)):
        if np.linalg.norm(quat[i] - quat[i - 1]) > threshold:
            quat[i:] *= -1
    return quat


def interpolate_nans(arr: np.ndarray, limit: int | None = None) -> np.ndarray:
    """Linear interpolation of NaN gaps, per column.

    Args:
        arr: 1-D or 2-D array with potential NaN values
        limit: if set, NaN runs longer than *limit* are left as NaN
    """
    arr = arr.copy()
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    for col in range(arr.shape[1]):
        y = arr[:, col]
        nans = np.isnan(y)
        if not nans.any():
            continue
        valid = ~nans
        if valid.sum() < 2:
            continue
        idx = np.arange(len(y))
        y[nans] = np.interp(idx[nans], idx[valid], y[valid])
        # Re-NaN runs that exceed the limit
        if limit is not None:
            nan_orig = nans.copy()
            i = 0
            while i < len(nan_orig):
                if nan_orig[i]:
                    j = i
                    while j < len(nan_orig) and nan_orig[j]:
                        j += 1
                    if j - i > limit:
                        arr[i:j, col] = np.nan
                    i = j
                else:
                    i += 1
    if squeeze:
        arr = arr[:, 0]
    return arr
