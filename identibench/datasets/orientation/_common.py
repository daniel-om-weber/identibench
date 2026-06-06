"""Shared infrastructure for the IMU-orientation datasets.

Holds the download/extract/HDF5 helpers used by each source preparer, the
standardized channel layout (:data:`IMU_U_COLS` / :data:`IMU_Y_COLS`), the
benchmark-spec factory (:func:`_spec`), the faithful masked per-source
evaluation (:func:`riann_eval`), and the download-and-route driver
(:func:`_prepare`) shared by the per-source loaders and the combined corpus.
"""

import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm

from identibench.benchmark import BenchmarkSpecSimulation
from identibench.metrics import _aligned_inclination_rad, aligned_inclination_rmse_deg

# Standardized channel layout written by :func:`write_hdf5` and read by the specs.
IMU_U_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "dt"]
IMU_Y_COLS = ["q_w", "q_x", "q_y", "q_z"]


def download_file(url: str, dest: Path, chunk_size: int = 8192, force: bool = False) -> Path:
    """Streaming HTTP download with progress bar; skip if file exists unless ``force``."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"  Already downloaded: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=(30, 300))
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as bar:
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
        f.create_dataset("q_w", data=quat[:, 0].astype(np.float32))
        f.create_dataset("q_x", data=quat[:, 1].astype(np.float32))
        f.create_dataset("q_y", data=quat[:, 2].astype(np.float32))
        f.create_dataset("q_z", data=quat[:, 3].astype(np.float32))
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


# ───────────────────────── benchmark spec factory ─────────────────────────


def _spec(name: str, dataset_id: str, download_func) -> BenchmarkSpecSimulation:
    """Build the orientation benchmark spec shared by every source.

    Headline ``metric_func`` is the first-sample-aligned inclination RMSE; the
    faithful masked + 99th-percentile per-source numbers come from
    :func:`riann_eval` via ``custom_test_evaluation``.
    """
    return BenchmarkSpecSimulation(
        name=name,
        dataset_id=dataset_id,
        u_cols=IMU_U_COLS,
        y_cols=IMU_Y_COLS,
        metric_func=aligned_inclination_rmse_deg,
        custom_test_evaluation=riann_eval,
        download_func=download_func,
        sampling_time=None,  # per-sample dt is the 7th u_col; rate is not constant across the corpus
        init_window=0,  # full-sequence orientation simulation
    )


# ───────────────────────── faithful evaluation ─────────────────────────


def _source_of(fpath) -> str:
    """Recover the source dataset name from a routed filename ``<Source>__<stem>``."""
    stem = Path(fpath).name
    return stem.split("__", 1)[0] if "__" in stem else Path(fpath).parent.name


def riann_eval(test_results, spec) -> dict:
    """Faithful RIANN evaluation: masked + first-sample-aligned inclination
    error and its 99th percentile, broken down per source dataset.

    ``test_results`` is the list of ``(y_pred, y_true)`` tuples in
    ``spec.test_files`` order; we re-open each file to recover ``movement_mask``
    (and to drop ground-truth NaN gaps). Returns ``{"<src>/incl_rmse_deg": …,
    "<src>/incl_p99_deg": …, "all/incl_rmse_deg": …, "all/incl_p99_deg": …}``.
    """
    files = list(spec.test_files)
    if len(files) != len(test_results):
        # Loader skips unreadable files, which would misalign the zip. Bail to a
        # single pooled "all" score rather than mis-attributing per-source numbers.
        files = [None] * len(test_results)

    per_source: dict[str, list[np.ndarray]] = defaultdict(list)
    for (y_pred, y_true), fpath in zip(test_results, files):
        incl = _aligned_inclination_rad(y_pred, y_true)  # radians, NaN where invalid
        mask = np.ones(len(incl))
        if fpath is not None:
            try:
                with h5py.File(fpath, "r") as f:
                    if "movement_mask" in f:
                        mask = np.asarray(f["movement_mask"][()], dtype=np.float64)
            except OSError:
                pass
        m = min(len(incl), len(mask))
        incl, mask = incl[:m], mask[:m]
        valid = np.isfinite(incl) & (mask > 0.5)
        source = _source_of(fpath) if fpath is not None else "all"
        per_source[source].append(incl[valid])

    scores: dict[str, float] = {}
    pooled: list[np.ndarray] = []
    for source, chunks in sorted(per_source.items()):
        v = np.concatenate(chunks) if chunks else np.empty(0)
        if v.size == 0:
            continue
        deg = v * 180.0 / np.pi
        scores[f"{source}/incl_rmse_deg"] = float(np.sqrt(np.mean(deg**2)))
        scores[f"{source}/incl_p99_deg"] = float(np.percentile(deg, 99))
        pooled.append(deg)
    if pooled:
        deg = np.concatenate(pooled)
        scores["all/incl_rmse_deg"] = float(np.sqrt(np.mean(deg**2)))
        scores["all/incl_p99_deg"] = float(np.percentile(deg, 99))
    return scores


# ───────────────────────── download / materialization ─────────────────────────


def _test_role(source: str, fname: str) -> str:
    """Role function for a standalone per-source dataset: every file is test."""
    return "test"


def _prepare(save_path, preparers, role_fn, force_download: bool = False) -> None:
    """Download + convert the given sources, then route every file into
    ``save_path/<role>/<Source>__<stem>.hdf5``.

    ``preparers`` is a list of ``(download_fn, convert_fn, source_dir)`` triples
    (each source module exposes these). Raw archives are cached in a shared
    ``_orientation_raw`` dir next to the dataset so they are not re-downloaded
    across the per-source and combined datasets. Conversion goes to a private
    staging dir which is removed afterwards. With ``force_download=True`` the
    routed train/valid/test dirs are cleared first and every source is
    re-downloaded and re-converted from scratch.
    """
    save_path = Path(save_path)
    raw = save_path.parent / "_orientation_raw"
    stage = save_path / "_stage"
    raw.mkdir(parents=True, exist_ok=True)
    if force_download:
        for role in ("train", "valid", "test"):
            shutil.rmtree(save_path / role, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)

    for download_fn, convert_fn, _ in preparers:
        download_fn(raw, force=force_download)
        convert_fn(raw, stage, force=force_download)

    wanted = {source_dir for _, _, source_dir in preparers}  # e.g. excludes Caruso-Sassari_orig
    for src_dir in sorted(p for p in stage.iterdir() if p.is_dir()):
        source = src_dir.name
        if source not in wanted:
            continue
        for f in sorted(src_dir.glob("*.hdf5")):
            role = role_fn(source, f.name)
            if role is None:
                continue
            dest_dir = save_path / role
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest_dir / f"{source}__{f.name}"))

    shutil.rmtree(stage, ignore_errors=True)
