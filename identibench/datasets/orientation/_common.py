"""Shared infrastructure for the IMU-orientation datasets.

Holds the download/extract/HDF5 helpers used by each source preparer, the
standardized channel layout (:data:`IMU_U_COLS` / :data:`IMU_Y_COLS`), the
benchmark-spec factory (:func:`_spec`), the faithful masked per-source
evaluation task (:class:`MaskedPooledInclination`), and the download-and-convert
driver (:func:`_prepare_sources`) shared by the per-source preparers.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from identibench.benchmark import BenchmarkSpec, EvalResult, pooled_scores_per_test_set
from identibench.dataset import Dataset
from identibench.metrics import _aligned_inclination_rad

# Standardized channel layout written by :func:`write_hdf5` and read by the specs.
IMU_U_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "dt"]
IMU_Y_COLS = ["q_w", "q_x", "q_y", "q_z"]


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


# ───────────────────────── faithful evaluation task ─────────────────────────


def _masked_inclination_deg(model, fpath: Path, spec: BenchmarkSpec) -> np.ndarray:
    """Run the model on one file and return its masked per-sample inclination errors.

    Full-sequence free run with empty ``y_init``; the prediction tail is aligned to
    the target length. The file's ``movement_mask`` is applied and non-finite errors
    (ground-truth NaN gaps) are dropped. Returns the remaining errors in degrees.
    """
    with h5py.File(fpath, "r") as f:
        u = np.stack([f[col][()] for col in spec.u_cols], axis=-1).astype(np.float32)
        y = np.stack([f[col][()] for col in spec.y_cols], axis=-1).astype(np.float32)
        attrs = dict(f.attrs)
        mask = np.asarray(f["movement_mask"][()], dtype=np.float64) if "movement_mask" in f else np.ones(len(y))
    y_pred = model(u, y[:0], attrs)
    incl = _aligned_inclination_rad(y_pred[-len(y) :], y)  # radians, NaN where invalid
    m = min(len(incl), len(mask))  # a model returning fewer samples shortens incl below the mask
    incl, mask = incl[:m], mask[:m]
    return incl[np.isfinite(incl) & (mask > 0.5)] * 180.0 / np.pi


@dataclass(frozen=True)
class MaskedPooledInclination:
    """Faithful RIANN evaluation as a task: masked, sample-pooled, first-sample-aligned
    inclination error in degrees.

    Each named test set (= source dataset) is scored as one sample pool with the RMS
    error and its ``percentile``-th percentile; a cross-set ``"all"`` pool is always
    appended and is the headline.
    """

    percentile: float = 99.0

    def _pool_scores(self, deg: np.ndarray) -> dict[str, float]:
        return {
            "incl_rmse_deg": float(np.sqrt(np.mean(deg**2))),
            f"incl_p{self.percentile:g}_deg": float(np.percentile(deg, self.percentile)),
        }

    def __call__(self, spec: BenchmarkSpec, model) -> EvalResult:
        scores = pooled_scores_per_test_set(
            spec, lambda f: _masked_inclination_deg(model, f, spec), self._pool_scores, all_set="all"
        )
        return EvalResult(scores=scores, headline=("all", "incl_rmse_deg"))


# ───────────────────────── benchmark spec factory ─────────────────────────


def _spec(name: str, dataset: Dataset) -> BenchmarkSpec:
    """Build the per-source orientation benchmark spec: every file of the source
    is one named test set, no training data is defined.

    The task is the faithful masked + sample-pooled evaluation; the headline is
    the cross-set ``"all"`` pool's ``incl_rmse_deg``.
    """
    return BenchmarkSpec(
        name=name,
        u_cols=IMU_U_COLS,
        y_cols=IMU_Y_COLS,
        train=[],
        valid=[],
        test_sets={dataset.dataset_id: [(dataset, "*.hdf5")]},
        task=MaskedPooledInclination(),
    )


# ───────────────────────── download / materialization ─────────────────────────


def _prepare_sources(save_path, preparers, force_download: bool = False) -> None:
    """Download + convert the given sources flat into ``save_path``.

    ``preparers`` is a list of ``(download_fn, convert_fn)`` pairs (each source
    module exposes these). Raw archives are cached in a shared
    ``_orientation_raw`` dir next to the dataset directory so they survive
    re-preparation and are shared across the orientation datasets; with
    ``force_download=True`` they are re-downloaded as well.
    """
    save_path = Path(save_path)
    raw = save_path.parent / "_orientation_raw"
    raw.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    for download_fn, convert_fn in preparers:
        download_fn(raw, force=force_download)
        convert_fn(raw, save_path, force=force_download)
