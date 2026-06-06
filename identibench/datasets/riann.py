"""RIANN IMU orientation-estimation datasets and benchmarks.

Ports the six datasets used by the RIANN paper into IdentiBench so anyone can
download them and evaluate their own model (neural network, complementary
filter, …) under RIANN's protocol.

    Weber, Gühmann, Seel. "RIANN — A Robust Neural Network Outperforms Attitude
    Estimation Filters." AI 2021, 2(3):444-463. doi:10.3390/ai2030028

Two flavours are exposed:

* **Per-source** datasets (``broad``, ``tumvi``, ``oxiod``, ``euroc``,
  ``repoimu``, ``caruso``) — general IMU orientation datasets, each downloaded
  under its own ``dataset_id``. Their files are all treated as *test*, so you can
  grab one small dataset and evaluate on it in isolation (mirrors the existing
  ``BenchmarkIMU_*`` style). These are not RIANN-specific; RIANN merely used them.
* **Combined** corpus (``riann``) — reproduces the paper's pooled-train /
  cross-dataset-test protocol in one ``dataset_id``.

Data format (per RIANN's ``write_hdf5``): each file holds 1-D float32 datasets
``acc_x..acc_z`` (m/s²), ``gyr_x..gyr_z`` (rad/s), ``dt`` (s, stored as an
``(N,)`` array), the ground-truth quaternion ``opt_a..opt_d`` (w,x,y,z), and a
``movement_mask`` (1=moving). So ``u_cols`` is the 7-tuple acc+gyr+dt and
``y_cols`` is the quaternion.

Evaluation. RIANN reports the *masked, first-sample-aligned* inclination error
plus its 99th percentile, per dataset. The IdentiBench ``metric_func`` slot only
sees ``(y_pred, y_true)`` (the loader drops ``movement_mask``), so it carries an
aligned-but-unmasked headline number; the faithful recipe lives in
``custom_test_evaluation`` (``riann_eval``), which re-opens each test file to
recover ``movement_mask`` and emits per-source RMSE and p99 into
``custom_scores`` (flattened to ``cs_*`` columns). The faithful numbers are the
``cs_*/incl_rmse_deg`` / ``cs_*/incl_p99_deg`` entries — not ``metric_score``.
"""

__all__ = [
    "aligned_inclination_rmse_deg",
    "riann_eval",
    "dl_riann",
    "dl_broad",
    "dl_tumvi",
    "dl_oxiod",
    "dl_euroc",
    "dl_repoimu",
    "dl_caruso",
    "BenchmarkRIANN_Inclination",
    "BenchmarkBROAD_Inclination",
    "BenchmarkTUMVI_Inclination",
    "BenchmarkOxIOD_Inclination",
    "BenchmarkEuRoC_Inclination",
    "BenchmarkRepoIMU_Inclination",
    "BenchmarkCaruso_Inclination",
    "RIANN_U_COLS",
    "RIANN_Y_COLS",
    "riann_benchmarks",
]

import shutil
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from ..benchmark import BenchmarkSpecSimulation
from ..metrics import _inclination_angle
from ._riann_prep import PREPARERS, SOURCE_DIRS

RIANN_U_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "dt"]
RIANN_Y_COLS = ["opt_a", "opt_b", "opt_c", "opt_d"]

# RIANN combined-corpus split rules (from riann/data.py).
MYON_VALID_IDS = {14, 39, 21}
MYON_TEST_IDS = {29, 22, 35}
TUMVI_TRAIN_ROOMS = {"room1", "room2", "room3"}

# ───────────────────────── quaternion helpers (pure numpy) ─────────────────────────


def _qnorm(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _qinv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (conjugate). Shape (..., 4), [w,x,y,z]."""
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def _qmult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b. Shapes broadcast on (..., 4), [w,x,y,z]."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _aligned_inclination_rad(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample inclination (tilt) error in radians, after removing the
    constant orientation offset between the two frames.

    The estimate is aligned to ground truth at the first sample where both are
    finite (``offset = gt0 ⊗ inv(est0)``), exactly as RIANN does before scoring.
    Samples where either quaternion is non-finite are returned as NaN.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = min(len(pred), len(true))
    pred, true = pred[:n], true[:n]

    out = np.full(n, np.nan)
    finite = np.isfinite(pred).all(-1) & np.isfinite(true).all(-1)
    if not finite.any():
        return out

    i0 = int(np.argmax(finite))  # first finite index
    offset = _qmult(_qnorm(true[i0]), _qinv(_qnorm(pred[i0])))  # (4,)
    pred_aligned = _qmult(offset[None, :], _qnorm(pred))  # (N, 4)
    ang = _inclination_angle(pred_aligned, true)  # (N,) radians
    out[finite] = ang[finite]
    return out


def aligned_inclination_rmse_deg(inp: np.ndarray, targ: np.ndarray) -> float:
    """RMS inclination error in degrees after first-sample alignment.

    Headline metric (``metric_func``). Aligned like RIANN but **not** masked by
    ``movement_mask`` (which the loader does not surface); the masked + 99th-pct
    numbers come from :func:`riann_eval` in ``custom_scores``. NaN ground-truth
    samples are ignored.
    """
    incl = _aligned_inclination_rad(inp, targ)
    incl = incl[np.isfinite(incl)]
    if incl.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(incl**2)) * 180.0 / np.pi)


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


def _myon_role(fname: str) -> str:
    i = int(fname.split("_")[0])
    if i in MYON_VALID_IDS:
        return "valid"
    if i in MYON_TEST_IDS:
        return "test"
    return "train"


def _tumvi_role(fname: str) -> str:
    return "train" if any(r in fname for r in TUMVI_TRAIN_ROOMS) else "valid"


def _riann_role(source: str, fname: str) -> str:
    if source == "Myon":
        return _myon_role(fname)
    if source == "TUM-VI":
        return _tumvi_role(fname)
    return "test"  # OxIOD, EuRoC-MAV, RepoIMU, Caruso-Sassari


def _prepare(save_path, sources, role_fn) -> None:
    """Download + convert the given sources, then route every file into
    ``save_path/<role>/<Source>__<stem>.hdf5``.

    Raw archives are cached in a shared ``_riann_raw`` dir next to the dataset so
    they are not re-downloaded across the per-source and combined datasets.
    Conversion goes to a private staging dir which is removed afterwards.
    """
    save_path = Path(save_path)
    raw = save_path.parent / "_riann_raw"
    stage = save_path / "_stage"
    raw.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)

    for name in sources:
        mod = PREPARERS[name]
        mod.download(raw)
        mod.convert(raw, stage)

    wanted = {SOURCE_DIRS[name] for name in sources}  # excludes Caruso-Sassari_orig
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


def dl_riann(save_path, force_download: bool = False) -> None:
    """Materialize the full combined RIANN corpus (all six sources) with the
    paper's cross-dataset train/valid/test split."""
    _prepare(save_path, list(SOURCE_DIRS), _riann_role)


def _dl_single(save_path, name: str) -> None:
    _prepare(save_path, [name], lambda source, fname: "test")


def dl_broad(save_path, force_download: bool = False) -> None:
    _dl_single(save_path, "broad")


def dl_tumvi(save_path, force_download: bool = False) -> None:
    _dl_single(save_path, "tumvi")


def dl_oxiod(save_path, force_download: bool = False) -> None:
    _dl_single(save_path, "oxiod")


def dl_euroc(save_path, force_download: bool = False) -> None:
    _dl_single(save_path, "euroc")


def dl_repoimu(save_path, force_download: bool = False) -> None:
    _dl_single(save_path, "repoimu")


def dl_caruso(save_path, force_download: bool = False) -> None:
    _dl_single(save_path, "caruso")


# ───────────────────────── benchmark specifications ─────────────────────────


def _spec(name: str, dataset_id: str, download_func) -> BenchmarkSpecSimulation:
    return BenchmarkSpecSimulation(
        name=name,
        dataset_id=dataset_id,
        u_cols=RIANN_U_COLS,
        y_cols=RIANN_Y_COLS,
        metric_func=aligned_inclination_rmse_deg,
        custom_test_evaluation=riann_eval,
        download_func=download_func,
        sampling_time=None,  # per-sample dt is the 7th u_col; rate is not constant across the corpus
        init_window=0,  # full-sequence orientation simulation
    )


BenchmarkRIANN_Inclination = _spec("BenchmarkRIANN_Inclination", "riann", dl_riann)
BenchmarkBROAD_Inclination = _spec("BenchmarkBROAD_Inclination", "broad", dl_broad)
BenchmarkTUMVI_Inclination = _spec("BenchmarkTUMVI_Inclination", "tumvi", dl_tumvi)
BenchmarkOxIOD_Inclination = _spec("BenchmarkOxIOD_Inclination", "oxiod", dl_oxiod)
BenchmarkEuRoC_Inclination = _spec("BenchmarkEuRoC_Inclination", "euroc", dl_euroc)
BenchmarkRepoIMU_Inclination = _spec("BenchmarkRepoIMU_Inclination", "repoimu", dl_repoimu)
BenchmarkCaruso_Inclination = _spec("BenchmarkCaruso_Inclination", "caruso", dl_caruso)

riann_benchmarks = {
    "RIANN_Inclination": BenchmarkRIANN_Inclination,
    "BROAD_Inclination": BenchmarkBROAD_Inclination,
    "TUMVI_Inclination": BenchmarkTUMVI_Inclination,
    "OxIOD_Inclination": BenchmarkOxIOD_Inclination,
    "EuRoC_Inclination": BenchmarkEuRoC_Inclination,
    "RepoIMU_Inclination": BenchmarkRepoIMU_Inclination,
    "Caruso_Inclination": BenchmarkCaruso_Inclination,
}
