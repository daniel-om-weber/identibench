"""Combined RIANN IMU orientation corpus (pooled-train / cross-dataset-test).

Reproduces the protocol from the RIANN paper in a single ``dataset_id``: it pools
all six source datasets (BROAD, TUM-VI, OxIOD, EuRoC-MAV, RepoIMU,
Caruso-Sassari — each a self-contained sibling module here) and assigns the
paper's train/valid/test roles across them.

    Weber, Gühmann, Seel. "RIANN — A Robust Neural Network Outperforms Attitude
    Estimation Filters." AI 2021, 2(3):444-463. doi:10.3390/ai2030028

Data format and evaluation are shared with the per-source datasets and live in
:mod:`._common`: each file holds 1-D float32 ``acc_x..acc_z`` (m/s²),
``gyr_x..gyr_z`` (rad/s), ``dt`` (s), the ground-truth quaternion ``q_w..q_z``,
and a ``movement_mask``. The headline ``metric_func`` is the aligned-but-unmasked
inclination RMSE; the faithful masked + 99th-percentile per-source numbers come
from :func:`._common.riann_eval` (re-exported here) via
``custom_test_evaluation`` and surface as ``cs_*`` columns.
"""

__all__ = [
    "dl_riann",
    "riann_eval",
    "BenchmarkRIANN_Inclination",
    "riann_benchmarks",
]

from . import broad, caruso, euroc, oxiod, repoimu, tumvi
from ._common import _prepare, _spec, riann_eval

# RIANN combined-corpus split rules (from riann/data.py).
MYON_VALID_IDS = {14, 39, 21}
MYON_TEST_IDS = {29, 22, 35}
TUMVI_TRAIN_ROOMS = {"room1", "room2", "room3"}

# (download, convert, source_dir) for every source pooled into the corpus.
_SOURCES = [
    (broad.download, broad.convert, broad.SOURCE_DIR),
    (tumvi.download, tumvi.convert, tumvi.SOURCE_DIR),
    (oxiod.download, oxiod.convert, oxiod.SOURCE_DIR),
    (euroc.download, euroc.convert, euroc.SOURCE_DIR),
    (repoimu.download, repoimu.convert, repoimu.SOURCE_DIR),
    (caruso.download, caruso.convert, caruso.SOURCE_DIR),
]


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
    if source == broad.SOURCE_DIR:  # "Myon"
        return _myon_role(fname)
    if source == tumvi.SOURCE_DIR:  # "TUM-VI"
        return _tumvi_role(fname)
    return "test"  # OxIOD, EuRoC-MAV, RepoIMU, Caruso-Sassari


def dl_riann(save_path, force_download: bool = False) -> None:
    """Materialize the full combined RIANN corpus (all six sources) with the
    paper's cross-dataset train/valid/test split."""
    _prepare(save_path, _SOURCES, _riann_role, force_download=force_download)


BenchmarkRIANN_Inclination = _spec("BenchmarkRIANN_Inclination", "riann", dl_riann)

# The RIANN family: the combined corpus + the six datasets it pools.
riann_benchmarks = {
    "RIANN_Inclination": BenchmarkRIANN_Inclination,
    "BROAD_Inclination": broad.BenchmarkBROAD_Inclination,
    "TUMVI_Inclination": tumvi.BenchmarkTUMVI_Inclination,
    "OxIOD_Inclination": oxiod.BenchmarkOxIOD_Inclination,
    "EuRoC_Inclination": euroc.BenchmarkEuRoC_Inclination,
    "RepoIMU_Inclination": repoimu.BenchmarkRepoIMU_Inclination,
    "Caruso_Inclination": caruso.BenchmarkCaruso_Inclination,
}
