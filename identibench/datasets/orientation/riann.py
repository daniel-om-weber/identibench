"""Combined RIANN IMU orientation benchmark (pooled-train / cross-dataset-test).

Reproduces the protocol from the RIANN paper as a single multi-dataset
benchmark: it pools all six source datasets (BROAD, TUM-VI, OxIOD, EuRoC-MAV,
RepoIMU, Caruso-Sassari — each a self-contained sibling module here) and
assigns the paper's train/valid/test roles via explicit file patterns. The
sources are stored once; this benchmark adds no data of its own.

    Weber, Gühmann, Seel. "RIANN — A Robust Neural Network Outperforms Attitude
    Estimation Filters." AI 2021, 2(3):444-463. doi:10.3390/ai2030028

Data format and evaluation are shared with the per-source benchmarks and live
in :mod:`._common`: each file holds 1-D float32 ``acc_x..acc_z`` (m/s²),
``gyr_x..gyr_z`` (rad/s), ``dt`` (s), the ground-truth quaternion ``q_w..q_z``,
and a ``movement_mask``. Evaluation is the
:class:`._common.MaskedPooledInclination` task: masked + first-sample-aligned
inclination errors, sample-pooled per source (one named test set per source)
plus a cross-set ``"all"`` pool, which is the headline.
"""

__all__ = [
    "BenchmarkRIANN_Inclination",
    "riann_benchmarks",
]

from identibench.benchmark import BenchmarkSpec

from . import broad, caruso, euroc, oxiod, repoimu, tumvi
from ._common import IMU_U_COLS, IMU_Y_COLS, MaskedPooledInclination

# RIANN split rules (from riann/data.py), keyed on the Myon trial number in the
# filename prefix (BROAD files are `NN_description.hdf5`, NN = 01..39) and the
# TUM-VI room number (`*room{N}.hdf5`, N = 1..6).
MYON_VALID_IDS = (14, 21, 39)
MYON_TEST_IDS = (22, 29, 35)
MYON_TRAIN_IDS = tuple(i for i in range(1, 40) if i not in MYON_VALID_IDS + MYON_TEST_IDS)
TUMVI_TRAIN_ROOMS = (1, 2, 3)
TUMVI_VALID_ROOMS = (4, 5, 6)

BenchmarkRIANN_Inclination = BenchmarkSpec(
    name="BenchmarkRIANN_Inclination",
    u_cols=IMU_U_COLS,
    y_cols=IMU_Y_COLS,
    train=[(broad.broad_dataset, f"{i:02d}_*.hdf5") for i in MYON_TRAIN_IDS]
    + [(tumvi.tumvi_dataset, f"*room{n}.hdf5") for n in TUMVI_TRAIN_ROOMS],
    valid=[(broad.broad_dataset, f"{i:02d}_*.hdf5") for i in MYON_VALID_IDS]
    + [(tumvi.tumvi_dataset, f"*room{n}.hdf5") for n in TUMVI_VALID_ROOMS],
    # One named test set per source; TUM-VI is train/valid-only in this split.
    test_sets={
        "broad": [(broad.broad_dataset, f"{i:02d}_*.hdf5") for i in MYON_TEST_IDS],
        "oxiod": [(oxiod.oxiod_dataset, "*.hdf5")],
        "euroc": [(euroc.euroc_dataset, "*.hdf5")],
        "repoimu": [(repoimu.repoimu_dataset, "*.hdf5")],
        "caruso": [(caruso.caruso_dataset, "*.hdf5")],
    },
    task=MaskedPooledInclination(),
)

# The RIANN family: the combined benchmark + the six per-source benchmarks.
riann_benchmarks = {
    "RIANN_Inclination": BenchmarkRIANN_Inclination,
    "BROAD_Inclination": broad.BenchmarkBROAD_Inclination,
    "TUMVI_Inclination": tumvi.BenchmarkTUMVI_Inclination,
    "OxIOD_Inclination": oxiod.BenchmarkOxIOD_Inclination,
    "EuRoC_Inclination": euroc.BenchmarkEuRoC_Inclination,
    "RepoIMU_Inclination": repoimu.BenchmarkRepoIMU_Inclination,
    "Caruso_Inclination": caruso.BenchmarkCaruso_Inclination,
}
