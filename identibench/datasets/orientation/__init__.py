"""IMU orientation-estimation datasets and benchmarks.

Groups every orientation-from-IMU benchmark in IdentiBench. The task is the same
throughout — estimate a unit quaternion (``[w, x, y, z]``) from 6-axis IMU data
and score it with the inclination (tilt) error in degrees — so the datasets share
one home:

* **dfjimu** (:mod:`~identibench.datasets.orientation.dfjimu`) — the Weygers & Kok
  (2020) two-sensor dataset, exposing inclination and relative-orientation
  benchmarks.
* **RIANN** (:mod:`~identibench.datasets.orientation.riann`) — six general IMU
  orientation datasets (BROAD, TUM-VI, OxIOD, EuRoC-MAV, RepoIMU, Caruso-Sassari)
  plus the combined ``riann`` corpus reproducing the paper's pooled-train /
  cross-dataset-test protocol.

Quaternion target columns are ``q_w, q_x, q_y, q_z`` across all datasets; the
download/conversion helpers live in the private :mod:`._common` module.
"""

from .dfjimu import (
    dl_dfjimu,
    BenchmarkDFJIMU_Inclination,
    BenchmarkDFJIMU_Relative,
)
from .broad import dl_broad, BenchmarkBROAD_Inclination
from .tumvi import dl_tumvi, BenchmarkTUMVI_Inclination
from .oxiod import dl_oxiod, BenchmarkOxIOD_Inclination
from .euroc import dl_euroc, BenchmarkEuRoC_Inclination
from .repoimu import dl_repoimu, BenchmarkRepoIMU_Inclination
from .caruso import dl_caruso, BenchmarkCaruso_Inclination
from .riann import (
    dl_riann,
    riann_eval,
    riann_benchmarks,
    BenchmarkRIANN_Inclination,
)

# All orientation benchmarks in one registry (mirrors ``simulation_benchmarks``).
orientation_benchmarks = {
    "DFJIMU_Inclination": BenchmarkDFJIMU_Inclination,
    "DFJIMU_Relative": BenchmarkDFJIMU_Relative,
    "RIANN_Inclination": BenchmarkRIANN_Inclination,
    "BROAD_Inclination": BenchmarkBROAD_Inclination,
    "TUMVI_Inclination": BenchmarkTUMVI_Inclination,
    "OxIOD_Inclination": BenchmarkOxIOD_Inclination,
    "EuRoC_Inclination": BenchmarkEuRoC_Inclination,
    "RepoIMU_Inclination": BenchmarkRepoIMU_Inclination,
    "Caruso_Inclination": BenchmarkCaruso_Inclination,
}

__all__ = [
    "dl_dfjimu",
    "BenchmarkDFJIMU_Inclination",
    "BenchmarkDFJIMU_Relative",
    "dl_riann",
    "dl_broad",
    "dl_tumvi",
    "dl_oxiod",
    "dl_euroc",
    "dl_repoimu",
    "dl_caruso",
    "riann_eval",
    "riann_benchmarks",
    "BenchmarkRIANN_Inclination",
    "BenchmarkBROAD_Inclination",
    "BenchmarkTUMVI_Inclination",
    "BenchmarkOxIOD_Inclination",
    "BenchmarkEuRoC_Inclination",
    "BenchmarkRepoIMU_Inclination",
    "BenchmarkCaruso_Inclination",
    "orientation_benchmarks",
]
