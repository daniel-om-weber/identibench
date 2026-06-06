__version__ = "0.3.1"

from .benchmark import (
    run_benchmark,
    run_benchmarks,
    BenchmarkSpecSimulation,
    BenchmarkSpecPrediction,
    TrainingContext,
    benchmark_results_to_dataframe,
    aggregate_benchmark_results,
)
from .utils import Sequence
from . import metrics
from . import datasets

# Workshop Benchmarks
from .datasets.workshop import (
    BenchmarkWH_Simulation,
    BenchmarkWH_Prediction,
    BenchmarkSilverbox_Simulation,
    BenchmarkSilverbox_Prediction,
    BenchmarkCascadedTanks_Simulation,
    BenchmarkCascadedTanks_Prediction,
    BenchmarkEMPS_Simulation,
    BenchmarkEMPS_Prediction,
    BenchmarkNoisyWH_Simulation,
    BenchmarkNoisyWH_Prediction,
    BenchmarkCED_Simulation,
    BenchmarkCED_Prediction,
)

# Robot Benchmarks
from .datasets.industrial_robot import (
    BenchmarkRobotForward_Simulation,
    BenchmarkRobotForward_Prediction,
    BenchmarkRobotInverse_Simulation,
    BenchmarkRobotInverse_Prediction,
)

# Ship Benchmark
from .datasets.ship import BenchmarkShip_Simulation, BenchmarkShip_Prediction

# Quadrotor Benchmarks
from .datasets.quad_pelican import BenchmarkQuadPelican_Simulation, BenchmarkQuadPelican_Prediction
from .datasets.quad_pi import BenchmarkQuadPi_Simulation, BenchmarkQuadPi_Prediction

# Orientation (IMU) Benchmarks — dfjimu dataset + the RIANN family
from .datasets.orientation import (
    BenchmarkDFJIMU_Inclination,
    BenchmarkDFJIMU_Relative,
    BenchmarkRIANN_Inclination,
    BenchmarkBROAD_Inclination,
    BenchmarkTUMVI_Inclination,
    BenchmarkOxIOD_Inclination,
    BenchmarkEuRoC_Inclination,
    BenchmarkRepoIMU_Inclination,
    BenchmarkCaruso_Inclination,
    orientation_benchmarks,
)

simulation_benchmarks = {
    "WH_Sim": BenchmarkWH_Simulation,
    "Silverbox_Sim": BenchmarkSilverbox_Simulation,
    "Tanks_Sim": BenchmarkCascadedTanks_Simulation,
    "CED_Sim": BenchmarkCED_Simulation,
    "EMPS_Sim": BenchmarkEMPS_Simulation,
    "NoisyWH_Sim": BenchmarkNoisyWH_Simulation,
    "RobotForward_Sim": BenchmarkRobotForward_Simulation,
    "RobotInverse_Sim": BenchmarkRobotInverse_Simulation,
    "Ship_Sim": BenchmarkShip_Simulation,
    "QuadPelican_Sim": BenchmarkQuadPelican_Simulation,
    "QuadPi_Sim": BenchmarkQuadPi_Simulation,
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

prediction_benchmarks = {
    "WH_Pred": BenchmarkWH_Prediction,
    "Silverbox_Pred": BenchmarkSilverbox_Prediction,
    "Tanks_Pred": BenchmarkCascadedTanks_Prediction,
    "CED_Pred": BenchmarkCED_Prediction,
    "EMPS_Pred": BenchmarkEMPS_Prediction,
    "NoisyWH_Pred": BenchmarkNoisyWH_Prediction,
    "RobotForward_Pred": BenchmarkRobotForward_Prediction,
    "RobotInverse_Pred": BenchmarkRobotInverse_Prediction,
    "Ship_Pred": BenchmarkShip_Prediction,
    "QuadPelican_Pred": BenchmarkQuadPelican_Prediction,
    "QuadPi_Pred": BenchmarkQuadPi_Prediction,
}

all_benchmarks = {**simulation_benchmarks, **prediction_benchmarks}

__all__ = [
    # Core API
    "run_benchmark",
    "run_benchmarks",
    "BenchmarkSpecSimulation",
    "BenchmarkSpecPrediction",
    "TrainingContext",
    "benchmark_results_to_dataframe",
    "aggregate_benchmark_results",
    "Sequence",
    # Subpackages / submodules
    "metrics",
    "datasets",
    # Benchmark registries
    "simulation_benchmarks",
    "prediction_benchmarks",
    "all_benchmarks",
    "orientation_benchmarks",
    # Workshop benchmark specs
    "BenchmarkWH_Simulation",
    "BenchmarkWH_Prediction",
    "BenchmarkSilverbox_Simulation",
    "BenchmarkSilverbox_Prediction",
    "BenchmarkCascadedTanks_Simulation",
    "BenchmarkCascadedTanks_Prediction",
    "BenchmarkEMPS_Simulation",
    "BenchmarkEMPS_Prediction",
    "BenchmarkNoisyWH_Simulation",
    "BenchmarkNoisyWH_Prediction",
    "BenchmarkCED_Simulation",
    "BenchmarkCED_Prediction",
    # Robot benchmark specs
    "BenchmarkRobotForward_Simulation",
    "BenchmarkRobotForward_Prediction",
    "BenchmarkRobotInverse_Simulation",
    "BenchmarkRobotInverse_Prediction",
    # Ship benchmark specs
    "BenchmarkShip_Simulation",
    "BenchmarkShip_Prediction",
    # Quadrotor benchmark specs
    "BenchmarkQuadPelican_Simulation",
    "BenchmarkQuadPelican_Prediction",
    "BenchmarkQuadPi_Simulation",
    "BenchmarkQuadPi_Prediction",
    # Orientation (IMU) benchmark specs
    "BenchmarkDFJIMU_Inclination",
    "BenchmarkDFJIMU_Relative",
    "BenchmarkRIANN_Inclination",
    "BenchmarkBROAD_Inclination",
    "BenchmarkTUMVI_Inclination",
    "BenchmarkOxIOD_Inclination",
    "BenchmarkEuRoC_Inclination",
    "BenchmarkRepoIMU_Inclination",
    "BenchmarkCaruso_Inclination",
]
