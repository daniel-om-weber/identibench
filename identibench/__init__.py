__version__ = "0.1.4"

from .benchmark import run_benchmark,run_multiple_benchmarks, BenchmarkSpecSimulation, BenchmarkSpecPrediction, TrainingContext
from . import metrics
from . import datasets

# Workshop Benchmarks
from .datasets.workshop import (
    BenchmarkWH_Simulation, BenchmarkWH_Prediction,
    BenchmarkSilverbox_Simulation, BenchmarkSilverbox_Prediction,
    BenchmarkCascadedTanks_Simulation, BenchmarkCascadedTanks_Prediction,
    BenchmarkEMPS_Simulation, BenchmarkEMPS_Prediction,
    BenchmarkNoisyWH_Simulation, BenchmarkNoisyWH_Prediction,
    BenchmarkCED_Simulation, BenchmarkCED_Prediction
)
# Robot Benchmarks
from .datasets.industrial_robot import (
    BenchmarkRobotForward_Simulation, BenchmarkRobotForward_Prediction,
    BenchmarkRobotInverse_Simulation, BenchmarkRobotInverse_Prediction
)
# Ship Benchmark
from .datasets.ship import BenchmarkShip_Simulation, BenchmarkShip_Prediction
# Quadrotor Benchmarks
from .datasets.quad_pelican import BenchmarkQuadPelican_Simulation, BenchmarkQuadPelican_Prediction
from .datasets.quad_pi import BenchmarkQuadPi_Simulation, BenchmarkQuadPi_Prediction
# Broad Benchmark
# from .datasets.broad import BenchmarkBroad_Simulation, BenchmarkBroad_Prediction

simulation_benchmarks = {
    'WH_Sim': BenchmarkWH_Simulation,
    'Silverbox_Sim': BenchmarkSilverbox_Simulation,
    'Tanks_Sim': BenchmarkCascadedTanks_Simulation,
    'EMPS_Sim': BenchmarkEMPS_Simulation,
    'NoisyWH_Sim': BenchmarkNoisyWH_Simulation,
    'RobotForward_Sim': BenchmarkRobotForward_Simulation,
    'RobotInverse_Sim': BenchmarkRobotInverse_Simulation,
    'Ship_Sim': BenchmarkShip_Simulation,
    'QuadPelican_Sim': BenchmarkQuadPelican_Simulation,
    'QuadPi_Sim': BenchmarkQuadPi_Simulation,
}

prediction_benchmarks = {
    'WH_Pred': BenchmarkWH_Prediction,
    'Silverbox_Pred': BenchmarkSilverbox_Prediction,
    'Tanks_Pred': BenchmarkCascadedTanks_Prediction,
    'EMPS_Pred': BenchmarkEMPS_Prediction,
    'NoisyWH_Pred': BenchmarkNoisyWH_Prediction,
    'RobotForward_Pred': BenchmarkRobotForward_Prediction,
    'RobotInverse_Pred': BenchmarkRobotInverse_Prediction,
    'Ship_Pred': BenchmarkShip_Prediction,
    'QuadPelican_Pred': BenchmarkQuadPelican_Prediction,
    'QuadPi_Pred': BenchmarkQuadPi_Prediction,
}

all_benchmarks = {**simulation_benchmarks, **prediction_benchmarks}

workshop_benchmarks = {
    'WH_Sim': BenchmarkWH_Simulation,
    'Silverbox_Sim': BenchmarkSilverbox_Simulation,
    'EMPS_Sim': BenchmarkEMPS_Simulation,
    'CED_Sim': BenchmarkCED_Simulation,
}