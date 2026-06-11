"""Ship dynamics benchmark dataset definition."""

__all__ = ["ship_dataset", "ship_u", "ship_y", "BenchmarkShip_Simulation", "BenchmarkShip_Prediction", "dl_ship"]

from nonlinear_benchmarks.utilities import get_tmp_benchmark_directory
import identibench.metrics
from ..benchmark import BenchmarkSpec, Prediction, Simulation
from ..dataset import Dataset
from pathlib import Path
import os
import h5py
import numpy as np
import pandas as pd
import shutil


def dl_ship(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
    remove_download: bool = True,
) -> None:
    save_path = Path(save_path)
    download_dir = Path(get_tmp_benchmark_directory()) / "Ship"

    if force_download and download_dir.exists():
        print(f"Force reload: Removing existing directory: {download_dir}")
        shutil.rmtree(download_dir)

    try:
        from easyDataverse import Dataverse
    except ImportError as e:
        raise ImportError(
            'easyDataverse is required for the Ship dataset. Install it with: pip install "identibench[ship]"'
        ) from e

    dataverse = Dataverse("https://darus.uni-stuttgart.de/")
    dataverse.load_dataset(
        pid="doi:10.18419/darus-2905",
        filedir=download_dir,
    )

    # str to Path to be plattform independent
    structure_mapping = {
        Path("patrol_ship_routine/processed/train"): "train",
        Path("patrol_ship_routine/processed/validation"): "valid",
        Path("patrol_ship_routine/processed/test"): "test",
        Path("patrol_ship_ood/processed/test"): "test_ood",
    }

    # Ensure desired directories exist
    for subdir in structure_mapping.values():
        os.makedirs(os.path.join(save_path, subdir), exist_ok=True)

    def convert_tab_to_hdf5(tab_path: Path, hdf5_path: Path) -> None:
        df = pd.read_csv(tab_path, sep="\t")
        with h5py.File(hdf5_path, "w") as hdf:
            for column in df.columns:
                data = df[column].astype(np.float32).values
                hdf.create_dataset(column, data=data, dtype="f4")

    # Walk through the current directory structure and process files
    for subdir, dirs, files in os.walk(download_dir):
        for file in files:
            if file.endswith(".tab"):
                current_file_path = os.path.join(subdir, file)

                # Determine the relative path
                relative_subdir = Path(os.path.relpath(subdir, download_dir))

                # Find the corresponding desired subdir
                if relative_subdir in structure_mapping:
                    desired_subdir = structure_mapping[relative_subdir]

                    # Construct desired file paths
                    base_filename = file.replace(".tab", "")
                    desired_hdf5_path = os.path.join(save_path, desired_subdir, base_filename + ".hdf5")

                    convert_tab_to_hdf5(current_file_path, desired_hdf5_path)

    # remove downloaded files
    if remove_download:
        shutil.rmtree(download_dir)


ship_dataset = Dataset("ship", prepare=dl_ship)

ship_u = ["n", "deltal", "deltar", "Vw"]
ship_y = ["alpha_x", "alpha_y", "u", "v", "p", "r", "phi"]

_ship = dict(
    u_cols=ship_u,
    y_cols=ship_y,
    train=[(ship_dataset, "train/*.hdf5")],
    valid=[(ship_dataset, "valid/*.hdf5")],
    # Routine-condition test set is the headline; the out-of-distribution
    # patrol_ship_ood recordings are the named set "ood".
    test_sets={
        "test": [(ship_dataset, "test/*.hdf5")],
        "ood": [(ship_dataset, "test_ood/*.hdf5")],
    },
)

BenchmarkShip_Simulation = BenchmarkSpec(
    name="BenchmarkShip_Simulation",
    task=Simulation(metric=identibench.metrics.rmse, init_window=100),
    **_ship,
)

BenchmarkShip_Prediction = BenchmarkSpec(
    name="BenchmarkShip_Prediction",
    task=Prediction(horizon=100, step=100, metric=identibench.metrics.rmse, init_window=100),
    **_ship,
)
