"""Workshop benchmark dataset definitions (WH, Silverbox, Tanks, EMPS, NoisyWH, CED)."""

__all__ = [
    "wh_dataset",
    "silverbox_dataset",
    "cascaded_tanks_dataset",
    "emps_dataset",
    "noisy_wh_dataset",
    "ced_dataset",
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
    "dl_wiener_hammerstein",
    "dl_silverbox",
    "dl_cascaded_tanks",
    "dl_emps",
    "dl_noisy_wh",
    "dl_ced",
]

from pathlib import Path

import nonlinear_benchmarks
import numpy as np
from nonlinear_benchmarks.utilities import Input_output_data
from scipy.io import loadmat

import identibench.metrics
from ..benchmark import BenchmarkSpec, Prediction, Simulation
from ..dataset import Dataset
from ..utils import dataset_to_hdf5, iodata_to_hdf5
from ._common import dl_split_by_index


def rmse_mV(inp: np.ndarray, targ: np.ndarray) -> float:
    return identibench.metrics.rmse(inp, targ) * 1000


# ───────────────────────── Wiener-Hammerstein ─────────────────────────


def dl_wiener_hammerstein(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
    save_train_valid: bool = True,  # save unsplitted train and valid datasets in 'train_valid' subdirectory
    split_idx: int = 80_000,  # split index for train and valid datasets
) -> None:
    dl_split_by_index(
        nonlinear_benchmarks.WienerHammerBenchMark, save_path, force_download, save_train_valid, split_idx
    )


wh_dataset = Dataset("wh", prepare=dl_wiener_hammerstein)

_wh = dict(
    u_cols=["u0"],
    y_cols=["y0"],
    train=[(wh_dataset, "train/*.hdf5")],
    valid=[(wh_dataset, "valid/*.hdf5")],
    train_valid=[(wh_dataset, "train_valid/*.hdf5")],
    test_sets={"test": [(wh_dataset, "test/*.hdf5")]},
)

BenchmarkWH_Simulation = BenchmarkSpec(
    name="BenchmarkWH_Simulation",
    task=Simulation(metric=rmse_mV, init_window=50),
    **_wh,
)

BenchmarkWH_Prediction = BenchmarkSpec(
    name="BenchmarkWH_Prediction",
    task=Prediction(horizon=100, step=100, metric=rmse_mV, init_window=50),
    **_wh,
)


# ───────────────────────── Silverbox ─────────────────────────


def dl_silverbox(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
    save_train_valid: bool = True,  # save unsplitted train and valid datasets in 'train_valid' subdirectory
    split_idx: int = 50_000,  # split index for train and valid datasets
) -> None:
    dl_split_by_index(nonlinear_benchmarks.Silverbox, save_path, force_download, save_train_valid, split_idx)


silverbox_dataset = Dataset("silverbox", prepare=dl_silverbox)

_silverbox = dict(
    u_cols=["u0"],
    y_cols=["y0"],
    train=[(silverbox_dataset, "train/*.hdf5")],
    valid=[(silverbox_dataset, "valid/*.hdf5")],
    train_valid=[(silverbox_dataset, "train_valid/*.hdf5")],
    # The three test records in deterministic enumerate order test_0/1/2.
    test_sets={
        "multisine": [(silverbox_dataset, "test/test_0.hdf5")],
        "arrow_full": [(silverbox_dataset, "test/test_1.hdf5")],
        "arrow_no_extrapolation": [(silverbox_dataset, "test/test_2.hdf5")],
    },
)

BenchmarkSilverbox_Simulation = BenchmarkSpec(
    name="BenchmarkSilverbox_Simulation",
    task=Simulation(metric=rmse_mV, init_window=50),
    **_silverbox,
)

BenchmarkSilverbox_Prediction = BenchmarkSpec(
    name="BenchmarkSilverbox_Prediction",
    task=Prediction(horizon=100, step=100, metric=rmse_mV, init_window=50),
    **_silverbox,
)


# ───────────────────────── Cascaded Tanks ─────────────────────────


def dl_cascaded_tanks(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
    save_train_valid: bool = True,  # save unsplitted train and valid datasets in 'train_valid' subdirectory
    split_idx: int = 160,  # split index for train and valid datasets
) -> None:
    # cascaded_tanks uses the reversed split: train=train_val[split_idx:], valid=train_val[:split_idx]
    dl_split_by_index(
        nonlinear_benchmarks.Cascaded_Tanks,
        save_path,
        force_download,
        save_train_valid,
        split_idx,
        reversed_split=True,
    )


cascaded_tanks_dataset = Dataset("cascaded_tanks", prepare=dl_cascaded_tanks)

_cascaded_tanks = dict(
    u_cols=["u0"],
    y_cols=["y0"],
    train=[(cascaded_tanks_dataset, "train/*.hdf5")],
    valid=[(cascaded_tanks_dataset, "valid/*.hdf5")],
    train_valid=[(cascaded_tanks_dataset, "train_valid/*.hdf5")],
    test_sets={"test": [(cascaded_tanks_dataset, "test/*.hdf5")]},
)

BenchmarkCascadedTanks_Simulation = BenchmarkSpec(
    name="BenchmarkCascadedTanks_Simulation",
    task=Simulation(metric=identibench.metrics.rmse, init_window=50),
    **_cascaded_tanks,
)

BenchmarkCascadedTanks_Prediction = BenchmarkSpec(
    name="BenchmarkCascadedTanks_Prediction",
    task=Prediction(horizon=100, step=100, metric=identibench.metrics.rmse, init_window=50),
    **_cascaded_tanks,
)


# ───────────────────────── EMPS ─────────────────────────


def dl_emps(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
    save_train_valid: bool = True,  # save unsplitted train and valid datasets in 'train_valid' subdirectory
    split_idx: int = 18_000,  # split index for train and valid datasets
) -> None:
    dl_split_by_index(nonlinear_benchmarks.EMPS, save_path, force_download, save_train_valid, split_idx)


emps_dataset = Dataset("emps", prepare=dl_emps)

_emps = dict(
    u_cols=["u0"],
    y_cols=["y0"],
    train=[(emps_dataset, "train/*.hdf5")],
    valid=[(emps_dataset, "valid/*.hdf5")],
    train_valid=[(emps_dataset, "train_valid/*.hdf5")],
    test_sets={"test": [(emps_dataset, "test/*.hdf5")]},
)

BenchmarkEMPS_Simulation = BenchmarkSpec(
    name="BenchmarkEMPS_Simulation",
    task=Simulation(metric=rmse_mV, init_window=20),
    **_emps,
)

BenchmarkEMPS_Prediction = BenchmarkSpec(
    name="BenchmarkEMPS_Prediction",
    task=Prediction(horizon=500, step=100, metric=rmse_mV, init_window=20),
    **_emps,
)


# ───────────────────────── Noisy Wiener-Hammerstein ─────────────────────────


def dl_noisy_wh(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
) -> None:
    "the wiener hammerstein dataset with process noise"

    # extract raw .mat files, to preserve filenames necessary for train, valid split
    matfiles = nonlinear_benchmarks.not_splitted_benchmarks.WienerHammerstein_Process_Noise(
        data_file_locations=True, train_test_split=False, force_download=force_download
    )

    for file in matfiles:
        f_path = Path(file)
        save_path = Path(save_path)

        if "Test" in f_path.stem:
            hdf_path = save_path / "test"
        elif "Combined" in f_path.stem:
            hdf_path = save_path / "valid"
        else:
            hdf_path = save_path / "train"

        out = loadmat(f_path)
        _, u, y, fs = out["dataMeas"][0, 0]
        fs = fs[0, 0]
        for idx, (ui, yi) in enumerate(zip(u.T, y.T)):
            iodata = Input_output_data(u=ui, y=yi, sampling_time=1 / fs)
            fname = f"{f_path.stem}_{idx + 1}"
            iodata_to_hdf5(iodata, hdf_path, fname)


noisy_wh_dataset = Dataset("noisy_wh", prepare=dl_noisy_wh)

_noisy_wh = dict(
    u_cols=["u0"],
    y_cols=["y0"],
    train=[(noisy_wh_dataset, "train/*.hdf5")],
    valid=[(noisy_wh_dataset, "valid/*.hdf5")],
    # The split is file-level, so the unsplit estimation data is simply both dirs.
    train_valid=[(noisy_wh_dataset, "train/*.hdf5"), (noisy_wh_dataset, "valid/*.hdf5")],
    test_sets={"test": [(noisy_wh_dataset, "test/*.hdf5")]},
)

BenchmarkNoisyWH_Simulation = BenchmarkSpec(
    name="BenchmarkNoisyWH_Simulation",
    task=Simulation(metric=rmse_mV, init_window=100),
    **_noisy_wh,
)

BenchmarkNoisyWH_Prediction = BenchmarkSpec(
    name="BenchmarkNoisyWH_Prediction",
    task=Prediction(horizon=100, step=100, metric=rmse_mV, init_window=100),
    **_noisy_wh,
)


# ───────────────────────── CED ─────────────────────────


def dl_ced(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
    save_train_valid: bool = True,  # save unsplitted train and valid datasets in 'train_valid' subdirectory
    split_idx: int = 300,  # split index for train and valid datasets
) -> None:
    train_val, test = nonlinear_benchmarks.CED(force_download=force_download, always_return_tuples_of_datasets=True)
    train = tuple(x[:split_idx] for x in train_val)
    valid = tuple(x[split_idx:] for x in train_val)

    dataset_to_hdf5(train, valid, test, save_path, train_valid=(train_val if save_train_valid else None))


ced_dataset = Dataset("ced", prepare=dl_ced)

_ced = dict(
    u_cols=["u0"],
    y_cols=["y0"],
    train=[(ced_dataset, "train/*.hdf5")],
    valid=[(ced_dataset, "valid/*.hdf5")],
    train_valid=[(ced_dataset, "train_valid/*.hdf5")],
    # The two test records in deterministic enumerate order test_0/1.
    test_sets={
        "test_1": [(ced_dataset, "test/test_0.hdf5")],
        "test_2": [(ced_dataset, "test/test_1.hdf5")],
    },
)

BenchmarkCED_Simulation = BenchmarkSpec(
    name="BenchmarkCED_Simulation",
    task=Simulation(metric=identibench.metrics.rmse, init_window=10),
    **_ced,
)

BenchmarkCED_Prediction = BenchmarkSpec(
    name="BenchmarkCED_Prediction",
    task=Prediction(horizon=30, step=30, metric=identibench.metrics.rmse, init_window=10),
    **_ced,
)
