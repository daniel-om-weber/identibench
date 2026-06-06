"""Shared helpers for building dataset benchmark specs and loaders."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import identibench.benchmark as idb
from ..utils import dataset_to_hdf5

__all__ = ["make_sim_pred", "dl_split_by_index"]


def make_sim_pred(
    name_base: str,  # benchmark name prefix; ``_Simulation`` / ``_Prediction`` is appended
    dataset_id: str,  # identifier for the raw dataset source
    u_cols: list[str],  # list of column names for input signals (u)
    y_cols: list[str],  # list of column names for output signals (y)
    metric_func: Callable[[Any, Any], float],  # primary metric: ``func(y_pred, y_true)``
    download_func: Callable[..., None],  # dataset preparation func
    init_window: int,  # steps for warm-up, potentially ignored in evaluation
    pred_horizon: int,  # the 'k' in k-step ahead prediction
    pred_step: int,  # step size for k-step ahead prediction
    *,
    custom_test_evaluation: Callable[..., dict[str, float]] | None = None,  # optional custom test evaluation
) -> tuple[idb.BenchmarkSpecSimulation, idb.BenchmarkSpecPrediction]:
    """Build the near-identical simulation/prediction benchmark twins for a dataset.

    Returns a ``(simulation_spec, prediction_spec)`` tuple. Both specs share the
    same dataset, columns, metric, downloader, warm-up window, and (optional)
    custom test evaluation; the prediction spec additionally carries the
    ``pred_horizon`` / ``pred_step`` k-step-ahead settings.

    Args:
        name_base: Benchmark name prefix; the suffixes ``_Simulation`` and
            ``_Prediction`` are appended for the two specs.
        dataset_id: Identifier for the raw dataset source.
        u_cols: Input signal (u) column names.
        y_cols: Output signal (y) column names.
        metric_func: Primary metric callable ``func(y_pred, y_true)``.
        download_func: Dataset preparation function.
        init_window: Warm-up window length in steps.
        pred_horizon: The 'k' in k-step ahead prediction.
        pred_step: Step size for k-step ahead prediction.
        custom_test_evaluation: Optional custom test-evaluation callable shared by
            both specs.

    Returns:
        A ``(simulation_spec, prediction_spec)`` tuple.
    """
    sim_spec = idb.BenchmarkSpecSimulation(
        name=f"{name_base}_Simulation",
        dataset_id=dataset_id,
        u_cols=u_cols,
        y_cols=y_cols,
        metric_func=metric_func,
        download_func=download_func,
        custom_test_evaluation=custom_test_evaluation,
        init_window=init_window,
    )
    pred_spec = idb.BenchmarkSpecPrediction(
        name=f"{name_base}_Prediction",
        dataset_id=dataset_id,
        u_cols=u_cols,
        y_cols=y_cols,
        metric_func=metric_func,
        download_func=download_func,
        custom_test_evaluation=custom_test_evaluation,
        init_window=init_window,
        pred_horizon=pred_horizon,
        pred_step=pred_step,
    )
    return sim_spec, pred_spec


def dl_split_by_index(
    benchmark_cls: Callable[..., tuple],  # nonlinear_benchmarks loader returning (train_val, test)
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool,  # force download the dataset
    save_train_valid: bool,  # save unsplit train and valid datasets in a 'train_valid' subdirectory
    split_idx: int,  # split index for train and valid datasets
    *,
    reversed_split: bool = False,  # if True use train=train_val[split_idx:], valid=train_val[:split_idx]
) -> None:
    """Load a ``nonlinear_benchmarks`` dataset, split it by index, and write HDF5.

    Captures the shared "load -> split by index -> ``dataset_to_hdf5``" pattern used
    by the wh, silverbox, emps and cascaded_tanks downloaders.

    Args:
        benchmark_cls: The ``nonlinear_benchmarks`` loader, called as
            ``benchmark_cls(force_download=force_download)`` and expected to return a
            ``(train_val, test)`` pair.
        save_path: Output directory; created if it does not exist.
        force_download: Whether to force a re-download of the raw dataset.
        save_train_valid: Whether to also save the unsplit train_valid subset.
        split_idx: Index at which ``train_val`` is split into train and valid.
        reversed_split: If ``True``, ``train = train_val[split_idx:]`` and
            ``valid = train_val[:split_idx]`` (used by cascaded_tanks); otherwise
            ``train = train_val[:split_idx]`` and ``valid = train_val[split_idx:]``.
    """
    train_val, test = benchmark_cls(force_download=force_download)
    if reversed_split:
        train = train_val[split_idx:]
        valid = train_val[:split_idx]
    else:
        train = train_val[:split_idx]
        valid = train_val[split_idx:]

    dataset_to_hdf5(train, valid, test, save_path, train_valid=(train_val if save_train_valid else None))
