"""Shared helpers for dataset download functions."""

from collections.abc import Callable
from pathlib import Path

from ..utils import dataset_to_hdf5

__all__ = ["dl_split_by_index"]


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
