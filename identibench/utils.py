"""Utility functions for data loading, HDF5 I/O, and downloads."""

__all__ = [
    "Sequence",
    "get_default_data_root",
    "hdf_files_from_path",
    "write_dataset",
    "write_array",
    "iodata_to_hdf5",
    "dataset_to_hdf5",
    "download_file",
    "extract_archive",
]

import os
import tarfile
import warnings
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np
import requests
from nonlinear_benchmarks import Input_output_data
from tqdm import tqdm


class Sequence(NamedTuple):
    """A single time-series sequence with per-file HDF5 attributes."""

    u: np.ndarray
    y: np.ndarray
    attrs: dict[str, Any]


def get_default_data_root() -> Path:
    """
    Returns the default root directory for datasets.

    Checks the 'IDENTIBENCH_DATA_ROOT' environment variable first (an empty
    value counts as unset), otherwise defaults to '~/.identibench_data'.
    """
    return Path(os.environ.get("IDENTIBENCH_DATA_ROOT") or Path.home() / ".identibench_data")


def _dummy_dataset_loader(
    save_path: Path,  # Directory where the dummy dataset files will be written
    force_download: bool = False,  # Argument for interface compatibility
    create_train_valid_dir: bool = False,  # If True, create a 'train_valid' subdir as well
) -> None:
    """Creates a dummy dataset structure with minimal HDF5 files for testing."""
    save_path = Path(save_path)
    if save_path.is_dir() and not force_download:
        return

    save_path.mkdir(parents=True, exist_ok=True)
    seq_len = 50
    subdirs = ["train", "valid", "test"]
    if create_train_valid_dir:
        subdirs.append("train_valid")

    for subdir in subdirs:
        subdir_path = save_path / subdir
        subdir_path.mkdir(exist_ok=True)
        n_files = 1 if subdir == "train_valid" else 2  # Create fewer files in train_valid for testing differentiation
        for i in range(n_files):
            dummy_file_path = subdir_path / f"{subdir}_{i}.hdf5"
            try:
                with h5py.File(dummy_file_path, "w") as f:
                    f.create_dataset("u0", data=np.random.rand(seq_len).astype(np.float32))
                    f.create_dataset("u1", data=np.random.rand(seq_len).astype(np.float32))
                    f.create_dataset("y0", data=np.random.rand(seq_len).astype(np.float32))
                    f.attrs["fs"] = 10.0
            except Exception as e:
                warnings.warn(f"Failed to create dummy file {dummy_file_path}: {e}", RuntimeWarning)


def hdf_files_from_path(fpath: Path) -> list[Path]:
    """Lists the HDF5 files in a directory, sorted by name.

    Args:
        fpath: Directory to search for ``*.hdf5`` files.

    Returns:
        Sorted list of paths to the HDF5 files found in ``fpath``.
    """
    return sorted(list(fpath.glob("*.hdf5")))


def _load_sequences_from_files(
    file_paths: list[Path],
    u_cols: list[str],
    y_cols: list[str],
    win_sz: int | None = None,
    stp_sz: int | None = None,
) -> Iterator[Sequence]:
    if not file_paths:
        return iter([])
    if win_sz is None and stp_sz is not None:
        raise ValueError("win_sz must be provided if stp_sz is provided")
    if stp_sz is None and win_sz is not None:
        raise ValueError("stp_sz must be provided if win_sz is provided")

    for file_path in file_paths:
        with h5py.File(file_path, "r") as f:
            u_data = np.stack([f[col][()] for col in u_cols], axis=-1).astype(np.float32)
            y_data = np.stack([f[col][()] for col in y_cols], axis=-1).astype(np.float32)
            attrs = dict(f.attrs)

        if win_sz is None:
            yield Sequence(u_data, y_data, attrs)
            continue

        seq_len = u_data.shape[0]
        if seq_len < win_sz:
            continue

        for start in range(0, seq_len - win_sz + 1, stp_sz):
            end = start + win_sz
            yield Sequence(u_data[start:end], y_data[start:end], attrs)


def write_dataset(
    group: h5py.File | h5py.Group,
    ds_name: str,
    data: np.ndarray,
    dtype: str = "f4",
    chunks: tuple[int, ...] | None = None,
) -> None:
    group.create_dataset(ds_name, data=data, dtype=dtype, chunks=chunks)


def write_array(
    group: h5py.File | h5py.Group,
    ds_name: str,
    data: np.ndarray,
    dtype: str = "f4",
    chunks: tuple[int, ...] | None = None,
) -> None:
    "Writes a 2d numpy array rowwise to a hdf5 file."
    for i in range(data.shape[1]):
        write_dataset(group, f"{ds_name}{i}", data[:, i], dtype, chunks)


def iodata_to_hdf5(
    iodata: Input_output_data,  # data to save to file
    hdf_dir: Path,  # Export directory for hdf5 files
    f_name: str | None = None,  # name of hdf5 file without '.hdf5' ending; defaults to iodata.name
) -> Path:
    """Writes a single Input_output_data record to an HDF5 file.

    The input and output channels are stored as row-wise datasets (``u0``,
    ``u1``, ... and ``y0``, ``y1``, ...). Sampling rate and state
    initialization window length, when available, are stored as file attributes.

    Args:
        iodata: Input/output data record to save.
        hdf_dir: Export directory for the HDF5 file; created if it does not exist.
        f_name: Name of the HDF5 file without the ``.hdf5`` ending. Defaults to
            ``iodata.name`` when ``None``.

    Returns:
        Path to the written HDF5 file.
    """
    data_2d = iodata.atleast_2d()
    u, y = data_2d.u, data_2d.y

    os.makedirs(hdf_dir, exist_ok=True)
    if f_name is None:
        f_name = iodata.name

    hdf_path = Path(hdf_dir) / f"{f_name}.hdf5".replace(" ", "_")
    with h5py.File(hdf_path, "w") as f:
        write_array(f, "u", u)
        write_array(f, "y", y)

        # Save sampling_rate and init_window_size as attributes
        if iodata.sampling_time is not None:
            f.attrs["fs"] = 1 / iodata.sampling_time
        if iodata.state_initialization_window_length is not None:
            f.attrs["init_sz"] = iodata.state_initialization_window_length

    return hdf_path


def dataset_to_hdf5(
    train: tuple,  # tuple of Input_output_data for training
    valid: tuple,  # tuple of Input_output_data for validation
    test: tuple,  # tuple of Input_output_data for test
    save_path: Path,  # directory the files are written to, created if it does not exist
    train_valid: tuple | None = None,  # optional tuple of unsplit Input_output_data for training and validation
) -> None:
    """Saves a dataset as HDF5 files in separate subdirectories per subset.

    Each subset is written to its own subdirectory (``train``, ``valid``,
    ``test`` and, when provided, ``train_valid``) of ``save_path``, with one
    HDF5 file per record. A single ``Input_output_data`` is accepted in place of
    a tuple for any subset.

    Args:
        train: Tuple of ``Input_output_data`` for training.
        valid: Tuple of ``Input_output_data`` for validation.
        test: Tuple of ``Input_output_data`` for testing.
        save_path: Directory the files are written to; created if it does not exist.
        train_valid: Optional tuple of unsplit ``Input_output_data`` for
            combined training and validation.

    Raises:
        ValueError: If a subset is not an ``Input_output_data`` or a tuple thereof.
    """
    save_path = Path(save_path)

    dict_data = {"train": train, "valid": valid, "test": test, "train_valid": train_valid}
    for subset, ds_entries in dict_data.items():
        if ds_entries is None:
            continue
        if isinstance(ds_entries, tuple):
            if not isinstance(ds_entries[0], Input_output_data):
                raise ValueError(f"Data has to be stored in tuples of Input_output_data. Got {type(ds_entries[0])}")
        else:
            if not isinstance(ds_entries, Input_output_data):
                raise ValueError(f"Data has to be stored in Input_output_data. Got {type(ds_entries)}")
            dict_data[subset] = (ds_entries,)

    os.makedirs(save_path, exist_ok=True)

    for subset, ds_entries in dict_data.items():
        if ds_entries is None:
            continue
        for idx, iodata in enumerate(ds_entries):
            iodata_to_hdf5(iodata, save_path / subset, f"{subset}_{idx}")


def download_file(
    url: str,  # URL of the file to download.
    dest: Path,  # Local file path the download is written to.
    chunk_size: int = 8192,  # Streaming chunk size in bytes.
    force: bool = False,  # Re-download even if ``dest`` already exists.
    headers: dict[str, str] | None = None,  # Optional HTTP request headers (e.g. User-Agent, Referer).
) -> Path:
    """Streaming HTTP download with progress bar; skip if file exists unless ``force``.

    Returns:
        Path to the downloaded (or pre-existing) file.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"  Already downloaded: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=(30, 300), headers=headers)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def extract_archive(
    path: Path,  # Archive file to extract (``.zip``, ``.tar``, ``.tar.gz``/``.tgz``).
    dest: Path,  # Destination directory; created if it does not exist.
    members: list[str] | None = None,  # Optional subset of archive members to extract.
) -> Path:
    """Extract zip, tar, or tar.gz archive to dest directory."""
    path, dest = Path(path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            targets = members or zf.namelist()
            for m in targets:
                zf.extract(m, dest)
    elif path.name.endswith(".tar.gz") or path.name.endswith(".tgz") or path.suffix == ".tar":
        mode = "r:gz" if (path.name.endswith(".tar.gz") or path.name.endswith(".tgz")) else "r"
        with tarfile.open(path, mode) as tf:
            if members:
                for m in members:
                    tf.extract(m, dest, filter="data")
            else:
                tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"Unsupported archive format: {path}")
    return dest
