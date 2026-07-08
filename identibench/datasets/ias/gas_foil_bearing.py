"""Gas foil bearing IAS estimation dataset (TU Berlin, pre-converted HDF5)."""

__all__ = [
    "gas_foil_bearing_dataset",
    "dl_gas_foil_bearing",
    "BenchmarkGasFoilBearing_Estimation",
    "BenchmarkGasFoilBearing_Simulation",
]

import re
import shutil
import tempfile
from pathlib import Path

import h5py
from tqdm import tqdm

from ...benchmark import BenchmarkSpec, Simulation, WindowedEstimation, GridwiseEstimation
from ...dataset import Dataset
from ...metrics import mae
from ._common import (
    DatasetInfo,
    _require_sklearn,
    download_and_unpack,
    ias_test_sets,
    write_disturbed_test_sets,
)

_INFO = DatasetInfo(
    name="Gas_Foil_Bearing",
    zip_url="https://tubcloud.tu-berlin.de/s/9emdeBacgRTo4mC/download",
)

# e.g. Load1_1_UpDown11s_03.hdf5  or  Load2_1_2_Random2_01.hdf5
_FILENAME_PATTERN = re.compile(r"^(Load\d+)_(\d+(?:_\d+)*)_([A-Za-z][A-Za-z0-9]*)_(\d{2})(\.hdf5)$")


def _copy_with_idb_attrs(src: Path, dest_dir: Path) -> None:
    """Copy one pre-converted file, renaming ``sampling_rate`` → ``fs`` and
    synthesizing ``gear_ratio=1`` (no gearing; kept for cross-dataset consistency)."""
    dest = dest_dir / src.name
    shutil.copy(str(src), str(dest))
    with h5py.File(dest, "r+") as f:
        if "sampling_rate" in f.attrs:
            f.attrs["fs"] = float(f.attrs["sampling_rate"])
            del f.attrs["sampling_rate"]
        f.attrs["gear_ratio"] = 1


def dl_gas_foil_bearing(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # unused; the framework only calls this when the dataset is missing or forced
) -> None:
    """Download the pre-converted HDF5 archive, split, and add disturbed test sets.

    The archive is already HDF5 (TDMS conversion happened upstream), so no
    signal preprocessing is needed. There is no wear condition for this dataset.
    """
    train_test_split = _require_sklearn()
    save_path = Path(save_path)
    for split in ("train", "valid", "test"):
        (save_path / split).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        download_and_unpack(_INFO, temp_dir)

        parsed_files = []
        for file in tqdm(sorted(temp_dir.rglob("*.hdf5")), desc="Processing Gas_Foil_Bearing files"):
            match = _FILENAME_PATTERN.match(file.name)
            if not match:
                print(f"Warning: Could not parse filename: {file.name}")
                continue
            load_str = match.group(1)  # e.g. "Load1"
            trajectory = match.group(3)  # e.g. "UpDown11s"
            if "anual" in trajectory:
                continue  # skip manual recordings
            parsed_files.append({"full_path": file, "trajectory": trajectory, "load_str": load_str})

        file_list = [f["full_path"] for f in parsed_files]
        strata_labels = []
        for f in parsed_files:
            if "Hold" in f["trajectory"]:
                strata_labels.append(f"Hold_{f['load_str']}")
            elif "Random" in f["trajectory"]:
                strata_labels.append(f["trajectory"])
            else:
                strata_labels.append(f"{f['trajectory']}_{f['load_str']}")

        train_valid_paths, test_basic_paths, train_valid_labels, _ = train_test_split(
            file_list, strata_labels, test_size=0.20, random_state=42, stratify=strata_labels
        )
        train_paths, valid_paths, _, _ = train_test_split(
            train_valid_paths, train_valid_labels, test_size=0.25, random_state=42, stratify=train_valid_labels
        )

        for paths, subdir in [(train_paths, "train"), (valid_paths, "valid"), (test_basic_paths, "test")]:
            for path in tqdm(paths, desc=f"Copying {subdir} files"):
                _copy_with_idb_attrs(path, save_path / subdir)

    write_disturbed_test_sets(save_path, vib_keys=["Acc_x", "Acc_y"])


gas_foil_bearing_dataset = Dataset("gas_foil_bearing", prepare=dl_gas_foil_bearing)

_gas_foil_bearing = dict(
    u_cols=["Acc_x", "Acc_y"],
    y_cols=["IAS"],
    train=[(gas_foil_bearing_dataset, "train/*.hdf5")],
    valid=[(gas_foil_bearing_dataset, "valid/*.hdf5")],
    # This dataset has no `wear` condition, so no such set is declared.
    test_sets=ias_test_sets(gas_foil_bearing_dataset, wear=False),
)

BenchmarkGasFoilBearing_Estimation = BenchmarkSpec(
    name="BenchmarkGasFoilBearing_Estimation",
    # window_sec = largest window any upstream method needed (MOPA 2.49 s — the order-tracking
    # methods decimate this fast signal), rounded to 2.5 s; smaller methods crop. See ias/__init__.
    task=WindowedEstimation(window_sec=2.5),
    **_gas_foil_bearing,
)

BenchmarkGasFoilBearing_GridwiseEstimation = BenchmarkSpec(
    name="BenchmarkGasFoilBearing_GridwiseEstimation",
    task=GridwiseEstimation(window_sec=3.0, step_sec=0.1),
    **_gas_foil_bearing,
)

# Dense free-run sibling (framework Simulation task): the model predicts one IAS estimate
# per sample over the full recording — the window lives in the model (e.g. a sliding window),
# not the benchmark — scored per-sample MAE in Hz.
BenchmarkGasFoilBearing_Simulation = BenchmarkSpec(
    name="BenchmarkGasFoilBearing_Simulation",
    task=Simulation(metric=mae),
    **_gas_foil_bearing,
)
