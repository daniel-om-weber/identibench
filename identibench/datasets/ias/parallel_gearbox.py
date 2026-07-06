"""Parallel gearbox IAS estimation dataset (MCC5-THU, Mendeley p92gj2732w)."""

__all__ = [
    "parallel_gearbox_dataset",
    "dl_parallel_gearbox",
    "BenchmarkParallelGearbox_Estimation",
    "BenchmarkParallelGearbox_Simulation",
]

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from ...benchmark import BenchmarkSpec, Simulation, WindowedEstimation, GridwiseEstimation
from ...dataset import Dataset
from ...metrics import mae
from ._common import (
    DatasetInfo,
    _require_sklearn,
    analog_pulse_to_ias,
    download_and_unpack,
    save_signals_hdf5,
    ias_test_sets,
    write_disturbed_test_sets,
)

_INFO = DatasetInfo(
    name="Parallel_Gearbox",
    zip_url="https://data.mendeley.com/public-api/zip/p92gj2732w/download/2",
)

_FS = 12800

# Skipped upstream because the IAS measurement stops early / is intermittent (verbatim).
_SKIPPED_STEMS = {
    "gear_pitting_M_torque_circulation_2000rpm_10Nm",
    "gear_pitting_M_speed_circulation_20Nm-1000rpm",
}

# Recordings whose speed channel is offset against the vibration channels; the
# IAS is shifted by -0.82 s to re-synchronize (verbatim upstream lookup table).
_FILES_TO_SHIFT = {
    "teeth_break_and_bearing_inner_H_torque_circulation_1000rpm_10Nm",
    "teeth_break_L_speed_circulation_20Nm-1000rpm",
    "gear_wear_M_speed_circulation_10Nm-1000rpm",
    "teeth_break_and_bearing_outer_L_speed_circulation_20Nm-1000rpm",
    "teeth_break_M_speed_circulation_20Nm-1000rpm",
    "teeth_break_and_bearing_outer_M_torque_circulation_1000rpm_10Nm",
    "teeth_break_and_bearing_outer_H_speed_circulation_10Nm-1000rpm",
    "teeth_break_and_bearing_inner_H_speed_circulation_20Nm-1000rpm",
    "teeth_crack_L_speed_circulation_20Nm-1000rpm",
    "health_torque_circulation_1000rpm_10Nm",
    "teeth_break_and_bearing_outer_H_torque_circulation_1000rpm_10Nm",
    "teeth_break_and_bearing_outer_M_torque_circulation_2000rpm_20Nm",
    "gear_pitting_H_speed_circulation_10Nm-1000rpm",
    "teeth_break_and_bearing_outer_H_speed_circulation_20Nm-1000rpm",
    "gear_wear_H_torque_circulation_2000rpm_20Nm",
    "teeth_break_and_bearing_inner_H_torque_circulation_2000rpm_10Nm",
    "gear_wear_L_torque_circulation_2000rpm_20Nm",
    "gear_pitting_M_torque_circulation_2000rpm_20Nm",
    "teeth_break_M_torque_circulation_3000rpm_10Nm",
    "teeth_break_H_speed_circulation_20Nm-2000rpm",
    "teeth_break_and_bearing_outer_H_speed_circulation_10Nm-2000rpm",
    "gear_wear_H_speed_circulation_20Nm-2000rpm",
    "miss_teeth_torque_circulation_3000rpm_10Nm",
    "teeth_break_H_torque_circulation_2000rpm_10Nm",
    "teeth_crack_L_torque_circulation_3000rpm_20Nm",
    "teeth_break_L_speed_circulation_10Nm-2000rpm",
    "gear_wear_L_speed_circulation_20Nm-2000rpm",
    "teeth_break_and_bearing_outer_H_torque_circulation_3000rpm_20Nm",
    "gear_pitting_L_speed_circulation_20Nm-2000rpm",
    "teeth_break_and_bearing_outer_M_torque_circulation_3000rpm_10Nm",
    "gear_pitting_H_speed_circulation_10Nm-2000rpm",
    "gear_pitting_M_speed_circulation_20Nm-2000rpm",
    "teeth_crack_M_speed_circulation_10Nm-2000rpm",
    "teeth_break_and_bearing_outer_H_speed_circulation_20Nm-2000rpm",
    "teeth_break_and_bearing_inner_H_speed_circulation_20Nm-3000rpm",
    "teeth_break_and_bearing_outer_L_speed_circulation_10Nm-3000rpm",
    "teeth_crack_H_torque_circulation_3000rpm_10Nm",
    "health_speed_circulation_10Nm-3000rpm",
    "teeth_break_L_speed_circulation_10Nm-3000rpm",
    "miss_teeth_speed_circulation_20Nm-3000rpm",
    "gear_pitting_L_speed_circulation_10Nm-3000rpm",
}

# Fault types whose M/H severities form the out-of-distribution wear set (verbatim).
_TEST_WEAR_TYPES = {"teeth_break", "miss_teeth", "teeth_break_and_bearing_inner", "teeth_break_and_bearing_outer"}

_FILENAME_PATTERN = re.compile(
    r"""
    ^([a-z_]+?)
    (?:_([HML]))?
    _(speed_circulation|torque_circulation)
    _(\d+(?:Nm|rpm))
    [-_](\d+(?:Nm|rpm))
    (\.hdf5)$
    """,
    re.VERBOSE,
)


def _shift_speed_by_lookup(calculated_speed: np.ndarray, file_name: str, fs: float) -> np.ndarray:
    """Align the speed to the vibration (in some files they are not synchronized)."""
    if file_name not in _FILES_TO_SHIFT:
        return calculated_speed
    signal_t = np.arange(len(calculated_speed)) / fs
    interpolator = interp1d(signal_t, np.array(calculated_speed), kind="linear", fill_value="extrapolate")
    return interpolator(signal_t - 0.82)


def dl_parallel_gearbox(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # unused; the framework only calls this when the dataset is missing or forced
) -> None:
    """Download, preprocess (encoder → IAS), split, and add disturbed test sets."""
    train_test_split = _require_sklearn()
    save_path = Path(save_path)
    for split in ("train", "valid", "test", "test_wear"):
        (save_path / split).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        download_and_unpack(_INFO, temp_dir, nested_zip=True)

        # important: the .zip also contains a __MACOSX folder with ._xxx.csv files to be ignored
        csv_files = list((temp_dir / "MCC5-THU gearbox fault diagnosis datasets").rglob("*.csv"))

        # Route every recording to its split BEFORE processing — the split needs only
        # the parsed filenames, so each HDF5 is written once, directly into place.
        # Sorted by the produced .hdf5 name to keep the seeded splits byte-identical
        # to the previous sort-processed-files implementation.
        parsed_files = []
        for fpath in sorted(csv_files, key=lambda p: f"{p.stem}.hdf5"):
            if fpath.stem in _SKIPPED_STEMS:
                continue
            match = _FILENAME_PATTERN.match(f"{fpath.stem}.hdf5")
            if not match:
                print(f"Warning: Could not parse filename: {fpath.name}")
                continue
            fault_type, severity, mode = match.group(1), match.group(2) or "N/A", match.group(3)
            parsed_files.append({"full_path": fpath, "fault_type": fault_type, "severity": severity, "mode": mode})

        test_wear_paths = []
        strata_labels = []
        file_list = []
        for f in parsed_files:
            if f["fault_type"] in _TEST_WEAR_TYPES and not f["severity"] == "L":
                test_wear_paths.append(f["full_path"])
            else:
                strata_labels.append(f"{f['fault_type']}_{f['severity']}_{f['mode']}")
                file_list.append(f["full_path"])

        # First split: 80% train/valid, 20% test
        train_valid_paths, test_basic_paths, train_valid_labels, _ = train_test_split(
            file_list, strata_labels, test_size=0.20, random_state=42, stratify=strata_labels
        )
        # Second split: 80% (of original) -> 75% train, 25% valid (= 20% of total)
        train_paths, valid_paths, _, _ = train_test_split(
            train_valid_paths, train_valid_labels, test_size=0.25, random_state=42, stratify=train_valid_labels
        )

        split_of = {}
        for paths, subdir in [
            (train_paths, "train"),
            (valid_paths, "valid"),
            (test_basic_paths, "test"),
            (test_wear_paths, "test_wear"),
        ]:
            for path in paths:
                split_of[path] = subdir

        for file in tqdm(sorted(split_of), desc="Preprocessing CSV files", unit="file"):
            data = pd.read_csv(file)
            ias = np.array(analog_pulse_to_ias(data["speed"], _FS, pulses_per_revolution=1))
            ias = _shift_speed_by_lookup(ias, file.stem, _FS)
            # recalculated, so the IAS is from the middle shaft (which the vibration sensor is
            # mounted on, whereas the speed sensor is mounted on the input shaft)
            ias = ias * 29 / 95
            signals = {
                "IAS": ias,
                "gearbox_vibration_x": np.array(data["gearbox_vibration_x"] * 9.81),
                "gearbox_vibration_y": np.array(data["gearbox_vibration_y"] * 9.81),
                "gearbox_vibration_z": np.array(data["gearbox_vibration_z"] * 9.81),
            }
            # gear ratio: input shaft, gear mesh, middle shaft (IAS), gear mesh, output shaft
            save_signals_hdf5(
                signals,
                save_path / split_of[file] / f"{file.stem}.hdf5",
                fs=_FS,
                gear_ratio=[95 / 29, 95, 1, 36, 36 / 90],
            )

    write_disturbed_test_sets(save_path, vib_keys=["gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z"])


parallel_gearbox_dataset = Dataset("parallel_gearbox", prepare=dl_parallel_gearbox)

_parallel_gearbox = dict(
    u_cols=["gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z"],
    y_cols=["IAS"],
    train=[(parallel_gearbox_dataset, "train/*.hdf5")],
    valid=[(parallel_gearbox_dataset, "valid/*.hdf5")],
    test_sets=ias_test_sets(parallel_gearbox_dataset),
)

BenchmarkParallelGearbox_Estimation = BenchmarkSpec(
    name="BenchmarkParallelGearbox_Estimation",
    # window_sec = largest window any upstream method needed (Ref-FFT-LSTM 2.13 s),
    # rounded to 2.2 s so every method has enough samples; smaller ones crop/decimate. See ias/__init__.
    task=WindowedEstimation(window_sec=2.2),
    **_parallel_gearbox,
)

BenchmarkParallelGearbox_GridwiseEstimation = BenchmarkSpec(
    name="BenchmarkParallelGearbox_GridwiseEstimation",
    task=GridwiseEstimation(window_sec=3.0, step_sec=0.1),
    **_parallel_gearbox,
)

# Dense free-run sibling (framework Simulation task): the model predicts one IAS estimate
# per sample over the full recording — the window lives in the model (e.g. a sliding window),
# not the benchmark — scored per-sample MAE in Hz. Same data and test sets as the windowed task.
BenchmarkParallelGearbox_Simulation = BenchmarkSpec(
    name="BenchmarkParallelGearbox_Simulation",
    task=Simulation(metric=mae),
    **_parallel_gearbox,
)
