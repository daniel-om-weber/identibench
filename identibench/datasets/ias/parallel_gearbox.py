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

import pandas as pd
from tqdm import tqdm

from ...benchmark import BenchmarkSpec, Simulation, WindowedEstimation, GridwiseEstimation
from ...dataset import Dataset
from ...metrics import mae
from ._common import (
    DatasetInfo,
    _require_sklearn,
    download_and_unpack,
    encoder_pulse_to_ias,
    save_signals_hdf5,
    ias_test_sets,
    write_disturbed_test_sets,
)

_INFO = DatasetInfo(
    name="Parallel_Gearbox",
    zip_url="https://data.mendeley.com/public-api/zip/p92gj2732w/download/2",
)

_FS = 12800

# Order-domain cutoff for the IAS, in orders of the *input* shaft (where the tacho sits).
# At 1 pulse/rev this is all the resolution the sensor supports: 0.08 orders means the label
# is smoothed over ~12.5 revolutions.
_CUTOFF_ORDER = 0.08
_PPR = 1

# Highest frequency the label retains. Careful with the shaft: the cutoff is in orders of the
# INPUT shaft (where the tacho is), while the stored label is the middle shaft, so the input-shaft
# max is the measured 15.38 Hz label max scaled back up by 95/29 = 50.39 Hz.
_IAS_BANDWIDTH_HZ = 15.38 * 95 / 29 * _CUTOFF_ORDER  # 4.03 Hz

# Seconds the speed channel lags the vibration in the `_FILES_TO_SHIFT` recordings.
_SPEED_LAG_SEC = 0.82

# Skipped upstream because the IAS measurement stops early / is intermittent (verbatim).
_SKIPPED_STEMS = {
    "gear_pitting_M_torque_circulation_2000rpm_10Nm",
    "gear_pitting_M_speed_circulation_20Nm-1000rpm",
}

# Recordings whose speed channel is offset against the vibration channels; the
# IAS is shifted by `_SPEED_LAG_SEC` to re-synchronize (verbatim upstream lookup table).
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
            # The cutoff is in input-shaft orders, which is where the tacho is, so the filter
            # runs before the transfer to the middle shaft below. `sl` truncates away the span
            # the 1 PPR tacho cannot resolve -- on this rig up to 3 s at the end, since a
            # ramping shaft can take that long to complete its final revolution.
            ias, sl = encoder_pulse_to_ias(
                data["speed"].to_numpy(), _FS, pulses_per_revolution=_PPR, cutoff_order=_CUTOFF_ORDER
            )
            if file.stem in _FILES_TO_SHIFT:
                # The speed channel lags the vibration in these recordings. `_SPEED_LAG_SEC * _FS`
                # is a whole number of samples, so this is an exact index shift; carrying the
                # valid span along with it keeps the newly exposed head from being extrapolated.
                lag = round(_SPEED_LAG_SEC * _FS)
                start, stop = sl.start + lag, min(sl.stop + lag, len(data))
                ias, sl = ias[: stop - start], slice(start, stop)
            # transfer to the middle shaft, which the vibration sensors are mounted on
            # (the tacho is on the input shaft)
            ias = ias * 29 / 95
            signals = {
                "IAS": ias,
                "gearbox_vibration_x": data["gearbox_vibration_x"].to_numpy()[sl] * 9.81,
                "gearbox_vibration_y": data["gearbox_vibration_y"].to_numpy()[sl] * 9.81,
                "gearbox_vibration_z": data["gearbox_vibration_z"].to_numpy()[sl] * 9.81,
            }
            # gear ratio: input shaft, gear mesh, middle shaft (IAS), gear mesh, output shaft
            save_signals_hdf5(
                signals,
                save_path / split_of[file] / f"{file.stem}.hdf5",
                fs=_FS,
                gear_ratio=[95 / 29, 95, 1, 36, 36 / 90],
                ias_bandwidth_hz=_IAS_BANDWIDTH_HZ,
            )

    write_disturbed_test_sets(save_path, vib_keys=["gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z"])


# version 2: order-domain IAS filtering (was a savgol + fixed 12.5 Hz time-domain low-pass).
parallel_gearbox_dataset = Dataset("parallel_gearbox", prepare=dl_parallel_gearbox, version="2")

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
    # window_sec=3.0: the largest single window across every upstream method's search space
    # over all four IAS datasets (unlike the per-dataset WindowedEstimation windows above,
    # this one is kept uniform — it's only a context guarantee, not a tuned averaging window).
    # step_sec: Nyquist for the label's retained band, _IAS_BANDWIDTH_HZ = 4.03 Hz -> 124 ms,
    # rounded down to 100 ms (1.24x margin). A 1 PPR tacho simply carries very little bandwidth,
    # so this is the one dataset whose grid does not get finer.
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
