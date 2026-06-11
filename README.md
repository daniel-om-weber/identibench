
<img src="https://raw.githubusercontent.com/daniel-om-weber/identibench/main/assets/logo.svg" width="200" align="left" alt="identibench logo">

## IdentiBench

[![PyPI version](https://badge.fury.io/py/identibench.svg)](https://badge.fury.io/py/identibench)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/daniel-om-weber/identibench/actions/workflows/test.yaml/badge.svg)](https://github.com/daniel-om-weber/identibench/actions/workflows/test.yaml)
[![Python Versions](https://img.shields.io/pypi/pyversions/identibench)](https://pypi.org/project/identibench/)

IdentiBench is a Python library designed to streamline and standardize
the benchmarking of system identification models. Evaluating and
comparing dynamic models often requires repetitive setup for data
handling, evaluation protocols, and metrics implementation, making fair
comparisons and reproducing results challenging. IdentiBench tackles
this by offering a collection of pre-defined benchmark specifications
for simulation and prediction tasks, built upon common datasets. It
automates data downloading and processing into a consistent format and
provides standard evaluation metrics via a simple interface
(run_benchmark). This allows you to focus your efforts on developing
innovative models, while relying on IdentiBench for robust and
reproducible evaluation.

## Key Features

- **Access Many Benchmarks from different systems:** Instantly utilize
  pre-configured benchmarks covering diverse domains like electronics
  (Silverbox), mechanics (Industrial Robot), process control (Cascaded
  Tanks), aerospace (Quadrotors), and more, available for both
  simulation and prediction tasks.
- **Automate Data Management:** Forget manual downloading and
  processing; the library handles fetching data from various sources
  (web, Drive, Dataverse), extracting archives (ZIP, RAR, MAT, BAG),
  converting to a standard HDF5 format, and caching locally.
- **Integrate Any Model to evaluate on all benchmarks:** Plug in your
  custom models, regardless of the Python framework used (NumPy, SciPy,
  PyTorch, TensorFlow, JAX, etc.), using a straightforward function
  interface (`build_model`) that receives all necessary context.
- **Capture Comprehensive Results:** Obtain detailed evaluation reports
  including standard metrics (RMSE, NRMSE, FIT%, etc.), per-test-set
  scores, execution timings, and configuration parameters
  (hyperparameters, seed) for thorough analysis.
- **Easily Define New Benchmarks:** Go beyond the included datasets:
  point a `Dataset` at a directory of HDF5 files (no manifest, no
  registration), select files with explicit glob patterns, and pick a
  `Simulation` or `Prediction` task (or any custom task callable) — a
  complete benchmark is one `BenchmarkSpec` literal.

## Installation

You can install `identibench` using pip:

```bash
pip install identibench
```

To install the latest development version directly from GitHub, use:

```bash
pip install git+https://github.com/daniel-om-weber/identibench.git
```

For development:

```bash
git clone https://github.com/daniel-om-weber/identibench.git
cd identibench
uv sync --extra dev
```

## Quickstart

A model is a `build_model(context)` function that trains on the benchmark's
training data and returns a *predictor* callable. For a simulation benchmark the
predictor is called as `model(u, y_init, attrs)` and returns the simulated
output. That's the whole interface — IdentiBench handles downloading the data,
running your predictor over every test sequence, and scoring it.

```python
import numpy as np
import identibench as idb

def build_model(context):
    # Train on the benchmark's training split. This trivial baseline just
    # predicts the mean of the training output — replace it with your own model.
    y_train = np.concatenate([seq.y for seq in context.get_train_sequences()])
    y_mean = y_train.mean(axis=0)

    def model(u, y_init, attrs):
        # Called once per test sequence; returns an array of shape (len(u), n_y).
        return np.tile(y_mean, (len(u), 1))

    return model

# The first call downloads and caches the dataset under ~/.identibench_data
# (override with the IDENTIBENCH_DATA_ROOT environment variable). WH is small
# (a few MB), so it's a good first benchmark.
result = idb.run_benchmark(idb.BenchmarkWH_Simulation, build_model)
print(result["metric_score"])   # primary metric (RMSE in mV) on the test set
```

For complete, runnable models see the [`examples/`](examples/) notebooks:
[`00_getting_started`](examples/00_getting_started.py) fits a linear ARX baseline
on a simulation benchmark, and
[`01_riann_orientation`](examples/01_riann_orientation.py) scores an orientation
model on the IMU benchmarks.

> **Datasets download on first use** and are cached locally, so you only pay for
> them once. The classic simulation/prediction benchmarks are small (a few MB
> each); the orientation/IMU and quadrotor datasets are larger (hundreds of MB —
> e.g. BROAD is ~0.8 GB), so start with a small benchmark such as `WH_Sim`.

### Training a model with hyperparameters

`build_model` receives the full `context`, so you can fit any model and read
settings from `context.hyperparameters`. This example trains a NARX model with
[`sysidentpy`](https://sysidentpy.org) (install it first: `pip install sysidentpy`):

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares

def build_frols_model(context):
    u_train, y_train, _ = next(context.get_train_sequences())

    ylag = context.hyperparameters.get('ylag', 5)
    xlag = context.hyperparameters.get('xlag', 5)
    n_terms = context.hyperparameters.get('n_terms', 10)
    estimator = context.hyperparameters.get('estimator', LeastSquares())

    _model = FROLS(xlag=xlag, ylag=ylag, n_terms=n_terms, estimator=estimator)
    _model.fit(X=u_train, y=y_train)

    def model(u_test, y_init, attrs):
        yhat_full = _model.predict(X=u_test, y=y_init[:_model.max_lag])
        return yhat_full[_model.max_lag:]

    return model
```

```python
hyperparams = {
    'ylag': 2,
    'xlag': 2,
    'n_terms': 10,            # number of terms for FROLS
    'estimator': LeastSquares(),
}

results = idb.run_benchmark(
    spec=idb.BenchmarkWH_Simulation,
    build_model=build_frols_model,
    hyperparameters=hyperparams,
)
```

## Simulation Benchmarks

| Key                | Benchmark Name                    |
|--------------------|-----------------------------------|
| `WH_Sim`           | BenchmarkWH_Simulation            |
| `Silverbox_Sim`    | BenchmarkSilverbox_Simulation     |
| `Tanks_Sim`        | BenchmarkCascadedTanks_Simulation |
| `CED_Sim`          | BenchmarkCED_Simulation           |
| `EMPS_Sim`         | BenchmarkEMPS_Simulation          |
| `NoisyWH_Sim`      | BenchmarkNoisyWH_Simulation       |
| `RobotForward_Sim` | BenchmarkRobotForward_Simulation  |
| `RobotInverse_Sim` | BenchmarkRobotInverse_Simulation  |
| `Ship_Sim`         | BenchmarkShip_Simulation          |
| `QuadPelican_Sim`  | BenchmarkQuadPelican_Simulation   |
| `QuadPi_Sim`       | BenchmarkQuadPi_Simulation        |

## IAS (Instantaneous Angular Speed) Benchmarks

Estimate the instantaneous angular speed `IAS` (Hz) of rotating machinery from
vibration/acceleration channels. All four use the `WindowedEstimation` task: the
model is applied to **non-overlapping windows** of a per-dataset `window_sec` and
emits one estimate per window, scored against the window-mean IAS. The per-window
absolute errors are pooled (micro) into MAE in Hz — the headline `metric_score`
— with `medae`/`std`/`max` reported alongside. The model is called as
`model(u_window, y_init, attrs)` with an empty `y_init`, and its per-window output
is mean-reduced.

This is a single **standardized** protocol that captures the shape of the upstream
IAS benchmark's windowed evaluation (windowed, window-mean target, pooled MAE in
Hz). It is *not* a drop-in reproduction of the upstream results table: the
original scored each model with its own window size (a per-model tuning knob), step
(overlapping for some models), and target granularity. Fixing one window per
dataset trades exact reproduction for a fairer apples-to-apples comparison; set
`window_sec` (a task parameter) to a model's tuned window to align with its upstream
run.

| Key                          | Benchmark Name                       | Inputs (u)                | `window_sec` |
|------------------------------|--------------------------------------|---------------------------|--------------|
| `BallBearing_Estimation`     | BenchmarkBallBearing_Estimation      | `Acc_x`                   | 2.0          |
| `ParallelGearbox_Estimation` | BenchmarkParallelGearbox_Estimation  | `gearbox_vibration_x/y/z` | 2.2          |
| `PlanetaryGearbox_Estimation`| BenchmarkPlanetaryGearbox_Estimation | `Acc_Carrier`, `Acc_Sun`  | 2.7          |
| `GasFoilBearing_Estimation`  | BenchmarkGasFoilBearing_Estimation   | `Acc_x`, `Acc_y`          | 2.5          |

Each `window_sec` is set to the **largest window any upstream method needed** on
that dataset (~2–2.7 s). This unifies the evaluation across method families: a
method needing fewer samples can always *crop or decimate* a longer window (the
FFT/order-tracking methods — FFT nets, MOPA, ViBES — decimate then STFT), but none
can run on a window that is too short. The largest windows turn out ~uniform in
time (~0.4 Hz spectral resolution), not in revolutions. Sizing this way means no
method is starved — e.g. MOPA needs ~2.5 s on the gas-foil bearing, which a
revolution-based window would have cut to a fraction of that.

Each dataset exposes several named test conditions: `basic` (in-distribution —
the headline `metric_score`), `wear` (out-of-distribution fault severities;
absent for the gas foil bearing), and `disturbed_{15,7.5,0}dB` (copies of
`basic` with reproducible synthetic sensor noise at decreasing SNR). All
conditions are scored into `result["test_sets"]`.

These live in `identibench.datasets.ias` and the `idb.ias_benchmarks` registry.
The stratified splits require `scikit-learn`
(`pip install "identibench[ias]"`); downloads are sizable (the ball bearing
dataset is recorded at 200 kHz).

## Orientation (IMU) Benchmarks

Estimate orientation (a unit quaternion) from 6-axis IMU data and score it with
the inclination (tilt) error in degrees. These are free-run estimation
benchmarks — plug in any model via `build_model` (neural network, complementary
filter, …).

The **RIANN** benchmarks port the six datasets from
[Weber et al. 2021](https://doi.org/10.3390/ai2030028). The combined RIANN
benchmark reproduces the paper's pooled-train / cross-dataset-test protocol as
explicit file patterns over the six source datasets (stored once — the
benchmark adds no data of its own); the per-source benchmarks evaluate a single
dataset in isolation. Each is downloaded from its original public source on
first use.

| Key                  | Benchmark Name                 | Files (role)            |
|----------------------|--------------------------------|-------------------------|
| `RIANN_Inclination`  | BenchmarkRIANN_Inclination     | 36 train / 6 valid / 119 test (all sources) |
| `BROAD_Inclination`  | BenchmarkBROAD_Inclination     | 39 test                 |
| `TUMVI_Inclination`  | BenchmarkTUMVI_Inclination     | 6 test                  |
| `OxIOD_Inclination`  | BenchmarkOxIOD_Inclination     | 71 test                 |
| `EuRoC_Inclination`  | BenchmarkEuRoC_Inclination     | 6 test                  |
| `RepoIMU_Inclination`| BenchmarkRepoIMU_Inclination   | 21 test                 |
| `Caruso_Inclination` | BenchmarkCaruso_Inclination    | 18 test                 |
| `DFJIMU_Inclination` | BenchmarkDFJIMU_Inclination    | Weygers & Kok (2020)    |
| `DFJIMU_Relative`    | BenchmarkDFJIMU_Relative       | Weygers & Kok (2020)    |

All of these live in `identibench.datasets.orientation` and are grouped in the
`idb.orientation_benchmarks` registry. Inputs are
`u_cols = [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, dt]` (acc m/s², gyr rad/s,
`dt` the per-sample sampling interval in seconds; the dfjimu benchmarks omit
`dt`) and the target is `y_cols = [q_w, q_x, q_y, q_z]` (quaternion `w, x, y, z`).

```python
import numpy as np
import identibench as idb

def build_model(context):
    # A trivial baseline that always predicts the identity orientation.
    def model(u, y_init, attrs):
        return np.tile([1.0, 0.0, 0.0, 0.0], (len(u), 1))
    return model

# Evaluate on a single small dataset...
result = idb.run_benchmark(idb.BenchmarkEuRoC_Inclination, build_model)

# ...or reproduce the full RIANN cross-dataset protocol in one run.
result = idb.run_benchmark(idb.BenchmarkRIANN_Inclination, build_model)
print(result["metric_score"])          # masked, sample-pooled inclination RMSE over all sources, deg
print(result["test_sets"])             # masked RMSE + 99th pct, per source + pooled "all"
```

> **Which number to report.** The headline `metric_score` is RIANN's faithful
> number: the *masked*, first-sample-aligned inclination RMSE, sample-pooled
> across all sources (the `"all"` test set). The per-source breakdown and the
> 99th percentiles live in `result["test_sets"]`
> (`{<source>: {incl_rmse_deg, incl_p99_deg}, "all": …}`), surfaced as
> `test_sets.<source>.<metric>` columns by `benchmark_results_to_dataframe`.

## Prediction Benchmarks

| Key                 | Benchmark Name                    |
|---------------------|-----------------------------------|
| `WH_Pred`           | BenchmarkWH_Prediction            |
| `Silverbox_Pred`    | BenchmarkSilverbox_Prediction     |
| `Tanks_Pred`        | BenchmarkCascadedTanks_Prediction |
| `CED_Pred`          | BenchmarkCED_Prediction           |
| `EMPS_Pred`         | BenchmarkEMPS_Prediction          |
| `NoisyWH_Pred`      | BenchmarkNoisyWH_Prediction       |
| `RobotForward_Pred` | BenchmarkRobotForward_Prediction  |
| `RobotInverse_Pred` | BenchmarkRobotInverse_Prediction  |
| `Ship_Pred`         | BenchmarkShip_Prediction          |
| `QuadPelican_Pred`  | BenchmarkQuadPelican_Prediction   |
| `QuadPi_Pred`       | BenchmarkQuadPi_Prediction        |

## Workflow Details

This section provides more detail on the core concepts and components of
the `identibench` workflow.

### Benchmark Types

Every benchmark is a single `BenchmarkSpec` (identity + data binding) carrying a
**task** — a callable that owns the whole evaluation, including its metric. The
library ships two built-in tasks; their parameters are readable from code
(`spec.task.init_window`, `spec.task.horizon`, … or generically via
`dataclasses.asdict(spec.task)`):

- **Simulation (`Simulation(metric=..., init_window=...)`)**:
  - **Goal:** Evaluate a model’s ability to perform a free-run
    simulation, predicting the system’s output over an extended period
    given the input sequence.
  - **Typical Input to Predictor:** The full input sequence (`u_test`)
    and potentially an initial segment of the output sequence
    (`y_test[:init_window]`) for warm-up or state initialization.
  - **Expected Output from Predictor:** The predicted output sequence
    (`y_pred`) corresponding to the input, usually excluding the warm-up
    period.
  - **Use Case:** Assessing models intended for long-term prediction,
    control simulation, or understanding overall system dynamics.
- **Prediction (`Prediction(horizon=..., step=..., metric=..., init_window=...)`)**:
  - **Goal:** Evaluate a model’s ability to predict the system’s output
    *k* steps into the future based on recent past data.
  - **Typical Input to Predictor:** Sliding windows of past inputs and
    outputs (e.g., `u[t:t+H]`, `y[t:t+H]`).
  - **Expected Output from Predictor:** The predicted output over the
    window. The `horizon` parameter defines ‘k’, and `step` defines how
    frequently prediction windows start.
  - **Use Case:** Evaluating models focused on short-to-medium term
    forecasting, state estimation, or receding horizon control.
- **`init_window`**: Both built-in tasks carry an `init_window`. This
  specifies an initial number of time steps whose data might be provided
  to the model for initialization or warm-up. Importantly, data within
  this window is *excluded* from the final performance metric calculation
  to ensure a fair evaluation of the model’s predictive capabilities
  beyond the initial transient. `init_window=0` is a valid free-run
  setting — the model then receives an *empty* `y_init`.
- **Custom tasks**: A task is any callable
  `(spec, model) -> EvalResult`. The `EvalResult` carries the full
  `{test_set: {metric: value}}` scores, an explicit
  `headline=(set, metric)` pair naming the leaderboard cell, and optional
  non-scalar `diagnostics` — so novel evaluations are defined without
  touching the library, e.g. the orientation benchmarks’
  `MaskedPooledInclination` task.
- **Named test sets**: Every spec names its test sets explicitly in
  `spec.test_sets`, each with its own file patterns (e.g. Silverbox’s
  `multisine` / `arrow_full` / `arrow_no_extrapolation` are three
  explicit files, Ship’s `ood` is the `test_ood/` directory). All named
  sets are scored into `result["test_sets"]`; the built-in tasks
  headline the first named set, and a task that pools across sets (the
  orientation benchmarks’ cross-set `"all"`) names its own pool in its
  `EvalResult.headline`.

### Model Interface (`build_model`)

The core of integrating your custom logic is the `build_model` function
you provide to `run_benchmark`.

- **Purpose:** This function is responsible for defining your model
  architecture, training it using the provided data, and returning a
  callable predictor function.
- **Input (`context: TrainingContext`):** Your `build_model` function
  receives a single argument, `context`, which is a `TrainingContext`
  object. This object gives you access to:
  - `context.spec`: The full specification of the current benchmark
    being run (dataset path, input/output columns, …). Evaluation
    parameters live on the task: `context.spec.task.init_window`,
    `context.spec.task.horizon`, etc.
  - `context.hyperparameters`: A dictionary containing any
    hyperparameters you passed to `run_benchmark`. Use this to configure
    your model or training process.
  - `context.seed`: A random seed for ensuring reproducibility.
  - Data Access Methods: Functions like `context.get_train_sequences()`
    and `context.get_valid_sequences()` provide iterators over the raw,
    full-length data sequences as `Sequence(u, y, attrs)` named tuples,
    where `u` and `y` are NumPy arrays and `attrs` is a dict of that
    file's HDF5 attributes (e.g. the sampling rate `fs`). **Note:** You
    need to handle any batching or windowing required for your specific
    training algorithm *within* your `build_model` function.
- **Output (Predictor `Callable`):** `build_model` *must* return a
  callable with the signature `model(u, y_init, attrs)` that returns the
  predicted output as a NumPy array of shape `(len(u), len(y_cols))`:
  - `u`: the input sequence to predict over — the full test signal for a
    simulation benchmark, or a single window for a prediction benchmark.
  - `y_init`: the first `task.init_window` ground-truth output samples,
    provided for warm-up / state initialization (empty when
    `init_window=0`).
  - `attrs`: a dict of the test file's HDF5 attributes (e.g. the sampling
    rate `fs`).

  `run_benchmark` calls this predictor on each test sequence and scores the
  returned predictions against the held-out targets.

### Running Multiple Benchmarks

To evaluate a model across several scenarios efficiently, use the
`run_benchmarks` function:

```python
# Example: Run on a subset of benchmarks
specs_to_run = {
    'WH_Sim': idb.simulation_benchmarks['WH_Sim'],
    'Silverbox_Sim': idb.simulation_benchmarks['Silverbox_Sim']
}

# Assume 'my_build_model' is your defined build function
all_results = idb.run_benchmarks(specs_to_run, build_model=build_frols_model,n_times=3)

all_results
```

    --- Starting benchmark run for 2 specifications, repeating each 3 times ---

    -- Repetition 1/3 --

    [1/6] Running: BenchmarkWH_Simulation (Rep 1)
      -> Success: BenchmarkWH_Simulation (Rep 1) completed.

    [2/6] Running: BenchmarkSilverbox_Simulation (Rep 1)
      -> Success: BenchmarkSilverbox_Simulation (Rep 1) completed.

    -- Repetition 2/3 --

    [3/6] Running: BenchmarkWH_Simulation (Rep 2)
      -> Success: BenchmarkWH_Simulation (Rep 2) completed.

    [4/6] Running: BenchmarkSilverbox_Simulation (Rep 2)
      -> Success: BenchmarkSilverbox_Simulation (Rep 2) completed.

    -- Repetition 3/3 --

    [5/6] Running: BenchmarkWH_Simulation (Rep 3)
      -> Success: BenchmarkWH_Simulation (Rep 3) completed.

    [6/6] Running: BenchmarkSilverbox_Simulation (Rep 3)
      -> Success: BenchmarkSilverbox_Simulation (Rep 3) completed.

    --- Benchmark run finished. 6/6 individual runs completed successfully. ---

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | benchmark_name | datasets | hyperparameters | seed | training_time_seconds | test_time_seconds | benchmark_type | metric_name | metric_score | test_sets.test.rmse_mV | test_sets.multisine.rmse_mV | test_sets.arrow_full.rmse_mV | test_sets.arrow_no_extrapolation.rmse_mV |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | BenchmarkWH_Simulation | [wh] | {} | 2406651230 | 4.944649 | 1.012850 | Simulation | rmse_mV | 42.161572 | 42.161572 | NaN | NaN | NaN |
| 1 | BenchmarkSilverbox_Simulation | [silverbox] | {} | 3813113752 | 2.839149 | 1.246224 | Simulation | rmse_mV | 8.501941 | NaN | 8.501941 | 16.154317 | 7.5409 |
| 2 | BenchmarkWH_Simulation | [wh] | {} | 1950649438 | 4.801520 | 1.034119 | Simulation | rmse_mV | 42.161572 | 42.161572 | NaN | NaN | NaN |
| 3 | BenchmarkSilverbox_Simulation | [silverbox] | {} | 1560698088 | 2.880391 | 1.217932 | Simulation | rmse_mV | 8.501941 | NaN | 8.501941 | 16.154317 | 7.5409 |
| 4 | BenchmarkWH_Simulation | [wh] | {} | 3258007268 | 4.916941 | 1.021927 | Simulation | rmse_mV | 42.161572 | 42.161572 | NaN | NaN | NaN |
| 5 | BenchmarkSilverbox_Simulation | [silverbox] | {} | 4194043971 | 2.937101 | 1.231710 | Simulation | rmse_mV | 8.501941 | NaN | 8.501941 | 16.154317 | 7.5409 |

</div>

This function iterates through the provided list or dictionary of
benchmark specifications, calling `run_benchmark` for each one using the same `build_model` function and hyperparameters.

```python
#calculate mean and std of the results
idb.aggregate_benchmark_results(all_results,agg_funcs=['mean','std'])
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead tr th {
        text-align: left;
    }
&#10;    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>

|  | training_time_seconds |  | test_time_seconds |  | metric_score |  | test_sets.multisine.rmse_mV |  | test_sets.arrow_full.rmse_mV |  | test_sets.arrow_no_extrapolation.rmse_mV |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | mean | std | mean | std | mean | std | mean | std | mean | std | mean | std |
| benchmark_name |  |  |  |  |  |  |  |  |  |  |  |  |
| BenchmarkSilverbox_Simulation | 2.885547 | 0.049179 | 1.231955 | 0.014147 | 8.501941 | 0.0 | 8.501941 | 0.0 | 16.154317 | 0.0 | 7.5409 | 0.0 |
| BenchmarkWH_Simulation | 4.887703 | 0.075912 | 1.022966 | 0.010673 | 42.161572 | 0.0 | NaN | NaN | NaN | NaN | NaN | NaN |

</div>

### Data Handling & Format

Understanding how `identibench` organizes and stores data is helpful for
direct interaction or adding new datasets.

- **Two levels, strictly separated:** A `Dataset` only downloads and
  prepares files — it carries no roles, splits, or test sets. A
  `BenchmarkSpec` defines everything else: which files play which role,
  selected by explicit `(dataset, glob)` patterns. The same files can be
  split differently by different benchmarks (e.g. the RIANN benchmark
  draws train/valid/test from six datasets that per-source benchmarks
  evaluate whole).
- **Directory Structure:** Datasets are stored under a root directory
  (default: `~/.identibench_data`, configurable via the
  `IDENTIBENCH_DATA_ROOT` environment variable, or temporarily via the
  `with idb.data_root(path): ...` context manager — handy in tests) as
  `DATA_ROOT / [dataset_id] / ...` — the layout below the dataset
  directory is whatever the preparer writes (most use `train/`, `valid/`,
  `test/` subdirectories; the orientation datasets are flat).
- **Preparation sentinel:** A successful preparation ends by writing a
  `.prepared` file containing the dataset’s format version. A directory
  without a matching sentinel is treated as absent and re-prepared from a
  clean slate, so an interrupted download can never masquerade as a ready
  dataset. (To adopt an externally prepared, layout-compatible cache:
  `echo -n 1 > <dataset_dir>/.prepared`.)
- **Download & Cache:** Data is downloaded automatically when a
  benchmark requires it and cached locally to avoid re-downloads. The
  `identibench.datasets.download_all_datasets` function can fetch all
  datasets at once.
- **File Format:** Processed time-series data is stored in the **HDF5
  (`.hdf5`)** format.
- **HDF5 Structure:**
  - Each `.hdf5` file typically represents one experimental run.
  - Signals (inputs, outputs, states) are stored as separate
    1-dimensional datasets within the file, named conventionally as
    `u0`, `u1`, …, `y0`, `y1`, …, `x0`, …
  - Data is usually stored as `float32` NumPy arrays.
  - Metadata like sampling frequency (`fs`) and suggested initialization
    window size (`init_sz`) are stored as attributes on the root group
    of the HDF5 file.
  - *Example Structure:*
    `my_dataset/       └── train/           └── train_run_1.hdf5               ├── u0 (Dataset: shape=(N,), dtype=float32)               ├── y0 (Dataset: shape=(N,), dtype=float32)               └── Attributes:                   └── fs (Attribute: float)`
- **Extensibility:** Adhering to this HDF5 format ensures compatibility
  when adding new dataset loaders. Helper functions like
  `identibench.utils.write_array`
  facilitate creating files in the correct format.

### Understanding Benchmark Results

The `run_benchmark` function returns a dictionary containing detailed results of the
experiment. Key entries include:

- `benchmark_name` (`str`): The unique name of the benchmark
  specification used.
- `datasets` (`list[str]`): The ids of every dataset the spec draws
  files from.
- `hyperparameters` (`dict`): The hyperparameters dictionary passed to
  the run.
- `seed` (`int`): The random seed used for the run.
- `training_time_seconds` (`float`): Wall-clock time spent inside your
  `build_model` function.
- `test_time_seconds` (`float`): Wall-clock time spent evaluating the
  returned predictor on the test set.
- `benchmark_type` (`str`): The name of the task that ran (e.g.,
  `'Simulation'`, `'Prediction'`, `'MaskedPooledInclination'`).
- `metric_name` (`str`): The headline metric named by the task.
- `metric_score` (`float`): The value of the headline `(set, metric)`
  cell the task names in its `EvalResult.headline`.
- `test_sets` (`dict`): The full `{test_set: {metric: value}}` scores —
  every named test set is scored, not just the primary one. Flattened to
  `test_sets.<set>.<metric>` columns by `benchmark_results_to_dataframe`.
- `diagnostics` (`dict`): Non-scalar artifacts a task chooses to return
  (e.g. raw predictions under the reserved key `"predictions"`); empty
  for the built-in tasks and dropped from the DataFrame.
