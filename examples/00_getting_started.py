# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="https://raw.githubusercontent.com/daniel-om-weber/identibench/main/assets/logo.svg" width="200" align="left" alt="identibench logo">

# %% [markdown]
# ## IdentiBench
# [![PyPI version](https://badge.fury.io/py/identibench.svg)](https://badge.fury.io/py/identibench)
# [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# [![Docs Status](https://img.shields.io/badge/docs-up_to_date-brightgreen.svg)](https://daniel-om-weber.github.io/identibench/)
# [![Python Versions](https://img.shields.io/pypi/pyversions/identibench)](https://pypi.org/project/identibench/)
#
# IdentiBench is a Python library designed to streamline and standardize the benchmarking of system identification models. Evaluating and comparing dynamic models often requires repetitive setup for data handling, evaluation protocols, and metrics implementation, making fair comparisons and reproducing results challenging. IdentiBench tackles this by offering a collection of pre-defined benchmark specifications for simulation and prediction tasks, built upon common datasets. It automates data downloading and processing into a consistent format and provides standard evaluation metrics via a simple interface (run_benchmark). This allows you to focus your efforts on developing innovative models, while relying on IdentiBench for robust and reproducible evaluation.

# %% [markdown]
# ## Key Features
#
# * **Access Many Benchmarks from different systems:** Instantly utilize pre-configured benchmarks covering diverse domains like electronics (Silverbox), mechanics (Industrial Robot), process control (Cascaded Tanks), aerospace (Quadrotors), and more, available for both simulation and prediction tasks.
# * **Automate Data Management:** Forget manual downloading and processing; the library handles fetching data from various sources (web, Drive, Dataverse), extracting archives (ZIP, RAR, MAT, BAG), converting to a standard HDF5 format, and caching locally.
# * **Integrate Any Model to evaluate on all benchmarks:** Plug in your custom models, regardless of the Python framework used (NumPy, SciPy, PyTorch, TensorFlow, JAX, etc.), using a straightforward function interface (`build_model`) that receives all necessary context.
# * **Capture Comprehensive Results:** Obtain detailed evaluation reports including standard metrics (RMSE, NRMSE, FIT%, etc.), per-test-set scores, execution timings, and configuration parameters (hyperparameters, seed) for thorough analysis.
# * **Easily Define New Benchmarks:** Go beyond the included datasets by creating your own benchmark specifications — a `BenchmarkSpec` carrying a `Simulation` or `Prediction` task — for private data or unique tasks, leveraging the library's structure and transparent data format.

# %% [markdown]
# ## Installation
# You can install `identibench` using pip:
# ```bash
# pip install identibench
# ```
# To install the latest development version directly from GitHub, use:
# ```bash
# pip install git+https://github.com/daniel-om-weber/identibench.git
# ```

# %%
# Basic usage
import identibench as idb
from pathlib import Path

# Example: Download a single dataset
# Note: Always use a Path object, not a string
save_path = Path("./tmp/wh")
idb.datasets.workshop.dl_wiener_hammerstein(save_path)

# %% [markdown]
# ## Defining a model: a linear ARX baseline
#
# To plug a model into a benchmark you provide a `build_model(context)` function
# that trains on `context.get_train_sequences()` and returns a predictor callable.
# For a simulation benchmark the predictor is called as `model(u, y_init, attrs)`
# and must return the simulated output.
#
# Here we use a **linear ARX model** (AutoRegressive with eXogenous input) — one of
# the oldest, most established system-identification baselines. Each output sample
# is a linear combination of past outputs and past inputs,
#
# $$y(t) = c + \sum_{i=1}^{n_a} a_i\,y(t-i) + \sum_{j=1}^{n_b} b_j\,u(t-j),$$
#
# with the coefficients found by a single ordinary-least-squares solve — nothing
# beyond NumPy required. A linear model can't capture the static nonlinearity of a
# Wiener–Hammerstein system, so expect it to trail a NARX method like FROLS; it's a
# deliberately simple reference point.

# %%
import numpy as np


def build_arx_model(context):
    """Fit a linear ARX model by ordinary least squares and return a simulator.

    Hyperparameters:
        na: number of past-output lags (default 5).
        nb: number of past-input lags (default 5).
    """
    na = context.hyperparameters.get("na", 5)
    nb = context.hyperparameters.get("nb", 5)
    lag = max(na, nb)

    # Stack the one-step regression [1, y(t-1..t-na), u(t-1..t-nb)] -> y(t) across
    # every training sequence, then solve for all coefficients in one lstsq call.
    rows, targets = [], []
    for u, y, _ in context.get_train_sequences():
        u = np.asarray(u, dtype=np.float64)  # (N, n_u)
        y = np.asarray(y, dtype=np.float64)  # (N, 1)
        n = len(y)
        if n <= lag:
            continue
        phi_y = np.column_stack([y[lag - i : n - i, 0] for i in range(1, na + 1)])
        phi_u = np.column_stack([u[lag - j : n - j, :] for j in range(1, nb + 1)])
        rows.append(np.hstack([np.ones((n - lag, 1)), phi_y, phi_u]))
        targets.append(y[lag:n, 0])

    theta, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets), rcond=None)
    c = theta[0]
    a_rev = theta[1 : 1 + na][::-1].copy()  # reversed: a_rev @ y[t-na:t] == sum a_i y(t-i)
    b = theta[1 + na :].reshape(nb, -1)  # (nb, n_u)

    def model(u, y_init, attrs):
        u = np.asarray(u, dtype=np.float64)
        m = len(u)
        y_sim = np.zeros(m)
        k = len(y_init)
        y_sim[:k] = np.asarray(y_init, dtype=np.float64)[:, 0]  # warm-up from given outputs
        # The input-driven term is known up front; precompute it for the whole run.
        drive = np.full(m, c)
        for j in range(1, nb + 1):
            drive[j:] += u[: m - j] @ b[j - 1]
        # Free simulation: past *predictions* feed back through the AR part.
        for t in range(max(k, na), m):
            y_sim[t] = drive[t] + a_rev @ y_sim[t - na : t]
        return y_sim.reshape(-1, 1)

    return model


# %%
hyperparams = {
    "na": 5,  # number of past-output lags
    "nb": 5,  # number of past-input lags
}

results = idb.run_benchmark(spec=idb.BenchmarkWH_Simulation, build_model=build_arx_model, hyperparameters=hyperparams)
results["metric_score"]

# %%
# Generate table of available benchmarks
sim_md = "## Simulation Benchmarks\n\n"
sim_md += "| Key | Benchmark Name |\n"
sim_md += "|---|---|\n"
for key, spec in idb.simulation_benchmarks.items():
    sim_md += f"| `{key}` | {getattr(spec, 'name', 'N/A')} |\n"

print(sim_md)

pred_md = "\n## Prediction Benchmarks\n\n"
pred_md += "| Key | Benchmark Name |\n"
pred_md += "|---|---|\n"
for key, spec in idb.prediction_benchmarks.items():
    pred_md += f"| `{key}` | {getattr(spec, 'name', 'N/A')} |\n"

print(pred_md)

# %% [markdown]
# ## Workflow Details
#
# This section provides more detail on the core concepts and components of the `identibench` workflow.
#
# ### Benchmark Types
#
# Every benchmark is a single `BenchmarkSpec` carrying a **task** — a callable that owns the whole evaluation, including its metric. The library ships two built-in tasks; their parameters are readable from code (`spec.task.init_window`, `spec.task.horizon`, ...):
#
# * **Simulation (`Simulation(metric=..., init_window=...)`)**:
#     * **Goal:** Evaluate a model's ability to perform a free-run simulation, predicting the system's output over an extended period given the input sequence.
#     * **Typical Input to Predictor:** The full input sequence (`u_test`) and potentially an initial segment of the output sequence (`y_test[:init_window]`) for warm-up or state initialization.
#     * **Expected Output from Predictor:** The predicted output sequence (`y_pred`) corresponding to the input, usually excluding the warm-up period.
#     * **Use Case:** Assessing models intended for long-term prediction, control simulation, or understanding overall system dynamics.
#
# * **Prediction (`Prediction(horizon=..., step=..., metric=..., init_window=...)`)**:
#     * **Goal:** Evaluate a model's ability to predict the system's output *k* steps into the future based on recent past data.
#     * **Typical Input to Predictor:** Sliding windows of past inputs and outputs (e.g., `u[t:t+H]`, `y[t:t+H]`).
#     * **Expected Output from Predictor:** The predicted output over the window. The `horizon` parameter defines 'k', and `step` defines how frequently prediction windows start.
#     * **Use Case:** Evaluating models focused on short-to-medium term forecasting, state estimation, or receding horizon control.
#
# * **`init_window`**: Both built-in tasks carry an `init_window`. This specifies an initial number of time steps whose data might be provided to the model for initialization or warm-up. Importantly, data within this window is *excluded* from the final performance metric calculation to ensure a fair evaluation of the model's predictive capabilities beyond the initial transient. `init_window=0` is a valid free-run setting — the model then receives an *empty* `y_init`.
#
# * **Named test sets**: Every spec names its test sets explicitly in `spec.test_sets`, each with its own file patterns (e.g. Silverbox's `multisine` / `arrow_full` / `arrow_no_extrapolation` are three explicit files). All named sets are scored into `result["test_sets"]`; the built-in tasks headline the first named set, and a task that pools across sets (e.g. the orientation benchmarks' cross-set `"all"`) names its own pool in its `EvalResult.headline`.
#
# ### Model Interface (`build_model`)
#
# The core of integrating your custom logic is the `build_model` function you provide to `run_benchmark`.
#
# * **Purpose:** This function is responsible for defining your model architecture, training it using the provided data, and returning a callable predictor function.
# * **Input (`context: TrainingContext`):** Your `build_model` function receives a single argument, `context`, which is a `TrainingContext` object. This object gives you access to:
#     * `context.spec`: The full specification of the current benchmark being run (dataset path, input/output columns, ...). Evaluation parameters live on the task: `context.spec.task.init_window`, `context.spec.task.horizon`, etc.
#     * `context.hyperparameters`: A dictionary containing any hyperparameters you passed to `run_benchmark`. Use this to configure your model or training process.
#     * `context.seed`: A random seed for ensuring reproducibility.
#     * Data Access Methods: Functions like `context.get_train_sequences()` and `context.get_valid_sequences()` provide iterators over the raw, full-length training and validation data sequences (as tuples of NumPy arrays `(u, y, x)`). **Note:** You need to handle any batching or windowing required for your specific training algorithm *within* your `build_model` function.
# * **Output (Predictor `Callable`):** `build_model` *must* return a callable object (e.g., a function, an object's method) that represents your trained model ready for prediction/simulation. This returned callable will be used internally by `run_benchmark` on the test set. Its expected signature depends on the benchmark type, but typically it accepts NumPy arrays for test inputs (and potentially initial outputs) and returns a NumPy array containing the predictions.
#
# ### Running Multiple Benchmarks
#
# To evaluate a model across several scenarios efficiently, use the `run_multiple_benchmarks` function:

# %%
# Example: Run on a subset of benchmarks
specs_to_run = {
    "WH_Sim": idb.simulation_benchmarks["WH_Sim"],
    "Silverbox_Sim": idb.simulation_benchmarks["Silverbox_Sim"],
}

# Assume 'my_build_model' is your defined build function
all_results = idb.run_benchmarks(specs_to_run, build_model=build_arx_model, n_times=3)

all_results

# %% [markdown]
# This function iterates through the provided list or dictionary of benchmark specifications, calling `run_benchmark` for each one using the same `build_model` function and hyperparameters.

# %%
# calculate mean and std of the results
idb.aggregate_benchmark_results(all_results, agg_funcs=["mean", "std"])

# %% [markdown]
# ### Data Handling & Format
#
# Understanding how `identibench` organizes and stores data is helpful for direct interaction or adding new datasets.
#
# * **Two levels, strictly separated:** A `Dataset` only downloads and prepares files — it carries no roles, splits, or test sets. A `BenchmarkSpec` defines everything else: which files play which role, selected by explicit `(dataset, glob)` patterns. The same files can be split differently by different benchmarks.
# * **Directory Structure:** Datasets are stored under a root directory (default: `~/.identibench_data`, configurable via the `IDENTIBENCH_DATA_ROOT` environment variable) as `DATA_ROOT / [dataset_id] / ...` — the layout below the dataset directory is whatever the preparer writes (most use `train/`, `valid/`, `test/` subdirectories).
# * **Preparation sentinel:** A successful preparation ends by writing a `.prepared` file containing the dataset's format version. A directory without a matching sentinel is treated as absent and re-prepared from a clean slate, so an interrupted download can never masquerade as a ready dataset.
# * **Download & Cache:** Data is downloaded automatically when a benchmark requires it and cached locally to avoid re-downloads. The `identibench.datasets.download_all_datasets` function can fetch all datasets at once.
# * **File Format:** Processed time-series data is stored in the **HDF5 (`.hdf5`)** format.
# * **HDF5 Structure:**
#     * Each `.hdf5` file typically represents one experimental run.
#     * Signals (inputs, outputs, states) are stored as separate 1-dimensional datasets within the file, named conventionally as `u0`, `u1`, ..., `y0`, `y1`, ..., `x0`, ...
#     * Data is usually stored as `float32` NumPy arrays.
#     * Metadata like sampling frequency (`fs`) and suggested initialization window size (`init_sz`) are stored as attributes on the root group of the HDF5 file.
#     * *Example Structure:*
#         ```
#         my_dataset/
#         └── train/
#             └── train_run_1.hdf5
#                 ├── u0 (Dataset: shape=(N,), dtype=float32)
#                 ├── y0 (Dataset: shape=(N,), dtype=float32)
#                 └── Attributes:
#                     └── fs (Attribute: float)
#         ```
# * **Extensibility:** Adhering to this HDF5 format ensures compatibility when adding new dataset loaders. Helper functions like `identibench.utils.write_array` facilitate creating files in the correct format.
#
# ### Understanding Benchmark Results
#
# The `run_benchmark` function returns a dictionary containing detailed results of the experiment. Key entries include:
#
# * `benchmark_name` (`str`): The unique name of the benchmark specification used.
# * `datasets` (`list[str]`): The ids of every dataset the spec draws files from.
# * `hyperparameters` (`dict`): The hyperparameters dictionary passed to the run.
# * `seed` (`int`): The random seed used for the run.
# * `training_time_seconds` (`float`): Wall-clock time spent inside your `build_model` function.
# * `test_time_seconds` (`float`): Wall-clock time spent evaluating the returned predictor on the test set.
# * `benchmark_type` (`str`): The name of the task that ran (e.g., `'Simulation'`, `'Prediction'`).
# * `metric_name` (`str`): The headline metric named by the task.
# * `metric_score` (`float`): The value of the headline `(set, metric)` cell the task names in its `EvalResult.headline`.
# * `test_sets` (`dict`): The full `{test_set: {metric: value}}` scores — every named test set is scored, not just the headline one. Flattened to `test_sets.<set>.<metric>` columns by `benchmark_results_to_dataframe`.
# * `diagnostics` (`dict`): Non-scalar artifacts a task chooses to return (e.g. raw predictions under the reserved key `"predictions"`); empty for the built-in tasks and dropped from the DataFrame.
