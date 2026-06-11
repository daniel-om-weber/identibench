"""Core benchmark specification, built-in task callables, and runner functions."""

__all__ = [
    "BenchmarkSpec",
    "Patterns",
    "Simulation",
    "Prediction",
    "WindowedEstimation",
    "EvalResult",
    "TestSetScores",
    "TrainingContext",
    "aggregate_metric_score",
    "evaluate_per_test_set",
    "pooled_scores_per_test_set",
    "run_benchmark",
    "benchmark_results_to_dataframe",
    "run_benchmarks",
    "aggregate_benchmark_results",
]

import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import Dataset
from .utils import Sequence, _load_sequences_from_files

# {test_set_name: {metric_name: value}} — the scalar result of one task evaluation.
TestSetScores = dict[str, dict[str, float]]

# A file selection: (dataset, glob) pairs, each glob relative to its dataset's directory.
Patterns = list[tuple[Dataset, str]]


def aggregate_metric_score(
    test_results: list,  # `(y_pred, y_true)` tuples (simulation) or nested lists thereof (prediction).
    metric_func: Callable[[np.ndarray, np.ndarray], float],  # Metric called as `func(y_pred, y_true)`.
    score_name: str | None = None,  # Key for the returned dict; defaults to `metric_func.__name__`.
    sequence_aggregation_func: Callable[..., float] = np.mean,  # Reduces per-sequence scores to a scalar.
    window_aggregation_func: Callable[..., float] = np.mean,  # Reduces per-window scores within a sequence.
) -> dict[str, float]:
    """Computes a single aggregated score from per-sequence test results.

    The metric is applied to each `(y_pred, y_true)` pair as `metric_func(y_pred, y_true)`
    (matching how the metrics in `metrics.py` are defined, `func(inp=pred, targ=true)`).
    Multi-channel metric outputs and the per-window/per-sequence axes are reduced to a
    single scalar (mean by default), so the returned score is averaged across both
    channels and sequences.

    Returns:
        A dict mapping `score_name` to the aggregated scalar score. For empty
        `test_results` the score is `np.nan`.
    """
    if score_name is None:
        score_name = metric_func.__name__
    if not test_results:
        return {score_name: np.nan}
    if isinstance(test_results[0], list):
        scores = []
        for windowed_sequence in test_results:
            scores.append(
                window_aggregation_func([metric_func(y_pred, y_test) for y_pred, y_test in windowed_sequence])
            )
    else:
        scores = [[metric_func(y_pred, y_test) for y_pred, y_test in test_results]]
    return {score_name: sequence_aggregation_func(scores)}


@dataclass
class EvalResult:
    """What a task returns: scalar ``scores`` feed the leaderboard, ``headline`` names its cell.

    ``scores`` is a ``{test_set_name: {metric_name: value}}`` mapping. ``headline``
    is the explicit ``(set_name, metric_name)`` pair `run_benchmark` reports as
    ``metric_score`` — the task states it outright; nothing is inferred from
    ordering or attributes. ``diagnostics`` holds non-scalar artifacts (curves,
    raw predictions under the reserved key ``"predictions"``, ...); it is kept in
    the result dict but never aggregated and is dropped from the results DataFrame.
    """

    scores: TestSetScores
    headline: tuple[str, str]  # (set_name, metric_name) reported as the leaderboard score
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (
            isinstance(self.headline, tuple)
            and len(self.headline) == 2
            and all(isinstance(part, str) for part in self.headline)
        ):
            raise ValueError(f"headline must be a (set_name, metric_name) tuple of strings, got {self.headline!r}")


def evaluate_per_test_set(
    spec: "BenchmarkSpec",  # Spec providing the named test sets and columns.
    model: Callable,  # Trained model `model(u, y_init, attrs) -> np.ndarray`.
    run_one: Callable[[Callable, Sequence], Any],  # Per-sequence runner `run_one(model, seq)`.
    metric: Callable[[np.ndarray, np.ndarray], float],  # Metric called as `func(y_pred, y_true)`.
    reduce: Callable[..., dict[str, float]] = aggregate_metric_score,  # Per-set reduction.
) -> TestSetScores:
    """Shared evaluation loop: run the model per sequence in every named test set, then reduce.

    ``run_one`` returns ONE result per sequence — a `(y_pred, y_true)` tuple for
    simulation, a list of per-window tuples for prediction. The results are passed to
    ``reduce`` without flattening; the per-sequence nesting is exactly what
    `aggregate_metric_score` keys on (list-of-lists ⇒ per-window-then-per-sequence,
    list-of-tuples ⇒ per-sequence).

    Returns:
        ``{test_set_name: {metric_name: value}}`` for every set the spec resolves.
    """
    out = {}
    for set_name, files in spec.test_set_files().items():
        results = [run_one(model, seq) for seq in _load_sequences_from_files(files, spec.u_cols, spec.y_cols)]
        out[set_name] = reduce(results, metric)
    return out


def pooled_scores_per_test_set(
    spec: "BenchmarkSpec",  # Spec providing the named test sets.
    file_errors: Callable[[Path], np.ndarray],  # Per-file 1-D array of error samples to pool.
    pool: Callable[[np.ndarray], dict[str, float]],  # Statistics over one pooled error array.
    all_set: str | None = None,  # Optional name for an extra cross-set pool (e.g. ``"all"``).
) -> TestSetScores:
    """Pooled (micro) evaluation loop: concatenate per-file error samples within every named test set.

    The counterpart of `evaluate_per_test_set` for tasks that score one pooled
    sample distribution per set instead of reducing per-sequence scores. A named
    set that yields zero error samples fails loudly — a silent gap in the scores
    would read as "not evaluated" rather than "broken data". With ``all_set``,
    an extra pool over every set's samples is appended under that name — a
    task-emitted set the spec does not name; a task makes such a pool the
    headline by naming it in its ``EvalResult.headline``.

    Returns:
        ``{test_set_name: {stat_name: value}}`` for every named set.
    """
    scores: TestSetScores = {}
    pooled: list[np.ndarray] = []
    for set_name, files in spec.test_set_files().items():
        chunks = [c for c in (file_errors(f) for f in files) if c.size]
        if not chunks:
            raise ValueError(f"{spec.name}: test set {set_name!r} produced no error samples from {len(files)} file(s)")
        errors = np.concatenate(chunks)
        scores[set_name] = pool(errors)
        pooled.append(errors)
    if all_set is not None:
        scores[all_set] = pool(np.concatenate(pooled))
    return scores


def _require_named_metric(metric: Callable) -> None:
    if not hasattr(metric, "__name__"):
        raise ValueError(
            "metric needs a stable __name__; use a named function instead of e.g. a bare functools.partial"
        )


class _MetricTask:
    """Shared shell of the built-in metric-driven tasks.

    Subclasses are dataclasses providing the ``metric`` field and a per-sequence
    ``_run_one(model, seq)``; the evaluation loop and the headline are identical:
    the spec's first named test set scored with the metric.
    """

    def __call__(self, spec: "BenchmarkSpec", model: Callable) -> EvalResult:
        scores = evaluate_per_test_set(spec, model, self._run_one, self.metric)
        return EvalResult(scores=scores, headline=(next(iter(spec.test_sets)), self.metric.__name__))


@dataclass(frozen=True)
class Simulation(_MetricTask):
    """Free-run simulation task: feed full ``u`` plus the first ``init_window`` samples
    of ``y``, score the prediction tail against the remaining ``y``.

    ``init_window=0`` is a valid free-run setting — the model receives an *empty*
    ``y_init`` (shape ``(0, n_y)``) and must not index into it.
    """

    metric: Callable[[np.ndarray, np.ndarray], float]  # The scoring function — part of the evaluation.
    init_window: int = 0  # Warm-up samples of `y` fed to the model and excluded from scoring.

    def __post_init__(self):
        if self.init_window < 0:
            raise ValueError("init_window must be >= 0")
        _require_named_metric(self.metric)

    def _run_one(self, model: Callable, seq: Sequence) -> tuple[np.ndarray, np.ndarray]:
        y_true = seq.y[self.init_window :]
        if y_true.shape[0] == 0:
            raise ValueError(
                f"test sequence has {seq.y.shape[0]} samples, not longer than init_window={self.init_window}"
            )
        y_pred = model(seq.u, seq.y[: self.init_window], seq.attrs)
        return (y_pred[-y_true.shape[0] :], y_true)


@dataclass(frozen=True)
class Prediction(_MetricTask):
    """Sliding-window k-step-ahead prediction task: feed ``init_window`` history plus
    ``horizon`` future inputs per window, score every window."""

    horizon: int  # The 'k' in k-step ahead prediction.
    step: int  # Stride between window starts.
    metric: Callable[[np.ndarray, np.ndarray], float]  # The scoring function — part of the evaluation.
    init_window: int = 0  # Warm-up samples of `y` fed to the model per window.

    def __post_init__(self):
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.init_window < 0:
            raise ValueError("init_window must be >= 0")
        _require_named_metric(self.metric)

    def _run_one(self, model: Callable, seq: Sequence) -> list[tuple[np.ndarray, np.ndarray]]:
        win = self.init_window + self.horizon
        if seq.u.shape[0] < win:
            raise ValueError(f"test sequence has {seq.u.shape[0]} samples, shorter than init_window + horizon = {win}")
        window_results = []
        for i in range(0, seq.u.shape[0] - win + 1, self.step):
            u_win = seq.u[i : i + win]
            y_win = seq.y[i : i + win]
            y_pred = model(u_win, y_win[: self.init_window], seq.attrs)
            window_results.append((y_pred, y_win))
        return window_results


@dataclass(frozen=True)
class WindowedEstimation:
    """Windowed signal-estimation task under a single standardized protocol.

    Each test sequence is tiled into **non-overlapping** windows of ``window_sec``
    seconds — converted to samples per file via that file's ``attrs['fs']``, so
    datasets with per-file rates (e.g. the IAS planetary gearbox) are handled
    correctly. The model is called on each window in isolation and produces **one
    estimate per window**: both the model's window output and the window's target
    are reduced to their mean. The per-window absolute errors are pooled (micro)
    across all windows and files; the headline is the pooled MAE on the spec's
    first named test set, with ``medae`` / ``std`` / ``max`` of the same pool
    reported alongside.

    This differs from :class:`Simulation` (dense free-run, scored per sample over
    the full sequence): here the model sees only one window of input at a time,
    which both bounds its context to ``window_sec`` and fixes the evaluation
    granularity to whole windows rather than individual samples.

    Relation to the upstream IAS benchmark this task standardizes: it is *not* a
    drop-in reproduction of the upstream results table. Upstream scored each model
    with its own test pipeline — a per-model HPO window size, a model-specific step
    (non-overlapping for the GRU reference, ~0.1 s overlapping for the FFT/TCN
    models), and a model-specific target granularity (window-mean for most, dense
    with a transient skip for the Many2Many variants). This task fixes one window
    per dataset, non-overlapping, with a window-mean target for every model — a
    fairer apples-to-apples comparison, but it will not match a specific upstream
    number unless ``window_sec`` is set to that model's tuned window (and even then
    only for the non-overlapping ``mean_IAS`` model family).

    Model contract: ``model(u_window, y_init, attrs) -> np.ndarray`` — the same
    ``(u, y_init, attrs)`` signature as the other tasks, called once per window with
    an empty ``y_init`` (shape ``(0, n_y)``); the output is mean-reduced, so a dense
    (one-value-per-sample) or a one-value-per-window model both work.
    """

    window_sec: float  # Window length in seconds; per-file fs converts it to samples.

    def __post_init__(self):
        if self.window_sec <= 0:
            raise ValueError("window_sec must be > 0")

    def _pool_stats(self, errors: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(np.mean(errors)),
            "medae": float(np.median(errors)),
            "std": float(np.std(errors)),
            "max": float(np.max(errors)),
        }

    def __call__(self, spec: "BenchmarkSpec", model: Callable) -> EvalResult:
        def file_errors(fpath: Path) -> np.ndarray:
            chunks = [
                self._window_errors(model, seq) for seq in _load_sequences_from_files([fpath], spec.u_cols, spec.y_cols)
            ]
            chunks = [c for c in chunks if c.size]
            return np.concatenate(chunks) if chunks else np.empty(0)

        scores = pooled_scores_per_test_set(spec, file_errors, self._pool_stats)
        return EvalResult(scores=scores, headline=(next(iter(spec.test_sets)), "mae"))

    def _window_errors(self, model: Callable, seq: Sequence) -> np.ndarray:
        """Per-window |mean(prediction) − mean(target)| over non-overlapping windows."""
        fs = float(seq.attrs["fs"])
        win = int(round(self.window_sec * fs))
        if win <= 0:
            raise ValueError(f"window_sec={self.window_sec!r} rounds to <1 sample at fs={fs}")
        y_init = seq.y[:0]  # empty (0, n_y) warm-up — estimation uses no output history
        preds, trues = [], []
        for i in range(0, seq.u.shape[0] - win + 1, win):
            y_pred = np.asarray(model(seq.u[i : i + win], y_init, seq.attrs), dtype=float)
            preds.append(np.mean(y_pred))
            trues.append(np.mean(seq.y[i : i + win]))
        if not preds:
            return np.empty(0)
        return np.abs(np.asarray(preds) - np.asarray(trues))


def _validate_patterns(spec_name: str, label: str, patterns: Patterns) -> None:
    if not isinstance(patterns, list):
        raise ValueError(f"{spec_name}: {label} must be a list of (Dataset, glob) tuples, got {patterns!r}")
    for entry in patterns:
        if not (
            isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], Dataset) and isinstance(entry[1], str)
        ):
            raise ValueError(f"{spec_name}: {label} entries must be (Dataset, glob) tuples, got {entry!r}")


@dataclass
class BenchmarkSpec:
    """One uniform benchmark specification: how files of datasets are used and evaluated.

    The strict two-level model: a :class:`~identibench.dataset.Dataset` only
    downloads and prepares files; the benchmark defines everything else — which
    files play which role (via explicit glob patterns), the column binding, and
    the task. The spec carries zero evaluation parameters — the whole evaluation
    (loop, metric, reduction, headline) lives on ``task``, any callable
    ``(spec, model) -> EvalResult``. The built-in tasks :class:`Simulation` and
    :class:`Prediction` are frozen dataclasses, so their parameters are readable
    as ``spec.task.init_window``, ``spec.task.horizon``, ... or generically via
    ``dataclasses.asdict(spec.task)``.

    File selection is a list of ``(dataset, glob)`` pairs, each glob relative to
    its dataset's directory. Resolution is strict: every pattern must match at
    least one file (`resolve`), so typos and stale caches fail loudly. ``train``
    and ``valid`` may be empty lists — an explicit "this benchmark defines no
    such data". ``test_sets`` names every evaluated test set with its own
    patterns; the built-in tasks make the first named set the headline.
    ``train_valid`` optionally selects unsplit estimation records (e.g. for
    models doing their own cross-validation split); ``None`` means the benchmark
    defines none — there is no implicit train+valid union.
    """

    name: str  # Unique name identifying this benchmark task.
    u_cols: list[str]  # Column names for input signals (u).
    y_cols: list[str]  # Column names for output signals (y).
    train: Patterns  # Training files; [] = this benchmark defines no training data.
    valid: Patterns  # Validation files; [] = this benchmark defines no validation data.
    test_sets: dict[str, Patterns]  # Named test sets; built-in tasks headline the first one.
    task: Callable[["BenchmarkSpec", Callable], EvalResult]  # Owns the evaluation: metric + params + loop.
    train_valid: Patterns | None = None  # Unsplit estimation records; None = the benchmark defines none.

    def __post_init__(self):
        if not self.test_sets or not all(isinstance(k, str) for k in self.test_sets):
            raise ValueError(f"{self.name}: test_sets must name at least one test set")
        _validate_patterns(self.name, "train", self.train)
        _validate_patterns(self.name, "valid", self.valid)
        if self.train_valid is not None:
            _validate_patterns(self.name, "train_valid", self.train_valid)
        for set_name, patterns in self.test_sets.items():
            _validate_patterns(self.name, f"test_sets[{set_name!r}]", patterns)
            if not patterns:
                raise ValueError(f"{self.name}: test set {set_name!r} has no patterns")

    @property
    def datasets(self) -> list[Dataset]:
        """Every dataset the spec's patterns draw files from, sorted by id."""
        groups = [self.train, self.valid, self.train_valid or [], *self.test_sets.values()]
        return sorted({ds for group in groups for ds, _ in group}, key=lambda d: d.dataset_id)

    def resolve(self, patterns: Patterns) -> list[Path]:
        """Resolves ``(dataset, glob)`` patterns to a sorted, deduplicated file list.

        Every pattern must match at least one file — a typo or a stale cache
        fails loudly instead of silently shrinking the selection. Non-file
        matches (directories) are ignored.

        Raises:
            FileNotFoundError: If a pattern matches no file.
        """
        files: set[Path] = set()
        for ds, pattern in patterns:
            matches = [m for m in ds.path.glob(pattern) if m.is_file()]
            if not matches:
                raise FileNotFoundError(
                    f"{self.name}: pattern {ds.dataset_id}/{pattern} matched no files under {ds.path} — "
                    f"run ensure_datasets_exist(force=True)?"
                )
            files.update(matches)
        return sorted(files)

    def test_set_files(self) -> dict[str, list[Path]]:
        """Resolves the named test sets to file lists, in declared order."""
        return {name: self.resolve(patterns) for name, patterns in self.test_sets.items()}

    def test_files(self) -> list[Path]:
        """Resolves "the test data" for file-level consumers: the sorted,
        deduplicated union of every named test set's files."""
        return sorted({f for files in self.test_set_files().values() for f in files})

    def ensure_datasets_exist(self, force: bool = False) -> None:
        """Prepares every dataset the spec draws from (see `Dataset.ensure_exists`)."""
        for ds in self.datasets:
            ds.ensure_exists(force)


class TrainingContext:
    """
    Context object passed to the user's training function (`build_model`).

    Holds the benchmark specification, hyperparameters, and seed.
    Provides methods to access the raw, full-length training and validation data sequences.
    Windowing/batching for training must be handled within the user's `build_model` function.

    Model contract:
        `build_model(context)` must return a callable `model(u, y_init, attrs) -> np.ndarray`.
        Given the input signal `u` (shape `(T, len(spec.u_cols))`), an initial-condition
        slice of the output `y_init` (empty when the task's `init_window` is 0), and the
        sequence's `attrs` dict, the model must return predictions of shape
        `(len(u), len(spec.y_cols))`. The evaluation loop aligns the prediction tail with
        the target, so returning predictions for the full `u` is expected.
    """

    def __init__(
        self,
        spec: BenchmarkSpec,  # The benchmark specification.
        hyperparameters: dict[str, Any],  # User-provided dictionary containing model and training hyperparameters.
        seed: int | None = None,  # Optional random seed for reproducibility.
    ):
        self.spec = spec
        self.hyperparameters = hyperparameters
        self.seed = seed

    # --- Data Access Methods ---

    def get_train_sequences(self) -> Iterator[Sequence]:
        """Returns a lazy iterator yielding Sequence objects for the training data."""
        if not self.spec.train:
            raise ValueError(f"{self.spec.name} defines no training data (train=[])")
        return _load_sequences_from_files(
            file_paths=self.spec.resolve(self.spec.train),
            u_cols=self.spec.u_cols,
            y_cols=self.spec.y_cols,
        )

    def get_valid_sequences(self) -> Iterator[Sequence]:
        """Returns a lazy iterator yielding Sequence objects for the validation data."""
        if not self.spec.valid:
            raise ValueError(f"{self.spec.name} defines no validation data (valid=[])")
        return _load_sequences_from_files(
            file_paths=self.spec.resolve(self.spec.valid),
            u_cols=self.spec.u_cols,
            y_cols=self.spec.y_cols,
        )

    def get_train_valid_sequences(self) -> Iterator[Sequence]:
        """Returns a lazy iterator yielding Sequence objects for the unsplit estimation records."""
        if self.spec.train_valid is None:
            raise ValueError(f"{self.spec.name} defines no train_valid data (train_valid=None)")
        return _load_sequences_from_files(
            file_paths=self.spec.resolve(self.spec.train_valid),
            u_cols=self.spec.u_cols,
            y_cols=self.spec.y_cols,
        )

    def get_test_sequences(self) -> Iterator[Sequence]:
        """Returns a lazy iterator yielding Sequence objects for every named test set."""
        return _load_sequences_from_files(
            file_paths=self.spec.test_files(),
            u_cols=self.spec.u_cols,
            y_cols=self.spec.y_cols,
        )


def _validate_test_set_scores(scores: Any) -> None:
    """Cheap shape check on a task's returned scores → leaderboard uniformity."""
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"task must return a non-empty {{test_set: {{metric: value}}}} mapping, got {scores!r}")
    for set_name, metric_scores in scores.items():
        if not isinstance(set_name, str) or not isinstance(metric_scores, dict) or not metric_scores:
            raise ValueError(f"invalid test-set entry {set_name!r}: {metric_scores!r}")
        for metric_name, value in metric_scores.items():
            if not isinstance(metric_name, str) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"invalid score {set_name!r}/{metric_name!r}: {value!r}")


def run_benchmark(
    spec: BenchmarkSpec,  # The benchmark specification to run.
    build_model: Callable[[TrainingContext], Callable],  # Builds the trained model from a TrainingContext.
    hyperparameters: dict[str, Any] | None = None,  # Model/training hyperparameters; defaults to empty.
    seed: int | None = None,  # Random seed; a random one is drawn when None.
) -> dict[str, Any]:
    """Trains and evaluates a single model against one benchmark specification.

    `build_model(context)` receives a `TrainingContext` and must return a callable
    `model(u, y_init, attrs) -> np.ndarray` producing predictions of shape
    `(len(u), len(spec.y_cols))` (see `TrainingContext` for the full model contract).
    Evaluation is one call: `spec.task(spec, model)`, which must return an
    `EvalResult` naming its own headline `(set, metric)` cell.

    Args:
        spec: The benchmark specification to run.
        build_model: User function that builds the trained model from a `TrainingContext`.
        hyperparameters: Model/training hyperparameters passed through the context.
            Defaults to an empty dict when None.
        seed: Random seed for reproducibility; a random seed is drawn when None.

    Returns:
        A result dict with the following keys:
            - `benchmark_name` (str): `spec.name`.
            - `datasets` (list[str]): The ids of every dataset the spec draws from.
            - `hyperparameters` (dict): The hyperparameters used.
            - `seed` (int): The seed used.
            - `training_time_seconds` (float): Wall-clock time spent in `build_model`.
            - `test_time_seconds` (float): Wall-clock time spent in the task.
            - `benchmark_type` (str): The task's name ("Simulation", "Prediction", ...).
            - `metric_name` (str): The headline metric named by the task.
            - `metric_score` (float): The headline `(set, metric)` cell's value.
            - `test_sets` (dict): The full `{test_set: {metric: value}}` scores.
            - `diagnostics` (dict): Non-scalar artifacts returned by the task.
    """
    hyperparameters = hyperparameters or {}

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    results = {
        "benchmark_name": spec.name,
        "datasets": [ds.dataset_id for ds in spec.datasets],
        "hyperparameters": hyperparameters,
        "seed": seed,
        "training_time_seconds": np.nan,
        "test_time_seconds": np.nan,
        "benchmark_type": getattr(spec.task, "__name__", type(spec.task).__name__),
        "metric_name": "",
        "metric_score": np.nan,
        "test_sets": {},
        "diagnostics": {},
    }

    spec.ensure_datasets_exist()

    context = TrainingContext(spec=spec, hyperparameters=hyperparameters, seed=seed)

    train_start_time = time.monotonic()
    model = build_model(context)
    train_end_time = time.monotonic()
    results["training_time_seconds"] = train_end_time - train_start_time

    if model is None:
        raise RuntimeError(f"build_model for {spec.name} did not return a model.")

    test_start_time = time.monotonic()
    ev = spec.task(spec, model)
    test_end_time = time.monotonic()
    results["test_time_seconds"] = test_end_time - test_start_time

    if not isinstance(ev, EvalResult):
        raise TypeError(f"{spec.name}: task must return an EvalResult, got {type(ev).__name__}")
    _validate_test_set_scores(ev.scores)

    headline_set, headline_metric = ev.headline
    if headline_set not in ev.scores:
        raise ValueError(f"headline set {headline_set!r} not in task scores {list(ev.scores)}")
    if headline_metric not in ev.scores[headline_set]:
        raise ValueError(
            f"headline metric {headline_metric!r} not in {headline_set!r} scores {list(ev.scores[headline_set])}"
        )

    results["metric_name"] = headline_metric
    results["metric_score"] = ev.scores[headline_set][headline_metric]
    results["test_sets"] = ev.scores
    results["diagnostics"] = ev.diagnostics

    return results


def benchmark_results_to_dataframe(
    results_list: list[dict[str, Any]],  # List of benchmark result dictionaries from `run_benchmark`.
) -> pd.DataFrame:
    """Transforms a list of benchmark result dictionaries into a pandas DataFrame.

    The nested `test_sets` scores are flattened into `test_sets.<set>.<metric>`
    columns; the non-scalar `diagnostics` are dropped.
    """
    if not results_list:
        return pd.DataFrame()

    df = pd.DataFrame(results_list)

    # Flatten the nested test-set scores into scalar columns.
    if "test_sets" in df.columns and df["test_sets"].apply(isinstance, args=(dict,)).any():
        test_sets_filled = df["test_sets"].apply(lambda x: x if isinstance(x, dict) else {})
        test_sets_df = pd.json_normalize(test_sets_filled).add_prefix("test_sets.")
        df = pd.concat([df.drop(columns=["test_sets"]), test_sets_df], axis=1)

    # Drop the 'diagnostics' column as it holds non-scalar artifacts unsuitable for a summary DataFrame.
    if "diagnostics" in df.columns:
        df = df.drop(columns=["diagnostics"])

    return df


def run_benchmarks(
    specs: list[BenchmarkSpec] | dict[str, BenchmarkSpec],  # Collection of specs to run.
    build_model: Callable[[TrainingContext], Callable],  # User function to build the model/predictor.
    hyperparameters: dict[str, Any]
    | list[dict[str, Any]]
    | None = None,  # Single dict, list of dicts (matching specs), or None.
    n_times: int = 1,  # Number of times to repeat each benchmark specification.
    continue_on_error: bool = True,  # If True, continue running benchmarks even if one fails.
    return_dataframe: bool = True,  # If True, return results as a pandas DataFrame, otherwise return a list of dicts.
) -> pd.DataFrame | list[dict[str, Any]]:
    """
    Runs multiple benchmarks sequentially, with repetitions and flexible hyperparameters.

    Returns either a pandas DataFrame summarizing the results (default)
    or a list of raw result dictionaries.
    """
    results_list = []
    spec_objects = list(specs.values()) if isinstance(specs, dict) else list(specs)
    num_specs = len(spec_objects)

    # Validate hyperparameters input
    if isinstance(hyperparameters, list):
        if len(hyperparameters) != num_specs:
            raise ValueError(
                f"If hyperparameters is a list, its length ({len(hyperparameters)}) must match the number of specs ({num_specs})."
            )
        get_hps = lambda i: hyperparameters[i]  # Function to get hp based on spec index
    else:
        # If None or a single dict, use the same for all. Ensure it's a dict or None.
        hps_single = hyperparameters or {}
        get_hps = lambda i: hps_single  # Function always returns the same hp dict

    print(f"--- Starting benchmark run for {num_specs} specifications, repeating each {n_times} times ---")

    total_runs = num_specs * n_times
    current_run = 0

    for repetition in range(n_times):
        print(f"\n-- Repetition {repetition + 1}/{n_times} --")
        for i, spec in enumerate(spec_objects):
            current_run += 1
            spec_name = getattr(spec, "name", f"Unnamed Spec {i + 1}")
            print(f"\n[{current_run}/{total_runs}] Running: {spec_name} (Rep {repetition + 1})")

            current_hyperparameters = get_hps(i)

            try:
                result = run_benchmark(spec=spec, build_model=build_model, hyperparameters=current_hyperparameters)
                results_list.append(result)
                print(f"  -> Success: {spec_name} (Rep {repetition + 1}) completed.")

            except Exception as e:
                print(f"  -> ERROR running benchmark '{spec_name}' (Rep {repetition + 1}): {e}")
                if not continue_on_error:
                    print("Stopping due to error (continue_on_error=False).")
                    raise

    print(f"\n--- Benchmark run finished. {len(results_list)}/{total_runs} individual runs completed successfully. ---")

    if return_dataframe:
        return benchmark_results_to_dataframe(results_list)
    else:
        return results_list


def aggregate_benchmark_results(
    results_df: pd.DataFrame,  # DataFrame returned by run_benchmarks (with return_dataframe=True).
    group_by_cols: str | list[str] = "benchmark_name",  # Column(s) to group by before aggregation.
    agg_funcs: str | list[str] = "mean",  # Aggregation function(s) ('mean', 'median', 'std', etc.) or list thereof.
) -> pd.DataFrame:
    """
    Aggregates numeric results from a benchmark DataFrame, grouped by specified columns.
    """
    if results_df.empty:
        return pd.DataFrame()  # Return empty if input is empty

    # Identify numeric columns suitable for aggregation
    numeric_cols = results_df.select_dtypes(include=np.number).columns.tolist()

    # Exclude columns that are typically identifiers or settings, even if numeric
    cols_to_exclude = ["seed", "repetition"]
    agg_cols = [
        col
        for col in numeric_cols
        if col not in cols_to_exclude
        and col not in (group_by_cols if isinstance(group_by_cols, list) else [group_by_cols])
    ]

    if not agg_cols:
        print("Warning: No numeric columns found to aggregate (excluding identifiers). Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        # Perform groupby and aggregation
        aggregated_df = results_df.groupby(group_by_cols)[agg_cols].agg(agg_funcs)
    except Exception as e:
        print(f"Error during aggregation: {e}")
        # Provide more context if grouping fails
        if isinstance(group_by_cols, list):
            missing_cols = [col for col in group_by_cols if col not in results_df.columns]
        else:
            missing_cols = [group_by_cols] if group_by_cols not in results_df.columns else []

        if missing_cols:
            print(f"  -> Grouping columns not found in DataFrame: {missing_cols}")
        print(f"  -> Available columns: {results_df.columns.tolist()}")
        print(f"  -> Columns selected for aggregation: {agg_cols}")
        return pd.DataFrame()  # Return empty on error

    return aggregated_df
