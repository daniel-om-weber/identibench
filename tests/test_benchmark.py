"""Tests for the core benchmark pipeline."""

import h5py
import numpy as np
import pandas as pd
import pytest

from identibench.benchmark import (
    BenchmarkSpec,
    EvalResult,
    Prediction,
    Simulation,
    TrainingContext,
    aggregate_benchmark_results,
    aggregate_metric_score,
    benchmark_results_to_dataframe,
    evaluate_per_test_set,
    run_benchmark,
    run_benchmarks,
)
from identibench.dataset import Dataset
from identibench.metrics import rmse
from identibench.utils import _dummy_dataset_loader, _load_sequences_from_files


# --- Fixtures ---


@pytest.fixture
def dummy_dataset(tmp_path, monkeypatch):
    """A prepared dummy dataset under a test-local data root."""
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _dummy_dataset_loader(tmp_path / "dummy")
    (tmp_path / "dummy" / ".prepared").write_text("1")
    return Dataset("dummy", prepare=_dummy_dataset_loader)


@pytest.fixture
def sim_spec(dummy_dataset):
    return BenchmarkSpec(
        name="TestSim",
        u_cols=["u0", "u1"],
        y_cols=["y0"],
        train=[(dummy_dataset, "train/*.hdf5")],
        valid=[(dummy_dataset, "valid/*.hdf5")],
        test_sets={"test": [(dummy_dataset, "test/*.hdf5")]},
        task=Simulation(metric=rmse, init_window=5),
    )


@pytest.fixture
def pred_spec(dummy_dataset):
    return BenchmarkSpec(
        name="TestPred",
        u_cols=["u0", "u1"],
        y_cols=["y0"],
        train=[(dummy_dataset, "train/*.hdf5")],
        valid=[(dummy_dataset, "valid/*.hdf5")],
        test_sets={"test": [(dummy_dataset, "test/*.hdf5")]},
        task=Prediction(horizon=10, step=10, metric=rmse, init_window=5),
    )


def dummy_build_model(context):
    output_dim = len(context.spec.y_cols)

    def model(u, y, attrs):
        return np.zeros((u.shape[0], output_dim))

    return model


def _write_flat_file(path, n=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("u0", data=np.random.rand(n).astype(np.float32))
        f.create_dataset("y0", data=np.random.rand(n).astype(np.float32))


@pytest.fixture
def flat_dataset(tmp_path, monkeypatch):
    """A flat user-managed dataset with four files c/d/e/f under sub/."""
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    for name in ["c.hdf5", "d.hdf5"]:
        _write_flat_file(tmp_path / "flat" / name)
    for name in ["e.hdf5", "f.hdf5"]:
        _write_flat_file(tmp_path / "flat" / "sub" / name)
    return Dataset("flat", prepare=None)


# --- Dataset ---


class TestDataset:
    def test_path_under_data_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        assert Dataset("abc", prepare=None).path == tmp_path / "abc"

    @pytest.mark.parametrize("bad_id", ["", ".", "..", "a/b"])
    def test_invalid_dataset_id_rejected(self, bad_id):
        with pytest.raises(ValueError, match="single path segment"):
            Dataset(bad_id, prepare=None)

    def test_user_managed_missing_dir_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="user-managed"):
            Dataset("nope", prepare=None).ensure_exists()

    def test_user_managed_existing_dir_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        (tmp_path / "mine").mkdir()
        Dataset("mine", prepare=None).ensure_exists()  # no error, no sentinel needed

    def test_matching_sentinel_skips_preparation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        ds_dir = tmp_path / "cached"
        ds_dir.mkdir()
        (ds_dir / ".prepared").write_text("1")
        marker = ds_dir / "untouched.txt"
        marker.write_text("data")

        def would_fail(path, force):
            raise RuntimeError("must not run")

        Dataset("cached", prepare=would_fail).ensure_exists()
        assert marker.read_text() == "data"

    def test_version_mismatch_reprepares(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        ds_dir = tmp_path / "dummy"
        _dummy_dataset_loader(ds_dir)
        (ds_dir / ".prepared").write_text("0")  # stale format version
        stray = ds_dir / "stray.txt"
        stray.write_text("old")

        Dataset("dummy", prepare=_dummy_dataset_loader, version="1").ensure_exists()

        assert not stray.exists()  # clean-slate re-preparation
        assert (ds_dir / ".prepared").read_text() == "1"
        assert (ds_dir / "train").is_dir()


# --- Task construction ---


class TestTaskConstruction:
    def test_simulation_defaults(self):
        task = Simulation(metric=rmse)
        assert task.init_window == 0

    def test_prediction_params(self):
        task = Prediction(horizon=10, step=5, metric=rmse, init_window=3)
        assert (task.horizon, task.step, task.init_window) == (10, 5, 3)

    def test_prediction_rejects_zero_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            Prediction(horizon=0, step=10, metric=rmse)

    def test_prediction_rejects_negative_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            Prediction(horizon=-1, step=10, metric=rmse)

    def test_prediction_rejects_non_positive_step(self):
        with pytest.raises(ValueError, match="step"):
            Prediction(horizon=10, step=0, metric=rmse)
        with pytest.raises(ValueError, match="step"):
            Prediction(horizon=10, step=-1, metric=rmse)

    def test_negative_init_window_rejected(self):
        with pytest.raises(ValueError, match="init_window"):
            Simulation(metric=rmse, init_window=-1)

    def test_metric_needs_name(self):
        class Nameless:
            def __call__(self, inp, targ):
                return 0.0

        with pytest.raises(ValueError, match="__name__"):
            Simulation(metric=Nameless())

    def test_tasks_are_frozen(self):
        task = Simulation(metric=rmse)
        with pytest.raises(AttributeError):
            task.init_window = 3


# --- EvalResult ---


class TestEvalResult:
    def test_headline_must_be_string_pair(self):
        with pytest.raises(ValueError, match="headline"):
            EvalResult(scores={"test": {"rmse": 1.0}}, headline="test")
        with pytest.raises(ValueError, match="headline"):
            EvalResult(scores={"test": {"rmse": 1.0}}, headline=("test",))

    def test_valid_headline(self):
        ev = EvalResult(scores={"test": {"rmse": 1.0}}, headline=("test", "rmse"))
        assert ev.headline == ("test", "rmse")
        assert ev.diagnostics == {}


# --- BenchmarkSpec construction & file resolution ---


class TestBenchmarkSpecResolver:
    def test_resolve_sorted_and_deduplicated(self, flat_dataset):
        spec = BenchmarkSpec(
            name="TestFlat",
            u_cols=["u0"],
            y_cols=["y0"],
            train=[],
            valid=[],
            test_sets={"s": [(flat_dataset, "*.hdf5"), (flat_dataset, "c.hdf5")]},
            task=Simulation(metric=rmse),
        )
        files = spec.resolve(spec.test_sets["s"])
        assert files == [flat_dataset.path / "c.hdf5", flat_dataset.path / "d.hdf5"]

    def test_resolve_zero_match_raises(self, flat_dataset):
        spec = BenchmarkSpec(
            name="TestMissing",
            u_cols=["u0"],
            y_cols=["y0"],
            train=[],
            valid=[],
            test_sets={"s": [(flat_dataset, "does_not_exist*.hdf5")]},
            task=Simulation(metric=rmse),
        )
        with pytest.raises(FileNotFoundError, match="matched no files"):
            spec.test_set_files()

    def test_resolve_ignores_directories(self, flat_dataset):
        spec = BenchmarkSpec(
            name="TestDirs",
            u_cols=["u0"],
            y_cols=["y0"],
            train=[],
            valid=[],
            # "sub" matches only the directory -> zero files -> loud failure.
            test_sets={"s": [(flat_dataset, "sub")]},
            task=Simulation(metric=rmse),
        )
        with pytest.raises(FileNotFoundError, match="matched no files"):
            spec.test_set_files()

    def test_resolve_recursive_pattern(self, flat_dataset):
        spec = BenchmarkSpec(
            name="TestSub",
            u_cols=["u0"],
            y_cols=["y0"],
            train=[],
            valid=[],
            test_sets={"s": [(flat_dataset, "sub/*.hdf5")]},
            task=Simulation(metric=rmse),
        )
        assert len(spec.test_set_files()["s"]) == 2

    def test_test_set_files_declared_order(self, flat_dataset):
        spec = BenchmarkSpec(
            name="TestOrder",
            u_cols=["u0"],
            y_cols=["y0"],
            train=[],
            valid=[],
            test_sets={
                "s2": [(flat_dataset, "d.hdf5")],
                "s1": [(flat_dataset, "c.hdf5")],
            },
            task=Simulation(metric=rmse),
        )
        assert list(spec.test_set_files()) == ["s2", "s1"]

    def test_test_files_union(self, flat_dataset):
        spec = BenchmarkSpec(
            name="TestUnion",
            u_cols=["u0"],
            y_cols=["y0"],
            train=[],
            valid=[],
            test_sets={
                "s1": [(flat_dataset, "c.hdf5")],
                "s2": [(flat_dataset, "c.hdf5"), (flat_dataset, "d.hdf5")],
            },
            task=Simulation(metric=rmse),
        )
        assert spec.test_files() == [flat_dataset.path / "c.hdf5", flat_dataset.path / "d.hdf5"]

    def test_datasets_derived_from_patterns(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        a = Dataset("a", prepare=None)
        b = Dataset("b", prepare=None)
        c = Dataset("c", prepare=None)
        spec = BenchmarkSpec(
            name="TestMulti",
            u_cols=["u0"],
            y_cols=["y0"],
            train=[(b, "*.hdf5")],
            valid=[],
            train_valid=[(c, "*.hdf5")],
            test_sets={"s": [(a, "*.hdf5"), (b, "x.hdf5")]},
            task=Simulation(metric=rmse),
        )
        assert [ds.dataset_id for ds in spec.datasets] == ["a", "b", "c"]

    def test_empty_test_sets_rejected(self):
        with pytest.raises(ValueError, match="test_sets"):
            BenchmarkSpec(
                name="TestEmpty",
                u_cols=["u0"],
                y_cols=["y0"],
                train=[],
                valid=[],
                test_sets={},
                task=Simulation(metric=rmse),
            )

    def test_test_set_without_patterns_rejected(self):
        with pytest.raises(ValueError, match="no patterns"):
            BenchmarkSpec(
                name="TestNoPatterns",
                u_cols=["u0"],
                y_cols=["y0"],
                train=[],
                valid=[],
                test_sets={"s": []},
                task=Simulation(metric=rmse),
            )

    def test_string_patterns_rejected(self, flat_dataset):
        with pytest.raises(ValueError, match=r"\(Dataset, glob\)"):
            BenchmarkSpec(
                name="TestBadPattern",
                u_cols=["u0"],
                y_cols=["y0"],
                train=["flat/train/*.hdf5"],  # plain string instead of (Dataset, glob)
                valid=[],
                test_sets={"s": [(flat_dataset, "c.hdf5")]},
                task=Simulation(metric=rmse),
            )


# --- Data loading ---


class TestLoadSequences:
    def test_load_sequences_shape(self, sim_spec):
        files = sim_spec.resolve(sim_spec.train)
        sequences = list(_load_sequences_from_files(files, sim_spec.u_cols, sim_spec.y_cols))
        seq = sequences[0]
        assert seq.u.shape == (50, 2)
        assert seq.y.shape == (50, 1)

    def test_load_sequences_yields_all_files(self, sim_spec):
        files = sim_spec.resolve(sim_spec.train)
        sequences = list(_load_sequences_from_files(files, sim_spec.u_cols, sim_spec.y_cols))
        assert len(sequences) == 2

    def test_load_sequences_windowing(self, sim_spec):
        files = sim_spec.resolve(sim_spec.train)
        sequences = list(_load_sequences_from_files(files, sim_spec.u_cols, sim_spec.y_cols, win_sz=20, stp_sz=10))
        for seq in sequences:
            assert seq.u.shape == (20, 2)
            assert seq.y.shape == (20, 1)
        # 2 files, each 50 samples: windows at offsets 0,10,20,30 = 4 windows per file
        assert len(sequences) == 2 * 4

    def test_load_sequences_attrs(self, sim_spec):
        files = sim_spec.resolve(sim_spec.train)
        sequences = list(_load_sequences_from_files(files, sim_spec.u_cols, sim_spec.y_cols))
        assert "fs" in sequences[0].attrs
        assert sequences[0].attrs["fs"] == 10.0

    def test_load_sequences_unpacking(self, sim_spec):
        files = sim_spec.resolve(sim_spec.train)
        sequences = list(_load_sequences_from_files(files, sim_spec.u_cols, sim_spec.y_cols))
        u, y, attrs = sequences[0]
        assert u.shape == (50, 2)
        assert isinstance(attrs, dict)

    def test_load_sequences_empty_path(self):
        sequences = list(_load_sequences_from_files([], ["u0"], ["y0"]))
        assert len(sequences) == 0

    def test_load_sequences_unreadable_file_raises(self, tmp_path):
        bad = tmp_path / "bad.hdf5"
        bad.write_text("not hdf5")
        with pytest.raises(Exception):
            list(_load_sequences_from_files([bad], ["u0"], ["y0"]))

    def test_load_sequences_win_without_step_raises(self, sim_spec):
        files = sim_spec.resolve(sim_spec.train)
        with pytest.raises(ValueError):
            list(_load_sequences_from_files(files, sim_spec.u_cols, sim_spec.y_cols, win_sz=20))

    def test_load_sequences_step_without_win_raises(self, sim_spec):
        files = sim_spec.resolve(sim_spec.train)
        with pytest.raises(ValueError):
            list(_load_sequences_from_files(files, sim_spec.u_cols, sim_spec.y_cols, stp_sz=10))


# --- TrainingContext ---


class TestTrainingContext:
    def test_train_sequences(self, sim_spec):
        ctx = TrainingContext(spec=sim_spec, hyperparameters={})
        sequences = list(ctx.get_train_sequences())
        assert len(sequences) == 2
        assert sequences[0].u.shape == (50, 2)
        assert "fs" in sequences[0].attrs

    def test_valid_sequences(self, sim_spec):
        ctx = TrainingContext(spec=sim_spec, hyperparameters={})
        sequences = list(ctx.get_valid_sequences())
        assert len(sequences) == 2

    def test_empty_train_raises(self, sim_spec):
        import dataclasses

        spec = dataclasses.replace(sim_spec, train=[])
        ctx = TrainingContext(spec=spec, hyperparameters={})
        with pytest.raises(ValueError, match="no training data"):
            ctx.get_train_sequences()

    def test_train_valid_none_raises(self, sim_spec):
        ctx = TrainingContext(spec=sim_spec, hyperparameters={})
        with pytest.raises(ValueError, match="train_valid"):
            ctx.get_train_valid_sequences()

    def test_train_valid_sequences(self, sim_spec, dummy_dataset):
        import dataclasses

        spec = dataclasses.replace(
            sim_spec, train_valid=[(dummy_dataset, "train/*.hdf5"), (dummy_dataset, "valid/*.hdf5")]
        )
        ctx = TrainingContext(spec=spec, hyperparameters={})
        assert len(list(ctx.get_train_valid_sequences())) == 4

    def test_test_sequences(self, sim_spec):
        ctx = TrainingContext(spec=sim_spec, hyperparameters={})
        sequences = list(ctx.get_test_sequences())
        assert len(sequences) == 2
        assert sequences[0].u.shape == (50, 2)

    def test_test_sequences_span_all_named_sets(self, flat_dataset):
        spec = BenchmarkSpec(
            name="TestPool",
            u_cols=["u0"],
            y_cols=["y0"],
            train=[],
            valid=[],
            test_sets={
                "s1": [(flat_dataset, "c.hdf5")],
                "s2": [(flat_dataset, "d.hdf5")],
            },
            task=Simulation(metric=rmse),
        )
        ctx = TrainingContext(spec=spec, hyperparameters={})
        assert len(list(ctx.get_test_sequences())) == 2


# --- Simulation task ---


class TestSimulationTask:
    def test_scores_every_test_set(self, sim_spec):
        model = lambda u, y, attrs: np.zeros((u.shape[0], 1))
        result = sim_spec.task(sim_spec, model)
        assert isinstance(result, EvalResult)
        assert list(result.scores) == ["test"]
        assert result.headline == ("test", "rmse")
        assert np.isfinite(result.scores["test"]["rmse"])

    def test_init_window_trims_target(self, sim_spec):
        seen_shapes = []

        def model(u, y_init, attrs):
            seen_shapes.append(y_init.shape)
            return np.zeros((u.shape[0], 1))

        results = [
            sim_spec.task._run_one(model, seq)
            for seq in _load_sequences_from_files(sim_spec.test_set_files()["test"], sim_spec.u_cols, sim_spec.y_cols)
        ]
        for y_pred, y_true in results:
            # y_true has init_window removed: 50 - 5 = 45; y_pred is tail-aligned
            assert y_true.shape[0] == 45
            assert y_pred.shape[0] == 45
        assert all(s == (5, 1) for s in seen_shapes)

    def test_empty_y_init_when_zero_window(self, sim_spec):
        task = Simulation(metric=rmse, init_window=0)
        y_init_shapes = []

        def model(u, y_init, attrs):
            y_init_shapes.append(y_init.shape)
            return np.zeros((u.shape[0], 1))

        task(sim_spec, model)
        assert all(s == (0, 1) for s in y_init_shapes)

    def test_perfect_model(self, sim_spec):
        # Precompute the full y for each test sequence (in file order) so the
        # model can return the ground truth and drive the metric to ~0.
        test_files = sim_spec.test_set_files()["test"]
        full_ys = [seq.y for seq in _load_sequences_from_files(test_files, sim_spec.u_cols, sim_spec.y_cols)]
        next_y = iter(full_ys)

        def perfect_model(u, y, attrs):
            return next(next_y)

        result = sim_spec.task(sim_spec, perfect_model)
        np.testing.assert_allclose(result.scores["test"]["rmse"], 0.0, atol=1e-10)

    def test_sequence_not_longer_than_init_window_raises(self):
        from identibench.utils import Sequence

        task = Simulation(metric=rmse, init_window=10)
        seq = Sequence(u=np.zeros((10, 1)), y=np.zeros((10, 1)), attrs={})
        model = lambda u, y, attrs: np.zeros((u.shape[0], 1))
        with pytest.raises(ValueError, match="init_window"):
            task._run_one(model, seq)


# --- Prediction task ---


class TestPredictionTask:
    def test_scores_every_test_set(self, pred_spec):
        model = lambda u, y, attrs: np.zeros((u.shape[0], 1))
        result = pred_spec.task(pred_spec, model)
        assert isinstance(result, EvalResult)
        assert result.headline == ("test", "rmse")
        assert np.isfinite(result.scores["test"]["rmse"])

    def test_window_count_and_shapes(self, pred_spec):
        model = lambda u, y, attrs: np.zeros((u.shape[0], 1))
        task = pred_spec.task
        seq_len = 50
        # Windows start at every `step` while a full window still fits (bound inclusive).
        expected_windows = len(range(0, seq_len - task.init_window - task.horizon + 1, task.step))
        for seq in _load_sequences_from_files(pred_spec.test_set_files()["test"], pred_spec.u_cols, pred_spec.y_cols):
            window_results = task._run_one(model, seq)
            assert len(window_results) == expected_windows
            for y_pred, y_true in window_results:
                assert y_pred.shape[0] == task.init_window + task.horizon
                assert y_true.shape[0] == task.init_window + task.horizon

    def test_exactly_fitting_window_is_scored(self):
        # A sequence of exactly init_window + horizon samples holds one valid window.
        from identibench.utils import Sequence

        task = Prediction(horizon=8, step=3, metric=rmse, init_window=2)
        seq = Sequence(u=np.zeros((10, 1)), y=np.zeros((10, 1)), attrs={})
        model = lambda u, y, attrs: np.zeros((u.shape[0], 1))
        assert len(task._run_one(model, seq)) == 1

    def test_too_short_sequence_raises(self):
        from identibench.utils import Sequence

        task = Prediction(horizon=8, step=3, metric=rmse, init_window=2)
        seq = Sequence(u=np.zeros((9, 1)), y=np.zeros((9, 1)), attrs={})
        model = lambda u, y, attrs: np.zeros((u.shape[0], 1))
        with pytest.raises(ValueError, match="shorter than"):
            task._run_one(model, seq)


# --- evaluate_per_test_set ---


class TestEvaluatePerTestSet:
    def test_custom_reduce(self, sim_spec):
        def count_results(results, metric):
            return {"n_sequences": float(len(results))}

        model = lambda u, y, attrs: np.zeros((u.shape[0], 1))
        scores = evaluate_per_test_set(sim_spec, model, sim_spec.task._run_one, rmse, reduce=count_results)
        assert scores == {"test": {"n_sequences": 2.0}}


# --- run_benchmark end-to-end ---


class TestRunBenchmark:
    def test_simulation(self, sim_spec):
        result = run_benchmark(sim_spec, dummy_build_model)
        assert result["benchmark_name"] == "TestSim"
        assert result["benchmark_type"] == "Simulation"
        assert result["datasets"] == ["dummy"]
        assert np.isfinite(result["metric_score"])
        assert result["training_time_seconds"] >= 0
        assert result["test_time_seconds"] >= 0
        assert result["metric_name"] == "rmse"
        assert result["test_sets"] == {"test": {"rmse": pytest.approx(result["metric_score"])}}
        assert result["diagnostics"] == {}

    def test_prediction(self, pred_spec):
        result = run_benchmark(pred_spec, dummy_build_model)
        assert result["benchmark_name"] == "TestPred"
        assert result["benchmark_type"] == "Prediction"
        assert np.isfinite(result["metric_score"])

    def test_none_model_raises(self, sim_spec):
        def bad_build(context):
            return None

        with pytest.raises(RuntimeError, match="did not return a model"):
            run_benchmark(sim_spec, bad_build)

    def test_custom_task_with_explicit_headline(self, sim_spec):
        def my_eval(spec, model):
            return EvalResult(scores={"test": {"foo": 42.0, "bar": 1.0}}, headline=("test", "foo"))

        sim_spec.task = my_eval
        result = run_benchmark(sim_spec, dummy_build_model)
        assert result["benchmark_type"] == "my_eval"
        assert result["metric_name"] == "foo"
        assert result["metric_score"] == 42.0
        assert result["test_sets"]["test"]["bar"] == 1.0

    def test_custom_task_eval_result_with_diagnostics(self, sim_spec):
        def my_eval(spec, model):
            return EvalResult(
                scores={"test": {"foo": 1.0}},
                headline=("test", "foo"),
                diagnostics={"predictions": {"test": [np.zeros(3)]}},
            )

        sim_spec.task = my_eval
        result = run_benchmark(sim_spec, dummy_build_model)
        assert "predictions" in result["diagnostics"]

    def test_bare_dict_task_return_rejected(self, sim_spec):
        sim_spec.task = lambda spec, model: {"test": {"foo": 1.0}}
        with pytest.raises(TypeError, match="EvalResult"):
            run_benchmark(sim_spec, dummy_build_model)

    def test_headline_set_must_be_scored(self, sim_spec):
        def my_eval(spec, model):
            return EvalResult(scores={"test": {"foo": 1.0}}, headline=("does_not_exist", "foo"))

        sim_spec.task = my_eval
        with pytest.raises(ValueError, match="headline set"):
            run_benchmark(sim_spec, dummy_build_model)

    def test_headline_metric_must_be_scored(self, sim_spec):
        def my_eval(spec, model):
            return EvalResult(scores={"test": {"foo": 1.0}}, headline=("test", "does_not_exist"))

        sim_spec.task = my_eval
        with pytest.raises(ValueError, match="headline metric"):
            run_benchmark(sim_spec, dummy_build_model)

    def test_invalid_task_scores_rejected(self, sim_spec):
        sim_spec.task = lambda spec, model: EvalResult(scores={"test": 1.0}, headline=("test", "x"))  # not nested
        with pytest.raises(ValueError, match="test-set entry"):
            run_benchmark(sim_spec, dummy_build_model)

    def test_seed_is_echoed(self, sim_spec):
        result = run_benchmark(sim_spec, dummy_build_model, seed=12345)
        assert result["seed"] == 12345

    def test_seed_defaults_to_non_none(self, sim_spec):
        result = run_benchmark(sim_spec, dummy_build_model)
        assert result["seed"] is not None
        assert isinstance(result["seed"], int)


# --- run_benchmarks ---


class TestRunBenchmarks:
    def test_multiple_specs(self, sim_spec, pred_spec):
        df = run_benchmarks([sim_spec, pred_spec], dummy_build_model)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_repetitions(self, sim_spec):
        df = run_benchmarks([sim_spec], dummy_build_model, n_times=3)
        assert len(df) == 3

    def test_hyperparams_list(self, sim_spec, pred_spec):
        hps = [{"lr": 0.1}, {"lr": 0.01}]
        df = run_benchmarks([sim_spec, pred_spec], dummy_build_model, hyperparameters=hps)
        assert len(df) == 2

    def test_hyperparams_mismatch_raises(self, sim_spec, pred_spec):
        with pytest.raises(ValueError, match="length"):
            run_benchmarks([sim_spec, pred_spec], dummy_build_model, hyperparameters=[{"lr": 0.1}])

    def test_continue_on_error(self, sim_spec, pred_spec):
        call_count = [0]

        def flaky_build(context):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure")
            return dummy_build_model(context)

        df = run_benchmarks([sim_spec, pred_spec], flaky_build, continue_on_error=True)
        assert len(df) == 1  # Only the second spec succeeded

    def test_return_list(self, sim_spec):
        results = run_benchmarks([sim_spec], dummy_build_model, return_dataframe=False)
        assert isinstance(results, list)
        assert isinstance(results[0], dict)

    def test_dict_input(self, sim_spec, pred_spec):
        specs = {"sim": sim_spec, "pred": pred_spec}
        df = run_benchmarks(specs, dummy_build_model)
        assert len(df) == 2


# --- benchmark_results_to_dataframe ---


class TestResultsToDataframe:
    def test_basic(self):
        results = [
            {
                "benchmark_name": "A",
                "metric_score": 1.0,
                "test_sets": {},
                "diagnostics": {},
            }
        ]
        df = benchmark_results_to_dataframe(results)
        assert "benchmark_name" in df.columns
        assert "metric_score" in df.columns

    def test_drops_diagnostics(self):
        results = [
            {
                "benchmark_name": "A",
                "metric_score": 1.0,
                "test_sets": {},
                "diagnostics": {"predictions": [np.zeros(10)]},
            }
        ]
        df = benchmark_results_to_dataframe(results)
        assert "diagnostics" not in df.columns

    def test_flattens_test_sets(self):
        results = [
            {
                "benchmark_name": "A",
                "metric_score": 1.0,
                "test_sets": {"basic": {"mae": 5.0}, "wear": {"mae": 7.0}},
                "diagnostics": {},
            }
        ]
        df = benchmark_results_to_dataframe(results)
        assert df["test_sets.basic.mae"].iloc[0] == 5.0
        assert df["test_sets.wear.mae"].iloc[0] == 7.0

    def test_empty_list(self):
        df = benchmark_results_to_dataframe([])
        assert df.empty


# --- aggregate_benchmark_results ---


class TestAggregateBenchmarkResults:
    def test_aggregate_by_name(self):
        df = pd.DataFrame(
            {
                "benchmark_name": ["A", "A", "B"],
                "metric_score": [1.0, 3.0, 5.0],
                "training_time_seconds": [0.1, 0.2, 0.3],
            }
        )
        agg = aggregate_benchmark_results(df)
        assert agg.loc["A", "metric_score"] == 2.0  # mean(1, 3)
        assert agg.loc["B", "metric_score"] == 5.0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        agg = aggregate_benchmark_results(df)
        assert agg.empty

    def test_custom_agg_funcs(self):
        df = pd.DataFrame(
            {
                "benchmark_name": ["A", "A", "B"],
                "metric_score": [1.0, 3.0, 5.0],
            }
        )
        agg = aggregate_benchmark_results(df, agg_funcs=["mean", "std"])
        assert agg.loc["A", ("metric_score", "mean")] == 2.0
        np.testing.assert_allclose(agg.loc["A", ("metric_score", "std")], np.std([1.0, 3.0], ddof=1))

    def test_no_aggregatable_numeric_columns(self):
        # Only the group key and excluded identifier columns are numeric.
        df = pd.DataFrame(
            {
                "benchmark_name": ["A", "B"],
                "seed": [1, 2],
            }
        )
        agg = aggregate_benchmark_results(df)
        assert isinstance(agg, pd.DataFrame)
        assert agg.empty

    def test_group_by_missing_column_returns_empty(self):
        df = pd.DataFrame(
            {
                "benchmark_name": ["A", "B"],
                "metric_score": [1.0, 2.0],
            }
        )
        agg = aggregate_benchmark_results(df, group_by_cols="does_not_exist")
        assert isinstance(agg, pd.DataFrame)
        assert agg.empty


# --- aggregate_metric_score ---


class TestAggregateMetricScore:
    def test_simulation_results(self):
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [2.1], [3.1]])
        test_results = [(y_pred, y_true)]
        scores = aggregate_metric_score(test_results, rmse)
        assert "rmse" in scores
        np.testing.assert_allclose(scores["rmse"], 0.1, atol=1e-6)

    def test_prediction_results_nested(self):
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        # Prediction results are nested: list of lists
        test_results = [[(y_pred, y_true), (y_pred, y_true)]]
        scores = aggregate_metric_score(test_results, rmse)
        assert "rmse" in scores
        np.testing.assert_allclose(scores["rmse"], 0.0, atol=1e-10)

    def test_custom_score_name(self):
        y = np.array([[1.0]])
        test_results = [(y, y)]
        scores = aggregate_metric_score(test_results, rmse, score_name="my_metric")
        assert "my_metric" in scores

    def test_empty_results_returns_nan(self):
        scores = aggregate_metric_score([], rmse)
        assert "rmse" in scores
        assert np.isnan(scores["rmse"])

    def test_empty_results_respects_score_name(self):
        scores = aggregate_metric_score([], rmse, score_name="metric_score")
        assert set(scores) == {"metric_score"}
        assert np.isnan(scores["metric_score"])
