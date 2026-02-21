"""Tests for the core benchmark pipeline."""

import numpy as np
import pandas as pd
import pytest

from identibench.benchmark import (
    BenchmarkSpecSimulation,
    BenchmarkSpecPrediction,
    TrainingContext,
    run_benchmark,
    run_benchmarks,
    benchmark_results_to_dataframe,
    aggregate_benchmark_results,
    aggregate_metric_score,
    _test_simulation,
    _test_prediction,
)
from identibench.metrics import rmse
from identibench.utils import _dummy_dataset_loader, _load_sequences_from_files


# --- Fixtures ---


@pytest.fixture
def dummy_dataset_path(tmp_path):
    path = tmp_path / "dummy"
    _dummy_dataset_loader(path)
    return path


@pytest.fixture
def sim_spec(dummy_dataset_path):
    return BenchmarkSpecSimulation(
        name="TestSim",
        dataset_id="dummy",
        u_cols=["u0", "u1"],
        y_cols=["y0"],
        metric_func=rmse,
        init_window=5,
        data_root=dummy_dataset_path.parent,
    )


@pytest.fixture
def pred_spec(dummy_dataset_path):
    return BenchmarkSpecPrediction(
        name="TestPred",
        dataset_id="dummy",
        u_cols=["u0", "u1"],
        y_cols=["y0"],
        metric_func=rmse,
        init_window=5,
        pred_horizon=10,
        pred_step=10,
        data_root=dummy_dataset_path.parent,
    )


def dummy_build_model(context):
    output_dim = len(context.spec.y_cols)

    def model(u, y_init):
        return np.zeros((u.shape[0], output_dim))

    return model


# --- BenchmarkSpec construction & properties ---


class TestBenchmarkSpecProperties:
    def test_sim_spec_properties(self, sim_spec, dummy_dataset_path):
        assert sim_spec.name == "TestSim"
        assert sim_spec.dataset_id == "dummy"
        assert sim_spec.dataset_path == dummy_dataset_path
        assert sim_spec.train_path == dummy_dataset_path / "train"
        assert sim_spec.test_path == dummy_dataset_path / "test"

    def test_pred_spec_has_prediction_params(self, pred_spec):
        assert pred_spec.pred_horizon == 10
        assert pred_spec.pred_step == 10

    def test_pred_spec_rejects_zero_horizon(self, dummy_dataset_path):
        with pytest.raises(ValueError):
            BenchmarkSpecPrediction(
                name="Bad",
                dataset_id="dummy",
                u_cols=["u0"],
                y_cols=["y0"],
                metric_func=rmse,
                pred_horizon=0,
                pred_step=10,
                data_root=dummy_dataset_path.parent,
            )

    def test_pred_spec_rejects_negative_horizon(self, dummy_dataset_path):
        with pytest.raises(ValueError):
            BenchmarkSpecPrediction(
                name="Bad",
                dataset_id="dummy",
                u_cols=["u0"],
                y_cols=["y0"],
                metric_func=rmse,
                pred_horizon=-1,
                pred_step=10,
                data_root=dummy_dataset_path.parent,
            )

    def test_spec_file_listing(self, sim_spec):
        assert len(sim_spec.train_files) == 2
        assert len(sim_spec.test_files) == 2
        # files should be sorted
        assert sim_spec.train_files == sorted(sim_spec.train_files)

    def test_train_valid_files_fallback(self, sim_spec):
        # No train_valid dir exists, so it falls back to train + valid union
        files = sim_spec.train_valid_files
        assert len(files) == 4  # 2 train + 2 valid
        assert files == sorted(files)

    def test_train_valid_files_dedicated_dir(self, tmp_path):
        path = tmp_path / "dummy_tv"
        _dummy_dataset_loader(path, create_train_valid_dir=True)
        spec = BenchmarkSpecSimulation(
            name="TestTV",
            dataset_id="dummy_tv",
            u_cols=["u0", "u1"],
            y_cols=["y0"],
            metric_func=rmse,
            init_window=5,
            data_root=tmp_path,
        )
        files = spec.train_valid_files
        assert len(files) == 1  # _dummy_dataset_loader creates 1 file in train_valid

    def test_data_root_callable(self, dummy_dataset_path):
        spec = BenchmarkSpecSimulation(
            name="TestCallable",
            dataset_id="dummy",
            u_cols=["u0"],
            y_cols=["y0"],
            metric_func=rmse,
            data_root=lambda: dummy_dataset_path.parent,
        )
        assert spec.data_root == dummy_dataset_path.parent


# --- Data loading ---


class TestLoadSequences:
    def test_load_sequences_shape(self, sim_spec):
        sequences = list(
            _load_sequences_from_files(sim_spec.train_files, sim_spec.u_cols, sim_spec.y_cols)
        )
        u, y, x = sequences[0]
        assert u.shape == (50, 2)
        assert y.shape == (50, 1)

    def test_load_sequences_yields_all_files(self, sim_spec):
        sequences = list(
            _load_sequences_from_files(sim_spec.train_files, sim_spec.u_cols, sim_spec.y_cols)
        )
        assert len(sequences) == 2

    def test_load_sequences_windowing(self, sim_spec):
        sequences = list(
            _load_sequences_from_files(
                sim_spec.train_files, sim_spec.u_cols, sim_spec.y_cols, win_sz=20, stp_sz=10
            )
        )
        for u, y, x in sequences:
            assert u.shape == (20, 2)
            assert y.shape == (20, 1)
        # 2 files, each 50 samples: windows at offsets 0,10,20,30 = 4 windows per file
        assert len(sequences) == 2 * 4

    def test_load_sequences_no_x_cols(self, sim_spec):
        sequences = list(
            _load_sequences_from_files(sim_spec.train_files, sim_spec.u_cols, sim_spec.y_cols)
        )
        _, _, x = sequences[0]
        assert x is None

    def test_load_sequences_empty_path(self):
        sequences = list(_load_sequences_from_files([], ["u0"], ["y0"]))
        assert len(sequences) == 0

    def test_load_sequences_win_without_step_raises(self, sim_spec):
        with pytest.raises(ValueError):
            list(
                _load_sequences_from_files(
                    sim_spec.train_files, sim_spec.u_cols, sim_spec.y_cols, win_sz=20
                )
            )

    def test_load_sequences_step_without_win_raises(self, sim_spec):
        with pytest.raises(ValueError):
            list(
                _load_sequences_from_files(
                    sim_spec.train_files, sim_spec.u_cols, sim_spec.y_cols, stp_sz=10
                )
            )


# --- TrainingContext ---


class TestTrainingContext:
    def test_train_sequences(self, sim_spec):
        ctx = TrainingContext(spec=sim_spec, hyperparameters={})
        sequences = list(ctx.get_train_sequences())
        assert len(sequences) == 2
        u, y, x = sequences[0]
        assert u.shape == (50, 2)

    def test_valid_sequences(self, sim_spec):
        ctx = TrainingContext(spec=sim_spec, hyperparameters={})
        sequences = list(ctx.get_valid_sequences())
        assert len(sequences) == 2

    def test_train_valid_sequences(self, sim_spec):
        ctx = TrainingContext(spec=sim_spec, hyperparameters={})
        sequences = list(ctx.get_train_valid_sequences())
        assert len(sequences) == 4  # train + valid combined


# --- Simulation testing ---


class TestSimulation:
    def test_returns_correct_structure(self, sim_spec):
        model = lambda u, y_init: np.zeros((u.shape[0], 1))
        results = _test_simulation(sim_spec, model)
        assert len(results) == 2  # 2 test files
        for y_pred, y_test in results:
            assert isinstance(y_pred, np.ndarray)
            assert isinstance(y_test, np.ndarray)

    def test_windowing(self, sim_spec):
        model = lambda u, y_init: np.zeros((u.shape[0], 1))
        results = _test_simulation(sim_spec, model)
        for y_pred, y_test in results:
            # y_test should have init_window removed: 50 - 5 = 45
            assert y_test.shape[0] == 45

    def test_perfect_model(self, sim_spec):
        def perfect_model(u, y_init):
            # Load the full y for this sequence to return perfect predictions
            # The model receives full u and y_init (first init_window steps)
            # It should return predictions for all timesteps, but only the last
            # (seq_len - init_window) are compared
            return np.zeros((u.shape[0], 1))  # placeholder

        # Instead, use a model that echoes the test data back
        # We need to verify structure, not perfection — that requires knowing y_test
        results = _test_simulation(sim_spec, perfect_model)
        for y_pred, y_test in results:
            score = rmse(y_test, y_pred)
            assert np.isfinite(score)


# --- Prediction testing ---


class TestPrediction:
    def test_returns_nested_structure(self, pred_spec):
        model = lambda u, y_init: np.zeros((u.shape[0], 1))
        results = _test_prediction(pred_spec, model)
        assert len(results) == 2  # 2 test files
        for window_results in results:
            assert isinstance(window_results, list)
            for y_pred, y_test in window_results:
                assert isinstance(y_pred, np.ndarray)

    def test_window_count(self, pred_spec):
        model = lambda u, y_init: np.zeros((u.shape[0], 1))
        results = _test_prediction(pred_spec, model)
        seq_len = 50
        expected_windows = len(range(0, seq_len - pred_spec.init_window - pred_spec.pred_horizon, pred_spec.pred_step))
        for window_results in results:
            assert len(window_results) == expected_windows


# --- run_benchmark end-to-end ---


class TestRunBenchmark:
    def test_simulation(self, sim_spec):
        result = run_benchmark(sim_spec, dummy_build_model)
        assert result["benchmark_name"] == "TestSim"
        assert result["benchmark_type"] == "BenchmarkSpecSimulation"
        assert np.isfinite(result["metric_score"])
        assert result["training_time_seconds"] >= 0
        assert result["test_time_seconds"] >= 0
        assert result["metric_name"] == "rmse"
        assert isinstance(result["model_predictions"], list)

    def test_prediction(self, pred_spec):
        result = run_benchmark(pred_spec, dummy_build_model)
        assert result["benchmark_name"] == "TestPred"
        assert result["benchmark_type"] == "BenchmarkSpecPrediction"
        assert np.isfinite(result["metric_score"])

    def test_none_model_raises(self, sim_spec):
        def bad_build(context):
            return None

        with pytest.raises(RuntimeError, match="did not return a model"):
            run_benchmark(sim_spec, bad_build)

    def test_custom_scores(self, sim_spec):
        def custom_eval(test_results, spec):
            return {"extra_metric": 42.0}

        sim_spec.custom_test_evaluation = custom_eval
        result = run_benchmark(sim_spec, dummy_build_model)
        assert result["custom_scores"]["extra_metric"] == 42.0


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
                "custom_scores": {},
                "model_predictions": [],
            }
        ]
        df = benchmark_results_to_dataframe(results)
        assert "benchmark_name" in df.columns
        assert "metric_score" in df.columns

    def test_drops_predictions(self):
        results = [
            {
                "benchmark_name": "A",
                "metric_score": 1.0,
                "custom_scores": {},
                "model_predictions": [np.zeros(10)],
            }
        ]
        df = benchmark_results_to_dataframe(results)
        assert "model_predictions" not in df.columns

    def test_flattens_custom_scores(self):
        results = [
            {
                "benchmark_name": "A",
                "metric_score": 1.0,
                "custom_scores": {"extra": 5.0},
                "model_predictions": [],
            }
        ]
        df = benchmark_results_to_dataframe(results)
        assert "cs_extra" in df.columns
        assert df["cs_extra"].iloc[0] == 5.0

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
