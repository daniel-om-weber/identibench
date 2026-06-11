"""Opt-in integration coverage for every registered benchmark.

Each test downloads the benchmark's real dataset and runs it end-to-end with a
trivial model, asserting the pipeline produces a finite score. Parametrization is
driven by the benchmark registries, so any dataset added later is covered
automatically.

These hit the network and some datasets are large; they are gated behind the
``slow`` mark — run with ``pytest --slow``.

Data caching
------------
All tests in a session share ONE data root (the ``integration_data_root``
fixture), so each dataset — and each raw archive — is downloaded only once and
reused. In particular the combined ``RIANN`` corpus reuses the per-source
archives already fetched by the individual orientation tests, instead of
re-downloading them. By default the root is a session temp dir (downloaded once
per session, removed afterward); set ``IDENTIBENCH_TEST_DATA_ROOT`` to a
persistent path to cache datasets across sessions so re-runs download nothing.

Because the cache is shared and the library's download path is not
concurrency-safe (a check-then-act existence test with no locking), run these
sequentially: parallel workers (e.g. ``pytest -n``) would race to download the
same archive into the shared root. Parallelize only against an already-warm
persistent cache.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import identibench as idb


@pytest.fixture(scope="session")
def integration_data_root(tmp_path_factory):
    """One shared data root for the whole --slow session (see module docstring)."""
    env = os.environ.get("IDENTIBENCH_TEST_DATA_ROOT")
    return Path(env) if env else tmp_path_factory.mktemp("idb_data")


def _dummy_build_model(context):
    """Return a trivial predictor.

    Outputs zeros, except for 4-channel (quaternion) orientation targets where it
    returns the identity quaternion ``[1, 0, 0, 0]`` so the inclination metric stays
    finite rather than dividing by a zero-norm quaternion.
    """
    n_y = len(context.spec.y_cols)

    def model(u, y_init, attrs):
        out = np.zeros((len(u), n_y), dtype=np.float32)
        if n_y == 4:
            out[:, 0] = 1.0
        return out

    return model


def _run(spec, data_root, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(data_root))
    result = idb.run_benchmark(spec, _dummy_build_model, seed=0)
    assert np.isfinite(result["metric_score"]), f"{spec.name}: non-finite metric_score"
    assert result["training_time_seconds"] >= 0
    assert result["test_time_seconds"] >= 0
    assert result["benchmark_type"] in ("Simulation", "Prediction", "MaskedPooledInclination", "WindowedEstimation")
    assert result["test_sets"], f"{spec.name}: no test sets scored"
    assert result["metric_name"], f"{spec.name}: no headline metric reported"
    for set_name, metric_scores in result["test_sets"].items():
        for metric_name, value in metric_scores.items():
            assert np.isfinite(value), f"{spec.name}: non-finite {set_name}/{metric_name}"


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(idb.simulation_benchmarks))
def test_simulation_benchmark_downloads_and_runs(key, integration_data_root, monkeypatch):
    _run(idb.simulation_benchmarks[key], integration_data_root, monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(idb.prediction_benchmarks))
def test_prediction_benchmark_downloads_and_runs(key, integration_data_root, monkeypatch):
    _run(idb.prediction_benchmarks[key], integration_data_root, monkeypatch)
