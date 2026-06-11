"""Tests for the IAS (instantaneous angular speed) datasets and benchmarks."""

import dataclasses

import h5py
import numpy as np
import pytest

import identibench as idb
from identibench.datasets.ias import WindowedEstimation, _common
from identibench.datasets.ias.ball_bearing import (
    BenchmarkBallBearing_Estimation,
    BenchmarkBallBearing_Simulation,
    ball_bearing_dataset,
)


# ───────────────────────── encoder → IAS ─────────────────────────


def test_analog_pulse_to_ias_recovers_constant_frequency():
    fs = 20_000.0
    f_rot = 20.0  # Hz
    ppr = 4
    t = np.arange(int(fs)) / fs  # 1 second
    # Phase offset avoids exact-zero samples, which the upstream zero-crossing
    # detector double-counts (real encoder signals never hit exact zeros).
    pulse = np.sin(2 * np.pi * f_rot * ppr * t + 0.3)

    ias = _common.analog_pulse_to_ias(pulse, fs, pulses_per_revolution=ppr)

    assert ias.shape == pulse.shape
    interior = ias[len(ias) // 4 : -len(ias) // 4]  # edges suffer from extrapolation
    np.testing.assert_allclose(interior, f_rot, rtol=0.02)


# ───────────────────────── seeded disturbances ─────────────────────────


def test_add_disturbances_deterministic_and_seed_sensitive():
    fs = 10_000.0
    sig = np.sin(2 * np.pi * 35.0 * np.arange(int(fs)) / fs)

    out_a = _common.add_disturbances(sig, fs, target_snr_db=15, rng=np.random.default_rng(7))
    out_b = _common.add_disturbances(sig, fs, target_snr_db=15, rng=np.random.default_rng(7))
    out_c = _common.add_disturbances(sig, fs, target_snr_db=15, rng=np.random.default_rng(8))

    np.testing.assert_array_equal(out_a, out_b)
    assert not np.array_equal(out_a, out_c)


def test_add_disturbances_hits_target_snr():
    fs = 10_000.0
    sig = np.sin(2 * np.pi * 35.0 * np.arange(int(fs)) / fs)

    for target_snr_db in (15, 7.5, 0):
        out = _common.add_disturbances(sig, fs, target_snr_db=target_snr_db, rng=np.random.default_rng(0))
        noise = out - sig
        snr_db = 10 * np.log10(np.mean(sig**2) / np.mean(noise**2))
        assert snr_db == pytest.approx(target_snr_db, abs=2.0)


def _write_ias_file(path, ias, vib_channels, fs=1000.0):
    # Write through the production writer so the fixtures track its layout.
    path.parent.mkdir(parents=True, exist_ok=True)
    _common.save_signals_hdf5({"IAS": ias, **vib_channels}, path, fs=fs, gear_ratio=1)


def test_write_disturbed_test_sets_reproducible(tmp_path):
    for run_dir in ("a", "b"):
        ds = tmp_path / run_dir
        for i in range(2):
            # identical per-file content in both dirs (deterministic per-index seed)
            ias = np.random.default_rng(i).standard_normal(500)
            _write_ias_file(ds / "test" / f"rec_{i}.hdf5", ias, {"Acc_x": np.sin(np.arange(500))})

    _common.write_disturbed_test_sets(tmp_path / "a", noise_levels=[15, 0], vib_keys=["Acc_x"])
    _common.write_disturbed_test_sets(tmp_path / "b", noise_levels=[0, 15], vib_keys=["Acc_x"])  # reversed order

    for level in (15, 0):
        for i in range(2):
            with (
                h5py.File(tmp_path / "a" / f"test_disturbed_{level}dB" / f"rec_{i}.hdf5") as fa,
                h5py.File(tmp_path / "b" / f"test_disturbed_{level}dB" / f"rec_{i}.hdf5") as fb,
            ):
                # per-(file, level) seeding makes output independent of iteration order
                np.testing.assert_array_equal(fa["Acc_x"][:], fb["Acc_x"][:])
                # IAS target is untouched
                np.testing.assert_array_equal(fa["IAS"][:], fb["IAS"][:])


def test_disturbed_copies_differ_per_file_and_level(tmp_path):
    sig = np.sin(np.arange(500) * 0.1)
    _write_ias_file(tmp_path / "test" / "rec_0.hdf5", np.ones(500), {"Acc_x": sig})
    _write_ias_file(tmp_path / "test" / "rec_1.hdf5", np.ones(500), {"Acc_x": sig})

    _common.write_disturbed_test_sets(tmp_path, noise_levels=[15, 7.5], vib_keys=["Acc_x"])

    with (
        h5py.File(tmp_path / "test_disturbed_15dB" / "rec_0.hdf5") as f0,
        h5py.File(tmp_path / "test_disturbed_15dB" / "rec_1.hdf5") as f1,
        h5py.File(tmp_path / "test_disturbed_7.5dB" / "rec_0.hdf5") as f2,
    ):
        a, b, c = f0["Acc_x"][:], f1["Acc_x"][:], f2["Acc_x"][:]
    assert not np.array_equal(a, b)  # same level, different file -> different noise
    assert not np.array_equal(a, c)  # same file, different level -> different noise


# ───────────────────────── HDF5 writing ─────────────────────────


def test_save_signals_hdf5_attrs_roundtrip(tmp_path):
    path = tmp_path / "rec.hdf5"
    _common.save_signals_hdf5({"IAS": np.ones(10), "Acc_x": np.zeros(10)}, path, fs=12800, gear_ratio=[95 / 29, 95, 1])

    with h5py.File(path) as f:
        attrs = dict(f.attrs)
        assert f["IAS"].dtype == np.float32
    assert attrs["fs"] == 12800
    np.testing.assert_allclose(np.asarray(attrs["gear_ratio"]), [95 / 29, 95, 1])


# ───────────────────────── test-set patterns + end-to-end ─────────────────────────


def test_ias_test_sets_patterns():
    sets = _common.ias_test_sets(ball_bearing_dataset)
    assert list(sets) == ["basic", "wear", "disturbed_15dB", "disturbed_7.5dB", "disturbed_0dB"]
    assert sets["basic"] == [(ball_bearing_dataset, "test/*.hdf5")]
    assert sets["wear"] == [(ball_bearing_dataset, "test_wear/*.hdf5")]
    assert sets["disturbed_7.5dB"] == [(ball_bearing_dataset, "test_disturbed_7.5dB/*.hdf5")]
    # A dataset without a wear condition simply does not declare the set.
    assert "wear" not in _common.ias_test_sets(ball_bearing_dataset, wear=False)


def _build_synthetic_ias_dataset(ds, with_wear=True):
    """A tiny ball-bearing-shaped dataset with known IAS targets, marked prepared."""
    rng = np.random.default_rng(0)
    for split, n_files in {"train": 1, "valid": 1, "test": 2}.items():
        for i in range(n_files):
            _write_ias_file(ds / split / f"{split}_{i}.hdf5", np.full(100, 5.0), {"Acc_x": rng.standard_normal(100)})
    if with_wear:
        _write_ias_file(ds / "test_wear" / "wear_0.hdf5", np.full(100, 9.0), {"Acc_x": rng.standard_normal(100)})
    _common.write_disturbed_test_sets(ds, noise_levels=[15], vib_keys=["Acc_x"])
    (ds / ".prepared").write_text("1")  # adopt the synthetic data as a prepared cache


def _narrowed_test_sets():
    """The synthetic dataset only materializes one disturbance level."""
    return {
        name: patterns
        for name, patterns in _common.ias_test_sets(ball_bearing_dataset).items()
        if name in ("basic", "wear", "disturbed_15dB")
    }


def test_run_benchmark_on_synthetic_ias_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    # 100-sample files at fs=1000 -> a 0.05 s window is 50 samples (2 windows/file).
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=WindowedEstimation(window_sec=0.05),
        test_sets=_narrowed_test_sets(),
    )

    def build_model(context):
        def model(u, y_init, attrs):
            assert y_init.shape == (0, 1)  # estimation -> empty warm-up, one window of u
            return np.zeros((len(u), 1))

        return model

    result = idb.run_benchmark(spec, build_model, seed=0)

    # Headline = basic pooled MAE; the zero-model's per-window error is the constant IAS.
    assert result["metric_name"] == "mae"
    assert result["metric_score"] == pytest.approx(5.0, abs=1e-5)
    assert set(result["test_sets"]) == {"basic", "wear", "disturbed_15dB"}
    assert result["test_sets"]["wear"]["mae"] == pytest.approx(9.0, abs=1e-5)
    assert result["test_sets"]["disturbed_15dB"]["mae"] == pytest.approx(5.0, abs=1e-5)
    # The pooled statistics ride alongside the headline MAE on every set.
    basic = result["test_sets"]["basic"]
    assert set(basic) == {"mae", "medae", "std", "max"}
    assert basic["std"] == pytest.approx(0.0, abs=1e-5)  # constant IAS -> identical errors
    assert basic["max"] == pytest.approx(5.0, abs=1e-5)


def test_run_simulation_benchmark_on_synthetic_ias_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")  # 100-sample files at fs=1000, basic IAS = 5.0 Hz
    spec = dataclasses.replace(BenchmarkBallBearing_Simulation, test_sets=_narrowed_test_sets())

    def build_model(context):
        def model(u, y_init, attrs):
            # Free-run: the model gets the WHOLE 100-sample recording at once with an empty
            # warm-up and returns one IAS estimate per sample (a real model would slide a
            # window over u; here a trivial zero output).
            assert y_init.shape == (0, 1)
            assert len(u) == 100
            return np.zeros((len(u), 1))

        return model

    result = idb.run_benchmark(spec, build_model, seed=0)

    # Per-sample MAE in Hz over the full sequence; only the headline `mae` is reported
    # (no pooled medae/std/max — those are WindowedEstimation-specific).
    assert result["benchmark_type"] == "Simulation"
    assert result["metric_name"] == "mae"
    assert result["metric_score"] == pytest.approx(5.0, abs=1e-5)  # |0 - 5| per sample
    assert set(result["test_sets"]) == {"basic", "wear", "disturbed_15dB"}
    assert set(result["test_sets"]["basic"]) == {"mae"}
    assert result["test_sets"]["wear"]["mae"] == pytest.approx(9.0, abs=1e-5)


def test_missing_wear_dir_fails_loudly(tmp_path, monkeypatch):
    """Declaring `wear` against a dataset without the condition must raise, not skip."""
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing", with_wear=False)
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=WindowedEstimation(window_sec=0.05),
        test_sets=_narrowed_test_sets(),  # declares wear
    )

    def build_model(context):
        return lambda u, y_init, attrs: np.zeros((len(u), 1))

    with pytest.raises(FileNotFoundError, match="test_wear"):
        idb.run_benchmark(spec, build_model, seed=0)


# ───────────────────────── registration ─────────────────────────


def test_registration():
    for key in (
        "BallBearing_Estimation",
        "ParallelGearbox_Estimation",
        "PlanetaryGearbox_Estimation",
        "GasFoilBearing_Estimation",
    ):
        assert key in idb.simulation_benchmarks
        assert key in idb.ias_benchmarks
        spec = idb.ias_benchmarks[key]
        assert isinstance(spec.task, WindowedEstimation)
        assert spec.task.window_sec > 0
        assert next(iter(spec.test_sets)) == "basic"  # built-in tasks headline the first set
        assert spec.y_cols == ["IAS"]
    # Each dataset also has a dense free-run Simulation sibling sharing the same data.
    for key in (
        "BallBearing_Simulation",
        "ParallelGearbox_Simulation",
        "PlanetaryGearbox_Simulation",
        "GasFoilBearing_Simulation",
    ):
        assert key in idb.simulation_benchmarks
        assert key in idb.ias_benchmarks
        sim = idb.ias_benchmarks[key]
        est = idb.ias_benchmarks[key.replace("_Simulation", "_Estimation")]
        assert isinstance(sim.task, idb.Simulation)
        assert sim.task.init_window == 0  # estimation: no output history fed as warm-up
        assert sim.task.metric.__name__ == "mae"
        assert next(iter(sim.test_sets)) == "basic"
        # The two variants differ only in the task; data binding is identical.
        assert sim.datasets == est.datasets
        assert sim.test_sets == est.test_sets
        assert sim.u_cols == est.u_cols
        assert sim.y_cols == est.y_cols == ["IAS"]
    for dataset_id in ("ball_bearing", "parallel_gearbox", "planetary_gearbox", "gas_foil_bearing"):
        assert dataset_id in idb.datasets.all_datasets
