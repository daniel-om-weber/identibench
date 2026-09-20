"""Tests for the IAS (instantaneous angular speed) datasets and benchmarks."""

import dataclasses

import h5py
import numpy as np
import pytest

import identibench as idb
from identibench.datasets.ias import GridwiseEstimation, WindowedEstimation, _common
from identibench.datasets.ias import planetary_gearbox as planetary
from identibench.datasets.ias.ball_bearing import (
    BenchmarkBallBearing_Estimation,
    BenchmarkBallBearing_Simulation,
    ball_bearing_dataset,
)


# ───────────────────────── encoder → IAS ─────────────────────────


def _pulse_train(ias_hz, fs, ppr):
    """An analog encoder pulse train whose instantaneous speed follows `ias_hz(t)`."""
    t = np.arange(int(fs * 2)) / fs
    phase = np.cumsum(2 * np.pi * ppr * ias_hz(t) / fs)
    return t, np.sin(phase)


def test_encoder_pulse_to_ias_recovers_constant_frequency():
    fs, f_rot, ppr = 20_000.0, 20.0, 64
    t, pulse = _pulse_train(lambda t: np.full_like(t, f_rot), fs, ppr)

    ias, sl = _common.encoder_pulse_to_ias(pulse, fs, pulses_per_revolution=ppr, cutoff_order=4.0)

    assert ias.shape == (sl.stop - sl.start,)
    # The IAS is defined only where the encoder measures; that span is nearly the whole
    # recording here (one pulse period at the head), and is never extrapolated past.
    assert sl.stop - sl.start > 0.99 * len(pulse)
    np.testing.assert_allclose(ias, f_rot, rtol=0.02)


def test_encoder_pulse_to_ias_tracks_a_speed_sweep():
    """Bandwidth is the point of the order-domain reference; a constant target cannot show it."""
    fs, ppr = 20_000.0, 64
    # 20 Hz mean with a 2 Hz peak-to-peak wobble at 2 orders -- inside a 4-order cutoff.
    profile = lambda t: 20.0 + 1.0 * np.sin(2 * np.pi * 2 * 20.0 * t)  # noqa: E731
    t, pulse = _pulse_train(profile, fs, ppr)
    ias, sl = _common.encoder_pulse_to_ias(pulse, fs, pulses_per_revolution=ppr, cutoff_order=4.0)

    expected = profile(t[sl])
    interior = slice(len(ias) // 10, -len(ias) // 10)  # filter transients at the ends
    np.testing.assert_allclose(ias[interior], expected[interior], rtol=0.05)
    # and it really is tracking, not just sitting at the mean
    assert np.ptp(ias[interior]) > 1.5


def test_encoder_pulse_to_ias_slice_keeps_channels_aligned():
    fs, ppr = 20_000.0, 64
    t, pulse = _pulse_train(lambda t: np.full_like(t, 20.0), fs, ppr)
    vibration = np.sin(2 * np.pi * 137.0 * t)

    ias, sl = _common.encoder_pulse_to_ias(pulse, fs, pulses_per_revolution=ppr, cutoff_order=4.0)

    assert len(vibration[sl]) == len(ias)
    assert 0 <= sl.start < sl.stop <= len(pulse)


def test_order_domain_lowpass_passes_and_stops():
    ppr = 128
    theta = np.arange(ppr * 200) / ppr  # 200 revolutions, uniform in angle
    for order, expect_pass in ((1.0, True), (40.0, False)):
        x = np.sin(2 * np.pi * order * theta)
        y = _common.order_domain_lowpass(x, cutoff_order=4.0, pulses_per_revolution=ppr)
        interior = y[len(y) // 10 : -len(y) // 10]
        gain = np.ptp(interior) / np.ptp(x)
        if expect_pass:
            assert gain > 0.9, f"order {order} should pass, gain={gain}"
        else:
            assert gain < 0.01, f"order {order} should be stopped, gain={gain}"


def test_order_domain_lowpass_rejects_impossible_cutoff():
    x = np.zeros(1000)
    with pytest.raises(ValueError, match="Nyquist"):
        _common.order_domain_lowpass(x, cutoff_order=15.0, pulses_per_revolution=20)
    with pytest.raises(ValueError, match="Nyquist"):
        _common.order_domain_lowpass(x, cutoff_order=0.0, pulses_per_revolution=1024)


def test_rising_edge_times_does_not_double_count_exact_zeros():
    """A sample landing exactly on zero is one edge, not two.

    The symmetric ramp is already normalized, so sample 5 of every repeat is exactly 0.0.
    A `np.diff(np.sign(...)) > 0` detector steps -1 -> 0 and 0 -> +1 separately and reports
    two rising edges there, halving the apparent period.
    """
    fs = 1000.0
    ramp = np.linspace(-1.0, 1.0, 11)  # ..., -0.2, 0.0, 0.2, ...
    signal = np.tile(ramp, 10)

    edges = _common.rising_edge_times(signal, fs)

    assert len(edges) == 10  # one upward crossing per repeat
    np.testing.assert_allclose(np.diff(edges) * fs, len(ramp), atol=1e-9)
    # the refined crossing lands exactly on the zero sample
    np.testing.assert_allclose(edges[0] * fs, 5.0, atol=1e-9)
    # the discarded alternative would have found twice as many
    assert len(np.argwhere(np.diff(np.sign(signal)) > 0).ravel()) == 20


# ───────────────────────── zebra tape reconstruction ─────────────────────────


def _synthetic_zebra(stripes, ias_profile, fs=5_000.0, duration=200.0, drop=0.15, jitter_s=2e-5, seed=0):
    """A synthetic planetary recording: 1PR carrier reference + irregular sun-shaft zebra tape.

    Returns ``(t_ref, t_zebra, ias_profile_on_grid, n_samples)``. Stripe detections are randomly
    dropped and their times jittered, so the fixture exercises the same gaps and timing noise the
    template estimator has to cope with on the real tape.

    The speed profiles passed in must vary slowly relative to one *carrier* revolution, as the
    real rigs do (they modulate over tens of seconds against a ~0.5 s carrier period). The whole
    method rests on interpolating sun angle between 1PR pulses, and the sun turns 5.77 times per
    carrier revolution, so an interpolation error worth 1% of a carrier revolution already smears
    a stripe across several of its neighbours.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    ias = ias_profile(t)
    theta = np.cumsum(ias) / fs  # sun revolutions

    ratio = planetary._SUN_PER_CARRIER_REV
    carrier = theta / ratio
    t_ref = np.interp(np.arange(1.0, np.floor(carrier[-1])), carrier, t)

    targets = (np.arange(0.0, np.floor(theta[-1]))[:, None] + np.asarray(stripes)[None, :]).ravel()
    targets = np.sort(targets[(targets > theta[0]) & (targets < theta[-1])])
    t_zebra = np.interp(targets, theta, t) + rng.normal(0, jitter_s, len(targets))
    t_zebra = np.sort(t_zebra[rng.random(len(t_zebra)) > drop])
    return t_ref, t_zebra, ias, len(t)


def _circular_spacings(angles):
    """Gap to the next stripe, wrapping around -- invariant to how the tape is clocked."""
    return np.sort(np.diff(np.concatenate([angles, [angles[0] + 1.0]])))


def test_zebra_round_trip_recovers_template_and_speed():
    fs, n_stripes = 5_000.0, 76
    rng = np.random.default_rng(7)
    # Irregular stripe angles, nudged off a perfect grid the way a real tape is. Kept within
    # +-0.12 of a spacing: the template fitter cannot resolve two stripes closer together than
    # half the mean spacing (see `_fit_template_from_phase`).
    planted = np.sort((np.arange(n_stripes) + rng.uniform(-0.12, 0.12, n_stripes)) / n_stripes) % 1.0

    def fast(t):
        return 10.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t)

    def slow(t):
        return 8.0 + 0.4 * np.sin(2 * np.pi * 0.015 * t)

    # The second recording is clocked 0.31 rev around from the first, as a rig reassembled
    # between test phases would be; pooling must align that out rather than splitting stripes.
    files = [
        _synthetic_zebra(planted, fast, fs=fs, seed=1),
        _synthetic_zebra((planted + 0.31) % 1.0, slow, fs=fs, seed=2),
    ]
    phases, revs = [], []
    for t_ref, t_zebra, _, _ in files:
        phase, n_revs = planetary._zebra_phase(t_ref, t_zebra)
        phases.append(phase)
        revs.append(n_revs)

    template, counts, pooled_hist = planetary.pool_zebra_templates(phases, revs)

    # Every stripe is found, and the tape's geometry is recovered up to how it is clocked.
    assert len(template) == n_stripes
    np.testing.assert_allclose(_circular_spacings(template), _circular_spacings(planted), atol=2e-3)
    assert counts.min() > 0.5 * sum(revs)  # ~85% of stripes detected per revolution

    # ...and the reconstructed IAS tracks the planted speed of each recording.
    for (t_ref, t_zebra, ias_true, n_samples), phase in zip(files, phases):
        t_matched, idx_matched = planetary.match_zebra_to_template(
            t_zebra, t_ref, template, phase_offset=planetary.zebra_phase_offset(phase, pooled_hist)
        )
        t_clean, idx_clean = planetary.drop_zebra_outliers(t_matched, idx_matched, template)
        ias, sl = planetary.reconstruct_ias(t_clean, idx_clean, template, fs, n_samples)

        assert len(ias) == sl.stop - sl.start
        interior = slice(len(ias) // 20, -len(ias) // 20)  # filter transients at the ends
        got, want = ias[interior], ias_true[sl][interior]

        # Judged on robust statistics, not sample-wise: stripe timing jitter propagates into
        # the finite-difference rate, and a 15-order cutoff against a 38-order Nyquist leaves
        # a few percent of it in by design -- that is the bandwidth/noise trade being made.
        rel_err = np.abs(got - want) / want
        assert np.median(rel_err) < 0.01
        assert np.percentile(rel_err, 99) < 0.03
        # unbiased, and actually following the profile rather than sitting at its mean
        assert abs(got.mean() - want.mean()) / want.mean() < 0.005
        assert np.corrcoef(got, want)[0, 1] > 0.95


def test_zebra_reconstruction_is_truncated_not_extrapolated():
    """The span before the first matched stripe is dropped, never filled in."""
    fs = 5_000.0
    rng = np.random.default_rng(3)
    planted = np.sort((np.arange(40) + rng.uniform(-0.12, 0.12, 40)) / 40) % 1.0
    t_ref, t_zebra, _, n_samples = _synthetic_zebra(planted, lambda t: np.full_like(t, 9.0), fs=fs, seed=4)

    phase, n_revs = planetary._zebra_phase(t_ref, t_zebra)
    template, _, pooled_hist = planetary.pool_zebra_templates([phase], [n_revs])
    t_matched, idx_matched = planetary.match_zebra_to_template(
        t_zebra, t_ref, template, phase_offset=planetary.zebra_phase_offset(phase, pooled_hist)
    )
    t_clean, idx_clean = planetary.drop_zebra_outliers(t_matched, idx_matched, template)
    _, sl = planetary.reconstruct_ias(t_clean, idx_clean, template, fs, n_samples)

    # `_zebra_phase` discards the first ten 1PR reference pulses, so a real head is cut off.
    assert sl.start / fs >= t_clean[0] - 1 / fs
    assert sl.stop / fs <= t_clean[-1] + 1 / fs
    assert sl.start > 0 and sl.stop < n_samples


# ───────────────────────── seeded disturbances ─────────────────────────


def test_add_disturbances_deterministic_and_seed_sensitive():
    fs = 10_000.0
    sig = np.sin(2 * np.pi * 35.0 * np.arange(int(fs)) / fs)
    ias_hz = np.full_like(sig, 20.0)

    out_a = _common.add_disturbances(sig, fs, target_snr_db=15, rng=np.random.default_rng(7), ias_hz=ias_hz)
    out_b = _common.add_disturbances(sig, fs, target_snr_db=15, rng=np.random.default_rng(7), ias_hz=ias_hz)
    out_c = _common.add_disturbances(sig, fs, target_snr_db=15, rng=np.random.default_rng(8), ias_hz=ias_hz)

    np.testing.assert_array_equal(out_a, out_b)
    assert not np.array_equal(out_a, out_c)


def test_add_disturbances_hits_target_snr():
    fs = 10_000.0
    sig = np.sin(2 * np.pi * 35.0 * np.arange(int(fs)) / fs)
    ias_hz = np.full_like(sig, 20.0)

    for target_snr_db in (15, 7.5, 0, -7.5):
        out = _common.add_disturbances(sig, fs, target_snr_db=target_snr_db, rng=np.random.default_rng(0), ias_hz=ias_hz)
        noise = out - sig
        snr_db = 10 * np.log10(np.mean(sig**2) / np.mean(noise**2))
        assert snr_db == pytest.approx(target_snr_db, abs=2.0)


def test_add_disturbances_single_type_hits_full_snr():
    """A single-element disturbance_types tuple gets 100% of the noise power."""
    fs = 10_000.0
    sig = np.sin(2 * np.pi * 35.0 * np.arange(int(fs)) / fs)
    ias_hz = np.full_like(sig, 20.0)

    for kind in ("harmonic", "gaussian", "impulsive"):
        out = _common.add_disturbances(
            sig, fs, target_snr_db=0, rng=np.random.default_rng(0), ias_hz=ias_hz, disturbance_types=(kind,)
        )
        noise = out - sig
        snr_db = 10 * np.log10(np.mean(sig**2) / np.mean(noise**2))
        assert snr_db == pytest.approx(0, abs=2.0)


def test_harmonic_disturbance_frequency_tracks_ias_and_varies():
    """The harmonic component's instantaneous frequency wanders with IAS(t), not constant."""
    fs = 5_000.0
    n = int(fs) * 4
    sig = np.ones(n)  # nonzero power, but far from the harmonic frequencies below
    t = np.arange(n) / fs
    # A slowly ramping IAS so the two halves of the recording have clearly different means.
    ias_hz = 20.0 + 10.0 * (t / t[-1])

    out = _common.add_disturbances(
        sig, fs, target_snr_db=0, rng=np.random.default_rng(0), ias_hz=ias_hz, disturbance_types=("harmonic",)
    )
    noise = out - sig  # isolate the added disturbance from the constant signal

    half = n // 2
    freqs = np.fft.rfftfreq(half, d=1 / fs)
    first_half_peak = freqs[np.argmax(np.abs(np.fft.rfft(noise[:half])))]
    second_half_peak = freqs[np.argmax(np.abs(np.fft.rfft(noise[half:])))]

    expected_first = _common.HARMONIC_IAS_RATIO * np.mean(ias_hz[:half])
    expected_second = _common.HARMONIC_IAS_RATIO * np.mean(ias_hz[half:])
    assert first_half_peak == pytest.approx(expected_first, rel=0.2)
    assert second_half_peak == pytest.approx(expected_second, rel=0.2)
    assert second_half_peak > first_half_peak  # tracks the rising IAS


def _write_ias_file(path, ias, vib_channels, fs=1000.0):
    # Write through the production writer so the fixtures track its layout.
    path.parent.mkdir(parents=True, exist_ok=True)
    _common.save_signals_hdf5({"IAS": ias, **vib_channels}, path, fs=fs, gear_ratio=1, ias_bandwidth_hz=100.0)


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
        for variant in _common.DISTURBANCE_VARIANTS:
            for i in range(2):
                with (
                    h5py.File(tmp_path / "a" / f"test_disturbed_{level}dB_{variant}" / f"rec_{i}.hdf5") as fa,
                    h5py.File(tmp_path / "b" / f"test_disturbed_{level}dB_{variant}" / f"rec_{i}.hdf5") as fb,
                ):
                    # per-(file, level, variant) seeding makes output independent of iteration order
                    np.testing.assert_array_equal(fa["Acc_x"][:], fb["Acc_x"][:])
                    # IAS target is untouched
                    np.testing.assert_array_equal(fa["IAS"][:], fb["IAS"][:])


def test_disturbed_copies_differ_per_file_and_level(tmp_path):
    sig = np.sin(np.arange(500) * 0.1)
    _write_ias_file(tmp_path / "test" / "rec_0.hdf5", np.ones(500), {"Acc_x": sig})
    _write_ias_file(tmp_path / "test" / "rec_1.hdf5", np.ones(500), {"Acc_x": sig})

    _common.write_disturbed_test_sets(tmp_path, noise_levels=[15, 7.5], vib_keys=["Acc_x"])

    with (
        h5py.File(tmp_path / "test_disturbed_15dB_combined" / "rec_0.hdf5") as f0,
        h5py.File(tmp_path / "test_disturbed_15dB_combined" / "rec_1.hdf5") as f1,
        h5py.File(tmp_path / "test_disturbed_7.5dB_combined" / "rec_0.hdf5") as f2,
    ):
        a, b, c = f0["Acc_x"][:], f1["Acc_x"][:], f2["Acc_x"][:]
    assert not np.array_equal(a, b)  # same level, different file -> different noise
    assert not np.array_equal(a, c)  # same file, different level -> different noise


# ───────────────────────── HDF5 writing ─────────────────────────


def test_save_signals_hdf5_attrs_roundtrip(tmp_path):
    path = tmp_path / "rec.hdf5"
    _common.save_signals_hdf5(
        {"IAS": np.ones(10), "Acc_x": np.zeros(10)},
        path,
        fs=12800,
        gear_ratio=[95 / 29, 95, 1],
        ias_bandwidth_hz=4.03,
    )

    with h5py.File(path) as f:
        attrs = dict(f.attrs)
        assert f["IAS"].dtype == np.float32
    assert attrs["fs"] == 12800
    np.testing.assert_allclose(np.asarray(attrs["gear_ratio"]), [95 / 29, 95, 1])
    assert attrs["ias_bandwidth_hz"] == pytest.approx(4.03)


# ───────────────────────── test-set patterns + end-to-end ─────────────────────────


def test_ias_test_sets_patterns():
    sets = _common.ias_test_sets(ball_bearing_dataset)
    expected = ["basic", "wear"] + [
        f"disturbed_{level}dB_{variant}"
        for level in _common.DISTURBANCE_LEVELS
        for variant in _common.DISTURBANCE_VARIANTS
    ]
    assert list(sets) == expected
    assert sets["basic"] == [(ball_bearing_dataset, "test/*.hdf5")]
    assert sets["wear"] == [(ball_bearing_dataset, "test_wear/*.hdf5")]
    assert sets["disturbed_7.5dB_combined"] == [(ball_bearing_dataset, "test_disturbed_7.5dB_combined/*.hdf5")]
    assert sets["disturbed_0dB_harmonic"] == [(ball_bearing_dataset, "test_disturbed_0dB_harmonic/*.hdf5")]
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
    # Adopt the synthetic data as a prepared cache. This must carry the dataset's *current*
    # version, or `ensure_exists` treats the directory as stale and downloads the real 200 kHz
    # archive instead of using the fixture.
    (ds / ".prepared").write_text(ball_bearing_dataset.version)


def _narrowed_test_sets():
    """The synthetic dataset only materializes one disturbance level."""
    return {
        name: patterns
        for name, patterns in _common.ias_test_sets(ball_bearing_dataset).items()
        if name in ("basic", "wear", "disturbed_15dB_combined")
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
    assert set(result["test_sets"]) == {"basic", "wear", "disturbed_15dB_combined"}
    assert result["test_sets"]["wear"]["mae"] == pytest.approx(9.0, abs=1e-5)
    assert result["test_sets"]["disturbed_15dB_combined"]["mae"] == pytest.approx(5.0, abs=1e-5)
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
    assert set(result["test_sets"]) == {"basic", "wear", "disturbed_15dB_combined"}
    assert set(result["test_sets"]["basic"]) == {"mae"}
    assert result["test_sets"]["wear"]["mae"] == pytest.approx(9.0, abs=1e-5)


def test_run_gridwise_benchmark_on_synthetic_ias_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    # 100-sample files at fs=1000 -> a 0.05 s window / 0.01 s step gives 6 query points/file.
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=GridwiseEstimation(window_sec=0.05, step_sec=0.01),
        test_sets=_narrowed_test_sets(),
    )

    def build_model(context):
        def model(u, y_init, attrs):
            assert y_init.shape == (0, 1)
            t_query = attrs["t_query"]
            # Returns a plain Python list, not an ndarray -- exercises the list-coercion fix.
            return [0.0] * len(t_query)

        return model

    result = idb.run_benchmark(spec, build_model, seed=0)

    # Headline = basic pooled MAE; the zero-model's per-query error is the constant IAS.
    assert result["metric_name"] == "mae"
    assert result["metric_score"] == pytest.approx(5.0, abs=1e-5)
    assert set(result["test_sets"]) == {"basic", "wear", "disturbed_15dB_combined"}
    assert result["test_sets"]["wear"]["mae"] == pytest.approx(9.0, abs=1e-5)
    basic = result["test_sets"]["basic"]
    assert set(basic) == {"mae", "medae", "std", "max"}
    assert basic["std"] == pytest.approx(0.0, abs=1e-5)  # constant IAS -> identical errors


def test_gridwise_estimation_rejects_wrong_shaped_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=GridwiseEstimation(window_sec=0.05, step_sec=0.01),
        test_sets=_narrowed_test_sets(),
    )

    def build_model(context):
        # Wrong length: one prediction regardless of how many query points were asked for.
        return lambda u, y_init, attrs: np.zeros(1)

    with pytest.raises(ValueError, match="expected"):
        idb.run_benchmark(spec, build_model, seed=0)


def test_gridwise_estimation_rejects_multi_output_specs(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=GridwiseEstimation(window_sec=0.05, step_sec=0.01),
        test_sets=_narrowed_test_sets(),
        y_cols=["IAS", "Extra"],
    )

    def build_model(context):
        return lambda u, y_init, attrs: np.zeros(len(attrs["t_query"]))

    with pytest.raises(ValueError, match="one y_col"):
        idb.run_benchmark(spec, build_model, seed=0)


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
    # Each dataset also has a GridwiseEstimation sibling sharing the same data. Its step_sec is
    # per-dataset and derived from the label bandwidth, so pin the values (see ias/__init__).
    for key, expected_step in (
        ("BallBearing_GridwiseEstimation", 0.003),
        ("ParallelGearbox_GridwiseEstimation", 0.1),
        ("PlanetaryGearbox_GridwiseEstimation", 0.003),
        ("GasFoilBearing_GridwiseEstimation", 0.003),
    ):
        assert key in idb.simulation_benchmarks
        assert key in idb.ias_benchmarks
        spec = idb.ias_benchmarks[key]
        assert isinstance(spec.task, GridwiseEstimation)
        assert spec.task.window_sec == 3.0  # uniform context guarantee, unrelated to resolution
        assert spec.task.step_sec == expected_step
        assert next(iter(spec.test_sets)) == "basic"  # built-in tasks headline the first set
        assert spec.y_cols == ["IAS"]


def test_gridwise_step_matches_declared_label_bandwidth():
    """Each grid is Nyquist for its dataset's retained band -- except the documented planetary cap."""
    from identibench.datasets.ias import ball_bearing, gas_foil_bearing, parallel_gearbox, planetary_gearbox

    for module, key in (
        (ball_bearing, "BallBearing_GridwiseEstimation"),
        (parallel_gearbox, "ParallelGearbox_GridwiseEstimation"),
        (gas_foil_bearing, "GasFoilBearing_GridwiseEstimation"),
    ):
        step = idb.ias_benchmarks[key].task.step_sec
        nyquist = 1 / (2 * module._IAS_BANDWIDTH_HZ)
        assert step <= nyquist, f"{key}: step {step} exceeds Nyquist {nyquist}"

    # The planetary is knowingly coarser; assert the cap so it cannot drift unnoticed in either
    # direction -- tightening it silently would multiply evaluation cost by over 100x.
    step = idb.ias_benchmarks["PlanetaryGearbox_GridwiseEstimation"].task.step_sec
    nyquist = 1 / (2 * planetary_gearbox._IAS_BANDWIDTH_HZ)
    assert nyquist == pytest.approx(0.000835, abs=1e-5)
    assert step == 0.003 > nyquist
    for dataset_id in ("ball_bearing", "parallel_gearbox", "planetary_gearbox", "gas_foil_bearing"):
        assert dataset_id in idb.datasets.all_datasets
