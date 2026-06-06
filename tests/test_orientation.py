"""Tests for the orientation (IMU) datasets and benchmarks (RIANN family)."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import identibench as idb
from identibench.datasets.orientation import riann as R
from identibench.datasets.orientation import _common
from identibench.metrics import _quat_mul


def _quat_x(angle_deg: float) -> np.ndarray:
    """Unit quaternion of a rotation by angle_deg about the x-axis."""
    a = np.deg2rad(angle_deg) / 2
    return np.array([np.cos(a), np.sin(a), 0.0, 0.0])


def _rand_quats(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _write_riann_file(path, true_quat, mask=None, seed=1):
    """Write a minimal RIANN-format HDF5 file (acc/gyr/dt/opt/movement_mask)."""
    n = len(true_quat)
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is None:
        mask = np.ones(n, dtype=np.float32)
    with h5py.File(path, "w") as f:
        for c in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]:
            f.create_dataset(c, data=rng.standard_normal(n).astype(np.float32))
        f.create_dataset("dt", data=np.full(n, 0.01, dtype=np.float32))
        for i, c in enumerate(_common.IMU_Y_COLS):
            f.create_dataset(c, data=true_quat[:, i].astype(np.float32))
        f.create_dataset("movement_mask", data=np.asarray(mask, dtype=np.float32))


# ───────────────────────── faithful evaluation ─────────────────────────


def test_riann_eval_masks_and_groups(tmp_path):
    q = _rand_quats(200)
    pred = q.copy()
    pred[50:60] = _quat_mul(_quat_x(20.0)[None, :], q[50:60])  # error only here

    # movement_mask = 0 exactly over the erroneous window -> masked score is ~0
    mask = np.ones(200, dtype=np.float32)
    mask[50:60] = 0.0
    fpath = tmp_path / "Caruso-Sassari__slow_v4_AP1.hdf5"
    _write_riann_file(fpath, q, mask=mask)

    spec = SimpleNamespace(test_files=[fpath])
    scores = R.riann_eval([(pred, q)], spec)

    assert "Caruso-Sassari/incl_rmse_deg" in scores
    assert "Caruso-Sassari/incl_p99_deg" in scores
    assert "all/incl_rmse_deg" in scores
    assert scores["Caruso-Sassari/incl_rmse_deg"] == pytest.approx(0.0, abs=1e-6)

    # Without masking the same error shows up.
    _write_riann_file(fpath, q, mask=np.ones(200, dtype=np.float32))
    scores_unmasked = R.riann_eval([(pred, q)], spec)
    assert scores_unmasked["Caruso-Sassari/incl_rmse_deg"] > 1.0


# ───────────────────────── quaternion sign flips ─────────────────────────


def test_fix_quaternion_flips_corrects_sign_flip():
    # A smooth, near-constant quaternion track with a single sign flip injected
    # halfway through. q and -q are the same rotation; the flip must be undone.
    q = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)).astype(np.float64)
    flipped = q.copy()
    flipped[5:] *= -1.0

    corrected = _common.fix_quaternion_flips(flipped)

    np.testing.assert_allclose(corrected, q)
    # Original input must be left untouched (function copies).
    assert flipped[5, 0] == -1.0


def test_fix_quaternion_flips_leaves_smooth_track_unchanged():
    # Small incremental rotations about x: consecutive quaternions stay close, so
    # every step is below the threshold and nothing should be flipped.
    angles = np.linspace(0.0, 30.0, 50)
    q = np.stack([_quat_x(a) for a in angles])
    assert np.max(np.linalg.norm(np.diff(q, axis=0), axis=1)) < 1.0  # genuinely smooth
    corrected = _common.fix_quaternion_flips(q)
    np.testing.assert_allclose(corrected, q)


# ───────────────────────── wiring / splits ─────────────────────────


def test_split_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    root = tmp_path / "riann"
    for sub, names in {
        "train": ["Myon__01_a", "TUM-VI__room1"],
        "valid": ["Myon__14_a", "TUM-VI__room4"],
        "test": ["OxIOD__handheld", "EuRoC-MAV__V1_01", "RepoIMU__t"],
    }.items():
        for nm in names:
            _write_riann_file(root / sub / f"{nm}.hdf5", _rand_quats(20))

    assert len(R.BenchmarkRIANN_Inclination.train_files) == 2
    assert len(R.BenchmarkRIANN_Inclination.valid_files) == 2
    assert len(R.BenchmarkRIANN_Inclination.test_files) == 3


def test_run_benchmark_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    root = tmp_path / "riann"
    ident = np.tile([1.0, 0.0, 0.0, 0.0], (120, 1))
    _write_riann_file(root / "test" / "OxIOD__a.hdf5", ident)
    _write_riann_file(root / "test" / "EuRoC-MAV__b.hdf5", ident)

    def build_model(context):
        def model(u, y_init, attrs):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(u), 1)).astype(np.float32)

        return model

    result = idb.run_benchmark(R.BenchmarkRIANN_Inclination, build_model, seed=0)

    assert np.isfinite(result["metric_score"])
    assert result["metric_score"] == pytest.approx(0.0, abs=1e-4)
    cs = result["custom_scores"]
    assert cs["OxIOD/incl_rmse_deg"] == pytest.approx(0.0, abs=1e-4)
    assert cs["EuRoC-MAV/incl_rmse_deg"] == pytest.approx(0.0, abs=1e-4)
    assert "all/incl_rmse_deg" in cs and "all/incl_p99_deg" in cs


def test_force_download_clears_and_regenerates(tmp_path):
    """force_download must wipe the routed role dirs and re-run download+convert
    (previously the flag was silently ignored)."""
    calls = {"download": 0, "convert": 0}

    def fake_download(raw, force=False):
        calls["download"] += 1

    def fake_convert(raw, out_dir, force=False):
        calls["convert"] += 1
        _write_riann_file(Path(out_dir) / "Fake" / "seq.hdf5", _rand_quats(20))

    preparers = [(fake_download, fake_convert, "Fake")]
    save_path = tmp_path / "fake_ds"

    # First materialization routes the converted file into test/.
    _common._prepare(save_path, preparers, _common._test_role)
    test_dir = save_path / "test"
    assert (test_dir / "Fake__seq.hdf5").exists()
    assert calls == {"download": 1, "convert": 1}

    # A stray leftover that a forced re-prep should remove.
    _write_riann_file(test_dir / "stray.hdf5", _rand_quats(20))

    _common._prepare(save_path, preparers, _common._test_role, force_download=True)
    assert not (test_dir / "stray.hdf5").exists()  # role dir was cleared
    assert (test_dir / "Fake__seq.hdf5").exists()  # regenerated
    assert calls == {"download": 2, "convert": 2}  # re-ran rather than skipping


def test_registration():
    for name in ["dfjimu", "riann", "broad", "oxiod", "euroc", "repoimu", "caruso", "tumvi"]:
        assert name in idb.datasets.all_dataset_loaders
    assert "RIANN_Inclination" in idb.simulation_benchmarks
    assert "DFJIMU_Inclination" in idb.simulation_benchmarks
    assert set(R.riann_benchmarks) >= {"RIANN_Inclination", "OxIOD_Inclination"}
    # The orientation package groups the dfjimu dataset and the whole RIANN family.
    assert set(idb.orientation_benchmarks) >= {"DFJIMU_Inclination", "DFJIMU_Relative", "RIANN_Inclination"}


# ───────────────────────── network smoke test (opt-in) ─────────────────────────


@pytest.mark.slow
def test_repoimu_download_and_eval(tmp_path, monkeypatch):
    """Downloads RepoIMU (small GitHub zip), materializes it, and runs the
    per-source benchmark with a trivial model. Network-gated via the 'slow' mark.
    """
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))

    def build_model(context):
        def model(u, y_init, attrs):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(u), 1)).astype(np.float32)

        return model

    result = idb.run_benchmark(idb.BenchmarkRepoIMU_Inclination, build_model, seed=0)
    assert np.isfinite(result["metric_score"])
    assert result["custom_scores"]["all/incl_rmse_deg"] > 0
    assert len(idb.BenchmarkRepoIMU_Inclination.test_files) == 21
