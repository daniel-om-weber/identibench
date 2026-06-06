"""Tests for the RIANN orientation-estimation datasets and benchmarks."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import identibench as idb
from identibench.datasets import riann as R


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
        for i, c in enumerate(R.RIANN_Y_COLS):
            f.create_dataset(c, data=true_quat[:, i].astype(np.float32))
        f.create_dataset("movement_mask", data=np.asarray(mask, dtype=np.float32))


# ───────────────────────── metric ─────────────────────────


def test_alignment_removes_constant_offset():
    q = _rand_quats(400)
    offset = _quat_x(25.0)
    pred = R._qmult(offset[None, :], q)  # constant rotation applied to all samples
    assert R.aligned_inclination_rmse_deg(pred, q) == pytest.approx(0.0, abs=1e-6)


def test_time_varying_tilt_is_measured():
    q = _rand_quats(400)
    pred = q.copy()
    # 30 deg tilt on the second half only -> a real residual after sample-0 alignment
    pred[200:] = R._qmult(_quat_x(30.0)[None, :], q[200:])
    rmse = R.aligned_inclination_rmse_deg(pred, q)
    # 30 deg on half the samples -> 30*sqrt(1/2) ≈ 21.2 deg RMS
    assert rmse == pytest.approx(30.0 * np.sqrt(0.5), rel=0.02)


def test_nan_ground_truth_ignored():
    q = _rand_quats(300)
    q_nan = q.copy()
    q_nan[10:40] = np.nan
    assert R.aligned_inclination_rmse_deg(q, q_nan) == pytest.approx(0.0, abs=1e-6)


def test_riann_eval_masks_and_groups(tmp_path):
    q = _rand_quats(200)
    pred = q.copy()
    pred[50:60] = R._qmult(_quat_x(20.0)[None, :], q[50:60])  # error only here

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


def test_registration():
    for name in ["riann", "broad", "oxiod", "euroc", "repoimu", "caruso", "tumvi"]:
        assert name in idb.datasets.all_dataset_loaders
    assert "RIANN_Inclination" in idb.simulation_benchmarks
    assert set(R.riann_benchmarks) >= {"RIANN_Inclination", "OxIOD_Inclination"}


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

    result = idb.run_benchmark(R.BenchmarkRepoIMU_Inclination, build_model, seed=0)
    assert np.isfinite(result["metric_score"])
    assert result["custom_scores"]["all/incl_rmse_deg"] > 0
    assert len(R.BenchmarkRepoIMU_Inclination.test_files) == 21
