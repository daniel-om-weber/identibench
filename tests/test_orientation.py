"""Tests for the orientation (IMU) datasets and benchmarks (RIANN family)."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import identibench as idb
from identibench.dataset import Dataset
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


# ───────────────────────── faithful evaluation task ─────────────────────────


def test_masked_pooled_inclination_masks_and_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    q = _rand_quats(200)
    pred = q.copy()
    pred[50:60] = _quat_mul(_quat_x(20.0)[None, :], q[50:60])  # error only here

    # movement_mask = 0 exactly over the erroneous window -> masked score is ~0
    mask = np.ones(200, dtype=np.float32)
    mask[50:60] = 0.0
    fpath = tmp_path / "caruso" / "slow_v4_AP1.hdf5"
    _write_riann_file(fpath, q, mask=mask)

    spec = _common._spec("Test_caruso", Dataset("caruso", prepare=None))
    model = lambda u, y_init, attrs: pred.astype(np.float32)
    result = spec.task(spec, model)

    assert result.headline == ("all", "incl_rmse_deg")
    assert "caruso" in result.scores
    assert set(result.scores["caruso"]) == {"incl_rmse_deg", "incl_p99_deg"}
    assert "all" in result.scores
    assert result.scores["caruso"]["incl_rmse_deg"] == pytest.approx(0.0, abs=1e-6)

    # Without masking the same error shows up.
    _write_riann_file(fpath, q, mask=np.ones(200, dtype=np.float32))
    result_unmasked = spec.task(spec, model)
    assert result_unmasked.scores["caruso"]["incl_rmse_deg"] > 1.0


def test_masked_pooled_inclination_pools_across_sets(tmp_path, monkeypatch):
    # Two single-file sets with disjoint error windows: "all" pools the samples.
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    q = _rand_quats(100, seed=2)
    pool = Dataset("pool", prepare=None)
    _write_riann_file(pool.path / "a.hdf5", q)
    _write_riann_file(pool.path / "b.hdf5", q)

    spec = idb.BenchmarkSpec(
        name="TestPool",
        u_cols=_common.IMU_U_COLS,
        y_cols=_common.IMU_Y_COLS,
        train=[],
        valid=[],
        test_sets={"A": [(pool, "a.hdf5")], "B": [(pool, "b.hdf5")]},
        task=_common.MaskedPooledInclination(),
    )
    model = lambda u, y_init, attrs: q.astype(np.float32)
    result = spec.task(spec, model)

    assert set(result.scores) == {"A", "B", "all"}
    assert result.scores["all"]["incl_rmse_deg"] == pytest.approx(0.0, abs=1e-4)


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


# ───────────────────────── wiring / pattern resolution ─────────────────────────


def _write_riann_layout(root: Path) -> None:
    """The flat per-source layout the preparers produce, complete enough that
    every RIANN train/valid/test pattern matches at least one file."""
    for i in range(1, 40):  # all 39 Myon trials, zero-padded prefixes
        _write_riann_file(root / "broad" / f"{i:02d}_trial.hdf5", _rand_quats(20))
    for n in range(1, 7):  # all six TUM-VI rooms
        _write_riann_file(root / "tumvi" / f"TumVI::room{n}.hdf5", _rand_quats(20))
    _write_riann_file(root / "oxiod" / "OxIOD::handheld_data1_1:fixed.hdf5", _rand_quats(20))
    _write_riann_file(root / "euroc" / "EurocMAV::V1_01_easy.hdf5", _rand_quats(20))
    _write_riann_file(root / "repoimu" / "RepoIMU::TStick_02_1.hdf5", _rand_quats(20))
    _write_riann_file(root / "caruso" / "Marco::slow_v4_AP1.hdf5", _rand_quats(20))


def test_riann_pattern_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _write_riann_layout(tmp_path)

    spec = R.BenchmarkRIANN_Inclination
    assert [ds.dataset_id for ds in spec.datasets] == ["broad", "caruso", "euroc", "oxiod", "repoimu", "tumvi"]
    assert len(spec.resolve(spec.train)) == 33 + 3  # Myon train trials + TUM-VI rooms 1-3
    assert len(spec.resolve(spec.valid)) == 3 + 3  # Myon valid trials + TUM-VI rooms 4-6
    test_sets = spec.test_set_files()
    assert set(test_sets) == {"broad", "oxiod", "euroc", "repoimu", "caruso"}
    assert len(test_sets["broad"]) == 3  # Myon test trials 22, 29, 35
    assert {f.name[:2] for f in test_sets["broad"]} == {"22", "29", "35"}
    assert sum(len(f) for f in test_sets.values()) == 7


def test_riann_split_is_disjoint_and_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _write_riann_layout(tmp_path)

    spec = R.BenchmarkRIANN_Inclination
    train = set(spec.resolve(spec.train))
    valid = set(spec.resolve(spec.valid))
    test = set(spec.test_files())
    assert train.isdisjoint(valid) and train.isdisjoint(test) and valid.isdisjoint(test)
    # Every Myon trial and every TUM-VI room is used exactly once.
    broad_files = set((tmp_path / "broad").glob("*.hdf5"))
    tumvi_files = set((tmp_path / "tumvi").glob("*.hdf5"))
    assert (train | valid | test) >= (broad_files | tumvi_files)


def test_run_benchmark_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _write_riann_layout(tmp_path)
    for ds in R.BenchmarkRIANN_Inclination.datasets:
        (ds.path / ".prepared").write_text("1")  # adopt the synthetic layout as prepared
    # Overwrite one source's test file with the identity quaternion as target.
    ident = np.tile([1.0, 0.0, 0.0, 0.0], (120, 1))
    _write_riann_file(tmp_path / "oxiod" / "OxIOD::handheld_data1_1:fixed.hdf5", ident)

    def build_model(context):
        def model(u, y_init, attrs):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(u), 1)).astype(np.float32)

        return model

    result = idb.run_benchmark(R.BenchmarkRIANN_Inclination, build_model, seed=0)

    # Headline = masked sample-pooled "all" inclination RMSE.
    assert result["benchmark_type"] == "MaskedPooledInclination"
    assert result["metric_name"] == "incl_rmse_deg"
    assert np.isfinite(result["metric_score"])
    sets = result["test_sets"]
    assert set(sets) == {"broad", "oxiod", "euroc", "repoimu", "caruso", "all"}
    # The identity-quaternion source scores ~0 against the identity model.
    assert sets["oxiod"]["incl_rmse_deg"] == pytest.approx(0.0, abs=1e-4)
    assert "incl_rmse_deg" in sets["all"] and "incl_p99_deg" in sets["all"]


def test_prepare_sources_uses_shared_raw_cache(tmp_path):
    """_prepare_sources downloads into the shared raw dir and converts flat into
    the dataset dir; force is forwarded to both steps."""
    calls = []

    def fake_download(raw, force=False):
        calls.append(("download", Path(raw).name, force))

    def fake_convert(raw, out_dir, force=False):
        calls.append(("convert", force))
        _write_riann_file(Path(out_dir) / "seq.hdf5", _rand_quats(20))

    save_path = tmp_path / "fake_ds"
    _common._prepare_sources(save_path, [(fake_download, fake_convert)])
    assert (save_path / "seq.hdf5").exists()  # flat, no role/source subdirs
    assert (tmp_path / "_orientation_raw").is_dir()  # shared raw cache next to the dataset
    assert calls == [("download", "_orientation_raw", False), ("convert", False)]

    calls.clear()
    _common._prepare_sources(save_path, [(fake_download, fake_convert)], force_download=True)
    assert calls == [("download", "_orientation_raw", True), ("convert", True)]


def test_registration():
    for name in ["dfjimu", "broad", "oxiod", "euroc", "repoimu", "caruso", "tumvi"]:
        assert name in idb.datasets.all_datasets
    assert "riann" not in idb.datasets.all_datasets  # riann is a benchmark, not a dataset
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
    assert result["test_sets"]["all"]["incl_rmse_deg"] > 0
    assert len(idb.BenchmarkRepoIMU_Inclination.test_set_files()["repoimu"]) == 21
