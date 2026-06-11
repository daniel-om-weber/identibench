"""Tests for dataset download pipeline, HDF5 writing utilities, and Dataset.ensure_exists."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from nonlinear_benchmarks.utilities import Input_output_data

from identibench.dataset import Dataset
from identibench.utils import _dummy_dataset_loader, dataset_to_hdf5, iodata_to_hdf5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iodata(n_samples: int = 100, name: str = "test_data", sampling_time: float = 0.01) -> Input_output_data:
    """Create a small Input_output_data object with known arrays."""
    u = np.random.default_rng(42).standard_normal(n_samples).astype(np.float32)
    y = np.random.default_rng(43).standard_normal(n_samples).astype(np.float32)
    return Input_output_data(u=u, y=y, sampling_time=sampling_time, name=name)


def _failing_loader(save_path: Path, force_download: bool = False) -> None:
    """A prepare function that always raises."""
    raise ValueError("intentional failure")


def _empty_loader(save_path: Path, force_download: bool = False) -> None:
    """A prepare function that exits cleanly without writing anything."""


# ---------------------------------------------------------------------------
# 1. Dataset.ensure_exists tests (subprocess mechanism)
# ---------------------------------------------------------------------------


class TestEnsureExists:
    def test_ensure_prepares_when_missing(self, tmp_path, monkeypatch):
        """Dataset dir doesn't exist -> ensure_exists creates it via subprocess and writes the sentinel."""
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        ds = Dataset("test_ds", prepare=_dummy_dataset_loader)
        assert not ds.path.exists()

        ds.ensure_exists()

        assert ds.path.is_dir()
        for subdir in ["train", "valid", "test"]:
            sub = ds.path / subdir
            assert sub.is_dir(), f"{subdir}/ not created"
            assert len(list(sub.glob("*.hdf5"))) > 0, f"No HDF5 files in {subdir}/"
        # The sentinel is written last, after a successful preparation.
        assert (ds.path / ".prepared").read_text() == ds.version

    def test_ensure_skips_when_sentinel_matches(self, tmp_path, monkeypatch):
        """Dataset dir with a matching sentinel -> no subprocess spawned."""
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        ds_path = tmp_path / "test_ds"
        ds_path.mkdir(parents=True)
        (ds_path / ".prepared").write_text("1")
        marker = ds_path / "marker.txt"
        marker.write_text("original")

        Dataset("test_ds", prepare=_failing_loader).ensure_exists()  # would raise if invoked

        assert marker.read_text() == "original"

    def test_ensure_reprepares_when_sentinel_missing(self, tmp_path, monkeypatch):
        """A cached dir without the sentinel (e.g. an interrupted preparation or an
        old-format cache) is cleared and re-prepared."""
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        ds_path = tmp_path / "test_ds"
        _dummy_dataset_loader(ds_path)
        stray = ds_path / "stray.txt"
        stray.write_text("leftover")

        Dataset("test_ds", prepare=_dummy_dataset_loader).ensure_exists()

        assert not stray.exists()  # clean slate
        assert (ds_path / ".prepared").is_file()
        assert (ds_path / "train").is_dir()

    def test_ensure_force(self, tmp_path, monkeypatch):
        """force=True -> re-prepares even with a matching sentinel."""
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        ds_path = tmp_path / "test_ds"
        ds_path.mkdir(parents=True)
        (ds_path / ".prepared").write_text("1")
        (ds_path / "marker.txt").write_text("stale")

        Dataset("test_ds", prepare=_dummy_dataset_loader).ensure_exists(force=True)

        assert not (ds_path / "marker.txt").exists()
        assert (ds_path / "train").is_dir()
        assert (ds_path / "valid").is_dir()
        assert (ds_path / "test").is_dir()

    def test_ensure_raises_on_failure(self, tmp_path, monkeypatch):
        """prepare that raises -> RuntimeError with exit code info, no sentinel."""
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        ds = Dataset("test_ds", prepare=_failing_loader)

        with pytest.raises(RuntimeError, match="failed.*exit code"):
            ds.ensure_exists()
        assert not (ds.path / ".prepared").exists()

    def test_ensure_raises_when_prepare_writes_nothing(self, tmp_path, monkeypatch):
        """prepare exiting 0 without creating the directory -> explicit error."""
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        ds = Dataset("test_ds", prepare=_empty_loader)

        with pytest.raises(RuntimeError, match="wrote nothing"):
            ds.ensure_exists()

    def test_ensure_refuses_to_clear_outside_data_root(self, tmp_path, monkeypatch):
        """A dataset dir that escapes the data root (e.g. via a symlink) is never cleared."""
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        (root / "test_ds").symlink_to(outside)
        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(root))

        with pytest.raises(RuntimeError, match="refusing to clear"):
            Dataset("test_ds", prepare=_dummy_dataset_loader).ensure_exists()
        assert outside.exists()


# ---------------------------------------------------------------------------
# 2. Direct utility tests (iodata_to_hdf5, dataset_to_hdf5)
# ---------------------------------------------------------------------------


class TestIodataToHdf5:
    """Tests for iodata_to_hdf5: write Input_output_data to HDF5 and read back."""

    def test_iodata_to_hdf5(self, tmp_path):
        iodata = _make_iodata(n_samples=50, name="my_signal", sampling_time=0.005)
        hdf_path = iodata_to_hdf5(iodata, tmp_path, f_name="my_signal")

        assert hdf_path.exists()
        assert hdf_path.suffix == ".hdf5"

        with h5py.File(hdf_path, "r") as f:
            assert "u0" in f
            assert "y0" in f
            np.testing.assert_array_equal(f["u0"][()].shape, (50,))
            np.testing.assert_array_equal(f["y0"][()].shape, (50,))
            assert f["u0"][()].dtype == np.float32
            assert f["y0"][()].dtype == np.float32
            assert f.attrs["fs"] == pytest.approx(1.0 / 0.005)

    def test_iodata_to_hdf5_default_name(self, tmp_path):
        """When f_name is None, uses iodata.name as filename."""
        iodata = _make_iodata(name="auto_name")
        hdf_path = iodata_to_hdf5(iodata, tmp_path)

        assert hdf_path.name == "auto_name.hdf5"

    def test_iodata_to_hdf5_2d(self, tmp_path):
        """Multi-channel input/output gets written as u0, u1, y0, y1, etc."""
        u = np.random.default_rng(0).standard_normal((30, 2)).astype(np.float32)
        y = np.random.default_rng(1).standard_normal((30, 3)).astype(np.float32)
        iodata = Input_output_data(u=u, y=y, sampling_time=0.01, name="multi")
        hdf_path = iodata_to_hdf5(iodata, tmp_path, f_name="multi")

        with h5py.File(hdf_path, "r") as f:
            assert "u0" in f and "u1" in f
            assert "y0" in f and "y1" in f and "y2" in f


class TestDatasetToHdf5:
    """Tests for dataset_to_hdf5: write train/valid/test splits."""

    def test_dataset_to_hdf5_basic(self, tmp_path):
        train = (_make_iodata(80, "tr0"), _make_iodata(80, "tr1"))
        valid = (_make_iodata(20, "va0"),)
        test = (_make_iodata(30, "te0"),)

        dataset_to_hdf5(train, valid, test, tmp_path)

        assert (tmp_path / "train").is_dir()
        assert (tmp_path / "valid").is_dir()
        assert (tmp_path / "test").is_dir()
        assert len(list((tmp_path / "train").glob("*.hdf5"))) == 2
        assert len(list((tmp_path / "valid").glob("*.hdf5"))) == 1
        assert len(list((tmp_path / "test").glob("*.hdf5"))) == 1

    def test_dataset_to_hdf5_deterministic_names(self, tmp_path):
        train = (_make_iodata(80, "tr0"), _make_iodata(80, "tr1"))
        valid = (_make_iodata(20, "va0"),)
        test = (_make_iodata(30, "te0"), _make_iodata(30, "te1"))

        dataset_to_hdf5(train, valid, test, tmp_path)

        # Enumerate order names — the contract benchmark patterns select on
        # (e.g. silverbox's test/test_0.hdf5 = multisine).
        assert sorted(p.name for p in (tmp_path / "train").glob("*.hdf5")) == ["train_0.hdf5", "train_1.hdf5"]
        assert sorted(p.name for p in (tmp_path / "test").glob("*.hdf5")) == ["test_0.hdf5", "test_1.hdf5"]

    def test_dataset_to_hdf5_with_train_valid(self, tmp_path):
        train = (_make_iodata(80, "tr"),)
        valid = (_make_iodata(20, "va"),)
        test = (_make_iodata(30, "te"),)
        train_valid = (_make_iodata(100, "tv"),)

        dataset_to_hdf5(train, valid, test, tmp_path, train_valid=train_valid)

        assert (tmp_path / "train_valid").is_dir()
        assert len(list((tmp_path / "train_valid").glob("*.hdf5"))) == 1

    def test_dataset_to_hdf5_single_iodata(self, tmp_path):
        """Passing a single Input_output_data (not a tuple) should also work."""
        single = _make_iodata(50, "single")
        dataset_to_hdf5(single, single, single, tmp_path)

        for subdir in ["train", "valid", "test"]:
            assert len(list((tmp_path / subdir).glob("*.hdf5"))) == 1


# ---------------------------------------------------------------------------
# 3. Mocked dl_wiener_hammerstein test
# ---------------------------------------------------------------------------


class TestDlWienerHammerstein:
    """Mock nonlinear_benchmarks.WienerHammerBenchMark and verify HDF5 output."""

    def test_dl_wiener_hammerstein_structure(self, tmp_path):
        n_samples = 200
        u = np.random.default_rng(0).standard_normal(n_samples).astype(np.float64)
        y = np.random.default_rng(1).standard_normal(n_samples).astype(np.float64)

        train_val = Input_output_data(u=u, y=y, sampling_time=1 / 51200, name="train_val")
        test_data = Input_output_data(u=u[:50], y=y[:50], sampling_time=1 / 51200, name="test")

        with patch("identibench.datasets.workshop.nonlinear_benchmarks.WienerHammerBenchMark") as mock_wh:
            mock_wh.return_value = (train_val, test_data)

            from identibench.datasets.workshop import dl_wiener_hammerstein

            save_path = tmp_path / "wh"
            dl_wiener_hammerstein(save_path, split_idx=100)

        for subdir in ["train", "valid", "test", "train_valid"]:
            d = save_path / subdir
            assert d.is_dir(), f"{subdir}/ not created"
            hdf5_files = list(d.glob("*.hdf5"))
            assert len(hdf5_files) >= 1, f"No HDF5 files in {subdir}/"

            # Verify HDF5 contents
            with h5py.File(hdf5_files[0], "r") as f:
                assert "u0" in f, f"u0 missing in {subdir}/"
                assert "y0" in f, f"y0 missing in {subdir}/"
                assert f["u0"][()].dtype == np.float32


# ---------------------------------------------------------------------------
# 4. dl_dfjimu tests (mocked)
# ---------------------------------------------------------------------------


class TestDlDfjimu:
    """Tests for dfjimu download and HDF5 conversion with mocked HTTP + scipy."""

    def test_dl_dfjimu_creates_flat_hdf5(self, tmp_path):
        from identibench.datasets.orientation.dfjimu import (
            dl_dfjimu,
            ALL_HDF5_FILES,
            ALL_HDF5_FILES_PERSENSOR,
            dfjimu_u_cols,
            dfjimu_y_q1_cols,
            dfjimu_y_q2_cols,
            dfjimu_y_rel_cols,
            dfjimu_u_generic,
            dfjimu_y_q_generic,
        )

        mock_data = MagicMock()
        mock_data.sensorData = np.random.default_rng(0).standard_normal((100, 12))
        mock_data.ref = np.random.default_rng(1).standard_normal((100, 17))
        mock_data.r_12 = np.array([0.1, 0.2, 0.3])
        mock_data.r_21 = np.array([0.4, 0.5, 0.6])
        mock_data.rate = 50.0

        mock_response = MagicMock()
        mock_response.content = b"fake"
        mock_response.raise_for_status = MagicMock()

        with patch("identibench.datasets.orientation.dfjimu.requests.get", return_value=mock_response):
            with patch("identibench.datasets.orientation.dfjimu.scipy.io.loadmat", return_value={"data": mock_data}):
                save_path = tmp_path / "dfjimu"
                dl_dfjimu(save_path, force_download=True)

        for fname in ALL_HDF5_FILES:
            fpath = save_path / fname
            assert fpath.exists(), f"{fname} not created"

            with h5py.File(fpath, "r") as f:
                for col in dfjimu_u_cols:
                    assert col in f, f"{col} missing in {fname}"
                for col in dfjimu_y_q1_cols + dfjimu_y_q2_cols + dfjimu_y_rel_cols:
                    assert col in f, f"{col} missing in {fname}"
                assert f.attrs["fs"] == 50.0
                assert "r_12" in f.attrs
                assert "r_21" in f.attrs

        for fname in ALL_HDF5_FILES_PERSENSOR:
            fpath = save_path / fname
            assert fpath.exists(), f"virtual {fname} not created"

            with h5py.File(fpath, "r") as f:
                for col in dfjimu_u_generic + dfjimu_y_q_generic:
                    assert col in f, f"{col} missing in virtual {fname}"
                    assert f[col].shape == (100,)
                assert f.attrs["fs"] == 50.0

    def test_dl_dfjimu_skips_existing(self, tmp_path):
        """When all HDF5 files exist and force_download=False, no download occurs."""
        from identibench.datasets.orientation.dfjimu import dl_dfjimu, ALL_HDF5_FILES, ALL_HDF5_FILES_PERSENSOR

        save_path = tmp_path / "dfjimu"
        save_path.mkdir()
        for fname in ALL_HDF5_FILES + ALL_HDF5_FILES_PERSENSOR:
            (save_path / fname).touch()

        with patch("identibench.datasets.orientation.dfjimu.requests.get") as mock_get:
            dl_dfjimu(save_path, force_download=False)
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Slow integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSlowIntegration:
    """Integration tests that download real data. Run with --slow."""

    def test_dl_cascaded_tanks_integration(self, tmp_path):
        from identibench.datasets.workshop import dl_cascaded_tanks

        dl_cascaded_tanks(tmp_path, force_download=False)

        for subdir in ["train", "valid", "test", "train_valid"]:
            d = tmp_path / subdir
            assert d.is_dir(), f"{subdir}/ not created"
            hdf5_files = list(d.glob("*.hdf5"))
            assert len(hdf5_files) >= 1, f"No HDF5 files in {subdir}/"

            with h5py.File(hdf5_files[0], "r") as f:
                assert "u0" in f
                assert "y0" in f
                assert f["u0"][()].dtype == np.float32
                assert f["y0"][()].dtype == np.float32

        # Verify valid data is first 160 samples (split_idx default)
        with h5py.File(list((tmp_path / "valid").glob("*.hdf5"))[0], "r") as f:
            valid_len = f["u0"][()].shape[0]

        # valid should be first 160 of train_val
        assert valid_len == 160

    def test_run_benchmark_cascaded_tanks(self, tmp_path, monkeypatch):
        """Full end-to-end: download + run_benchmark with a dummy model."""
        import dataclasses

        from identibench.benchmark import run_benchmark
        from identibench.datasets.workshop import BenchmarkCascadedTanks_Simulation

        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        spec = dataclasses.replace(BenchmarkCascadedTanks_Simulation, name="test_tanks")

        def build_model(context):
            def model(u, y, attrs):
                return np.zeros((u.shape[0], len(context.spec.y_cols)))

            return model

        result = run_benchmark(spec, build_model)
        assert result["benchmark_name"] == "test_tanks"
        assert np.isfinite(result["metric_score"])
        assert result["training_time_seconds"] >= 0
        assert result["test_time_seconds"] >= 0

    def test_dl_dfjimu_integration(self, tmp_path):
        """Download real dfjimu data and verify HDF5 + virtual files."""
        from identibench.datasets.orientation.dfjimu import (
            dl_dfjimu,
            ALL_HDF5_FILES,
            ALL_HDF5_FILES_PERSENSOR,
            dfjimu_u_cols,
            dfjimu_u_generic,
            dfjimu_y_q_generic,
        )

        dl_dfjimu(tmp_path, force_download=False)

        for fname in ALL_HDF5_FILES:
            fpath = tmp_path / fname
            assert fpath.exists(), f"{fname} not created"
            with h5py.File(fpath, "r") as f:
                for col in dfjimu_u_cols:
                    assert col in f

        for fname in ALL_HDF5_FILES_PERSENSOR:
            fpath = tmp_path / fname
            assert fpath.exists(), f"virtual {fname} not created"
            with h5py.File(fpath, "r") as f:
                for col in dfjimu_u_generic + dfjimu_y_q_generic:
                    assert col in f
                    assert f[col][()].dtype == np.float32

    def test_run_benchmark_dfjimu_inclination(self, tmp_path, monkeypatch):
        """Full end-to-end: download + run BenchmarkDFJIMU_Inclination with a dummy model."""
        import dataclasses

        from identibench.benchmark import run_benchmark
        from identibench.datasets.orientation.dfjimu import BenchmarkDFJIMU_Inclination

        monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
        spec = dataclasses.replace(BenchmarkDFJIMU_Inclination, name="test_dfjimu_inclination")

        def build_model(context):
            def model(u, y, attrs):
                # Return identity quaternions [1, 0, 0, 0]
                out = np.zeros((u.shape[0], len(context.spec.y_cols)))
                out[:, 0] = 1.0
                return out

            return model

        result = run_benchmark(spec, build_model)
        assert result["benchmark_name"] == "test_dfjimu_inclination"
        assert np.isfinite(result["metric_score"])
        assert result["training_time_seconds"] >= 0
        assert result["test_time_seconds"] >= 0
