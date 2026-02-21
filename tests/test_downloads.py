"""Tests for dataset download pipeline, HDF5 writing utilities, and ensure_dataset_exists."""

import multiprocessing
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from nonlinear_benchmarks.utilities import Input_output_data

from identibench.benchmark import BenchmarkSpecSimulation
from identibench.metrics import rmse
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
    """A download function that always raises."""
    raise ValueError("intentional failure")


# ---------------------------------------------------------------------------
# 1. ensure_dataset_exists tests
# ---------------------------------------------------------------------------


class TestEnsureDatasetExists:
    """Tests for BenchmarkSpecBase.ensure_dataset_exists subprocess mechanism."""

    def _make_spec(self, data_root: Path, download_func=_dummy_dataset_loader, **kwargs):
        return BenchmarkSpecSimulation(
            name="test_spec",
            dataset_id="test_ds",
            u_cols=["u0"],
            y_cols=["y0"],
            metric_func=rmse,
            download_func=download_func,
            data_root=data_root,
            **kwargs,
        )

    def test_ensure_downloads_when_missing(self, tmp_path):
        """Dataset dir doesn't exist -> ensure_dataset_exists creates it via subprocess."""
        spec = self._make_spec(tmp_path)
        ds_path = tmp_path / "test_ds"
        assert not ds_path.exists()

        spec.ensure_dataset_exists()

        assert ds_path.is_dir()
        for subdir in ["train", "valid", "test"]:
            sub = ds_path / subdir
            assert sub.is_dir(), f"{subdir}/ not created"
            hdf5_files = list(sub.glob("*.hdf5"))
            assert len(hdf5_files) > 0, f"No HDF5 files in {subdir}/"

    def test_ensure_skips_when_exists(self, tmp_path):
        """Dataset dir already exists -> no subprocess spawned."""
        ds_path = tmp_path / "test_ds"
        ds_path.mkdir(parents=True)
        marker = ds_path / "marker.txt"
        marker.write_text("original")

        spec = self._make_spec(tmp_path)
        spec.ensure_dataset_exists()

        # Marker file should be untouched (download_func was never called)
        assert marker.read_text() == "original"

    def test_ensure_force_download(self, tmp_path):
        """force_download=True -> re-downloads even when dir exists."""
        spec = self._make_spec(tmp_path)
        ds_path = tmp_path / "test_ds"
        ds_path.mkdir(parents=True)
        # Place a marker that will be absent after re-download
        marker = ds_path / "marker.txt"
        marker.write_text("should_survive")

        spec.ensure_dataset_exists(force_download=True)

        # _dummy_dataset_loader creates train/valid/test dirs
        assert (ds_path / "train").is_dir()
        assert (ds_path / "valid").is_dir()
        assert (ds_path / "test").is_dir()

    def test_ensure_raises_on_failure(self, tmp_path):
        """download_func that raises -> RuntimeError with exit code info."""
        spec = self._make_spec(tmp_path, download_func=_failing_loader)

        with pytest.raises(RuntimeError, match="failed.*exit code"):
            spec.ensure_dataset_exists()

    def test_ensure_no_download_func(self, tmp_path, capsys):
        """download_func=None -> prints warning, doesn't crash."""
        spec = self._make_spec(tmp_path, download_func=None)
        spec.ensure_dataset_exists()

        captured = capsys.readouterr()
        assert "Warning" in captured.out


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
        test_data = Input_output_data(
            u=u[:50], y=y[:50], sampling_time=1 / 51200, name="test"
        )

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
# 4. Slow integration test
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

    def test_run_benchmark_cascaded_tanks(self, tmp_path):
        """Full end-to-end: download + run_benchmark with a dummy model."""
        from identibench.benchmark import BenchmarkSpecSimulation, run_benchmark
        from identibench.datasets.workshop import dl_cascaded_tanks

        spec = BenchmarkSpecSimulation(
            name="test_tanks",
            dataset_id="cascaded_tanks",
            u_cols=["u0"],
            y_cols=["y0"],
            metric_func=rmse,
            download_func=dl_cascaded_tanks,
            init_window=10,
            data_root=tmp_path,
        )

        def build_model(context):
            def model(u, y_init):
                return np.zeros((u.shape[0], len(context.spec.y_cols)))
            return model

        result = run_benchmark(spec, build_model)
        assert result["benchmark_name"] == "test_tanks"
        assert np.isfinite(result["metric_score"])
        assert result["training_time_seconds"] >= 0
        assert result["test_time_seconds"] >= 0
