"""Tests for identibench.metrics module."""

import numpy as np
import pytest
from identibench.metrics import rmse, nrmse, fit_index, mae, r_squared, inclination_rmse_deg, orientation_rmse_deg


# --- Test data ---

@pytest.fixture
def data_1d():
    y_true = np.array([1., 2., 3., 4., 5.])
    return {
        'y_true': y_true,
        'y_perfect': np.array([1., 2., 3., 4., 5.]),
        'y_offset': np.array([1.1, 2.1, 3.1, 4.1, 5.1]),  # Error=0.1
        'y_noise': np.array([1.1, 1.9, 3.1, 4.1, 4.9]),  # Errors: 0.1, -0.1, 0.1, 0.1, -0.1
        'y_neg_offset': np.array([0., 1., 2., 3., 4.]),  # Error=-1.0
    }


@pytest.fixture
def data_2d():
    return {
        'y_true': np.array([[1., 10.], [2., 20.], [3., 30.]]),  # time x features
        'y_perfect': np.array([[1., 10.], [2., 20.], [3., 30.]]),
        'y_offset': np.array([[1., 11.], [2., 21.], [3., 31.]]),  # Error=[0, 1]
        'y_mixed_err': np.array([[1.1, 10.], [1.9, 21.], [3.1, 30.]]),
    }


@pytest.fixture
def data_const():
    return {
        'y_true': np.array([5., 5., 5.]),
        'y_err': np.array([5., 5., 6.]),  # Error=[0, 0, 1]
    }


# --- RMSE Tests ---

class TestRMSE:
    def test_perfect_1d(self, data_1d):
        np.testing.assert_allclose(rmse(data_1d['y_true'], data_1d['y_perfect']), 0.0, atol=1e-10)

    def test_offset_1d(self, data_1d):
        np.testing.assert_allclose(rmse(data_1d['y_true'], data_1d['y_offset']), 0.1, atol=1e-10)

    def test_perfect_2d(self, data_2d):
        np.testing.assert_allclose(rmse(data_2d['y_true'], data_2d['y_perfect'], time_axis=0), np.array([0., 0.]), atol=1e-10)

    def test_offset_2d(self, data_2d):
        np.testing.assert_allclose(rmse(data_2d['y_true'], data_2d['y_offset'], time_axis=0), np.array([0., 1.]), atol=1e-10)


# --- NRMSE Tests ---

class TestNRMSE:
    def test_offset_1d(self, data_1d):
        std_1d = np.std(data_1d['y_true'])
        expected = 0.1 / std_1d
        np.testing.assert_allclose(nrmse(data_1d['y_true'], data_1d['y_offset']), expected, atol=1e-10)

    def test_offset_2d(self, data_2d):
        std_2d = np.std(data_2d['y_true'], axis=0)
        expected = np.array([0., 1.]) / std_2d
        np.testing.assert_allclose(nrmse(data_2d['y_true'], data_2d['y_offset'], time_axis=0), expected, atol=1e-10)

    def test_zero_std_returns_nan(self, data_const):
        with pytest.warns(RuntimeWarning, match="Standard deviation of y_true is below tolerance"):
            result = nrmse(data_const['y_true'], data_const['y_err'])
        assert np.isnan(result).all()


# --- Fit Index Tests ---

class TestFitIndex:
    def test_offset_1d(self, data_1d):
        std_1d = np.std(data_1d['y_true'])
        nrmse_expected = 0.1 / std_1d
        expected = 100.0 * (1.0 - nrmse_expected)
        np.testing.assert_allclose(fit_index(data_1d['y_true'], data_1d['y_offset']), expected, atol=1e-10)

    def test_offset_2d(self, data_2d):
        std_2d = np.std(data_2d['y_true'], axis=0)
        nrmse_expected = np.array([0., 1.]) / std_2d
        expected = 100.0 * (1.0 - nrmse_expected)
        np.testing.assert_allclose(fit_index(data_2d['y_true'], data_2d['y_offset'], time_axis=0), expected, atol=1e-10)

    def test_zero_std_returns_nan(self, data_const):
        with pytest.warns(RuntimeWarning, match="Standard deviation of y_true is below tolerance"):
            result = fit_index(data_const['y_true'], data_const['y_err'])
        assert np.isnan(result).all()


# --- MAE Tests ---

class TestMAE:
    def test_perfect_1d(self, data_1d):
        np.testing.assert_allclose(mae(data_1d['y_true'], data_1d['y_perfect']), 0.0, atol=1e-10)

    def test_offset_1d(self, data_1d):
        np.testing.assert_allclose(mae(data_1d['y_true'], data_1d['y_offset']), 0.1, atol=1e-10)

    def test_neg_offset_1d(self, data_1d):
        np.testing.assert_allclose(mae(data_1d['y_true'], data_1d['y_neg_offset']), 1.0, atol=1e-10)

    def test_noise_1d(self, data_1d):
        np.testing.assert_allclose(mae(data_1d['y_true'], data_1d['y_noise']), np.mean([0.1, 0.1, 0.1, 0.1, 0.1]), atol=1e-10)

    def test_perfect_2d(self, data_2d):
        np.testing.assert_allclose(mae(data_2d['y_true'], data_2d['y_perfect'], time_axis=0), np.array([0., 0.]), atol=1e-10)

    def test_offset_2d(self, data_2d):
        np.testing.assert_allclose(mae(data_2d['y_true'], data_2d['y_offset'], time_axis=0), np.array([0., 1.]), atol=1e-10)

    def test_mixed_err_2d(self, data_2d):
        # abs errors are [[0.1, 0], [0.1, 1], [0.1, 0]]
        # Mean along axis 0: [mean(0.1, 0.1, 0.1), mean(0, 1, 0)] = [0.1, 1/3]
        np.testing.assert_allclose(mae(data_2d['y_true'], data_2d['y_mixed_err'], time_axis=0), np.array([0.1, 1./3.]), atol=1e-10)


# --- R-squared Tests ---

class TestRSquared:
    def test_perfect_1d(self, data_1d):
        np.testing.assert_allclose(r_squared(data_1d['y_true'], data_1d['y_perfect']), 1.0, atol=1e-10)

    def test_offset_1d(self, data_1d):
        std_1d = np.std(data_1d['y_true'])
        nrmse_expected = 0.1 / std_1d
        expected = 1.0 - nrmse_expected**2
        np.testing.assert_allclose(r_squared(data_1d['y_true'], data_1d['y_offset']), expected, atol=1e-10)

    def test_offset_2d(self, data_2d):
        std_2d = np.std(data_2d['y_true'], axis=0)
        nrmse_expected = np.array([0., 1.]) / std_2d
        expected = 1.0 - nrmse_expected**2
        np.testing.assert_allclose(r_squared(data_2d['y_true'], data_2d['y_offset'], time_axis=0), expected, atol=1e-10)

    def test_zero_std_returns_nan(self, data_const):
        with pytest.warns(RuntimeWarning, match="Standard deviation of y_true is below tolerance"):
            result = r_squared(data_const['y_true'], data_const['y_err'])
        assert np.isnan(result).all()


# --- Inclination RMSE Tests ---

class TestInclinationRmseDeg:
    def test_identical_quaternions(self):
        q = np.array([[1, 0, 0, 0]] * 10, dtype=np.float64)
        assert inclination_rmse_deg(q, q) == pytest.approx(0.0, abs=1e-10)

    def test_90_deg_tilt_about_x(self):
        """90-degree rotation about x-axis: q = [cos(45°), sin(45°), 0, 0]."""
        q_id = np.array([[1, 0, 0, 0]] * 10, dtype=np.float64)
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        q_90 = np.array([[c, s, 0, 0]] * 10, dtype=np.float64)
        assert inclination_rmse_deg(q_id, q_90) == pytest.approx(90.0, abs=0.1)

    def test_pure_heading_rotation_gives_zero(self):
        """Rotation purely about z-axis should give zero inclination error."""
        q_id = np.array([[1, 0, 0, 0]] * 10, dtype=np.float64)
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        q_z90 = np.array([[c, 0, 0, s]] * 10, dtype=np.float64)
        assert inclination_rmse_deg(q_id, q_z90) == pytest.approx(0.0, abs=1e-6)

    def test_antipodal_quaternions(self):
        """q and -q represent the same rotation, error should be zero."""
        q = np.array([[1, 0, 0, 0]] * 5, dtype=np.float64)
        assert inclination_rmse_deg(q, -q) == pytest.approx(0.0, abs=1e-6)

    def test_wrong_last_dim_raises(self):
        with pytest.raises(ValueError, match="last dimension 4"):
            inclination_rmse_deg(np.zeros((10, 3)), np.zeros((10, 3)))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Input shapes must match"):
            inclination_rmse_deg(np.zeros((10, 4)), np.zeros((5, 4)))

    def test_returns_float(self):
        q = np.random.randn(50, 4)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        assert isinstance(inclination_rmse_deg(q, q), float)


# --- Orientation RMSE Tests ---

class TestOrientationRmseDeg:
    def test_identical_quaternions(self):
        q = np.array([[1, 0, 0, 0]] * 10, dtype=np.float64)
        assert orientation_rmse_deg(q, q) == pytest.approx(0.0, abs=1e-10)

    def test_180_deg_about_z(self):
        """180-degree rotation about z-axis: q = [0, 0, 0, 1]."""
        q_id = np.array([[1, 0, 0, 0]] * 10, dtype=np.float64)
        q_180z = np.array([[0, 0, 0, 1]] * 10, dtype=np.float64)
        assert orientation_rmse_deg(q_id, q_180z) == pytest.approx(180.0, abs=0.1)

    def test_90_deg_about_x(self):
        q_id = np.array([[1, 0, 0, 0]] * 10, dtype=np.float64)
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        q_90 = np.array([[c, s, 0, 0]] * 10, dtype=np.float64)
        assert orientation_rmse_deg(q_id, q_90) == pytest.approx(90.0, abs=0.1)

    def test_antipodal_quaternions(self):
        q = np.array([[1, 0, 0, 0]] * 5, dtype=np.float64)
        assert orientation_rmse_deg(q, -q) == pytest.approx(0.0, abs=1e-6)

    def test_wrong_last_dim_raises(self):
        with pytest.raises(ValueError, match="last dimension 4"):
            orientation_rmse_deg(np.zeros((10, 3)), np.zeros((10, 3)))

    def test_returns_float(self):
        q = np.random.randn(50, 4)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        assert isinstance(orientation_rmse_deg(q, q), float)
