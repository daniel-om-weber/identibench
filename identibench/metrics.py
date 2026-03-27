"""Metric functions for evaluating system identification models."""

__all__ = ["rmse", "nrmse", "fit_index", "mae", "r_squared", "inclination_rmse_deg", "orientation_rmse_deg"]

import numpy as np
import warnings


def rmse(
    inp: np.ndarray,  # Predicted / estimated values.
    targ: np.ndarray,  # Ground truth target values.
    time_axis: int = 0,  # Axis representing time or samples.
) -> np.ndarray:  # Root Mean Squared Error for each channel.
    """
    Computes the Root Mean Square Error (RMSE) along a specified time axis.

    Calculates RMSE = sqrt(mean((inp - targ)**2)) separately for each channel
    defined by the remaining axes.
    """
    inp = np.asarray(inp)
    targ = np.asarray(targ)
    if inp.shape != targ.shape:
        raise ValueError(f"Input shapes must match. Got {inp.shape} and {targ.shape}")

    # Ensure time_axis is valid
    if not (0 <= time_axis < inp.ndim):
        raise ValueError(f"Invalid time_axis {time_axis} for array with {inp.ndim} dimensions")

    # Calculate RMSE
    try:
        rmse_val = np.sqrt(np.mean((inp - targ) ** 2, axis=time_axis))
    except FloatingPointError as e:
        warnings.warn(f"Floating point error during RMSE calculation: {e}. Check for NaNs or Infs.", RuntimeWarning)
        raise e
    return rmse_val


def nrmse(
    inp: np.ndarray,  # Predicted / estimated values.
    targ: np.ndarray,  # Ground truth target values.
    time_axis: int = 0,  # Axis representing time or samples.
    std_tolerance: float = 1e-9,  # Minimum standard deviation allowed for targ to avoid division by zero.
) -> np.ndarray:  # Normalized Root Mean Squared Error for each channel.
    """
    Computes the Normalized Root Mean Square Error (NRMSE).

    Calculates NRMSE = RMSE / std(targ) separately for each channel.
    Returns NaN for channels where std(targ) is close to zero (below std_tolerance).
    """
    rmse_val = rmse(inp, targ, time_axis=time_axis)
    std_targ = np.std(targ, axis=time_axis)

    # Initialize nrmse_val with NaNs or another placeholder
    nrmse_val = np.full_like(std_targ, fill_value=np.nan, dtype=np.float64)

    # Identify channels with standard deviation above the tolerance
    valid_std_mask = std_targ > std_tolerance

    # Calculate NRMSE only for valid channels
    if np.any(valid_std_mask):
        nrmse_val[valid_std_mask] = rmse_val[valid_std_mask] / std_targ[valid_std_mask]

    # Warn if any channels had std below tolerance
    if not np.all(valid_std_mask):
        warnings.warn(
            f"Standard deviation of targ is below tolerance ({std_tolerance}) for some channels. NRMSE set to NaN for these channels.",
            RuntimeWarning,
        )

    return nrmse_val


def fit_index(
    inp: np.ndarray,  # Predicted / estimated values.
    targ: np.ndarray,  # Ground truth target values.
    time_axis: int = 0,  # Axis representing time or samples.
    std_tolerance: float = 1e-9,  # Minimum standard deviation allowed for targ.
) -> np.ndarray:  # Fit index (in percent) for each channel.
    """
    Computes the Fit Index (FIT) commonly used in System Identification.

    Calculates FIT = 100 * (1 - NRMSE) separately for each channel.
    Returns NaN for channels where NRMSE could not be calculated (e.g., std(targ) near zero).
    """
    nrmse_val = nrmse(inp, targ, time_axis=time_axis, std_tolerance=std_tolerance)

    # Fit index calculation, handles potential NaNs from nrmse
    fit_val = 100.0 * (1.0 - nrmse_val)

    return fit_val


def mae(
    inp: np.ndarray,  # Predicted / estimated values.
    targ: np.ndarray,  # Ground truth target values.
    time_axis: int = 0,  # Axis representing time or samples.
) -> np.ndarray:  # Mean Absolute Error for each channel.
    """
    Computes the Mean Absolute Error (MAE) along a specified time axis.

    Calculates MAE = mean(abs(inp - targ)) separately for each channel
    defined by the remaining axes.
    """
    inp = np.asarray(inp)
    targ = np.asarray(targ)
    if inp.shape != targ.shape:
        raise ValueError(f"Input shapes must match. Got {inp.shape} and {targ.shape}")

    # Ensure time_axis is valid
    if not (0 <= time_axis < inp.ndim):
        raise ValueError(f"Invalid time_axis {time_axis} for array with {inp.ndim} dimensions")

    # Calculate MAE
    try:
        mae_val = np.mean(np.abs(inp - targ), axis=time_axis)
    except FloatingPointError as e:
        warnings.warn(f"Floating point error during MAE calculation: {e}. Check for NaNs or Infs.", RuntimeWarning)
        raise e
    return mae_val


def r_squared(
    inp: np.ndarray,  # Predicted / estimated values.
    targ: np.ndarray,  # Ground truth target values.
    time_axis: int = 0,  # Axis representing time or samples.
    std_tolerance: float = 1e-9,  # Minimum standard deviation allowed for targ.
) -> np.ndarray:  # R-squared (coefficient of determination) for each channel.
    """
    Computes the R-squared (coefficient of determination) score.

    Calculates R^2 = 1 - NRMSE^2 separately for each channel.
    Returns NaN for channels where NRMSE could not be calculated (e.g., std(targ) near zero).
    A constant model that always predicts the mean of targ would get R^2=0.
    """
    nrmse_val = nrmse(inp, targ, time_axis=time_axis, std_tolerance=std_tolerance)

    r2 = 1.0 - nrmse_val**2

    return r2


# --- Quaternion orientation metrics ---


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions to unit norm. Input shape: (..., 4)."""
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _quat_diff(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute difference quaternion q1 * inv(q2). Inputs shape: (..., 4), [w,x,y,z] convention."""
    q1 = _quat_normalize(q1)
    q2 = _quat_normalize(q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    ow = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    ox = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
    oy = -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2
    oz = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2
    return np.stack([ow, ox, oy, oz], axis=-1)


def _inclination_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Inclination (tilt) angle in radians between two quaternion arrays."""
    q = _quat_diff(q1, q2)
    return 2 * np.arctan2(
        np.sqrt(q[..., 1] ** 2 + q[..., 2] ** 2),
        np.sqrt(q[..., 0] ** 2 + q[..., 3] ** 2),
    )


def _relative_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Full 3D rotation angle in radians between two quaternion arrays."""
    q = _quat_diff(q1, q2)
    return 2 * np.arctan2(
        np.linalg.norm(q[..., 1:], axis=-1),
        np.abs(q[..., 0]),
    )


def inclination_rmse_deg(
    inp: np.ndarray,  # Predicted quaternions, shape (N, 4), [w, x, y, z].
    targ: np.ndarray,  # Ground truth quaternions, shape (N, 4), [w, x, y, z].
) -> float:  # RMS inclination error in degrees.
    """
    Computes the RMS inclination (tilt) error in degrees between two quaternion time series.

    Measures only the tilt component of orientation error, ignoring heading.
    Uses atan2 for numerical stability.
    """
    inp = np.asarray(inp, dtype=np.float64)
    targ = np.asarray(targ, dtype=np.float64)
    if inp.shape != targ.shape:
        raise ValueError(f"Input shapes must match. Got {inp.shape} and {targ.shape}")
    if inp.shape[-1] != 4:
        raise ValueError(f"Expected quaternion arrays with last dimension 4, got {inp.shape[-1]}")
    angles_rad = _inclination_angle(inp, targ)
    return float(np.sqrt(np.mean(angles_rad**2)) * 180.0 / np.pi)


def orientation_rmse_deg(
    inp: np.ndarray,  # Predicted quaternions, shape (N, 4), [w, x, y, z].
    targ: np.ndarray,  # Ground truth quaternions, shape (N, 4), [w, x, y, z].
) -> float:  # RMS full rotation error in degrees.
    """
    Computes the RMS full 3D rotation error in degrees between two quaternion time series.

    Measures the complete rotation angle between predicted and true orientations.
    Uses atan2 for numerical stability.
    """
    inp = np.asarray(inp, dtype=np.float64)
    targ = np.asarray(targ, dtype=np.float64)
    if inp.shape != targ.shape:
        raise ValueError(f"Input shapes must match. Got {inp.shape} and {targ.shape}")
    if inp.shape[-1] != 4:
        raise ValueError(f"Expected quaternion arrays with last dimension 4, got {inp.shape[-1]}")
    angles_rad = _relative_angle(inp, targ)
    return float(np.sqrt(np.mean(angles_rad**2)) * 180.0 / np.pi)
