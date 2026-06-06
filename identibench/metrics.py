"""Metric functions for evaluating system identification models."""

__all__ = [
    "rmse",
    "nrmse",
    "fit_index",
    "mae",
    "r_squared",
    "inclination_rmse_deg",
    "orientation_rmse_deg",
    "aligned_inclination_rmse_deg",
]

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
    rmse_val = np.sqrt(np.mean((inp - targ) ** 2, axis=time_axis))
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
    mae_val = np.mean(np.abs(inp - targ), axis=time_axis)
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


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (the inverse for a unit quaternion). Shape: (..., 4), [w,x,y,z]."""
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b. Shapes broadcast on (..., 4), [w,x,y,z] convention."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_diff(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Difference quaternion q1 ⊗ inv(q2). Inputs shape: (..., 4), [w,x,y,z] convention."""
    return _quat_mul(_quat_normalize(q1), _quat_conj(_quat_normalize(q2)))


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


def _aligned_inclination_rad(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample inclination (tilt) error in radians, after removing the constant
    orientation offset between the two frames.

    The estimate is aligned to ground truth at the first sample where both are
    finite (``offset = gt0 ⊗ inv(est0)``, applied to all samples). This removes the
    fixed rotation between, e.g., an IMU's gravity-defined frame and an optical
    reference frame. Samples where either quaternion is non-finite are returned as
    NaN (so callers can mask them out).
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = min(len(pred), len(true))
    pred, true = pred[:n], true[:n]

    out = np.full(n, np.nan)
    finite = np.isfinite(pred).all(-1) & np.isfinite(true).all(-1)
    if not finite.any():
        return out

    i0 = int(np.argmax(finite))  # first finite index
    offset = _quat_mul(_quat_normalize(true[i0]), _quat_conj(_quat_normalize(pred[i0])))  # (4,)
    pred_aligned = _quat_mul(offset[None, :], _quat_normalize(pred))  # (N, 4)
    ang = _inclination_angle(pred_aligned, true)  # (N,) radians
    out[finite] = ang[finite]
    return out


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


def aligned_inclination_rmse_deg(
    inp: np.ndarray,  # Predicted quaternions, shape (N, 4), [w, x, y, z].
    targ: np.ndarray,  # Ground truth quaternions, shape (N, 4), [w, x, y, z].
) -> float:  # RMS inclination error in degrees after first-sample alignment.
    """
    Computes the RMS inclination (tilt) error in degrees after first-sample alignment.

    Like :func:`inclination_rmse_deg`, but first removes the constant orientation
    offset between the two frames by aligning the estimate to ground truth at the
    first finite sample. Use this when the prediction lives in a different fixed
    reference frame than the target (e.g. an IMU gravity frame vs. an optical mocap
    frame). Non-finite ground-truth samples are ignored; no movement masking is
    applied.
    """
    incl = _aligned_inclination_rad(inp, targ)
    incl = incl[np.isfinite(incl)]
    if incl.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(incl**2)) * 180.0 / np.pi)


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
