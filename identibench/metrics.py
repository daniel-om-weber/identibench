"""Metric functions for evaluating system identification models."""

__all__ = ["rmse", "nrmse", "fit_index", "mae", "r_squared"]

import numpy as np
import warnings


def rmse(
    y_true: np.ndarray,  # Ground truth target values.
    y_pred: np.ndarray,  # Estimated target values.
    time_axis: int = 0,  # Axis representing time or samples.
) -> np.ndarray:  # Root Mean Squared Error for each channel.
    """
    Computes the Root Mean Square Error (RMSE) along a specified time axis.

    Calculates RMSE = sqrt(mean((y_pred - y_true)**2)) separately for each channel
    defined by the remaining axes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Input shapes must match. Got {y_true.shape} and {y_pred.shape}")

    # Ensure time_axis is valid
    if not (0 <= time_axis < y_true.ndim):
        raise ValueError(f"Invalid time_axis {time_axis} for array with {y_true.ndim} dimensions")

    # Calculate RMSE
    try:
        rmse_val = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=time_axis))
    except FloatingPointError as e:
        warnings.warn(f"Floating point error during RMSE calculation: {e}. Check for NaNs or Infs.", RuntimeWarning)
        raise e
    return rmse_val


def nrmse(
    y_true: np.ndarray,  # Ground truth target values.
    y_pred: np.ndarray,  # Estimated target values.
    time_axis: int = 0,  # Axis representing time or samples.
    std_tolerance: float = 1e-9,  # Minimum standard deviation allowed for y_true to avoid division by zero.
) -> np.ndarray:  # Normalized Root Mean Squared Error for each channel.
    """
    Computes the Normalized Root Mean Square Error (NRMSE).

    Calculates NRMSE = RMSE / std(y_true) separately for each channel.
    Returns NaN for channels where std(y_true) is close to zero (below std_tolerance).
    """
    rmse_val = rmse(y_true, y_pred, time_axis=time_axis)
    std_true = np.std(y_true, axis=time_axis)

    # Initialize nrmse_val with NaNs or another placeholder
    nrmse_val = np.full_like(std_true, fill_value=np.nan, dtype=np.float64)

    # Identify channels with standard deviation above the tolerance
    valid_std_mask = std_true > std_tolerance

    # Calculate NRMSE only for valid channels
    if np.any(valid_std_mask):
        nrmse_val[valid_std_mask] = rmse_val[valid_std_mask] / std_true[valid_std_mask]

    # Warn if any channels had std below tolerance
    if not np.all(valid_std_mask):
        warnings.warn(
            f"Standard deviation of y_true is below tolerance ({std_tolerance}) for some channels. NRMSE set to NaN for these channels.",
            RuntimeWarning,
        )

    return nrmse_val


def fit_index(
    y_true: np.ndarray,  # Ground truth target values.
    y_pred: np.ndarray,  # Estimated target values.
    time_axis: int = 0,  # Axis representing time or samples.
    std_tolerance: float = 1e-9,  # Minimum standard deviation allowed for y_true.
) -> np.ndarray:  # Fit index (in percent) for each channel.
    """
    Computes the Fit Index (FIT) commonly used in System Identification.

    Calculates FIT = 100 * (1 - NRMSE) separately for each channel.
    Returns NaN for channels where NRMSE could not be calculated (e.g., std(y_true) near zero).
    """
    nrmse_val = nrmse(y_true, y_pred, time_axis=time_axis, std_tolerance=std_tolerance)

    # Fit index calculation, handles potential NaNs from nrmse
    fit_val = 100.0 * (1.0 - nrmse_val)

    return fit_val


def mae(
    y_true: np.ndarray,  # Ground truth target values.
    y_pred: np.ndarray,  # Estimated target values.
    time_axis: int = 0,  # Axis representing time or samples.
) -> np.ndarray:  # Mean Absolute Error for each channel.
    """
    Computes the Mean Absolute Error (MAE) along a specified time axis.

    Calculates MAE = mean(abs(y_pred - y_true)) separately for each channel
    defined by the remaining axes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Input shapes must match. Got {y_true.shape} and {y_pred.shape}")

    # Ensure time_axis is valid
    if not (0 <= time_axis < y_true.ndim):
        raise ValueError(f"Invalid time_axis {time_axis} for array with {y_true.ndim} dimensions")

    # Calculate MAE
    try:
        mae_val = np.mean(np.abs(y_pred - y_true), axis=time_axis)
    except FloatingPointError as e:
        warnings.warn(f"Floating point error during MAE calculation: {e}. Check for NaNs or Infs.", RuntimeWarning)
        raise e
    return mae_val


def r_squared(
    y_true: np.ndarray,  # Ground truth target values.
    y_pred: np.ndarray,  # Estimated target values.
    time_axis: int = 0,  # Axis representing time or samples.
    std_tolerance: float = 1e-9,  # Minimum standard deviation allowed for y_true.
) -> np.ndarray:  # R-squared (coefficient of determination) for each channel.
    """
    Computes the R-squared (coefficient of determination) score.

    Calculates R^2 = 1 - NRMSE^2 separately for each channel.
    Returns NaN for channels where NRMSE could not be calculated (e.g., std(y_true) near zero).
    A constant model that always predicts the mean of y_true would get R^2=0.
    """
    nrmse_val = nrmse(y_true, y_pred, time_axis=time_axis, std_tolerance=std_tolerance)

    r2 = 1.0 - nrmse_val**2

    return r2
