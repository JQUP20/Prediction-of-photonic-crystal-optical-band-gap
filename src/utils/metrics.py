"""
Evaluation metrics for photonic band gap prediction
"""

import numpy as np
import torch
from typing import Union


def calculate_band_gap(frequencies: np.ndarray, threshold: float = 0.01) -> float:
    """
    Calculate band gap from frequency array

    Args:
        frequencies: Array of frequencies (num_k_points, num_bands)
        threshold: Minimum gap size to consider

    Returns:
        Band gap value (largest gap found)
    """
    if frequencies.ndim == 1:
        frequencies = frequencies.reshape(-1, 1)

    max_gap = 0.0

    for band_idx in range(frequencies.shape[1] - 1):
        lower_band_max = np.max(frequencies[:, band_idx])
        upper_band_min = np.min(frequencies[:, band_idx + 1])

        gap = upper_band_min - lower_band_max

        if gap > threshold and gap > max_gap:
            max_gap = gap

    return max_gap


def mean_absolute_error(y_true: Union[np.ndarray, torch.Tensor],
                       y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate mean absolute error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true: Union[np.ndarray, torch.Tensor],
                      y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate mean squared error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: Union[np.ndarray, torch.Tensor],
                           y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate root mean squared error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true: Union[np.ndarray, torch.Tensor],
             y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate R² coefficient of determination

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def mean_absolute_percentage_error(y_true: Union[np.ndarray, torch.Tensor],
                                  y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate mean absolute percentage error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value (in percentage)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_predictions(y_true: Union[np.ndarray, torch.Tensor],
                        y_pred: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Evaluate predictions using multiple metrics

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of evaluation metrics
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }
