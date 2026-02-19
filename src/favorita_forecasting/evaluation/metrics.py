"""Evaluation metrics for sales forecasting.

The primary metric is NWRMSLE (Normalized Weighted Root Mean Squared Logarithmic Error),
which is the official Kaggle Favorita competition metric.
"""

import numpy as np


def nwrmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Normalized Weighted Root Mean Squared Logarithmic Error.

    Perishable items should have weight=1.25, non-perishable weight=1.0.
    """
    if weights is None:
        weights = np.ones_like(y_true)

    y_pred_clipped = np.maximum(y_pred, 0)
    log_diff = np.log1p(y_pred_clipped) - np.log1p(np.maximum(y_true, 0))
    weighted_sq = weights * (log_diff ** 2)
    return float(np.sqrt(np.sum(weighted_sq) / np.sum(weights)))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error (unweighted)."""
    y_pred_clipped = np.maximum(y_pred, 0)
    log_diff = np.log1p(y_pred_clipped) - np.log1p(np.maximum(y_true, 0))
    return float(np.sqrt(np.mean(log_diff ** 2)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_perishable_weights(perishable: np.ndarray) -> np.ndarray:
    """Compute sample weights: 1.25 for perishable, 1.0 for non-perishable."""
    return np.where(perishable, 1.25, 1.0).astype(np.float32)
