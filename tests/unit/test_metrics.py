"""Tests for evaluation metrics."""

import numpy as np
import pytest

from favorita_forecasting.evaluation.metrics import (
    compute_perishable_weights,
    mae,
    nwrmsle,
    rmse,
    rmsle,
)


def test_nwrmsle_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert nwrmsle(y, y) == pytest.approx(0.0, abs=1e-10)


def test_nwrmsle_with_weights():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    weights = np.array([1.0, 1.25])

    result = nwrmsle(y_true, y_pred, weights)
    assert result > 0
    assert isinstance(result, float)


def test_nwrmsle_clips_negative_predictions():
    y_true = np.array([5.0, 10.0])
    y_pred = np.array([-1.0, 10.0])
    result = nwrmsle(y_true, y_pred)
    assert result > 0  # -1 clipped to 0, so error exists


def test_rmsle_symmetric():
    y = np.array([1.0, 5.0, 10.0])
    assert rmsle(y, y) == pytest.approx(0.0, abs=1e-10)


def test_rmse_known_value():
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([1.0, 6.0])
    # ((3-1)^2 + (4-6)^2) / 2 = (4+4)/2 = 4 -> sqrt(4) = 2
    assert rmse(y_true, y_pred) == pytest.approx(2.0)


def test_mae_known_value():
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([1.0, 6.0])
    # (|3-1| + |4-6|) / 2 = (2+2)/2 = 2
    assert mae(y_true, y_pred) == pytest.approx(2.0)


def test_perishable_weights():
    perishable = np.array([0, 1, 0, 1])
    weights = compute_perishable_weights(perishable)
    assert weights.tolist() == [1.0, 1.25, 1.0, 1.25]
