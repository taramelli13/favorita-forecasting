"""Weighted ensemble of multiple models."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
from scipy.optimize import minimize

from favorita_forecasting.evaluation.metrics import nwrmsle
from favorita_forecasting.models.base import BaseModel

logger = structlog.get_logger()


class WeightedEnsemble:
    """Combine predictions from multiple models with optimized weights."""

    def __init__(self) -> None:
        self.weights: np.ndarray = np.array([])
        self.models: list[BaseModel] = []

    def optimize_weights(
        self,
        predictions: list[np.ndarray],
        y_true: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Find optimal blend weights by minimizing NWRMSLE."""
        n_models = len(predictions)

        def objective(w: np.ndarray) -> float:
            # Normalize weights to sum to 1
            w_norm = w / w.sum()
            blended = sum(w_norm[i] * predictions[i] for i in range(n_models))
            return nwrmsle(y_true, blended, weights)

        # Initial equal weights
        x0 = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0)] * n_models
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        self.weights = result.x

        logger.info(
            "ensemble_weights_optimized",
            weights=self.weights.tolist(),
            nwrmsle=f"{result.fun:.6f}",
        )
        return self.weights

    def predict(
        self,
        models: list[BaseModel],
        X: pl.DataFrame,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate blended predictions from multiple models."""
        if weights is not None:
            self.weights = weights
        assert len(self.weights) == len(models), "Weights must match number of models"

        predictions = [model.predict(X) for model in models]
        blended = sum(self.weights[i] * predictions[i] for i in range(len(models)))
        return np.maximum(blended, 0)
