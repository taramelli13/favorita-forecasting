"""Abstract base model interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract interface for all forecasting models."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weight: np.ndarray | None = None,
        categorical_feature: list[int] | None = None,
        feature_name: list[str] | None = None,
    ) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: Path) -> "BaseModel":
        """Load model from disk."""

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
