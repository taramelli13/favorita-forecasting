"""LightGBM model wrapper using the native training API for memory efficiency."""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import structlog

from favorita_forecasting.models.base import BaseModel

logger = structlog.get_logger()

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
    "device": "gpu",
}


class LightGBMModel(BaseModel):
    """LightGBM wrapper using native lgb.train() API for memory efficiency and GPU support."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []
        self._n_estimators = self.params.pop("n_estimators", 3000)
        self._early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)
        # Remove sklearn-specific params that native API doesn't accept
        self.params.pop("random_state", None)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weight: np.ndarray | None = None,
        categorical_feature: list[int] | None = None,
        feature_name: list[str] | None = None,
        verbose_eval: int = 100,
    ) -> None:
        """Train LightGBM using native API with optional early stopping."""
        self.feature_names = feature_name or [f"f{i}" for i in range(X.shape[1])]
        train_data = lgb.Dataset(
            X, label=y, weight=sample_weight,
            feature_name=self.feature_names,
            categorical_feature=categorical_feature or "auto",
            free_raw_data=True,
        )

        callbacks: list = []
        if verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))
        valid_sets = [train_data]
        valid_names = ["train"]

        if eval_set is not None:
            X_val, y_val = eval_set
            val_data = lgb.Dataset(
                X_val, label=y_val, reference=train_data,
                free_raw_data=True,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self._early_stopping_rounds))

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self._n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.info(
            "lightgbm_training_complete",
            best_iteration=self.model.best_iteration,
            best_score=self.model.best_score.get("valid", {}).get("rmse"),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        assert self.model is not None, "Model not trained"
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        return np.maximum(preds, 0).astype(np.float32)

    def save(self, path: Path) -> None:
        """Save model to file (handles Unicode paths on Windows)."""
        assert self.model is not None, "Model not trained"
        model_str = self.model.model_to_string()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(model_str)
        logger.info("model_saved", path=str(path))

    def load(self, path: Path) -> "LightGBMModel":
        """Load model from file (handles Unicode paths on Windows)."""
        with open(path, encoding="utf-8") as f:
            model_str = f.read()
        self.model = lgb.Booster(model_str=model_str)
        # Restore feature names from loaded model
        self.feature_names = self.model.feature_name()
        logger.info("model_loaded", path=str(path))
        return self

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance as name->score dict."""
        assert self.model is not None, "Model not trained"
        importance = self.model.feature_importance(importance_type="gain")
        names = self.model.feature_name()
        return dict(sorted(zip(names, importance.tolist()), key=lambda x: -x[1]))

    def get_params(self) -> dict[str, Any]:
        return {
            **self.params,
            "n_estimators": self._n_estimators,
            "early_stopping_rounds": self._early_stopping_rounds,
        }

    @property
    def best_iteration(self) -> int | None:
        if self.model is None:
            return None
        return self.model.best_iteration
