"""XGBoost model wrapper using the native training API for memory efficiency."""

from pathlib import Path
from typing import Any

import numpy as np
import structlog
import xgboost as xgb

from favorita_forecasting.models.base import BaseModel

logger = structlog.get_logger()

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5,
    "eval_metric": "rmse",
    "learning_rate": 0.02,
    "max_depth": 8,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "tree_method": "hist",
    "device": "cuda",
    "seed": 42,
    "verbosity": 0,
}


class XGBoostModel(BaseModel):
    """XGBoost wrapper using native xgb.train() API for memory efficiency and GPU support."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: xgb.Booster | None = None
        self.feature_names: list[str] = []
        self._n_estimators = self.params.pop("n_estimators", 3000)
        self._early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)
        # Remove sklearn-specific params
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
        """Train XGBoost using native API with optional early stopping."""
        self.feature_names = feature_name or [f"f{i}" for i in range(X.shape[1])]
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight, feature_names=self.feature_names)

        evals = [(dtrain, "train")]
        if eval_set is not None:
            X_val, y_val = eval_set
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self._n_estimators,
            evals=evals,
            early_stopping_rounds=self._early_stopping_rounds,
            verbose_eval=verbose_eval if verbose_eval > 0 else False,
        )

        logger.info(
            "xgboost_training_complete",
            best_iteration=self.model.best_iteration,
            best_score=self.model.best_score,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        assert self.model is not None, "Model not trained"
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names or None)
        preds = self.model.predict(dmatrix, iteration_range=(0, self.model.best_iteration))
        return np.maximum(preds, 0).astype(np.float32)

    def save(self, path: Path) -> None:
        """Save model to file (handles Unicode paths on Windows)."""
        assert self.model is not None, "Model not trained"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.model.save_raw()
        with open(path, "wb") as f:
            f.write(raw)
        logger.info("model_saved", path=str(path))

    def load(self, path: Path) -> "XGBoostModel":
        """Load model from file (handles Unicode paths on Windows)."""
        self.model = xgb.Booster()
        with open(path, "rb") as f:
            self.model.load_model(bytearray(f.read()))
        # Restore feature names from loaded model
        if self.model.feature_names:
            self.feature_names = list(self.model.feature_names)
        logger.info("model_loaded", path=str(path))
        return self

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance as name->score dict."""
        assert self.model is not None, "Model not trained"
        importance = self.model.get_score(importance_type="gain")
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

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
