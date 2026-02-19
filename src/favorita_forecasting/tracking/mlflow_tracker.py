"""MLflow experiment tracking wrapper.

MLflow is an optional dependency. All tracking calls are no-ops if MLflow
is not installed, allowing the pipeline to work without it.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import structlog

logger = structlog.get_logger()

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Wrapper for MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "favorita-forecasting",
        tracking_uri: str = "sqlite:///mlflow.db",
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("mlflow_initialized", experiment=experiment_name)
        else:
            logger.warning("mlflow_not_available", msg="Install with: pip install mlflow")

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Generator[None, None, None]:
        """Context manager for an MLflow run."""
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name):
                yield
        else:
            yield

    @contextmanager
    def nested_run(self, run_name: str | None = None) -> Generator[None, None, None]:
        """Context manager for a nested MLflow run (child of current run)."""
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name, nested=True):
                yield
        else:
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if MLFLOW_AVAILABLE:
            # MLflow doesn't support nested dicts, flatten them
            flat = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}.{kk}"] = vv
                else:
                    flat[k] = v
            mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log evaluation metrics."""
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)

    def log_cv_fold(self, fold: int, metrics: dict[str, float]) -> None:
        """Log metrics for a single CV fold (prefixed with fold number)."""
        if MLFLOW_AVAILABLE:
            prefixed = {f"cv_fold_{fold}_{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(prefixed)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Set tags on the current run."""
        if MLFLOW_AVAILABLE:
            mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path) -> None:
        """Log a file artifact."""
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(path))

    def log_model_info(self, model_type: str, feature_importance: dict[str, float]) -> None:
        """Log model metadata."""
        if MLFLOW_AVAILABLE:
            mlflow.log_param("model_type", model_type)
            # Log top 20 features as metrics
            for name, score in list(feature_importance.items())[:20]:
                mlflow.log_metric(f"feature_importance_{name}", score)
