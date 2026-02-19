"""Optuna hyperparameter optimization — memory-safe implementation.

Uses numpy arrays loaded once via _load_to_numpy() and zero-copy CV slicing,
identical to train_pipeline.py. SQLite storage allows resuming interrupted studies.
"""

import gc
import json
import time
from datetime import date

import numpy as np
import optuna
import structlog

from favorita_forecasting.config import load_settings
from favorita_forecasting.data.splitter import ExpandingWindowCV
from favorita_forecasting.evaluation.metrics import nwrmsle
from favorita_forecasting.models.lightgbm_model import LightGBMModel
from favorita_forecasting.models.xgboost_model import XGBoostModel
from favorita_forecasting.pipeline.train_pipeline import _load_to_numpy
from favorita_forecasting.tracking.mlflow_tracker import ExperimentTracker

logger = structlog.get_logger()

MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
}


def _lightgbm_search_space(trial: optuna.Trial) -> dict:
    """Define LightGBM hyperparameter search space."""
    return {
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.01, 1.99),
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),
        "max_depth": -1,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 3000,
        "early_stopping_rounds": 50,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
        "device": "cpu",  # GPU bug with 90+ features
    }


def _xgboost_search_space(trial: optuna.Trial) -> dict:
    """Define XGBoost hyperparameter search space."""
    return {
        "objective": "reg:tweedie",
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.01, 1.99),
        "eval_metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 3000,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "device": "cuda",
        "seed": 42,
        "verbosity": 0,
    }


SEARCH_SPACES = {
    "lightgbm": _lightgbm_search_space,
    "xgboost": _xgboost_search_space,
}


def _precompute_fold_indices(
    dates: np.ndarray,
    n_splits: int,
    forecast_horizon: int,
) -> list[tuple[int, int, int]]:
    """Pre-compute CV fold indices using numpy searchsorted (zero-copy)."""
    cv = ExpandingWindowCV(n_splits=n_splits, forecast_horizon=forecast_horizon)
    splits = cv.get_splits()

    fold_indices = []
    for split in splits:
        train_end_np = np.datetime64(split.train_end)
        val_start_np = np.datetime64(split.val_start)
        val_end_np = np.datetime64(split.val_end)

        train_idx_end = int(np.searchsorted(dates, train_end_np, side="right"))
        val_idx_start = int(np.searchsorted(dates, val_start_np, side="left"))
        val_idx_end = int(np.searchsorted(dates, val_end_np, side="right"))

        fold_indices.append((train_idx_end, val_idx_start, val_idx_end))

    return fold_indices


def run_hyperopt(
    model_type: str = "lightgbm",
    n_trials: int = 100,
    timeout: int | None = None,
) -> optuna.Study:
    """Run Optuna hyperparameter optimization with memory-safe numpy CV.

    Data is loaded once to numpy arrays and CV folds use zero-copy slices.
    SQLite storage allows resuming interrupted studies.
    """
    t0 = time.time()
    settings = load_settings()

    model_class = MODEL_REGISTRY[model_type]
    search_space_fn = SEARCH_SPACES[model_type]

    # --- Step 1: Load data ONCE to numpy ---
    features_path = str(settings.paths.data_processed / "features_train.parquet")
    min_date = date.fromisoformat(settings.data.min_train_date)

    logger.info("hyperopt_loading_data", path=features_path, min_date=str(min_date))
    X, y, dates, weights, feature_cols, cat_indices = _load_to_numpy(
        features_path, min_date=min_date
    )
    logger.info(
        "hyperopt_data_loaded",
        X_shape=X.shape,
        X_gb=f"{X.nbytes / 1e9:.2f}",
    )

    # --- Step 2: Pre-compute fold indices (zero-copy) ---
    fold_indices = _precompute_fold_indices(
        dates,
        n_splits=settings.train.cv_splits,
        forecast_horizon=settings.train.forecast_horizon,
    )
    logger.info("hyperopt_folds_precomputed", n_folds=len(fold_indices))

    for fi, (t_end, v_start, v_end) in enumerate(fold_indices):
        logger.info(
            "fold_info",
            fold=fi + 1,
            train_rows=t_end,
            val_rows=v_end - v_start,
        )

    # --- MLflow tracking ---
    tracker = ExperimentTracker(experiment_name="favorita-tuning")

    # --- Step 3: Objective function ---
    def objective(trial: optuna.Trial) -> float:
        params = search_space_fn(trial)
        fold_scores = []

        for fi, (t_end, v_start, v_end) in enumerate(fold_indices):
            # Zero-copy numpy slices
            X_train_cv = X[:t_end]
            y_train_cv = y[:t_end]
            X_val_cv = X[v_start:v_end]
            y_val_cv = y[v_start:v_end]
            w_val_cv = weights[v_start:v_end] if weights is not None else None

            model = model_class(params=params.copy())
            model.fit(
                X_train_cv,
                y_train_cv,
                eval_set=(X_val_cv, y_val_cv),
                categorical_feature=cat_indices if cat_indices else None,
                feature_name=feature_cols,
                verbose_eval=0,
            )

            preds = model.predict(X_val_cv)
            score = nwrmsle(y_val_cv, preds, w_val_cv)
            fold_scores.append(score)

            # Report running mean for pruning
            trial.report(np.mean(fold_scores), fi)
            if trial.should_prune():
                del model, preds
                gc.collect()
                raise optuna.TrialPruned()

            del model, preds
            gc.collect()

        mean_score = float(np.mean(fold_scores))

        # Log trial as nested MLflow run
        with tracker.nested_run(run_name=f"trial-{trial.number}"):
            tracker.log_params(params)
            tracker.log_metrics({"cv_mean_nwrmsle": mean_score})
            for fi2, s in enumerate(fold_scores):
                tracker.log_metrics({f"cv_fold_{fi2 + 1}_nwrmsle": s})

        return mean_score

    # --- Step 4: Create study with SQLite persistence ---
    metrics_dir = settings.paths.project_root / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    db_path = metrics_dir / f"optuna_{model_type}.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        storage=storage,
        study_name=f"favorita-{model_type}",
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_completed > 0:
        logger.info(
            "hyperopt_resuming",
            completed_trials=n_completed,
            best_value=f"{study.best_value:.6f}",
        )

    # Use timeout from params.yaml if not passed explicitly
    effective_timeout = timeout if timeout is not None else settings.tuning.timeout

    logger.info(
        "hyperopt_starting",
        model_type=model_type,
        n_trials=n_trials,
        timeout=effective_timeout,
        storage=str(db_path),
    )

    # Wrap study.optimize in a parent MLflow run
    with tracker.start_run(run_name=f"tuning-{model_type}"):
        tracker.set_tags({
            "model_type": model_type,
            "study_name": f"favorita-{model_type}",
            "n_trials_requested": str(n_trials),
        })

        study.optimize(objective, n_trials=n_trials, timeout=effective_timeout)

        # --- Step 5: Save best params to JSON ---
        elapsed = time.time() - t0
        n_complete = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        n_pruned = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        )

        # Log summary to parent run
        tracker.log_metrics({
            "best_value": study.best_value,
            "n_complete": float(n_complete),
            "n_pruned": float(n_pruned),
        })
        tracker.log_params({"best_" + k: v for k, v in study.best_params.items()})

        best_params_path = metrics_dir / f"best_params_{model_type}.json"
        best_result = {
            "model_type": model_type,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials_complete": n_complete,
            "n_trials_pruned": n_pruned,
            "elapsed_seconds": round(elapsed, 1),
        }
        with open(best_params_path, "w") as f:
            json.dump(best_result, f, indent=2)

        tracker.log_artifact(best_params_path)

    logger.info(
        "hyperopt_complete",
        best_value=f"{study.best_value:.6f}",
        best_params=study.best_params,
        n_complete=n_complete,
        n_pruned=n_pruned,
        elapsed=f"{elapsed / 60:.1f} min",
        saved_to=str(best_params_path),
    )

    return study
