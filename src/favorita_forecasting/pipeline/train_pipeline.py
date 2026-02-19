"""End-to-end training pipeline with memory-efficient GPU support."""

import gc
import json
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from favorita_forecasting.config import load_settings
from favorita_forecasting.data.splitter import ExpandingWindowCV
from favorita_forecasting.evaluation.metrics import compute_perishable_weights, nwrmsle
from favorita_forecasting.models.lightgbm_model import LightGBMModel
from favorita_forecasting.models.xgboost_model import XGBoostModel
from favorita_forecasting.tracking.mlflow_tracker import ExperimentTracker

logger = structlog.get_logger()

MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
}

# Columns that are never needed during training
_SKIP_COLS = {"id", "unit_sales_log1p", "city", "state", "store_nbr", "item_nbr", "description"}

# Non-feature columns (target, metadata)
_NON_FEATURE_COLS = {"date", "unit_sales", "perishable"}


def _copy_to_local(src: str) -> str:
    """Copy parquet to a local temp directory for faster I/O (avoids OneDrive overhead)."""
    src_path = Path(src)
    local_dir = Path(tempfile.gettempdir()) / "favorita_cache"
    local_dir.mkdir(exist_ok=True)
    local_path = local_dir / src_path.name

    # Only copy if source is newer or local doesn't exist
    if not local_path.exists() or src_path.stat().st_mtime > local_path.stat().st_mtime:
        logger.info("copying_parquet_to_local", src=src, dst=str(local_path))
        shutil.copy2(src, local_path)

    return str(local_path)


def _forward_fill_sorted(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values in a sorted (by date) numpy array."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = arr.copy()
    out[mask] = arr[idx[mask]]
    return np.nan_to_num(out, nan=0.0)


def _load_to_numpy(
    features_path: str,
    min_date: date | None = None,
    batch_size: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, list[str], list[int]]:
    """Load training features directly to numpy with minimal peak memory.

    Reads the parquet in column batches to avoid holding the full DataFrame
    and the full numpy array simultaneously.

    Returns: (X, y, dates, weights, feature_cols, cat_indices)
    """
    # Copy to local temp for fast I/O
    local_path = _copy_to_local(features_path)

    schema = pl.read_parquet_schema(local_path)
    keep_cols = [c for c in schema if c not in _SKIP_COLS]

    # Determine feature columns
    feature_cols = [c for c in keep_cols if c not in _NON_FEATURE_COLS]
    n_features = len(feature_cols)

    # Identify categorical (string) columns
    cat_indices = []
    cat_cols = set()
    for i, col in enumerate(feature_cols):
        if schema[col] in (pl.Utf8, pl.String):
            cat_indices.append(i)
            cat_cols.add(col)

    # --- Pass 1: Read date column → filter, sort order ---
    date_series = pl.read_parquet(local_path, columns=["date"])["date"]
    date_np = date_series.to_numpy()
    del date_series

    # Build combined filter + sort index
    if min_date is not None:
        min_date_np = np.datetime64(min_date)
        row_mask = date_np >= min_date_np
        filtered_dates = date_np[row_mask]
    else:
        row_mask = None
        filtered_dates = date_np

    sort_order = filtered_dates.argsort(kind="stable")
    dates = filtered_dates[sort_order]
    n_rows = len(dates)

    # Combined index: filter then sort in one step
    if row_mask is not None:
        all_idx = np.arange(len(date_np), dtype=np.int32)
        filtered_idx = all_idx[row_mask]
        combined_idx = filtered_idx[sort_order].astype(np.int32)
        del all_idx, filtered_idx
    else:
        combined_idx = sort_order.astype(np.int32)

    del date_np, filtered_dates, sort_order
    gc.collect()
    logger.info("loaded_dates", n_rows=n_rows, n_features=n_features)

    # --- Pass 2: Read target + weights ---
    meta_cols = ["unit_sales"]
    if "perishable" in schema:
        meta_cols.append("perishable")
    meta = pl.read_parquet(local_path, columns=meta_cols)
    y = meta["unit_sales"].fill_null(0).to_numpy().astype(np.float32)[combined_idx]
    weights = (
        compute_perishable_weights(meta["perishable"].to_numpy()[combined_idx])
        if "perishable" in meta.columns
        else None
    )
    del meta
    gc.collect()

    # --- Pass 3: Pre-allocate X and read features in column batches ---
    X = np.empty((n_rows, n_features), dtype=np.float32)

    for bi in range(0, n_features, batch_size):
        batch_cols = feature_cols[bi : bi + batch_size]
        batch_df = pl.read_parquet(local_path, columns=batch_cols)

        for j, col in enumerate(batch_cols):
            series = batch_df[col]

            if col in cat_cols:
                # String → integer code (LightGBM needs integer categoricals)
                series = series.cast(pl.Categorical).to_physical().cast(pl.Int32)
                X[:, bi + j] = series.fill_null(0).to_numpy().astype(np.float32)[combined_idx]

            elif col == "dcoilwtico":
                # Oil: forward fill by date order, then fill remaining with 0
                raw = series.to_numpy().astype(np.float64)[combined_idx]
                X[:, bi + j] = _forward_fill_sorted(raw).astype(np.float32)

            elif series.dtype == pl.Boolean:
                X[:, bi + j] = series.cast(pl.Int8).fill_null(0).to_numpy().astype(np.float32)[combined_idx]

            elif series.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                # Integer features: fill null with 0 (flags, counts, etc.)
                X[:, bi + j] = series.fill_null(0).to_numpy().astype(np.float32)[combined_idx]

            else:
                # Float features (lags, rolling, target encoding): keep NaN
                # LightGBM handles NaN natively as missing values
                X[:, bi + j] = series.to_numpy().astype(np.float32)[combined_idx]

        del batch_df
        gc.collect()

    del combined_idx
    gc.collect()

    logger.info(
        "loaded_to_numpy",
        X_gb=f"{X.nbytes / 1e9:.2f}",
        n_features=n_features,
        n_categorical=len(cat_indices),
    )

    return X, y, dates, weights, feature_cols, cat_indices


def run_training(model_type: str = "lightgbm", skip_cv: bool = False) -> None:
    """Train a model with optional cross-validation and save the final model."""
    t0 = time.time()
    settings = load_settings()

    features_path = str(settings.paths.data_processed / "features_train.parquet")
    min_date = date.fromisoformat(settings.data.min_train_date)

    # --- Step 1: Load data directly to numpy ---
    logger.info("loading_features", path=features_path, min_date=str(min_date))
    X, y, dates, weights, feature_cols, cat_indices = _load_to_numpy(
        features_path, min_date=min_date
    )
    logger.info(
        "numpy_conversion_done",
        X_shape=X.shape,
        X_gb=f"{X.nbytes / 1e9:.2f}",
    )

    # Get model class and params
    model_class = MODEL_REGISTRY[model_type]
    model_config = getattr(settings.train, model_type, None)
    model_params = model_config.model_dump() if model_config else None

    # --- MLflow tracking ---
    tracker = ExperimentTracker()

    with tracker.start_run(run_name=f"train-{model_type}"):
        # Log params and tags
        if model_params:
            tracker.log_params(model_params)
        tracker.set_tags({
            "model_type": model_type,
            "min_train_date": str(min_date),
            "n_features": len(feature_cols),
            "n_rows": X.shape[0],
            "skip_cv": str(skip_cv),
        })

        # --- Step 2: Optional cross-validation (numpy zero-copy slices) ---
        cv_results = None
        if not skip_cv:
            cv = ExpandingWindowCV(
                n_splits=settings.train.cv_splits,
                forecast_horizon=settings.train.forecast_horizon,
            )
            splits = cv.get_splits()

            fold_metrics = []
            best_iterations = []

            for split in splits:
                logger.info(
                    "cv_fold_start",
                    fold=split.fold,
                    train_end=str(split.train_end),
                    val_start=str(split.val_start),
                    val_end=str(split.val_end),
                )

                # Convert Python dates to numpy datetime64 for searchsorted
                train_end_np = np.datetime64(split.train_end)
                val_start_np = np.datetime64(split.val_start)
                val_end_np = np.datetime64(split.val_end)

                # Zero-copy slicing on the sorted dates array
                train_idx_end = int(np.searchsorted(dates, train_end_np, side="right"))
                val_idx_start = int(np.searchsorted(dates, val_start_np, side="left"))
                val_idx_end = int(np.searchsorted(dates, val_end_np, side="right"))

                X_train_cv = X[:train_idx_end]
                y_train_cv = y[:train_idx_end]
                w_train_cv = weights[:train_idx_end] if weights is not None else None

                X_val_cv = X[val_idx_start:val_idx_end]
                y_val_cv = y[val_idx_start:val_idx_end]
                w_val_cv = weights[val_idx_start:val_idx_end] if weights is not None else None

                logger.info(
                    "cv_fold_data",
                    fold=split.fold,
                    train_rows=len(y_train_cv),
                    val_rows=len(y_val_cv),
                )

                if len(y_val_cv) == 0:
                    logger.warning("cv_fold_empty_validation", fold=split.fold)
                    continue

                # Train fold model
                fold_model = model_class(params=model_params)
                fold_model.fit(
                    X_train_cv,
                    y_train_cv,
                    eval_set=(X_val_cv, y_val_cv),
                    sample_weight=w_train_cv,
                    categorical_feature=cat_indices if cat_indices else None,
                    feature_name=feature_cols,
                )

                # Evaluate
                preds = fold_model.predict(X_val_cv)
                fold_nwrmsle = nwrmsle(y_val_cv, preds, w_val_cv)
                fold_metrics.append(fold_nwrmsle)
                best_iterations.append(fold_model.best_iteration)

                # Log fold metrics to MLflow
                tracker.log_cv_fold(split.fold, {"nwrmsle": fold_nwrmsle})
                tracker.log_metrics(
                    {f"cv_fold_{split.fold}_best_iter": fold_model.best_iteration}
                )

                logger.info(
                    "cv_fold_complete",
                    fold=split.fold,
                    nwrmsle=f"{fold_nwrmsle:.6f}",
                    best_iteration=fold_model.best_iteration,
                )

                del fold_model, preds
                gc.collect()

            mean_score = float(np.mean(fold_metrics))
            logger.info(
                "cv_complete", mean_nwrmsle=f"{mean_score:.6f}", fold_scores=fold_metrics
            )

            # Log CV summary to MLflow
            tracker.log_metrics({"cv_mean_nwrmsle": mean_score})

            cv_results = {
                "fold_metrics": fold_metrics,
                "mean_nwrmsle": mean_score,
                "best_iterations": best_iterations,
            }

            # Save CV results
            metrics_dir = settings.paths.project_root / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            cv_results_path = metrics_dir / f"cv_results_{model_type}.json"
            with open(cv_results_path, "w") as f:
                json.dump(
                    {
                        "model_type": model_type,
                        "mean_nwrmsle": cv_results["mean_nwrmsle"],
                        "fold_metrics": cv_results["fold_metrics"],
                        "best_iterations": cv_results["best_iterations"],
                    },
                    f,
                    indent=2,
                )
            tracker.log_artifact(cv_results_path)

        # --- Step 3: Split train/val for final model (last 16 days) ---
        max_date = dates.max()
        val_start = max_date - np.timedelta64(15, "D")
        split_idx = np.searchsorted(dates, val_start)

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        w_train = weights[:split_idx] if weights is not None else None
        w_val = weights[split_idx:] if weights is not None else None
        del X, y, dates, weights
        gc.collect()

        logger.info(
            "train_val_split",
            train_rows=len(y_train),
            val_rows=len(y_val),
            train_gb=f"{X_train.nbytes / 1e9:.2f}",
            val_gb=f"{X_val.nbytes / 1e9:.2f}",
        )

        # --- Step 4: Train final model ---
        logger.info("training_final_model", model_type=model_type)

        if cv_results is not None:
            valid_iters = [b for b in cv_results["best_iterations"] if b is not None]
            final_n_estimators = int(np.mean(valid_iters) * 1.1) if valid_iters else 3000
        else:
            final_n_estimators = (
                model_params.get("n_estimators", 3000) if model_params else 3000
            )

        final_params = {**(model_params or {}), "n_estimators": final_n_estimators}

        final_model = model_class(params=final_params)
        final_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            sample_weight=w_train,
            categorical_feature=cat_indices if cat_indices else None,
            feature_name=feature_cols,
        )

        # Evaluate on validation set
        val_preds = final_model.predict(X_val)
        val_nwrmsle = nwrmsle(y_val, val_preds, w_val)
        logger.info("final_model_validation", nwrmsle=f"{val_nwrmsle:.6f}")

        # Log final metrics to MLflow
        tracker.log_metrics({"final_val_nwrmsle": val_nwrmsle})

        # --- Step 5: Save model and metrics ---
        model_ext = ".txt" if model_type == "lightgbm" else ".json"
        model_path = settings.paths.models / f"{model_type}_final{model_ext}"
        settings.paths.models.mkdir(parents=True, exist_ok=True)
        final_model.save(model_path)
        tracker.log_artifact(model_path)

        elapsed = time.time() - t0
        logger.info(
            "training_complete",
            model_path=str(model_path),
            cv_mean_nwrmsle=f"{cv_results['mean_nwrmsle']:.6f}" if cv_results else "skipped",
            final_val_nwrmsle=f"{val_nwrmsle:.6f}",
            n_estimators=final_n_estimators,
            elapsed_seconds=f"{elapsed:.1f}",
        )

        # Save and log feature importance
        importance = final_model.get_feature_importance()
        metrics_dir = settings.paths.project_root / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        importance_path = metrics_dir / f"{model_type}_importance.json"
        with open(importance_path, "w") as f:
            json.dump(importance, f, indent=2)
        logger.info("top_10_features", features=dict(list(importance.items())[:10]))

        tracker.log_model_info(model_type, importance)
        tracker.log_artifact(importance_path)
