"""Expanding window cross-validation for time series."""

import gc

import numpy as np
import polars as pl
import structlog

from favorita_forecasting.data.splitter import ExpandingWindowCV
from favorita_forecasting.evaluation.metrics import compute_perishable_weights, nwrmsle
from favorita_forecasting.models.base import BaseModel

logger = structlog.get_logger()


# Columns to exclude from features
NON_FEATURE_COLS = {
    "id", "date", "store_nbr", "item_nbr", "unit_sales",
    "unit_sales_log1p", "perishable", "city", "state",
    "description",
}


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Determine feature columns by excluding non-feature columns."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def encode_categoricals(df: pl.DataFrame) -> tuple[pl.DataFrame, list[int]]:
    """Encode string/boolean columns as integers and return categorical indices.

    Returns the modified DataFrame and a list of column indices that are categorical.
    """
    feature_cols = get_feature_columns(df)
    cat_indices = []
    for i, col in enumerate(feature_cols):
        dtype = df[col].dtype
        if dtype == pl.Utf8 or dtype == pl.String:
            df = df.with_columns(
                pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(col)
            )
            cat_indices.append(i)
        elif dtype == pl.Boolean:
            df = df.with_columns(pl.col(col).cast(pl.Int8).alias(col))
    return df, cat_indices


def df_to_numpy(df: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Convert Polars DataFrame columns to a float32 numpy array."""
    return df.select(columns).to_numpy().astype(np.float32)


def run_cross_validation(
    df: pl.DataFrame,
    model_factory: type[BaseModel],
    model_params: dict | None = None,
    n_splits: int = 4,
    forecast_horizon: int = 16,
    target_col: str = "unit_sales",
) -> dict:
    """Run expanding window CV and return fold metrics."""
    cv = ExpandingWindowCV(n_splits=n_splits, forecast_horizon=forecast_horizon)
    splits = cv.split(df.lazy())

    # Encode categoricals once
    df, cat_indices = encode_categoricals(df)
    feature_cols = get_feature_columns(df)

    fold_metrics = []
    best_iterations = []

    for i, (train_lf, val_lf) in enumerate(splits, 1):
        logger.info("cv_fold_start", fold=i, n_splits=n_splits)

        train_df = train_lf.collect()
        val_df = val_lf.collect()

        # Convert to numpy float32
        X_train = df_to_numpy(train_df, feature_cols)
        y_train = train_df[target_col].to_numpy().astype(np.float32)
        X_val = df_to_numpy(val_df, feature_cols)
        y_val = val_df[target_col].to_numpy().astype(np.float32)

        # Compute sample weights
        if "perishable" in train_df.columns:
            train_weights = compute_perishable_weights(train_df["perishable"].to_numpy())
            val_weights = compute_perishable_weights(val_df["perishable"].to_numpy())
        else:
            train_weights = None
            val_weights = np.ones(len(y_val), dtype=np.float32)

        # Free Polars DataFrames
        del train_df, val_df
        gc.collect()

        # Train model
        model = model_factory(params=model_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            sample_weight=train_weights,
            categorical_feature=cat_indices if cat_indices else None,
        )

        # Predict and evaluate
        preds = model.predict(X_val)
        fold_nwrmsle = nwrmsle(y_val, preds, val_weights)

        fold_metrics.append(fold_nwrmsle)
        best_iterations.append(model.best_iteration)

        logger.info("cv_fold_complete", fold=i, nwrmsle=f"{fold_nwrmsle:.6f}")

        # Free fold data
        del X_train, y_train, X_val, y_val, preds, model
        gc.collect()

    mean_score = float(np.mean(fold_metrics))
    logger.info("cv_complete", mean_nwrmsle=f"{mean_score:.6f}", fold_scores=fold_metrics)

    return {
        "fold_metrics": fold_metrics,
        "mean_nwrmsle": mean_score,
        "best_iterations": best_iterations,
    }
