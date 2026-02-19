"""Model evaluation and error analysis."""

from pathlib import Path

import numpy as np
import polars as pl
import structlog

from favorita_forecasting.config import load_settings
from favorita_forecasting.evaluation.cross_validation import (
    df_to_numpy,
    encode_categoricals,
    get_feature_columns,
)
from favorita_forecasting.evaluation.metrics import mae, nwrmsle, rmse, rmsle
from favorita_forecasting.models.lightgbm_model import LightGBMModel

logger = structlog.get_logger()


def analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics."""
    return {
        "nwrmsle": nwrmsle(y_true, y_pred, weights),
        "rmsle": rmsle(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }


def run_evaluation(model_path: str | None = None) -> None:
    """Evaluate a trained model on validation data."""
    settings = load_settings()
    model_path = model_path or str(settings.paths.models / "lightgbm_final.txt")

    model = LightGBMModel()
    model.load(Path(model_path))

    # Load features with date filter
    from datetime import date
    features_path = settings.paths.data_processed / "features_train.parquet"
    min_date = date.fromisoformat(settings.data.min_train_date)
    df = pl.scan_parquet(str(features_path)).filter(
        pl.col("date") >= min_date
    ).collect()

    # Encode categoricals
    df, _ = encode_categoricals(df)

    # Fill nulls
    if "dcoilwtico" in df.columns:
        df = df.with_columns(pl.col("dcoilwtico").fill_null(strategy="forward").fill_null(0))
    if "transactions" in df.columns:
        df = df.with_columns(pl.col("transactions").fill_null(0))

    # Use last 16 days as validation
    max_date = df["date"].max()
    val_df = df.filter(pl.col("date") > max_date - pl.duration(days=16))

    feature_cols = get_feature_columns(val_df)
    X_val = df_to_numpy(val_df, feature_cols)
    y_val = val_df["unit_sales"].to_numpy().astype(np.float32)

    preds = model.predict(X_val)
    metrics = analyze_predictions(y_val, preds)

    for name, value in metrics.items():
        logger.info("evaluation_metric", metric=name, value=f"{value:.6f}")

    # Feature importance
    importance = model.get_feature_importance()
    top_features = dict(list(importance.items())[:20])
    logger.info("top_20_features", features=top_features)
