"""Prediction pipeline: generate forecasts on test data."""

from pathlib import Path

import numpy as np
import polars as pl
import structlog

from favorita_forecasting.config import load_settings
from favorita_forecasting.models.lightgbm_model import LightGBMModel
from favorita_forecasting.models.xgboost_model import XGBoostModel
from favorita_forecasting.pipeline.submission import create_submission

logger = structlog.get_logger()


def run_prediction(model_path: str | None = None) -> None:
    """Load test features, run model inference, and create submission."""
    settings = load_settings()

    # Determine model type and path
    if model_path is None:
        model_type = settings.train.model_type
        ext = ".txt" if model_type == "lightgbm" else ".json"
        model_path = str(settings.paths.models / f"{model_type}_final{ext}")
    else:
        model_type = "lightgbm" if model_path.endswith(".txt") else "xgboost"

    # Load model
    logger.info("loading_model", path=model_path)
    if model_type == "lightgbm":
        model = LightGBMModel()
    else:
        model = XGBoostModel()
    model.load(Path(model_path))

    # Load test features
    test_path = settings.paths.data_processed / "features_test.parquet"
    logger.info("loading_test_features", path=str(test_path))
    test_df = pl.read_parquet(test_path)

    # Use model's feature names to ensure correct column ordering
    feature_cols = model.feature_names
    logger.info("feature_columns", n_features=len(feature_cols))

    # Encode categoricals and build numpy array column-by-column
    # (same logic as _load_to_numpy in train_pipeline)
    n_rows = len(test_df)
    X_test = np.empty((n_rows, len(feature_cols)), dtype=np.float32)

    for j, col in enumerate(feature_cols):
        series = test_df[col]

        if series.dtype in (pl.Utf8, pl.String):
            series = series.cast(pl.Categorical).to_physical().cast(pl.Int32)
            X_test[:, j] = series.fill_null(0).to_numpy().astype(np.float32)

        elif col == "dcoilwtico":
            raw = series.to_numpy().astype(np.float64)
            # Forward fill then fill remaining with 0
            mask = np.isnan(raw)
            if mask.any():
                idx = np.where(~mask, np.arange(len(raw)), 0)
                np.maximum.accumulate(idx, out=idx)
                raw[mask] = raw[idx[mask]]
                raw = np.nan_to_num(raw, nan=0.0)
            X_test[:, j] = raw.astype(np.float32)

        elif series.dtype == pl.Boolean:
            X_test[:, j] = series.cast(pl.Int8).fill_null(0).to_numpy().astype(np.float32)

        elif series.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
            X_test[:, j] = series.fill_null(0).to_numpy().astype(np.float32)

        else:
            # Float features: keep NaN (models handle natively)
            X_test[:, j] = series.to_numpy().astype(np.float32)

    # Generate predictions
    logger.info("generating_predictions", n_rows=X_test.shape[0])
    predictions = model.predict(X_test)

    # Create submission
    submission_df = pl.DataFrame({
        "id": test_df["id"],
        "unit_sales": predictions,
    })

    output_path = settings.paths.data_submissions / "submission.csv"
    create_submission(submission_df, output_path)
    logger.info("prediction_complete", output_path=str(output_path))
