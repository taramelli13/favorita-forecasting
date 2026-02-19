"""Kaggle submission file generation."""

from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger()

EXPECTED_ROWS = 3_370_464


def create_submission(df: pl.DataFrame, output_path: Path) -> None:
    """Format and save predictions as a Kaggle submission CSV.

    Expected format: id, unit_sales (matching sample_submission.csv).
    """
    # Validate
    assert "id" in df.columns, "Missing 'id' column"
    assert "unit_sales" in df.columns, "Missing 'unit_sales' column"

    # Clip negative predictions
    submission = df.select(
        pl.col("id").cast(pl.Int64),
        pl.col("unit_sales").clip(lower_bound=0),
    )

    # Check for NaN/null
    null_count = submission["unit_sales"].null_count()
    nan_count = submission["unit_sales"].is_nan().sum()
    if null_count > 0 or nan_count > 0:
        logger.warning("null_predictions_found", null=null_count, nan=nan_count)
        submission = submission.with_columns(
            pl.col("unit_sales").fill_null(0).fill_nan(0)
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(output_path)

    logger.info(
        "submission_created",
        path=str(output_path),
        n_rows=submission.shape[0],
        mean_sales=f"{submission['unit_sales'].mean():.2f}",
    )
