"""Data cleaning: handle negatives, missing values, and outliers."""

import polars as pl
import structlog

logger = structlog.get_logger()


def clean_train(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the training data.

    - Clip negative unit_sales to 0 (negatives are returns)
    - Fill null onpromotion with False (not tracked before 2014)
    """
    return lf.with_columns(
        pl.col("unit_sales").clip(lower_bound=0).alias("unit_sales"),
        pl.col("onpromotion").fill_null(False).alias("onpromotion"),
    )


def clean_oil(df: pl.DataFrame) -> pl.DataFrame:
    """Clean oil prices: forward-fill then backward-fill gaps."""
    return df.sort("date").with_columns(
        pl.col("dcoilwtico").forward_fill().backward_fill().alias("dcoilwtico")
    )


def clean_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure transactions are sorted by date and store."""
    return df.sort(["date", "store_nbr"])


def clean_all(
    train_lf: pl.LazyFrame,
    oil: pl.DataFrame,
    transactions: pl.DataFrame,
) -> tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame]:
    """Apply all cleaning steps."""
    logger.info("cleaning_data")
    return (
        clean_train(train_lf),
        clean_oil(oil),
        clean_transactions(transactions),
    )
