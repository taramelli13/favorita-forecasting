"""Merge all dimension tables into the main sales data."""

import polars as pl
import structlog

logger = structlog.get_logger()


def merge_dimensions(
    sales_lf: pl.LazyFrame,
    stores: pl.DataFrame,
    items: pl.DataFrame,
    oil: pl.DataFrame,
    transactions: pl.DataFrame,
) -> pl.LazyFrame:
    """Join stores, items, oil, and transactions onto the sales LazyFrame.

    Holidays are NOT joined here — they require locale-aware matching
    which is handled in features/holiday_features.py.
    """
    logger.info("merging_dimensions")

    merged = (
        sales_lf
        .join(stores.lazy(), on="store_nbr", how="left")
        .join(items.lazy(), on="item_nbr", how="left")
        .join(oil.lazy(), on="date", how="left")
        .join(transactions.lazy(), on=["date", "store_nbr"], how="left")
    )

    return merged
