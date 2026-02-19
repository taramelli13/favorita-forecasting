"""Lag features: shifted sales values per store-item group."""

import polars as pl

# Lags >= 16 to avoid data leakage for 16-day forecast horizon
DEFAULT_LAGS = [16, 17, 18, 19, 20, 21, 28, 30, 60, 90, 180, 365]


def build_lag_features(
    lf: pl.LazyFrame,
    lags: list[int] | None = None,
    target_col: str = "unit_sales",
    group_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Create lag features for the target variable.

    All lags are computed per (store_nbr, item_nbr) group to capture
    item-store specific sales patterns.
    """
    lags = lags or DEFAULT_LAGS
    group_cols = group_cols or ["store_nbr", "item_nbr"]

    lag_exprs = [
        pl.col(target_col)
        .shift(lag)
        .over(group_cols)
        .alias(f"sales_lag_{lag}")
        for lag in lags
    ]

    # Diff features: change between consecutive lags
    diff_exprs = [
        (
            pl.col(target_col).shift(lags[0]).over(group_cols)
            - pl.col(target_col).shift(lags[1]).over(group_cols)
        ).alias(f"sales_diff_{lags[0]}_{lags[1]}")
    ] if len(lags) >= 2 else []

    return lf.with_columns(lag_exprs + diff_exprs)
