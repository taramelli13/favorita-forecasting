"""Rolling window aggregation features."""

import polars as pl

DEFAULT_WINDOWS = [7, 14, 28, 60, 90, 180, 365]
DEFAULT_STATS = ["mean", "std", "min", "max"]
DEFAULT_SHIFT = 16  # Prevent leakage for 16-day horizon


def _rolling_expr(
    target_col: str,
    window: int,
    stat: str,
    shift: int,
    group_cols: list[str],
) -> pl.Expr:
    """Build a single rolling statistic expression."""
    base = pl.col(target_col).shift(shift).over(group_cols)

    stat_map = {
        "mean": lambda: pl.col(target_col).shift(shift).rolling_mean(window_size=window).over(group_cols),
        "std": lambda: pl.col(target_col).shift(shift).rolling_std(window_size=window).over(group_cols),
        "min": lambda: pl.col(target_col).shift(shift).rolling_min(window_size=window).over(group_cols),
        "max": lambda: pl.col(target_col).shift(shift).rolling_max(window_size=window).over(group_cols),
    }

    return stat_map[stat]().alias(f"sales_rolling_{stat}_{window}")


def build_rolling_features(
    lf: pl.LazyFrame,
    windows: list[int] | None = None,
    stats: list[str] | None = None,
    shift: int = DEFAULT_SHIFT,
    target_col: str = "unit_sales",
    group_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Create rolling window features with proper shift to prevent leakage."""
    windows = windows or DEFAULT_WINDOWS
    stats = stats or DEFAULT_STATS
    group_cols = group_cols or ["store_nbr", "item_nbr"]

    exprs = [
        _rolling_expr(target_col, window, stat, shift, group_cols)
        for window in windows
        for stat in stats
    ]

    return lf.with_columns(exprs)
