"""Calendar and temporal features extracted from the date column."""

import math
from datetime import date

import polars as pl

# Reference date for trend feature
_EPOCH = date(2013, 1, 1)


def build_time_features(lf: pl.LazyFrame, date_col: str = "date") -> pl.LazyFrame:
    """Extract temporal features from the date column."""
    d = pl.col(date_col)

    lf = lf.with_columns(
        # Basic calendar
        d.dt.weekday().alias("day_of_week"),         # 1=Monday .. 7=Sunday
        d.dt.day().alias("day_of_month"),
        d.dt.ordinal_day().alias("day_of_year"),
        d.dt.week().alias("week_of_year"),
        d.dt.month().alias("month"),
        d.dt.quarter().alias("quarter"),
        d.dt.year().alias("year"),

        # Binary flags
        (d.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        (d.dt.day() == 1).cast(pl.Int8).alias("is_month_start"),

        # Payday indicator (15th and last day — Ecuadorian paydays)
        (d.dt.day() == 15).cast(pl.Int8).alias("is_payday_15"),

        # Cyclical encoding for day of week (period=7)
        (d.dt.weekday().cast(pl.Float32) * 2 * math.pi / 7).sin().alias("sin_day_of_week"),
        (d.dt.weekday().cast(pl.Float32) * 2 * math.pi / 7).cos().alias("cos_day_of_week"),

        # Cyclical encoding for month (period=12)
        (d.dt.month().cast(pl.Float32) * 2 * math.pi / 12).sin().alias("sin_month"),
        (d.dt.month().cast(pl.Float32) * 2 * math.pi / 12).cos().alias("cos_month"),

        # Trend: days since reference date
        (d.cast(pl.Date) - pl.lit(_EPOCH)).dt.total_days().cast(pl.Int32).alias("days_since_epoch"),
    )

    # Month-end detection (compare day to month-end day)
    lf = lf.with_columns(
        (pl.col(date_col).dt.month_end().dt.day() == pl.col(date_col).dt.day())
        .cast(pl.Int8)
        .alias("is_month_end"),
    )

    # Payday: 15th OR last day of month
    lf = lf.with_columns(
        pl.max_horizontal("is_payday_15", "is_month_end").alias("is_payday"),
    ).drop("is_payday_15")

    return lf
