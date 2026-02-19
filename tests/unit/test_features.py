"""Tests for feature engineering modules."""

from datetime import date

import polars as pl
import pytest

from favorita_forecasting.features.time_features import build_time_features
from favorita_forecasting.features.holiday_features import build_holiday_features


def test_time_features_adds_expected_columns():
    df = pl.DataFrame({
        "date": [date(2017, 7, 10), date(2017, 7, 15), date(2017, 7, 31)],
        "value": [1.0, 2.0, 3.0],
    })
    result = build_time_features(df.lazy()).collect()

    expected_cols = [
        "day_of_week", "day_of_month", "day_of_year",
        "week_of_year", "month", "quarter", "year",
        "is_weekend", "is_month_end", "is_payday",
        "sin_day_of_week", "cos_day_of_week",
        "sin_month", "cos_month",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_time_features_weekend_detection():
    df = pl.DataFrame({
        "date": [
            date(2017, 7, 10),  # Monday
            date(2017, 7, 15),  # Saturday
            date(2017, 7, 16),  # Sunday
        ],
    })
    result = build_time_features(df.lazy()).collect()
    assert result["is_weekend"].to_list() == [0, 1, 1]


def test_time_features_payday():
    df = pl.DataFrame({
        "date": [
            date(2017, 7, 14),  # Not payday
            date(2017, 7, 15),  # Payday (15th)
            date(2017, 7, 31),  # Payday (last day)
        ],
    })
    result = build_time_features(df.lazy()).collect()
    assert result["is_payday"].to_list() == [0, 1, 1]


def test_holiday_features_national(sample_stores_df, sample_holidays_df):
    df = pl.DataFrame({
        "date": [date(2017, 7, 4), date(2017, 7, 5)],
        "store_nbr": [1, 1],
    }).cast({"store_nbr": pl.Int16})

    result = build_holiday_features(
        df.lazy(), sample_holidays_df, sample_stores_df
    ).collect()

    assert "is_holiday" in result.columns
    # July 4 is a national holiday — should be 1
    assert result.filter(pl.col("date") == date(2017, 7, 4))["is_holiday"][0] == 1
    # July 5 is not a holiday — should be 0
    assert result.filter(pl.col("date") == date(2017, 7, 5))["is_holiday"][0] == 0
