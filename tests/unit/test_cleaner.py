"""Tests for data cleaning."""

import polars as pl
import pytest

from favorita_forecasting.preprocessing.cleaner import clean_oil, clean_train


def test_clean_train_clips_negative_sales():
    df = pl.DataFrame({
        "id": [0, 1, 2],
        "date": ["2017-01-01", "2017-01-02", "2017-01-03"],
        "store_nbr": [1, 1, 1],
        "item_nbr": [100, 100, 100],
        "unit_sales": [-5.0, 0.0, 10.0],
        "onpromotion": [None, True, False],
    }).cast({"unit_sales": pl.Float32, "store_nbr": pl.Int16, "item_nbr": pl.Int32})

    result = clean_train(df.lazy()).collect()
    assert result["unit_sales"].to_list() == [0.0, 0.0, 10.0]


def test_clean_train_fills_null_promotion():
    df = pl.DataFrame({
        "id": [0, 1],
        "date": ["2017-01-01", "2017-01-02"],
        "store_nbr": [1, 1],
        "item_nbr": [100, 100],
        "unit_sales": [5.0, 10.0],
        "onpromotion": [None, True],
    }).cast({"unit_sales": pl.Float32, "store_nbr": pl.Int16, "item_nbr": pl.Int32})

    result = clean_train(df.lazy()).collect()
    assert result["onpromotion"].null_count() == 0
    assert result["onpromotion"][0] == False
    assert result["onpromotion"][1] == True


def test_clean_oil_fills_gaps():
    df = pl.DataFrame({
        "date": ["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"],
        "dcoilwtico": [50.0, None, None, 55.0],
    }).cast({"dcoilwtico": pl.Float32, "date": pl.Date})

    result = clean_oil(df)
    assert result["dcoilwtico"].null_count() == 0
    # Forward fill: 50, 50, 50, 55
    assert result["dcoilwtico"][1] == 50.0
    assert result["dcoilwtico"][2] == 50.0
