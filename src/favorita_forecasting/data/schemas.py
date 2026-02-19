"""Polars schema definitions for all Favorita CSV datasets."""

import polars as pl

TRAIN_SCHEMA = {
    "id": pl.Int64,
    "date": pl.Date,
    "store_nbr": pl.Int16,
    "item_nbr": pl.Int32,
    "unit_sales": pl.Float32,
    "onpromotion": pl.String,  # Raw: empty/True/False — parsed in loader
}

TEST_SCHEMA = {
    "id": pl.Int64,
    "date": pl.Date,
    "store_nbr": pl.Int16,
    "item_nbr": pl.Int32,
    "onpromotion": pl.String,
}

STORES_SCHEMA = {
    "store_nbr": pl.Int16,
    "city": pl.String,
    "state": pl.String,
    "type": pl.String,
    "cluster": pl.Int8,
}

ITEMS_SCHEMA = {
    "item_nbr": pl.Int32,
    "family": pl.String,
    "class": pl.Int32,
    "perishable": pl.Int8,
}

OIL_SCHEMA = {
    "date": pl.Date,
    "dcoilwtico": pl.Float32,
}

TRANSACTIONS_SCHEMA = {
    "date": pl.Date,
    "store_nbr": pl.Int16,
    "transactions": pl.Int32,
}

HOLIDAYS_SCHEMA = {
    "date": pl.Date,
    "type": pl.String,
    "locale": pl.String,
    "locale_name": pl.String,
    "description": pl.String,
    "transferred": pl.Boolean,
}

# Categorical columns for each dataset
STORE_CATEGORICALS = ["city", "state", "type"]
ITEM_CATEGORICALS = ["family"]
HOLIDAY_CATEGORICALS = ["type", "locale", "locale_name"]
