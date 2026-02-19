"""Shared test fixtures."""

from datetime import date

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_train_df() -> pl.DataFrame:
    """Small training DataFrame for testing."""
    n_days = 30
    n_stores = 2
    n_items = 3
    dates = pl.date_range(date(2017, 7, 1), date(2017, 7, 30), eager=True)

    rows = []
    idx = 0
    for d in dates:
        for store in range(1, n_stores + 1):
            for item in [100001, 100002, 100003]:
                rows.append({
                    "id": idx,
                    "date": d,
                    "store_nbr": store,
                    "item_nbr": item,
                    "unit_sales": float(np.random.default_rng(idx).integers(0, 50)),
                    "onpromotion": bool(np.random.default_rng(idx).integers(0, 2)),
                })
                idx += 1

    return pl.DataFrame(rows).cast({
        "id": pl.Int64,
        "store_nbr": pl.Int16,
        "item_nbr": pl.Int32,
        "unit_sales": pl.Float32,
    })


@pytest.fixture
def sample_stores_df() -> pl.DataFrame:
    return pl.DataFrame({
        "store_nbr": [1, 2],
        "city": ["Quito", "Guayaquil"],
        "state": ["Pichincha", "Guayas"],
        "type": ["A", "B"],
        "cluster": [1, 2],
    }).cast({"store_nbr": pl.Int16, "cluster": pl.Int8})


@pytest.fixture
def sample_items_df() -> pl.DataFrame:
    return pl.DataFrame({
        "item_nbr": [100001, 100002, 100003],
        "family": ["GROCERY I", "BREAD/BAKERY", "CLEANING"],
        "class": [1001, 1002, 1003],
        "perishable": [0, 1, 0],
    }).cast({"item_nbr": pl.Int32, "class": pl.Int32, "perishable": pl.Int8})


@pytest.fixture
def sample_oil_df() -> pl.DataFrame:
    dates = pl.date_range(date(2017, 7, 1), date(2017, 7, 30), eager=True)
    return pl.DataFrame({
        "date": dates,
        "dcoilwtico": np.random.default_rng(42).uniform(40, 60, len(dates)).astype(np.float32),
    })


@pytest.fixture
def sample_holidays_df() -> pl.DataFrame:
    return pl.DataFrame({
        "date": [date(2017, 7, 4), date(2017, 7, 24), date(2017, 7, 25)],
        "type": ["Holiday", "Holiday", "Bridge"],
        "locale": ["National", "Local", "National"],
        "locale_name": ["Ecuador", "Guayaquil", "Ecuador"],
        "description": ["Batalla de Pichincha", "Fundacion de Guayaquil", "Bridge Day"],
        "transferred": [False, False, False],
    })


@pytest.fixture
def sample_transactions_df() -> pl.DataFrame:
    dates = pl.date_range(date(2017, 7, 1), date(2017, 7, 30), eager=True)
    rows = []
    for d in dates:
        for store in [1, 2]:
            rows.append({
                "date": d,
                "store_nbr": store,
                "transactions": int(np.random.default_rng(hash(str(d) + str(store)) % 2**31).integers(500, 3000)),
            })
    return pl.DataFrame(rows).cast({"store_nbr": pl.Int16, "transactions": pl.Int32})
