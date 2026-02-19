"""Data loading functions for all Favorita CSV datasets."""

from pathlib import Path

import polars as pl

from favorita_forecasting.config import load_settings
from favorita_forecasting.data.schemas import (
    HOLIDAYS_SCHEMA,
    ITEMS_SCHEMA,
    OIL_SCHEMA,
    STORES_SCHEMA,
    TEST_SCHEMA,
    TRAIN_SCHEMA,
    TRANSACTIONS_SCHEMA,
)


def _raw_path() -> Path:
    return load_settings().paths.data_raw


def load_train(path: Path | None = None) -> pl.LazyFrame:
    """Load train.csv as a LazyFrame (4.7GB — never fully in memory)."""
    path = path or _raw_path() / "train.csv"
    lf = pl.scan_csv(path, schema_overrides=TRAIN_SCHEMA)
    # Parse onpromotion: empty/"" -> null, "True"->True, "False"->False
    lf = lf.with_columns(
        pl.when(pl.col("onpromotion") == "True")
        .then(True)
        .when(pl.col("onpromotion") == "False")
        .then(False)
        .otherwise(None)
        .alias("onpromotion")
    )
    return lf


def load_test(path: Path | None = None) -> pl.LazyFrame:
    """Load test.csv as a LazyFrame."""
    path = path or _raw_path() / "test.csv"
    lf = pl.scan_csv(path, schema_overrides=TEST_SCHEMA)
    lf = lf.with_columns(
        pl.when(pl.col("onpromotion") == "True")
        .then(True)
        .when(pl.col("onpromotion") == "False")
        .then(False)
        .otherwise(None)
        .alias("onpromotion")
    )
    return lf


def load_stores(path: Path | None = None) -> pl.DataFrame:
    """Load stores.csv (54 rows)."""
    path = path or _raw_path() / "stores.csv"
    return pl.read_csv(path, schema_overrides=STORES_SCHEMA)


def load_items(path: Path | None = None) -> pl.DataFrame:
    """Load items.csv (4100 rows)."""
    path = path or _raw_path() / "items.csv"
    return pl.read_csv(path, schema_overrides=ITEMS_SCHEMA)


def load_oil(path: Path | None = None) -> pl.DataFrame:
    """Load oil.csv (1218 rows)."""
    path = path or _raw_path() / "oil.csv"
    return pl.read_csv(path, schema_overrides=OIL_SCHEMA)


def load_holidays(path: Path | None = None) -> pl.DataFrame:
    """Load holidays_events.csv (350 rows)."""
    path = path or _raw_path() / "holidays_events.csv"
    return pl.read_csv(path, schema_overrides=HOLIDAYS_SCHEMA)


def load_transactions(path: Path | None = None) -> pl.DataFrame:
    """Load transactions.csv (83K rows)."""
    path = path or _raw_path() / "transactions.csv"
    return pl.read_csv(path, schema_overrides=TRANSACTIONS_SCHEMA)
