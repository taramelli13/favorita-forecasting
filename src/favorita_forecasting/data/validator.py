"""Data quality validation checks."""

from datetime import date

import polars as pl
import structlog

logger = structlog.get_logger()


def validate_train(lf: pl.LazyFrame) -> None:
    """Run quality checks on the training LazyFrame."""
    sample = lf.head(1000).collect()

    # Check expected columns
    expected = {"id", "date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"}
    assert expected.issubset(set(sample.columns)), f"Missing columns: {expected - set(sample.columns)}"

    # Check date range
    dates = lf.select(pl.col("date").min().alias("min"), pl.col("date").max().alias("max")).collect()
    min_date = dates["min"][0]
    max_date = dates["max"][0]
    assert min_date <= date(2013, 2, 1), f"Train start date too late: {min_date}"
    assert max_date >= date(2017, 8, 1), f"Train end date too early: {max_date}"

    logger.info("train_validation_passed", min_date=str(min_date), max_date=str(max_date))


def validate_stores(df: pl.DataFrame) -> None:
    """Validate stores dimension table."""
    assert df.shape[0] == 54, f"Expected 54 stores, got {df.shape[0]}"
    assert df["store_nbr"].null_count() == 0, "Null store_nbr found"
    assert df["store_nbr"].min() >= 1, "store_nbr below 1"
    assert df["store_nbr"].max() <= 54, "store_nbr above 54"
    logger.info("stores_validation_passed", n_stores=df.shape[0])


def validate_items(df: pl.DataFrame) -> None:
    """Validate items dimension table."""
    assert df.shape[0] > 4000, f"Expected 4000+ items, got {df.shape[0]}"
    assert df["item_nbr"].null_count() == 0, "Null item_nbr found"
    assert df["perishable"].is_in([0, 1]).all(), "Unexpected perishable values"
    families = df["family"].unique()
    assert len(families) > 30, f"Expected 30+ families, got {len(families)}"
    logger.info("items_validation_passed", n_items=df.shape[0], n_families=len(families))


def validate_oil(df: pl.DataFrame) -> None:
    """Validate oil price data."""
    assert df.shape[0] > 1000, f"Expected 1000+ oil records, got {df.shape[0]}"
    non_null = df.filter(pl.col("dcoilwtico").is_not_null())
    assert non_null["dcoilwtico"].min() > 0, "Negative oil price found"
    logger.info("oil_validation_passed", n_records=df.shape[0])


def validate_all(
    train_lf: pl.LazyFrame,
    stores: pl.DataFrame,
    items: pl.DataFrame,
    oil: pl.DataFrame,
) -> None:
    """Run all validation checks."""
    validate_train(train_lf)
    validate_stores(stores)
    validate_items(items)
    validate_oil(oil)
    logger.info("all_validations_passed")
