"""Locale-aware holiday features.

Holidays in Ecuador operate at three levels:
- National: affects all stores
- Regional: affects stores in that state
- Local: affects stores in that city

This module matches each store to its applicable holidays based on location.
"""

import polars as pl
import structlog

logger = structlog.get_logger()


def _build_holiday_lookup(
    holidays: pl.DataFrame,
    stores: pl.DataFrame,
) -> pl.DataFrame:
    """Build a store-date holiday indicator table.

    For each store, determine which holidays apply based on:
    - National holidays -> all stores
    - Regional holidays -> stores in that state
    - Local holidays -> stores in that city
    """
    # National holidays (apply to all stores)
    national = holidays.filter(pl.col("locale") == "National").select(
        "date", "type", "transferred"
    )
    # Cross join with all store numbers
    store_nbrs = stores.select("store_nbr")
    national_expanded = national.join(store_nbrs, how="cross")

    # Regional holidays (match by state)
    regional = holidays.filter(pl.col("locale") == "Regional").select(
        "date", "type", "transferred", pl.col("locale_name").alias("state")
    )
    regional_expanded = regional.join(
        stores.select("store_nbr", "state"), on="state", how="inner"
    ).drop("state")

    # Local holidays (match by city)
    local = holidays.filter(pl.col("locale") == "Local").select(
        "date", "type", "transferred", pl.col("locale_name").alias("city")
    )
    local_expanded = local.join(
        stores.select("store_nbr", "city"), on="city", how="inner"
    ).drop("city")

    # Combine all applicable holidays
    all_holidays = pl.concat(
        [national_expanded, regional_expanded, local_expanded],
        how="diagonal",
    )

    return all_holidays


def build_holiday_features(
    lf: pl.LazyFrame,
    holidays: pl.DataFrame,
    stores: pl.DataFrame,
) -> pl.LazyFrame:
    """Add holiday indicator features to the main LazyFrame."""
    logger.info("building_holiday_features")

    holiday_lookup = _build_holiday_lookup(holidays, stores)

    # Create binary indicators per holiday type
    holiday_types = ["Holiday", "Transfer", "Bridge", "Additional", "Work Day", "Event"]

    # Aggregate: for each (date, store_nbr), create binary flags per type
    agg_exprs = [
        (pl.col("type") == ht).any().cast(pl.Int8).alias(f"is_{ht.lower().replace(' ', '_')}")
        for ht in holiday_types
    ]
    agg_exprs.append(
        pl.col("transferred").any().cast(pl.Int8).alias("is_transferred")
    )

    holiday_indicators = (
        holiday_lookup
        .group_by(["date", "store_nbr"])
        .agg(agg_exprs)
    )

    # Join to main data and fill nulls with 0 (no holiday)
    lf = lf.join(holiday_indicators.lazy(), on=["date", "store_nbr"], how="left")

    fill_cols = [f"is_{ht.lower().replace(' ', '_')}" for ht in holiday_types] + ["is_transferred"]
    lf = lf.with_columns([pl.col(c).fill_null(0) for c in fill_cols])

    # Combined "any holiday" flag
    lf = lf.with_columns(
        pl.max_horizontal([pl.col(f"is_{ht.lower().replace(' ', '_')}") for ht in ["Holiday", "Transfer", "Bridge", "Additional"]])
        .alias("is_any_holiday")
    )

    return lf
