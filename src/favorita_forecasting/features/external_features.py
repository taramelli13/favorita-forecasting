"""External data features: oil prices, transactions, and promotions."""

import polars as pl


def build_oil_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Build oil price features (already joined via merger)."""
    return lf.with_columns(
        # Oil lag features
        pl.col("dcoilwtico").shift(7).over("date").alias("oil_lag_7"),
        pl.col("dcoilwtico").shift(14).over("date").alias("oil_lag_14"),
        pl.col("dcoilwtico").shift(30).over("date").alias("oil_lag_30"),

        # Oil rolling averages (using sorted date column)
        pl.col("dcoilwtico").rolling_mean(window_size=7).alias("oil_rolling_mean_7"),
        pl.col("dcoilwtico").rolling_mean(window_size=30).alias("oil_rolling_mean_30"),

        # Oil price change
        (
            (pl.col("dcoilwtico") - pl.col("dcoilwtico").shift(7))
            / pl.col("dcoilwtico").shift(7)
        ).alias("oil_pct_change_7"),
    )


def build_transaction_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Build transaction count features (already joined via merger)."""
    return lf.with_columns(
        # Fill missing transactions (store closed) with 0
        pl.col("transactions").fill_null(0).alias("transactions"),
    ).with_columns(
        # Transaction lags per store
        pl.col("transactions").shift(7).over("store_nbr").alias("transactions_lag_7"),
        pl.col("transactions").shift(14).over("store_nbr").alias("transactions_lag_14"),
        pl.col("transactions").rolling_mean(window_size=7).over("store_nbr").alias("transactions_rolling_mean_7"),
        pl.col("transactions").rolling_mean(window_size=14).over("store_nbr").alias("transactions_rolling_mean_14"),
    )


def build_promotion_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Build promotion-related features."""
    return lf.with_columns(
        # Promotion as integer
        pl.col("onpromotion").cast(pl.Int8).alias("onpromotion_int"),

        # Number of items on promotion per store-date
        pl.col("onpromotion").cast(pl.Int8).sum().over(["date", "store_nbr"]).alias("store_promo_count"),

        # Percentage of items in family on promo per store-date
        pl.col("onpromotion").cast(pl.Float32).mean().over(["date", "store_nbr", "family"]).alias("family_promo_pct"),
    )


def build_external_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Apply all external feature builders."""
    lf = build_oil_features(lf)
    lf = build_transaction_features(lf)
    lf = build_promotion_features(lf)
    return lf
