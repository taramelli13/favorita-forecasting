"""Feature pipeline orchestrator: chains all feature builders."""

import gc
import time
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import structlog

from favorita_forecasting.config import Settings, load_settings
from favorita_forecasting.data.loader import (
    load_holidays,
    load_items,
    load_oil,
    load_stores,
    load_test,
    load_train,
    load_transactions,
)
from favorita_forecasting.data.validator import validate_all
from favorita_forecasting.features.external_features import build_external_features
from favorita_forecasting.features.holiday_features import build_holiday_features
from favorita_forecasting.features.lag_features import build_lag_features
from favorita_forecasting.features.rolling_features import build_rolling_features
from favorita_forecasting.features.target_encoding import build_target_encodings
from favorita_forecasting.features.time_features import build_time_features
from favorita_forecasting.preprocessing.cleaner import clean_all
from favorita_forecasting.preprocessing.merger import merge_dimensions
from favorita_forecasting.preprocessing.transformer import apply_log1p_target

logger = structlog.get_logger()


def _ensure_dirs(settings: object) -> None:
    """Create output directories if they don't exist."""
    for attr in ["data_interim", "data_processed", "data_submissions", "models"]:
        path = getattr(settings.paths, attr)
        path.mkdir(parents=True, exist_ok=True)


def _filter_date_range(lf: pl.LazyFrame, start: str, end: str) -> pl.LazyFrame:
    """Filter LazyFrame to a date range."""
    return lf.filter(
        (pl.col("date") >= date.fromisoformat(start))
        & (pl.col("date") <= date.fromisoformat(end))
    )


def build_features(
    sales_lf: pl.LazyFrame,
    stores: pl.DataFrame,
    items: pl.DataFrame,
    oil: pl.DataFrame,
    transactions: pl.DataFrame,
    holidays: pl.DataFrame,
    settings: object | None = None,
) -> pl.LazyFrame:
    """Run the full feature engineering pipeline on a LazyFrame."""
    settings = settings or load_settings()

    # 1. Merge dimension tables
    lf = merge_dimensions(sales_lf, stores, items, oil, transactions)

    # 2. Sort by store, item, date (required for lag/rolling)
    lf = lf.sort(["store_nbr", "item_nbr", "date"])

    # 3. Time features (no data dependency)
    lf = build_time_features(lf)

    # 4. Holiday features (locale-aware)
    lf = build_holiday_features(lf, holidays, stores)

    # 5. External features (oil, transactions, promotions)
    lf = build_external_features(lf)

    # 6. Lag features
    lf = build_lag_features(lf, lags=settings.features.lags)

    # 7. Rolling features
    lf = build_rolling_features(
        lf,
        windows=settings.features.rolling_windows,
        stats=settings.features.rolling_stats,
        shift=settings.features.rolling_shift,
    )

    # 8. Target encodings (if unit_sales present — training only)
    if "unit_sales" in lf.collect_schema().names():
        lf = build_target_encodings(lf, shift=settings.features.rolling_shift)

    return lf


def _build_features_chunked(
    sales_lf: pl.LazyFrame,
    stores: pl.DataFrame,
    items: pl.DataFrame,
    oil: pl.DataFrame,
    transactions: pl.DataFrame,
    holidays: pl.DataFrame,
    settings: Settings,
    store_ids: list[int],
    out_path: Path,
    filter_start: date,
    filter_end: date | None = None,
) -> None:
    """Build features store by store, writing to parquet incrementally.

    Processing per store keeps peak memory low (~1-2 GB per store) instead
    of 60+ GB for all stores at once.
    """
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    t0 = time.time()

    for i, store_nbr in enumerate(store_ids, 1):
        logger.info("processing_store", store=store_nbr, progress=f"{i}/{len(store_ids)}")

        store_lf = sales_lf.filter(pl.col("store_nbr") == store_nbr)
        features_lf = build_features(
            store_lf, stores, items, oil, transactions, holidays, settings
        )

        # Filter dates AFTER computing features (lags need full history)
        date_filter = pl.col("date") >= filter_start
        if filter_end is not None:
            date_filter = date_filter & (pl.col("date") <= filter_end)
        features_df = features_lf.filter(date_filter).collect()

        if features_df.height == 0:
            logger.warning("empty_store_chunk", store=store_nbr)
            continue

        total_rows += features_df.height
        arrow_table = features_df.to_arrow()

        if writer is None:
            writer = pq.ParquetWriter(str(out_path), arrow_table.schema)
        writer.write_table(arrow_table)

        del features_df, arrow_table
        gc.collect()

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    logger.info(
        "saved_features",
        path=str(out_path),
        total_rows=total_rows,
        elapsed_min=f"{elapsed / 60:.1f}",
    )


def run_feature_pipeline() -> None:
    """Execute the full pipeline: load -> clean -> features -> save to Parquet.

    Processes one store at a time to stay within 16 GB RAM.
    """
    settings = load_settings()
    _ensure_dirs(settings)

    logger.info("loading_raw_data")
    train_lf = load_train()
    stores = load_stores()
    items = load_items()
    oil = load_oil()
    holidays = load_holidays()
    transactions = load_transactions()

    # Validate
    validate_all(train_lf, stores, items, oil)

    # Clean
    train_lf, oil, transactions = clean_all(train_lf, oil, transactions)

    # Filter to training date range (full history for lag computation)
    train_lf = _filter_date_range(
        train_lf, settings.data.train_start_date, settings.data.train_end_date
    )

    # Apply log1p to target
    train_lf = apply_log1p_target(train_lf)

    store_ids = sorted(stores["store_nbr"].to_list())
    min_date = date.fromisoformat(settings.data.min_train_date)

    # ---- Train features (chunked by store) ----
    train_out = settings.paths.data_processed / "features_train.parquet"
    logger.info("building_train_features", n_stores=len(store_ids))
    _build_features_chunked(
        sales_lf=train_lf,
        stores=stores, items=items, oil=oil,
        transactions=transactions, holidays=holidays,
        settings=settings, store_ids=store_ids,
        out_path=train_out,
        filter_start=min_date,
    )

    # ---- Test features (chunked by store) ----
    logger.info("building_test_features")
    test_lf = load_test()
    test_lf = test_lf.with_columns(pl.col("onpromotion").fill_null(False))

    # Concatenate train + test so lags can reach into training data
    combined_lf = pl.concat([train_lf, test_lf], how="diagonal")

    test_out = settings.paths.data_processed / "features_test.parquet"
    test_start = date.fromisoformat(settings.data.test_start_date)
    test_end = date.fromisoformat(settings.data.test_end_date)
    _build_features_chunked(
        sales_lf=combined_lf,
        stores=stores, items=items, oil=oil,
        transactions=transactions, holidays=holidays,
        settings=settings, store_ids=store_ids,
        out_path=test_out,
        filter_start=test_start,
        filter_end=test_end,
    )
