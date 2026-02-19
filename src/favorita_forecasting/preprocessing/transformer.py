"""Data transformations: log transforms and type casting."""

import numpy as np
import polars as pl


def apply_log1p_target(lf: pl.LazyFrame, col: str = "unit_sales") -> pl.LazyFrame:
    """Apply log1p transform to the target variable."""
    return lf.with_columns(
        pl.col(col).log1p().alias(f"{col}_log1p"),
    )


def inverse_log1p(values: np.ndarray) -> np.ndarray:
    """Inverse log1p transform: expm1."""
    return np.expm1(values)


def cast_categoricals(lf: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    """Cast string columns to Polars Categorical type."""
    return lf.with_columns(
        [pl.col(c).cast(pl.Categorical) for c in columns if c in lf.collect_schema().names()]
    )
