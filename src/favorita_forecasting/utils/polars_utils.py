"""Polars utility functions."""

import polars as pl


def reduce_memory(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast numeric columns to reduce memory usage."""
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if dtype == pl.Float64:
            df = df.with_columns(pl.col(col_name).cast(pl.Float32))
        elif dtype == pl.Int64:
            col_min = df[col_name].min()
            col_max = df[col_name].max()
            if col_min is not None and col_max is not None:
                if col_min >= -128 and col_max <= 127:
                    df = df.with_columns(pl.col(col_name).cast(pl.Int8))
                elif col_min >= -32768 and col_max <= 32767:
                    df = df.with_columns(pl.col(col_name).cast(pl.Int16))
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df = df.with_columns(pl.col(col_name).cast(pl.Int32))
    return df
