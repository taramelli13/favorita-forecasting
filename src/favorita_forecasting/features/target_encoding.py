"""Target encoding: expanding mean of sales for categorical groups."""

import polars as pl


def build_target_encodings(
    lf: pl.LazyFrame,
    target_col: str = "unit_sales",
    shift: int = 16,
) -> pl.LazyFrame:
    """Create target encoding features using expanding mean with shift.

    The shift prevents data leakage by only using data from at least
    `shift` days before the current date.
    """
    group_configs = [
        (["store_nbr", "item_nbr"], "store_item_mean_sales"),
        (["store_nbr", "family"], "store_family_mean_sales"),
        (["store_nbr"], "store_mean_sales"),
        (["item_nbr"], "item_mean_sales"),
        (["family"], "family_mean_sales"),
    ]

    exprs = []
    for group_cols, alias in group_configs:
        expr = (
            pl.col(target_col)
            .shift(shift)
            .mean()
            .over(group_cols)
            .alias(alias)
        )
        exprs.append(expr)

    return lf.with_columns(exprs)
