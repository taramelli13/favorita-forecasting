"""Temporal train/validation splitting for time series cross-validation."""

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


@dataclass
class TimeSeriesSplit:
    """A single train/validation split defined by date boundaries."""

    train_end: date
    val_start: date
    val_end: date
    fold: int


class ExpandingWindowCV:
    """Expanding window cross-validation for time series.

    Each fold uses all data up to a cutoff date for training,
    then validates on the next `forecast_horizon` days.
    """

    def __init__(
        self,
        n_splits: int = 4,
        forecast_horizon: int = 16,
        final_date: date = date(2017, 8, 15),
    ) -> None:
        self.n_splits = n_splits
        self.forecast_horizon = forecast_horizon
        self.final_date = final_date

    def get_splits(self) -> list[TimeSeriesSplit]:
        """Generate expanding window splits working backwards from final_date."""
        splits = []
        for i in range(self.n_splits, 0, -1):
            offset = i * self.forecast_horizon
            val_end = self.final_date - timedelta(days=(i - 1) * self.forecast_horizon)
            val_start = val_end - timedelta(days=self.forecast_horizon - 1)
            train_end = val_start - timedelta(days=1)

            splits.append(
                TimeSeriesSplit(
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    fold=self.n_splits - i + 1,
                )
            )
        return splits

    def split(
        self, lf: pl.LazyFrame, date_col: str = "date"
    ) -> list[tuple[pl.LazyFrame, pl.LazyFrame]]:
        """Split a LazyFrame into (train, val) pairs for each fold."""
        result = []
        for s in self.get_splits():
            train = lf.filter(pl.col(date_col) <= s.train_end)
            val = lf.filter(
                (pl.col(date_col) >= s.val_start) & (pl.col(date_col) <= s.val_end)
            )
            result.append((train, val))
        return result
