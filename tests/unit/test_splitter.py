"""Tests for temporal data splitting."""

from datetime import date

from favorita_forecasting.data.splitter import ExpandingWindowCV


def test_expanding_window_creates_correct_number_of_splits():
    cv = ExpandingWindowCV(n_splits=4, forecast_horizon=16)
    splits = cv.get_splits()
    assert len(splits) == 4


def test_expanding_window_no_overlap():
    cv = ExpandingWindowCV(n_splits=4, forecast_horizon=16)
    splits = cv.get_splits()

    for split in splits:
        # Train end must be before val start
        assert split.train_end < split.val_start
        # Val period should be exactly forecast_horizon days
        val_days = (split.val_end - split.val_start).days + 1
        assert val_days == 16


def test_expanding_window_monotonic_train_end():
    cv = ExpandingWindowCV(n_splits=4, forecast_horizon=16)
    splits = cv.get_splits()

    train_ends = [s.train_end for s in splits]
    for i in range(1, len(train_ends)):
        assert train_ends[i] > train_ends[i - 1]


def test_expanding_window_last_val_ends_at_final_date():
    final = date(2017, 8, 15)
    cv = ExpandingWindowCV(n_splits=4, forecast_horizon=16, final_date=final)
    splits = cv.get_splits()
    assert splits[-1].val_end == final
