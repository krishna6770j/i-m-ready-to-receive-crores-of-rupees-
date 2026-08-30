"""Validator tests.

Each anomaly class the validator claims to detect gets a test that injects that
specific defect and asserts it is found. A validator that silently misses a
defect is worse than none, because it manufactures false confidence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from marketdata.schemas import CLOSE, HIGH, LOW, OPEN, TS, VOLUME
from marketdata.validator import Severity, validate
from tests.conftest import make_ohlcv


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def run(frame, **kwargs):
    return validate(frame, symbol="TEST", resolution="1", **kwargs)


def test_clean_data_produces_no_errors(ohlcv):
    report = run(ohlcv, expected_interval_minutes=1)
    assert report.is_usable
    assert not report.errors


def test_detects_empty_dataset():
    from marketdata.schemas import empty_ohlcv

    report = run(empty_ohlcv())
    assert "EMPTY_DATASET" in codes(report)
    assert not report.is_usable


def test_detects_high_below_open_close():
    frame = make_ohlcv(50)
    frame.loc[10, HIGH] = frame.loc[10, [OPEN, CLOSE]].max() - 1.0
    report = run(frame)
    assert "OHLC_HIGH_TOO_LOW" in codes(report)
    assert not report.is_usable


def test_detects_low_above_open_close():
    frame = make_ohlcv(50)
    frame.loc[7, LOW] = frame.loc[7, [OPEN, CLOSE]].min() + 1.0
    report = run(frame)
    assert "OHLC_LOW_TOO_HIGH" in codes(report)


def test_detects_high_below_low():
    frame = make_ohlcv(50)
    frame.loc[5, HIGH] = 100.0
    frame.loc[5, LOW] = 200.0
    report = run(frame)
    assert "OHLC_HIGH_BELOW_LOW" in codes(report)


def test_detects_non_positive_price():
    frame = make_ohlcv(50)
    frame.loc[3, LOW] = 0.0
    report = run(frame)
    assert "NON_POSITIVE_PRICE" in codes(report)


def test_detects_negative_price():
    frame = make_ohlcv(50)
    frame.loc[3, LOW] = -5.0
    report = run(frame)
    assert "NON_POSITIVE_PRICE" in codes(report)


def test_detects_null_price():
    frame = make_ohlcv(50)
    frame.loc[9, CLOSE] = float("nan")
    report = run(frame)
    assert "NULL_PRICE" in codes(report)


def test_detects_negative_volume():
    frame = make_ohlcv(50)
    frame.loc[4, VOLUME] = -1
    report = run(frame)
    assert "NEGATIVE_VOLUME" in codes(report)


def test_detects_missing_volume():
    """Missing volume must be reported, never filled with 0."""
    frame = make_ohlcv(50)
    frame.loc[6, VOLUME] = pd.NA
    report = run(frame)
    assert "NULL_VOLUME" in codes(report)
    assert not report.is_usable


def test_detects_positive_infinity_price():
    """+inf passes every comparison rule, so it needs its own check."""
    frame = make_ohlcv(50)
    frame.loc[11, HIGH] = float("inf")
    report = run(frame)
    assert "NON_FINITE_PRICE" in codes(report)
    assert not report.is_usable


def test_detects_negative_infinity_price():
    frame = make_ohlcv(50)
    frame.loc[12, LOW] = float("-inf")
    report = run(frame)
    assert "NON_FINITE_PRICE" in codes(report)
    assert not report.is_usable


def test_nan_is_reported_as_null_not_as_non_finite():
    """NaN and inf are different defects and must not be conflated."""
    frame = make_ohlcv(50)
    frame.loc[13, CLOSE] = float("nan")
    report = run(frame)
    assert "NULL_PRICE" in codes(report)
    assert "NON_FINITE_PRICE" not in codes(report)


def test_detects_duplicate_timestamps():
    frame = make_ohlcv(20)
    dup = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    dup = dup.sort_values(TS).reset_index(drop=True)
    report = run(dup)
    assert "DUPLICATE_TIMESTAMPS" in codes(report)
    assert not report.is_usable


def test_detects_unsorted_timestamps():
    frame = make_ohlcv(20)
    order = list(range(20))
    order[3], order[4] = order[4], order[3]
    swapped = frame.iloc[order].reset_index(drop=True)
    report = run(swapped)
    assert "TS_NOT_SORTED" in codes(report)


def test_detects_naive_timestamps():
    frame = make_ohlcv(20)
    frame[TS] = frame[TS].dt.tz_localize(None)
    report = run(frame)
    assert "TZ_NAIVE" in codes(report)
    assert not report.is_usable


def test_detects_non_ist_timezone():
    frame = make_ohlcv(20)
    frame[TS] = frame[TS].dt.tz_convert("UTC")
    report = run(frame)
    assert "TZ_NOT_IST" in codes(report)


def test_detects_within_day_gap():
    """A missing minute inside a session must be reported, not filled."""
    frame = make_ohlcv(30)
    frame = frame.drop(index=[10, 11]).reset_index(drop=True)
    report = run(frame, expected_interval_minutes=1)
    assert "WITHIN_DAY_GAPS" in codes(report)
    gap = next(i for i in report.issues if i.code == "WITHIN_DAY_GAPS")
    assert gap.severity is Severity.WARNING


def test_gap_detection_skipped_without_interval():
    """Interval is not guessed; without it, gap checks do not run."""
    frame = make_ohlcv(30).drop(index=[10]).reset_index(drop=True)
    report = run(frame, expected_interval_minutes=None)
    assert "WITHIN_DAY_GAPS" not in codes(report)


def test_detects_misaligned_seconds():
    frame = make_ohlcv(20)
    frame[TS] = frame[TS] + pd.Timedelta(seconds=17)
    report = run(frame, expected_interval_minutes=1)
    assert "TS_NOT_MINUTE_ALIGNED" in codes(report)


def test_detects_extreme_return():
    frame = make_ohlcv(200)
    frame.loc[100, CLOSE] = frame.loc[100, CLOSE] * 3.0
    frame.loc[100, HIGH] = frame.loc[100, CLOSE] + 1
    report = run(frame, expected_interval_minutes=1)
    assert "EXTREME_RETURN" in codes(report)


def _two_blocks(d1: str, d2: str) -> pd.DataFrame:
    return pd.concat(
        [
            make_ohlcv(10, start=f"{d1} 09:15", seed=1),
            make_ohlcv(10, start=f"{d2} 09:15", seed=2),
        ],
        ignore_index=True,
    )


def test_cross_day_gap_reports_magnitude_when_no_calendar_configured():
    """Without a calendar, a cross-day gap must not be called 'expected'.

    Regression for the defect where every cross-day gap -- overnight or
    four months -- was reported identically as an expected day boundary.
    """
    report = run(_two_blocks("2026-01-01", "2026-01-02"), expected_interval_minutes=1)
    assert "TRADING_CALENDAR_NOT_CONFIGURED" in codes(report)
    issue = next(
        i for i in report.issues if i.code == "TRADING_CALENDAR_NOT_CONFIGURED"
    )
    assert issue.severity is Severity.WARNING
    assert "calendar days" in issue.message
    assert "WITHIN_DAY_GAPS" not in codes(report)


def test_multi_month_hole_is_an_error_even_without_a_calendar():
    """A four-month hole is missing data under any calendar. THE key regression."""
    report = run(_two_blocks("2026-01-05", "2026-05-05"), expected_interval_minutes=1)
    assert "IMPLAUSIBLE_DATA_GAP" in codes(report)
    assert not report.is_usable, "a four-month hole must not be classified usable"


def test_overnight_gap_is_not_an_implausible_gap():
    """The absurdity ceiling must not fire on ordinary breaks."""
    report = run(_two_blocks("2026-01-01", "2026-01-02"), expected_interval_minutes=1)
    assert "IMPLAUSIBLE_DATA_GAP" not in codes(report)
    assert report.is_usable


def test_weekend_gap_is_not_an_implausible_gap():
    report = run(_two_blocks("2026-01-02", "2026-01-05"), expected_interval_minutes=1)
    assert "IMPLAUSIBLE_DATA_GAP" not in codes(report)
    assert report.is_usable


def test_configured_calendar_flags_excessive_gap_as_error():
    """With a calendar bound, gaps beyond it are ERRORs."""
    report = run(
        _two_blocks("2026-01-05", "2026-01-12"),
        expected_interval_minutes=1,
        max_session_gap_days=4,
    )
    assert "EXCESSIVE_DATA_GAP" in codes(report)
    assert not report.is_usable
    assert "TRADING_CALENDAR_NOT_CONFIGURED" not in codes(report)


def test_configured_calendar_accepts_gap_within_limit():
    report = run(
        _two_blocks("2026-01-02", "2026-01-05"),
        expected_interval_minutes=1,
        max_session_gap_days=4,
    )
    assert "EXCESSIVE_DATA_GAP" not in codes(report)
    assert report.is_usable


def test_session_window_check_is_skipped_when_not_supplied(ohlcv):
    """Unconfirmed NSE hours must not be guessed at."""
    report = run(ohlcv, expected_interval_minutes=1)
    assert "SESSION_WINDOW_NOT_CHECKED" in codes(report)
    issue = next(i for i in report.issues if i.code == "SESSION_WINDOW_NOT_CHECKED")
    assert issue.severity is Severity.INFO


def test_session_window_accepts_candles_inside_it(ohlcv):
    from datetime import time as t

    report = run(
        ohlcv, expected_interval_minutes=1, session_window=(t(9, 15), t(15, 40))
    )
    assert "OUTSIDE_SESSION_WINDOW" not in codes(report)
    assert "SESSION_WINDOW_NOT_CHECKED" not in codes(report)


def test_session_window_flags_candles_outside_it():
    from datetime import time as t

    frame = make_ohlcv(30, start="2026-01-01 15:35")
    report = run(
        frame, expected_interval_minutes=1, session_window=(t(9, 15), t(15, 40))
    )
    assert "OUTSIDE_SESSION_WINDOW" in codes(report)
    issue = next(i for i in report.issues if i.code == "OUTSIDE_SESSION_WINDOW")
    assert issue.count == 24  # 15:41..16:04 fall outside


def test_report_serialises_and_renders(ohlcv):
    report = run(ohlcv, expected_interval_minutes=1)
    assert isinstance(report.to_dict(), dict)
    assert "DATA QUALITY REPORT" in report.to_text()
    assert report.to_dict()["timezone"] == "Asia/Kolkata"


def test_samples_are_bounded():
    """A multi-year dataset must not produce an unreadable report."""
    frame = make_ohlcv(200)
    frame.loc[:, LOW] = frame[[OPEN, CLOSE]].min(axis=1) + 1.0
    report = run(frame)
    issue = next(i for i in report.issues if i.code == "OHLC_LOW_TOO_HIGH")
    assert issue.count == 200
    assert len(issue.samples) <= 5
