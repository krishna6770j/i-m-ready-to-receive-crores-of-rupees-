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


def test_session_boundary_is_info_not_warning():
    """Overnight boundaries are expected and must not look like defects."""
    day1 = make_ohlcv(10, start="2026-01-01 09:15", seed=1)
    day2 = make_ohlcv(10, start="2026-01-02 09:15", seed=2)
    frame = pd.concat([day1, day2], ignore_index=True)
    report = run(frame, expected_interval_minutes=1)
    assert "SESSION_BOUNDARIES" in codes(report)
    boundary = next(i for i in report.issues if i.code == "SESSION_BOUNDARIES")
    assert boundary.severity is Severity.INFO
    assert "WITHIN_DAY_GAPS" not in codes(report)


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
