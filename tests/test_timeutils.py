"""Timezone handling tests. Getting IST wrong shifts every candle by 5h30m."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import pytest

from core.timeutils import (
    IST,
    IST_NAME,
    NaiveTimestampError,
    ensure_ist,
    epoch_series_to_ist,
    epoch_to_ist,
    is_tz_aware_ist,
    ist_datetime,
    to_api_date,
)


def test_ist_offset_is_five_thirty():
    ts = datetime(2026, 8, 29, 9, 15, tzinfo=IST)
    assert ts.utcoffset() == timedelta(hours=5, minutes=30)


def test_india_has_no_dst_across_the_year():
    """A DST assumption would silently shift half the dataset."""
    offsets = {
        datetime(2026, m, 15, 12, 0, tzinfo=IST).utcoffset() for m in range(1, 13)
    }
    assert offsets == {timedelta(hours=5, minutes=30)}


def test_epoch_to_ist_known_value():
    # 1767239100 == 2026-01-01T09:15:00+05:30 (NSE session open)
    got = epoch_to_ist(1767239100)
    assert got.isoformat() == "2026-01-01T09:15:00+05:30"


def test_epoch_series_round_trips_through_utc():
    series = epoch_series_to_ist([1767239100, 1767239160])
    assert str(series.dtype.tz) == IST_NAME
    assert series.iloc[0].isoformat() == "2026-01-01T09:15:00+05:30"
    assert series.iloc[1].isoformat() == "2026-01-01T09:16:00+05:30"


def test_ensure_ist_rejects_naive_datetime():
    with pytest.raises(NaiveTimestampError):
        ensure_ist(datetime(2026, 1, 1, 9, 15))


def test_ensure_ist_converts_utc_correctly():
    utc = datetime(2026, 1, 1, 3, 45, tzinfo=timezone.utc)
    assert ensure_ist(utc).isoformat() == "2026-01-01T09:15:00+05:30"


def test_is_tz_aware_ist_detects_naive_series():
    naive = pd.Series(pd.date_range("2026-01-01", periods=3, freq="1min"))
    assert not is_tz_aware_ist(naive)


def test_is_tz_aware_ist_rejects_other_timezone():
    utc = pd.Series(pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC"))
    assert not is_tz_aware_ist(utc)


def test_is_tz_aware_ist_accepts_ist():
    ist = pd.Series(pd.date_range("2026-01-01", periods=3, freq="1min", tz=IST_NAME))
    assert is_tz_aware_ist(ist)


def test_ist_datetime_builds_aware_value():
    got = ist_datetime(date(2026, 8, 3), time(15, 40))
    assert got.isoformat() == "2026-08-03T15:40:00+05:30"


def test_to_api_date_format_matches_sdk_docstring():
    """SDK docstring specifies 'yyyy-mm-dd' when date_format=1."""
    assert to_api_date(date(2026, 1, 5)) == "2026-01-05"
