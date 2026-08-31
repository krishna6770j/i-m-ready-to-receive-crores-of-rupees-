"""Continuity/session certification primitive tests.

``FakeCalendar`` below is a deterministic, OBVIOUSLY FAKE test fixture only
(Monday-Friday, 09:15-15:30) -- it is never imported by production code and
must never be confused with a real NSE calendar. No NSE hours are claimed
anywhere in this file.
"""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.continuity import (
    CalendarExplanation,
    CertificationStatus,
    ContinuityCertification,
    GapExplanation,
    NullCalendar,
    ObservedGap,
    SessionCertification,
    certify_continuity,
    certify_sessions,
)
from marketdata.schemas import CLOSE, HIGH, LOW, OPEN, TS, VOLUME, SchemaError


# ---------------------------------------------------------------------------
# Test-only fake calendar -- NOT NSE, NOT production
# ---------------------------------------------------------------------------


class FakeCalendar:
    """Obviously-fake, deterministic Monday-Friday 09:15-15:30 session
    calendar. Test fixture only -- never imported outside this test file.
    """

    calendar_id = "fake-test-calendar"
    calendar_version = "1"

    _SESSION_START = time(9, 15)
    _SESSION_END = time(15, 30)

    def is_session_day(self, day: date) -> bool:
        return day.weekday() < 5  # Mon=0 .. Fri=4

    def is_valid_bar(self, ts: pd.Timestamp, resolution: str) -> bool:
        if not self.is_session_day(ts.date()):
            return False
        t = ts.time()
        return self._SESSION_START <= t <= self._SESSION_END

    def _interval(self, resolution: str) -> pd.Timedelta:
        if resolution == "1D":
            return pd.Timedelta(days=1)
        return pd.Timedelta(minutes=int(resolution))

    def _next_session_day(self, day: date) -> date:
        nxt = day + pd.Timedelta(days=1)
        nxt = nxt if isinstance(nxt, date) else nxt.date()
        while not self.is_session_day(nxt):
            nxt = nxt + pd.Timedelta(days=1)
            nxt = nxt if isinstance(nxt, date) else nxt.date()
        return nxt

    def expected_next_bar(self, ts: pd.Timestamp, resolution: str) -> pd.Timestamp:
        if resolution == "1D":
            next_day = self._next_session_day(ts.date())
            return pd.Timestamp(next_day, tz=IST_NAME)
        candidate = ts + self._interval(resolution)
        if candidate.time() > self._SESSION_END:
            next_day = self._next_session_day(ts.date())
            return pd.Timestamp(
                f"{next_day.isoformat()} {self._SESSION_START.isoformat()}", tz=IST_NAME
            )
        return candidate


class NaiveTimestampCalendar(FakeCalendar):
    """Protocol violation: returns a naive (non-tz-aware) timestamp."""

    def expected_next_bar(self, ts, resolution):
        return super().expected_next_bar(ts, resolution).tz_localize(None)


class NonIstTimestampCalendar(FakeCalendar):
    """Protocol violation: returns a tz-aware but non-IST timestamp."""

    def expected_next_bar(self, ts, resolution):
        return super().expected_next_bar(ts, resolution).tz_convert("UTC")


class EmptyIdCalendar(FakeCalendar):
    calendar_id = ""


class EmptyVersionCalendar(FakeCalendar):
    calendar_version = ""


# ---------------------------------------------------------------------------
# Frame-building helpers
# ---------------------------------------------------------------------------


def _frame_from_timestamps(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    """Build a frame with exactly canonical dtypes/column order, WITHOUT
    going through ``canonicalise()`` -- that would silently sort the rows,
    which would defeat the no-repair test's ability to feed in a
    deliberately out-of-order (non-canonical) frame.
    """
    n = len(timestamps)
    bases = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            TS: pd.Series(list(timestamps), dtype=f"datetime64[ns, {IST_NAME}]"),
            OPEN: pd.Series(bases, dtype="float64"),
            HIGH: pd.Series([b + 5 for b in bases], dtype="float64"),
            LOW: pd.Series([b - 5 for b in bases], dtype="float64"),
            CLOSE: pd.Series([b + 1 for b in bases], dtype="float64"),
            VOLUME: pd.Series([1000] * n, dtype="Int64"),
        }
    )


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=IST_NAME)


def _find_friday(start_year: int = 2026) -> date:
    d = date(start_year, 1, 1)
    while d.weekday() != 4:
        d = d + pd.Timedelta(days=1)
        d = d if isinstance(d, date) else d.date()
    return d


# ---------------------------------------------------------------------------
# Continuity: NullCalendar
# ---------------------------------------------------------------------------


def test_continuity_null_calendar_no_gaps_not_certified():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15"), _ts("2026-01-01 09:16")])
    result = certify_continuity(frame, "1", NullCalendar())
    assert result.status is CertificationStatus.NOT_CERTIFIED
    assert result.gap_explanations == ()


def test_continuity_null_calendar_regular_step_plus_one_gap_unknown():
    frame = _frame_from_timestamps(
        [_ts("2026-01-01 09:15"), _ts("2026-01-01 09:16"), _ts("2026-01-01 09:18")]
    )
    result = certify_continuity(frame, "1", NullCalendar())
    assert result.status is CertificationStatus.NOT_CERTIFIED
    assert len(result.gap_explanations) == 1
    assert result.gap_explanations[0].explanation is CalendarExplanation.UNKNOWN
    assert result.gap_explanations[0].gap.elapsed == pd.Timedelta(minutes=2)


def test_continuity_null_calendar_90_day_gap_still_not_certified_unknown():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15"), _ts("2026-04-01 09:15")])
    result = certify_continuity(frame, "1", NullCalendar())
    assert result.status is CertificationStatus.NOT_CERTIFIED
    assert result.gap_explanations[0].explanation is CalendarExplanation.UNKNOWN


# ---------------------------------------------------------------------------
# Continuity: configured calendar
# ---------------------------------------------------------------------------


def test_continuity_consecutive_expected_bars_certified():
    frame = _frame_from_timestamps(
        [_ts("2026-01-01 09:15"), _ts("2026-01-01 09:16"), _ts("2026-01-01 09:17")]
    )
    result = certify_continuity(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.CERTIFIED
    assert result.gap_explanations == ()


def test_continuity_explained_overnight_transition_certified():
    day1 = _find_friday()
    # last bar of day1, first bar of the SAME (non-weekend) next session day
    day2 = day1 + pd.Timedelta(days=3)  # Friday -> Monday, both session days if day1 is Friday
    frame = _frame_from_timestamps(
        [
            pd.Timestamp(f"{day1.isoformat()} 15:30", tz=IST_NAME),
            pd.Timestamp(f"{(day1 + pd.Timedelta(days=1)).isoformat()} 09:15", tz=IST_NAME)
            if (day1 + pd.Timedelta(days=1)).weekday() < 5
            else pd.Timestamp(f"{day2.isoformat()} 09:15", tz=IST_NAME),
        ]
    )
    result = certify_continuity(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.CERTIFIED


def test_critical_friday_to_monday_certified_friday_to_tuesday_failed():
    """The exact manager-specified critical test: two datasets with
    DIFFERENT elapsed gap sizes, distinguished ONLY by calendar knowledge --
    there is zero numeric 'max gap days' threshold involved anywhere.
    """
    friday = _find_friday()
    monday = friday + pd.Timedelta(days=3)
    tuesday = friday + pd.Timedelta(days=4)
    assert monday.weekday() == 0
    assert tuesday.weekday() == 1

    friday_close = pd.Timestamp(f"{friday.isoformat()} 15:30", tz=IST_NAME)
    monday_open = pd.Timestamp(f"{monday.isoformat()} 09:15", tz=IST_NAME)
    tuesday_open = pd.Timestamp(f"{tuesday.isoformat()} 09:15", tz=IST_NAME)

    dataset_a = _frame_from_timestamps([friday_close, monday_open])
    dataset_b = _frame_from_timestamps([friday_close, tuesday_open])

    result_a = certify_continuity(dataset_a, "1", FakeCalendar())
    result_b = certify_continuity(dataset_b, "1", FakeCalendar())

    assert result_a.status is CertificationStatus.CERTIFIED
    assert result_b.status is CertificationStatus.FAILED
    # Both gaps span a similar multi-day elapsed period; only the calendar's
    # opinion about the SPECIFIC expected next bar differs.
    assert result_b.gap_explanations[0].explanation is CalendarExplanation.UNEXPLAINED


def test_continuity_missing_expected_intraday_bar_fails():
    frame = _frame_from_timestamps(
        [_ts("2026-01-01 09:15"), _ts("2026-01-01 09:17")]  # skipped 09:16
    )
    result = certify_continuity(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.FAILED
    assert result.gap_explanations[0].explanation is CalendarExplanation.UNEXPLAINED


def test_continuity_unexplained_multiday_gap_fails():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15"), _ts("2026-03-01 09:15")])
    result = certify_continuity(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.FAILED


def test_no_arbitrary_30_day_rule_exists():
    """A gap far larger than any historical 30-day threshold must still be
    CERTIFIED if the calendar says it is exactly the expected next bar --
    using a genuine two-observation transition, not a vacuous single row."""
    friday = _find_friday()
    next_session_day = friday + pd.Timedelta(days=3)  # Friday -> Monday
    frame_daily = _frame_from_timestamps(
        [
            pd.Timestamp(friday.isoformat(), tz=IST_NAME),
            pd.Timestamp(next_session_day.isoformat(), tz=IST_NAME),
        ]
    )
    result = certify_continuity(frame_daily, "1D", FakeCalendar())
    assert result.status is CertificationStatus.CERTIFIED


def test_continuity_frame_not_mutated():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15"), _ts("2026-01-01 09:20")])
    before = frame.copy(deep=True)
    certify_continuity(frame, "1", FakeCalendar())
    pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Session certification
# ---------------------------------------------------------------------------


def test_session_null_calendar_not_certified():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    result = certify_sessions(frame, "1", NullCalendar())
    assert result.status is CertificationStatus.NOT_CERTIFIED


def test_session_valid_fake_session_candles_certified():
    frame = _frame_from_timestamps(
        [_ts("2026-01-01 09:15"), _ts("2026-01-01 09:16"), _ts("2026-01-01 15:30")]
    )
    result = certify_sessions(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.CERTIFIED
    assert result.invalid_timestamps == ()
    assert result.checked_count == 3


def test_session_candle_on_non_session_day_fails():
    saturday = _find_friday() + pd.Timedelta(days=1)
    frame = _frame_from_timestamps([pd.Timestamp(f"{saturday.isoformat()} 10:00", tz=IST_NAME)])
    result = certify_sessions(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.FAILED
    assert len(result.invalid_timestamps) == 1


def test_session_out_of_session_candle_on_valid_day_fails():
    frame = _frame_from_timestamps([_ts("2026-01-01 16:00")])  # after 15:30 close
    result = certify_sessions(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.FAILED
    assert len(result.invalid_timestamps) == 1


def test_session_first_candle_independently_checked_via_is_valid_bar():
    """A single-candle frame has no predecessor -- is_valid_bar is the ONLY
    way to certify it, proving is_session_day alone is not enough."""
    frame = _frame_from_timestamps([_ts("2026-01-01 16:00")])  # valid day, invalid time
    result = certify_sessions(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.FAILED
    assert result.invalid_timestamps[0].startswith("2026-01-01T16:00")


def test_session_frame_not_mutated():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15"), _ts("2026-01-01 09:16")])
    before = frame.copy(deep=True)
    certify_sessions(frame, "1", FakeCalendar())
    pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Protocol/error cases
# ---------------------------------------------------------------------------


def test_naive_timestamp_from_expected_next_bar_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15"), _ts("2026-01-01 09:16")])
    with pytest.raises(ValueError, match="naive"):
        certify_continuity(frame, "1", NaiveTimestampCalendar())


def test_non_ist_timestamp_from_expected_next_bar_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15"), _ts("2026-01-01 09:16")])
    with pytest.raises(ValueError, match="IST"):
        certify_continuity(frame, "1", NonIstTimestampCalendar())


def test_empty_calendar_id_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    with pytest.raises(ValueError, match="calendar_id"):
        certify_continuity(frame, "1", EmptyIdCalendar())
    with pytest.raises(ValueError, match="calendar_id"):
        certify_sessions(frame, "1", EmptyIdCalendar())


def test_empty_calendar_version_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    with pytest.raises(ValueError, match="calendar_version"):
        certify_continuity(frame, "1", EmptyVersionCalendar())
    with pytest.raises(ValueError, match="calendar_version"):
        certify_sessions(frame, "1", EmptyVersionCalendar())


def test_unsupported_resolution_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    with pytest.raises(ValueError, match="not in the documented set"):
        certify_continuity(frame, "45", FakeCalendar())
    with pytest.raises(ValueError, match="not in the documented set"):
        certify_sessions(frame, "45", FakeCalendar())


def test_noncanonical_frame_rejected_never_repaired():
    """A non-canonical (unsorted) frame must be REJECTED outright, never
    silently sorted/repaired before certification runs."""
    frame = _frame_from_timestamps([_ts("2026-01-01 09:20"), _ts("2026-01-01 09:15")])
    with pytest.raises(SchemaError):
        certify_continuity(frame, "1", FakeCalendar())
    with pytest.raises(SchemaError):
        certify_sessions(frame, "1", FakeCalendar())


# ---------------------------------------------------------------------------
# Unit 13A final hardening: <2-observation continuity rule, and strict
# non-short-circuiting bool checks on is_session_day/is_valid_bar.
# ---------------------------------------------------------------------------


def test_continuity_empty_canonical_frame_not_certified():
    frame = _frame_from_timestamps([])
    result_null = certify_continuity(frame, "1", NullCalendar())
    assert result_null.status is CertificationStatus.NOT_CERTIFIED
    result_calendar = certify_continuity(frame, "1", FakeCalendar())
    assert result_calendar.status is CertificationStatus.NOT_CERTIFIED


def test_continuity_one_row_with_null_calendar_not_certified():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    result = certify_continuity(frame, "1", NullCalendar())
    assert result.status is CertificationStatus.NOT_CERTIFIED


def test_continuity_one_row_with_configured_calendar_not_certified():
    """The overcertification regression: a single-row frame has no
    transition to evaluate at all, so it must be NOT_CERTIFIED
    (insufficient evidence) -- never vacuously CERTIFIED, never FAILED."""
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    result = certify_continuity(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.NOT_CERTIFIED
    assert result.gap_explanations == ()


def test_continuity_two_expected_consecutive_bars_certified():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15"), _ts("2026-01-01 09:16")])
    result = certify_continuity(frame, "1", FakeCalendar())
    assert result.status is CertificationStatus.CERTIFIED


# --- strict, non-short-circuiting bool checks -------------------------------


class SessionDayReturnsStringCalendar(FakeCalendar):
    def is_session_day(self, day):
        return "yes"


class SessionDayReturnsIntCalendar(FakeCalendar):
    def is_session_day(self, day):
        return 1


class ValidBarReturnsStringCalendar(FakeCalendar):
    def is_valid_bar(self, ts, resolution):
        return "yes"


class ValidBarReturnsIntCalendar(FakeCalendar):
    def is_valid_bar(self, ts, resolution):
        return 1


class SessionDayFalseButValidBarBrokenCalendar(FakeCalendar):
    """is_session_day() correctly returns False (a real bool) for every
    day, but is_valid_bar() is broken (wrong type). If session certification
    ever short-circuited on ``is_session_day(...) and is_valid_bar(...)``,
    this broken is_valid_bar() would never even be called, and its protocol
    violation would go completely undetected.
    """

    def is_session_day(self, day):
        return False

    def is_valid_bar(self, ts, resolution):
        return "not-a-bool"


def test_session_day_returns_string_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    with pytest.raises(TypeError, match="is_session_day"):
        certify_sessions(frame, "1", SessionDayReturnsStringCalendar())


def test_session_day_returns_int_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    with pytest.raises(TypeError, match="is_session_day"):
        certify_sessions(frame, "1", SessionDayReturnsIntCalendar())


def test_valid_bar_returns_string_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    with pytest.raises(TypeError, match="is_valid_bar"):
        certify_sessions(frame, "1", ValidBarReturnsStringCalendar())


def test_valid_bar_returns_int_rejected():
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    with pytest.raises(TypeError, match="is_valid_bar"):
        certify_sessions(frame, "1", ValidBarReturnsIntCalendar())


def test_valid_bar_protocol_violation_detected_even_when_session_day_false():
    """Proves session certification never short-circuits: is_session_day()
    returning a genuine False must NOT skip calling/type-checking
    is_valid_bar() -- a broken is_valid_bar() must still be caught."""
    frame = _frame_from_timestamps([_ts("2026-01-01 09:15")])
    with pytest.raises(TypeError, match="is_valid_bar"):
        certify_sessions(frame, "1", SessionDayFalseButValidBarBrokenCalendar())
