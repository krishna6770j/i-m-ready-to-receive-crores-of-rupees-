"""Continuity and session certification primitives, per the frozen
architecture (docs/architecture/phase1-trust-hardening.md, section 19).

**Pure functions only.** ``certify_continuity()``/``certify_sessions()`` take
an already-canonical frame and a ``TradingCalendar`` and return an immutable
certification result. Neither function sorts, repairs, or otherwise
canonicalises its input -- ``assert_canonical()`` is called first and raises
outright if the frame is not already canonical.

**No integration with ``TrustedDataset``/``ResearchDataPolicy`` in this
unit.** These are standalone primitives; wiring them into the trust/research
layers is explicitly a later, separately-authorised unit.

**Architecture correction (section 19, this unit).** ``is_session_day(date)``
and ``expected_next_bar(ts, resolution)`` are both RELATIONAL -- they answer
questions about a calendar day as a whole, or about the transition between
two consecutive observations. Neither can answer, for one arbitrary
timestamp in isolation (most importantly the FIRST candle in a dataset,
which has no predecessor to relate it to), whether that exact instant is a
valid session bar: ``is_session_day`` only confirms the calendar DAY trades
at all, not that a given intraday timestamp falls inside session hours or on
a valid bar boundary for the resolution. Session certification needs that
per-timestamp answer for every candle, including the first. ``TradingCalendar``
therefore gains ``is_valid_bar(ts, resolution) -> bool`` -- a protocol
capability only; no NSE session-hour data is invented anywhere in this
module. Only ``NullCalendar`` ships, and it answers ``False``
unconditionally (no calendar knowledge).

**No arbitrary day-count threshold anywhere in this module.** The frozen
architecture removed ``ABSURD_GAP_DAYS = 30`` entirely (section 19): an
overnight or weekend-sized elapsed gap is exactly as EXPLAINED or
UNEXPLAINED as any other gap -- purely by whether a configured calendar says
the next observed candle is the expected next bar. There is no numeric
elapsed-time cutoff distinguishing "acceptable" from "suspicious" gaps
anywhere in this code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd

from core.timeutils import IST_NAME
from core.types import Resolution
from marketdata.schemas import TS, assert_canonical


class CertificationStatus(str, Enum):
    """The only three outcomes a certification may report. ``NOT_CERTIFIED``
    is not a failure -- it is an honest statement that the question cannot
    be answered (no calendar configured).
    """

    CERTIFIED = "CERTIFIED"
    NOT_CERTIFIED = "NOT_CERTIFIED"
    FAILED = "FAILED"


class CalendarExplanation(str, Enum):
    """Whether a configured calendar explains one observed transition.
    Without a calendar, every gap is ``UNKNOWN`` -- uniformly, regardless of
    elapsed size.
    """

    EXPLAINED = "EXPLAINED"
    UNEXPLAINED = "UNEXPLAINED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ObservedGap:
    """Pure fact: elapsed time between two consecutive observations whose
    gap differs from the resolution's normal interval. NO severity, NO
    ERROR/WARNING classification -- what the gap MEANS (explained by a
    calendar, or not) is recorded separately, never on this object.
    """

    previous_ts: str
    current_ts: str
    elapsed: pd.Timedelta


@dataclass(frozen=True, slots=True)
class GapExplanation:
    """One ``ObservedGap`` paired with whether a calendar explains it."""

    gap: ObservedGap
    explanation: CalendarExplanation


@dataclass(frozen=True, slots=True)
class ContinuityCertification:
    """Result of :func:`certify_continuity`. ``gap_explanations`` covers
    every observed transition whose elapsed time differed from the
    resolution's normal interval -- an ordinary back-to-back bar produces no
    entry at all, since there is no gap fact worth recording for it.
    """

    status: CertificationStatus
    calendar_id: str
    calendar_version: str
    gap_explanations: tuple[GapExplanation, ...]


@dataclass(frozen=True, slots=True)
class SessionCertification:
    """Result of :func:`certify_sessions`. ``invalid_timestamps`` records
    (as ISO-8601 strings) every candle that failed ``is_session_day`` and/or
    ``is_valid_bar`` -- immutably, so the specific failing instants survive
    past the boolean verdict.
    """

    status: CertificationStatus
    calendar_id: str
    calendar_version: str
    invalid_timestamps: tuple[str, ...]
    checked_count: int


@runtime_checkable
class TradingCalendar(Protocol):
    """Protocol only -- no NSE calendar data is invented anywhere in this
    codebase. ``NullCalendar`` (below) is the only shipped implementation.

    ``calendar_id``/``calendar_version`` identify the specific calendar
    implementation (they enter provenance when a real calendar arrives);
    they are never a market-hours claim by themselves.
    """

    calendar_id: str
    calendar_version: str

    def is_session_day(self, day: _date) -> bool:
        """Whether ``day`` is a trading session day at all."""
        ...

    def is_valid_bar(self, ts: pd.Timestamp, resolution: str) -> bool:
        """Whether ``ts`` is, independently, a valid bar timestamp for
        ``resolution`` -- inside session hours and on a valid bar boundary.
        Needed because ``is_session_day`` alone cannot answer this for a
        single arbitrary timestamp (see the module docstring).
        """
        ...

    def expected_next_bar(self, ts: pd.Timestamp, resolution: str) -> pd.Timestamp:
        """The next bar timestamp the calendar expects after ``ts``, given
        ``resolution``. Must return a tz-aware IST timestamp."""
        ...


class NullCalendar:
    """Represents NO CALENDAR KNOWLEDGE. Must never accidentally certify
    anything -- ``is_session_day``/``is_valid_bar`` both answer ``False``
    unconditionally, and ``expected_next_bar`` raises outright rather than
    guessing, since :func:`certify_continuity` special-cases ``NullCalendar``
    before ever calling it.

    ``calendar_id``/``calendar_version`` identify THIS IMPLEMENTATION only
    ("null"/"1") -- never a market calendar. No weekends, holidays, or NSE
    hours are encoded anywhere in this class.
    """

    calendar_id = "null"
    calendar_version = "1"

    def is_session_day(self, day: _date) -> bool:
        return False

    def is_valid_bar(self, ts: pd.Timestamp, resolution: str) -> bool:
        return False

    def expected_next_bar(self, ts: pd.Timestamp, resolution: str) -> pd.Timestamp:
        raise NotImplementedError(
            "NullCalendar carries no calendar knowledge; expected_next_bar() "
            "must never actually be called on it. certify_continuity() "
            "special-cases NullCalendar before ever reaching this method."
        )


_REQUIRED_CALENDAR_ATTRS = (
    "calendar_id",
    "calendar_version",
    "is_session_day",
    "is_valid_bar",
    "expected_next_bar",
)


def _require_calendar_shape(calendar: object) -> None:
    missing = [a for a in _REQUIRED_CALENDAR_ATTRS if not hasattr(calendar, a)]
    if missing:
        raise TypeError(
            "calendar does not implement the TradingCalendar protocol; "
            f"missing: {missing}"
        )


def _require_calendar_identity(calendar: TradingCalendar) -> None:
    if not isinstance(calendar.calendar_id, str) or calendar.calendar_id == "":
        raise ValueError(
            f"calendar_id must be a non-empty str, got {calendar.calendar_id!r}"
        )
    if not isinstance(calendar.calendar_version, str) or calendar.calendar_version == "":
        raise ValueError(
            "calendar_version must be a non-empty str, got "
            f"{calendar.calendar_version!r}"
        )


def _require_ist_timestamp(ts: object, label: str) -> pd.Timestamp:
    if not isinstance(ts, pd.Timestamp):
        raise TypeError(f"{label} must be a pandas Timestamp, got {type(ts).__name__}")
    if ts.tz is None:
        raise ValueError(f"{label} must be tz-aware IST, got a naive timestamp: {ts!r}")
    if str(ts.tz) != IST_NAME:
        raise ValueError(f"{label} must be tz-aware IST ({IST_NAME}), got tz={ts.tz!r}")
    return ts


def _require_bool(value: object, label: str) -> bool:
    """Exact ``bool`` only -- ``"yes"``, ``1``, or any other truthy-but-not-
    ``bool`` value is rejected outright rather than coerced with
    ``bool(...)``. A calendar returning the wrong type is a protocol
    violation, not something to silently paper over.
    """
    if type(value) is not bool:
        raise TypeError(
            f"{label} must return an actual bool, got {value!r} ({type(value).__name__})"
        )
    return value


def _require_resolution(resolution: str) -> Resolution:
    try:
        return Resolution(resolution)
    except ValueError as exc:
        raise ValueError(
            f"Resolution {resolution!r} is not in the documented set "
            f"{sorted(r.value for r in Resolution)}. Undocumented resolutions "
            "are not supported."
        ) from exc


def _normal_interval(resolution_enum: Resolution) -> pd.Timedelta:
    if resolution_enum is Resolution.DAY:
        return pd.Timedelta(days=1)
    return pd.Timedelta(minutes=resolution_enum.minutes)


def certify_continuity(
    frame: pd.DataFrame, resolution: str, calendar: TradingCalendar
) -> ContinuityCertification:
    """Certify whether consecutive observations in ``frame`` are all
    calendar-explained. ``frame`` must already be canonical -- this
    function never sorts or repairs it (``assert_canonical`` raises
    outright if it is not).

    With ``NullCalendar``: always ``NOT_CERTIFIED``; every recorded gap's
    explanation is ``UNKNOWN``.

    With a configured calendar: for every consecutive pair,
    ``expected = calendar.expected_next_bar(previous_ts, resolution)``. If
    ``current_ts == expected`` the transition is ``EXPLAINED``; otherwise
    ``UNEXPLAINED``. All transitions explained -> ``CERTIFIED``; any
    unexplained transition -> ``FAILED``. There is no elapsed-time
    threshold: an overnight/weekend-sized gap is fully ``CERTIFIED`` if the
    calendar says the observed candle is exactly the expected next bar.

    With FEWER THAN 2 observations there is no transition to evaluate at
    all -- continuity answers whether transitions between observations are
    complete, and with 0 or 1 candles that question has no evidence to
    answer it either way. This is ``NOT_CERTIFIED`` (insufficient evidence),
    never ``CERTIFIED`` (vacuously) and never ``FAILED`` (there is nothing
    contradictory), for BOTH ``NullCalendar`` and a configured calendar.
    """
    assert_canonical(frame)
    resolution_enum = _require_resolution(resolution)
    normal_interval = _normal_interval(resolution_enum)
    timestamps: list[pd.Timestamp] = list(frame[TS])

    gap_explanations: list[GapExplanation] = []

    if isinstance(calendar, NullCalendar):
        for i in range(1, len(timestamps)):
            previous_ts, current_ts = timestamps[i - 1], timestamps[i]
            elapsed = current_ts - previous_ts
            if elapsed != normal_interval:
                gap_explanations.append(
                    GapExplanation(
                        gap=ObservedGap(
                            previous_ts=previous_ts.isoformat(),
                            current_ts=current_ts.isoformat(),
                            elapsed=elapsed,
                        ),
                        explanation=CalendarExplanation.UNKNOWN,
                    )
                )
        return ContinuityCertification(
            status=CertificationStatus.NOT_CERTIFIED,
            calendar_id=calendar.calendar_id,
            calendar_version=calendar.calendar_version,
            gap_explanations=tuple(gap_explanations),
        )

    _require_calendar_shape(calendar)
    _require_calendar_identity(calendar)

    if len(timestamps) < 2:
        return ContinuityCertification(
            status=CertificationStatus.NOT_CERTIFIED,
            calendar_id=calendar.calendar_id,
            calendar_version=calendar.calendar_version,
            gap_explanations=(),
        )

    all_explained = True
    for i in range(1, len(timestamps)):
        previous_ts, current_ts = timestamps[i - 1], timestamps[i]
        elapsed = current_ts - previous_ts
        expected = _require_ist_timestamp(
            calendar.expected_next_bar(previous_ts, resolution),
            "expected_next_bar(...) return value",
        )
        explanation = (
            CalendarExplanation.EXPLAINED
            if current_ts == expected
            else CalendarExplanation.UNEXPLAINED
        )
        if explanation is CalendarExplanation.UNEXPLAINED:
            all_explained = False
        if elapsed != normal_interval:
            gap_explanations.append(
                GapExplanation(
                    gap=ObservedGap(
                        previous_ts=previous_ts.isoformat(),
                        current_ts=current_ts.isoformat(),
                        elapsed=elapsed,
                    ),
                    explanation=explanation,
                )
            )

    status = CertificationStatus.CERTIFIED if all_explained else CertificationStatus.FAILED
    return ContinuityCertification(
        status=status,
        calendar_id=calendar.calendar_id,
        calendar_version=calendar.calendar_version,
        gap_explanations=tuple(gap_explanations),
    )


def certify_sessions(
    frame: pd.DataFrame, resolution: str, calendar: TradingCalendar
) -> SessionCertification:
    """Certify whether every candle in ``frame`` is independently a valid
    session bar. ``frame`` must already be canonical -- this function never
    sorts or repairs it.

    With ``NullCalendar``: always ``NOT_CERTIFIED``.

    With a configured calendar: each candle must satisfy BOTH
    ``calendar.is_session_day(ts.date())`` and
    ``calendar.is_valid_bar(ts, resolution)``. Session validity is never
    inferred from ``is_session_day`` alone -- this is exactly why
    ``is_valid_bar`` exists on the protocol (see the module docstring).
    """
    assert_canonical(frame)
    _require_resolution(resolution)
    timestamps: list[pd.Timestamp] = list(frame[TS])

    if isinstance(calendar, NullCalendar):
        return SessionCertification(
            status=CertificationStatus.NOT_CERTIFIED,
            calendar_id=calendar.calendar_id,
            calendar_version=calendar.calendar_version,
            invalid_timestamps=(),
            checked_count=len(timestamps),
        )

    _require_calendar_shape(calendar)
    _require_calendar_identity(calendar)

    invalid: list[str] = []
    for ts in timestamps:
        # Both facts are evaluated and type-checked EXPLICITLY, never via
        # `a() and b()` short-circuiting -- if is_session_day() returns
        # False first, short-circuiting would skip calling is_valid_bar()
        # entirely, letting a broken is_valid_bar() implementation go
        # completely undetected on that candle.
        session_day_result = _require_bool(
            calendar.is_session_day(ts.date()), "is_session_day(...) return value"
        )
        valid_bar_result = _require_bool(
            calendar.is_valid_bar(ts, resolution), "is_valid_bar(...) return value"
        )
        valid = session_day_result and valid_bar_result
        if not valid:
            invalid.append(ts.isoformat())

    status = CertificationStatus.CERTIFIED if not invalid else CertificationStatus.FAILED
    return SessionCertification(
        status=status,
        calendar_id=calendar.calendar_id,
        calendar_version=calendar.calendar_version,
        invalid_timestamps=tuple(invalid),
        checked_count=len(timestamps),
    )
