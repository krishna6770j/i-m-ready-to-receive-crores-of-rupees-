"""Data quality validation.

Design rule (requirement 27): this module FLAGS anomalies. It never repairs
them. Silent repair is how corrupt data reaches a backtest and produces a
result that looks plausible and is wrong. Repair, when justified, is an
explicit and logged decision made in ``cleaner.py``.

Every check reports a count and a bounded sample of offending timestamps, so a
report stays readable on a multi-year 1-minute dataset while still pointing at
specific rows to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from enum import Enum

import numpy as np
import pandas as pd

from core.timeutils import IST_NAME
from marketdata.schemas import (
    CLOSE,
    HIGH,
    LOW,
    OHLCV_COLUMNS,
    OPEN,
    PRICE_COLUMNS,
    TS,
    VOLUME,
)

MAX_SAMPLES = 5


class Severity(str, Enum):
    """How seriously to take an issue.

    ERROR   -- data is unusable as-is; a backtest on it would be invalid.
    WARNING -- data is usable but a human must understand the anomaly.
    INFO    -- an observation worth recording, expected in normal data.
    """

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    count: int
    samples: tuple[str, ...] = ()

    def __str__(self) -> str:
        base = f"[{self.severity.value}] {self.code}: {self.message} (count={self.count})"
        if self.samples:
            base += f"\n        e.g. {', '.join(self.samples)}"
        return base


@dataclass
class ValidationReport:
    """Structured outcome of validating one dataset."""

    symbol: str
    resolution: str
    row_count: int
    first_ts: str | None
    last_ts: str | None
    timezone: str
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        if issue.count > 0:
            self.issues.append(issue)

    def by_severity(self, severity: Severity) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is severity]

    @property
    def errors(self) -> list[ValidationIssue]:
        return self.by_severity(Severity.ERROR)

    @property
    def warnings(self) -> list[ValidationIssue]:
        return self.by_severity(Severity.WARNING)

    @property
    def is_usable(self) -> bool:
        """True when no ERROR-severity issue was found.

        Deliberately not called ``is_valid``: warnings can still make a dataset
        unsuitable for a given purpose. This only says nothing fatal was found.
        """
        return not self.errors

    def to_text(self) -> str:
        lines = [
            "=" * 72,
            "DATA QUALITY REPORT",
            "=" * 72,
            f"  symbol      : {self.symbol}",
            f"  resolution  : {self.resolution}",
            f"  rows        : {self.row_count}",
            f"  first candle: {self.first_ts}",
            f"  last candle : {self.last_ts}",
            f"  timezone    : {self.timezone}",
            "-" * 72,
        ]
        if not self.issues:
            lines.append("  No issues detected.")
        else:
            counts = {
                s.value: len(self.by_severity(s))
                for s in Severity
                if self.by_severity(s)
            }
            lines.append(f"  Issues: {counts}")
            lines.append("")
            for sev in (Severity.ERROR, Severity.WARNING, Severity.INFO):
                for issue in self.by_severity(sev):
                    lines.append(f"  {issue}")
        lines.append("-" * 72)
        lines.append(
            f"  USABLE (no ERRORs): {self.is_usable}"
        )
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "row_count": self.row_count,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "timezone": self.timezone,
            "is_usable": self.is_usable,
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity.value,
                    "message": i.message,
                    "count": i.count,
                    "samples": list(i.samples),
                }
                for i in self.issues
            ],
        }


def _samples(timestamps: pd.Series) -> tuple[str, ...]:
    return tuple(
        pd.Timestamp(t).isoformat() for t in timestamps.head(MAX_SAMPLES).tolist()
    )


def validate(
    frame: pd.DataFrame,
    *,
    symbol: str,
    resolution: str,
    expected_interval_minutes: int | None = None,
    sigma_threshold: float = 10.0,
    session_window: tuple[time, time] | None = None,
    max_session_gap_days: float | None = None,
) -> ValidationReport:
    """Run all data quality checks and return a structured report.

    Args:
        frame: Canonical OHLCV frame.
        symbol: Symbol, recorded in the report.
        resolution: Resolution string, recorded in the report.
        expected_interval_minutes: Bar length, used for gap detection. When
            None, gap detection is skipped rather than guessed at.
        sigma_threshold: Bar-to-bar return z-score beyond which a move is
            flagged for human review.
        session_window: Optional (start, end) wall-clock IST times that every
            candle must fall within, inclusive. When None the check is SKIPPED
            and recorded as skipped -- NSE session boundaries changed on
            2026-08-03 and the exact windows differ by segment and remain
            unconfirmed, so guessing them would manufacture false failures.
        max_session_gap_days: LEGACY, schema-v1-compatibility field only.
            Elapsed calendar-day gap size, by itself, can NEVER make
            MarketDataValidity INVALID -- gap MEANING (explained by a
            calendar, or not) now belongs entirely to
            ``marketdata.continuity.certify_continuity``, which requires an
            actual ``TradingCalendar`` rather than a bare day-count. When
            this field is supplied and cross-day gaps exist, a WARNING
            records that it is present but NON-GATING; when it is None,
            cross-day gaps are reported with their magnitude as a WARNING
            stating that no calendar is configured. Neither branch ever
            produces an ERROR from gap size alone.
    """
    has_ts = TS in frame.columns
    tz_attr = getattr(frame[TS].dtype, "tz", None) if has_ts else None
    report = ValidationReport(
        symbol=symbol,
        resolution=resolution,
        row_count=len(frame),
        first_ts=frame[TS].iloc[0].isoformat() if has_ts and len(frame) else None,
        last_ts=frame[TS].iloc[-1].isoformat() if has_ts and len(frame) else None,
        timezone=str(tz_attr) if tz_attr is not None else "naive/unknown",
    )

    # --- structural checks -------------------------------------------------
    missing_cols = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing_cols:
        report.add(
            ValidationIssue(
                "SCHEMA_MISSING_COLUMNS",
                Severity.ERROR,
                f"Missing required columns: {missing_cols}",
                count=len(missing_cols),
            )
        )
        return report  # nothing further is meaningful

    if len(frame) == 0:
        report.add(
            ValidationIssue(
                "EMPTY_DATASET",
                Severity.ERROR,
                "Dataset contains no candles.",
                count=1,
            )
        )
        return report

    # --- timezone ----------------------------------------------------------
    if not isinstance(frame[TS].dtype, pd.DatetimeTZDtype):
        report.add(
            ValidationIssue(
                "TZ_NAIVE",
                Severity.ERROR,
                f"Timestamps are not tz-aware (dtype={frame[TS].dtype}).",
                count=len(frame),
            )
        )
        return report
    if str(frame[TS].dtype.tz) != IST_NAME:
        report.add(
            ValidationIssue(
                "TZ_NOT_IST",
                Severity.ERROR,
                f"Timestamps are {frame[TS].dtype.tz}, expected {IST_NAME}.",
                count=len(frame),
            )
        )

    # --- ordering and duplicates -------------------------------------------
    if not frame[TS].is_monotonic_increasing:
        out_of_order = int((frame[TS].diff() < pd.Timedelta(0)).sum())
        report.add(
            ValidationIssue(
                "TS_NOT_SORTED",
                Severity.ERROR,
                "Timestamps are not in ascending order.",
                count=out_of_order,
                samples=_samples(frame.loc[frame[TS].diff() < pd.Timedelta(0), TS]),
            )
        )

    dup_mask = frame[TS].duplicated(keep=False)
    if dup_mask.any():
        dup_ts = frame.loc[dup_mask, TS].drop_duplicates()
        report.add(
            ValidationIssue(
                "DUPLICATE_TIMESTAMPS",
                Severity.ERROR,
                f"{int(dup_mask.sum())} rows share {len(dup_ts)} duplicated timestamps.",
                count=len(dup_ts),
                samples=_samples(dup_ts),
            )
        )

    # --- OHLC integrity ----------------------------------------------------
    # Rules from requirement 27:
    #   high >= max(open, close);  low <= min(open, close);  high >= low
    oc_max = frame[[OPEN, CLOSE]].max(axis=1)
    oc_min = frame[[OPEN, CLOSE]].min(axis=1)

    bad_high = frame[HIGH] < oc_max
    if bad_high.any():
        report.add(
            ValidationIssue(
                "OHLC_HIGH_TOO_LOW",
                Severity.ERROR,
                "high < max(open, close): the high does not contain the bar.",
                count=int(bad_high.sum()),
                samples=_samples(frame.loc[bad_high, TS]),
            )
        )

    bad_low = frame[LOW] > oc_min
    if bad_low.any():
        report.add(
            ValidationIssue(
                "OHLC_LOW_TOO_HIGH",
                Severity.ERROR,
                "low > min(open, close): the low does not contain the bar.",
                count=int(bad_low.sum()),
                samples=_samples(frame.loc[bad_low, TS]),
            )
        )

    bad_range = frame[HIGH] < frame[LOW]
    if bad_range.any():
        report.add(
            ValidationIssue(
                "OHLC_HIGH_BELOW_LOW",
                Severity.ERROR,
                "high < low: impossible bar.",
                count=int(bad_range.sum()),
                samples=_samples(frame.loc[bad_range, TS]),
            )
        )

    nonpositive = (frame[list(PRICE_COLUMNS)] <= 0).any(axis=1)
    if nonpositive.any():
        report.add(
            ValidationIssue(
                "NON_POSITIVE_PRICE",
                Severity.ERROR,
                "One or more OHLC values are zero or negative.",
                count=int(nonpositive.sum()),
                samples=_samples(frame.loc[nonpositive, TS]),
            )
        )

    nulls = frame[list(PRICE_COLUMNS)].isna().any(axis=1)
    if nulls.any():
        report.add(
            ValidationIssue(
                "NULL_PRICE",
                Severity.ERROR,
                "One or more OHLC values are null/NaN.",
                count=int(nulls.sum()),
                samples=_samples(frame.loc[nulls, TS]),
            )
        )

    # Non-finite prices. +inf passes every comparison-based rule above
    # (it is "positive", it is >= everything), so it needs its own check or an
    # infinite price would be classified as a valid bar.
    non_finite = ~np.isfinite(frame[list(PRICE_COLUMNS)].to_numpy(dtype="float64"))
    non_finite_rows = pd.Series(non_finite.any(axis=1), index=frame.index)
    only_inf = non_finite_rows & ~nulls
    if only_inf.any():
        report.add(
            ValidationIssue(
                "NON_FINITE_PRICE",
                Severity.ERROR,
                "One or more OHLC values are +inf or -inf. An infinite price is "
                "not a market observation.",
                count=int(only_inf.sum()),
                samples=_samples(frame.loc[only_inf, TS]),
            )
        )

    # Volume is nullable Int64 so that missing volume stays missing. Both the
    # missing case and the negative case are reported rather than repaired.
    null_volume = frame[VOLUME].isna()
    if null_volume.any():
        report.add(
            ValidationIssue(
                "NULL_VOLUME",
                Severity.ERROR,
                "Volume is missing. It is reported, never filled with 0, because "
                "0 would assert that no trading occurred.",
                count=int(null_volume.sum()),
                samples=_samples(frame.loc[null_volume, TS]),
            )
        )

    negative_volume = (frame[VOLUME] < 0).fillna(False)
    if negative_volume.any():
        report.add(
            ValidationIssue(
                "NEGATIVE_VOLUME",
                Severity.ERROR,
                "Negative volume.",
                count=int(negative_volume.sum()),
                samples=_samples(frame.loc[negative_volume, TS]),
            )
        )

    # Unit 13B contract: whenever max_session_gap_days is supplied, exactly
    # one LEGACY_MAX_SESSION_GAP_POLICY_NON_GATING issue is recorded,
    # regardless of whether any cross-day gap (or any gap at all) actually
    # exists -- a supplied legacy field must never be silently ignored.
    # Set True the moment it is emitted below, from whichever path reaches
    # it first, so it is never emitted twice.
    legacy_gap_policy_issue_emitted = False

    # --- minute alignment --------------------------------------------------
    if expected_interval_minutes is not None:
        seconds = frame[TS].dt.second
        misaligned = seconds != 0
        if misaligned.any():
            report.add(
                ValidationIssue(
                    "TS_NOT_MINUTE_ALIGNED",
                    Severity.WARNING,
                    "Timestamps have non-zero seconds; expected minute-aligned bars.",
                    count=int(misaligned.sum()),
                    samples=_samples(frame.loc[misaligned, TS]),
                )
            )

        # --- gaps ----------------------------------------------------------
        # Gaps are reported, never filled. In a 1-minute index series, gaps
        # occur legitimately (no trades in a minute, lunch-time thinness,
        # session boundaries), so this is INFO within a session and the caller
        # decides what matters. Filling them would invent prices.
        step = pd.Timedelta(minutes=expected_interval_minutes)
        deltas = frame[TS].diff()
        intraday_gaps = deltas > step
        # Overnight boundaries are expected; separate them from within-day gaps.
        same_day = frame[TS].dt.date == frame[TS].shift().dt.date
        within_day_gaps = intraday_gaps & same_day
        overnight = intraday_gaps & ~same_day

        if within_day_gaps.any():
            missing_bars = int(
                ((deltas[within_day_gaps] / step) - 1).round().sum()
            )
            report.add(
                ValidationIssue(
                    "WITHIN_DAY_GAPS",
                    Severity.WARNING,
                    f"{int(within_day_gaps.sum())} within-session gaps, "
                    f"approximately {missing_bars} missing bars. Not filled: "
                    "gaps are characterised, never interpolated.",
                    count=int(within_day_gaps.sum()),
                    samples=_samples(frame.loc[within_day_gaps, TS]),
                )
            )
        if overnight.any():
            # A cross-day gap may be an ordinary overnight break, a weekend, a
            # market holiday, or a large block of missing data. Without a
            # trading calendar this project cannot tell those apart, so it
            # reports the MAGNITUDE and refuses to call them all "expected".
            gap_days = (deltas[overnight] / pd.Timedelta(days=1)).astype(float)
            largest = float(gap_days.max())

            if max_session_gap_days is None:
                report.add(
                    ValidationIssue(
                        "TRADING_CALENDAR_NOT_CONFIGURED",
                        Severity.WARNING,
                        f"{int(overnight.sum())} cross-day gap(s); largest spans "
                        f"{largest:.1f} calendar days. No trading calendar is "
                        "configured, so weekends, holidays and genuinely missing "
                        "data CANNOT be distinguished. Configure a "
                        "TradingCalendar and use continuity/session "
                        "certification (marketdata.continuity) when "
                        "calendar-aware research suitability is required.",
                        count=int(overnight.sum()),
                        samples=_samples(frame.loc[overnight, TS]),
                    )
                )
            else:
                # LEGACY, schema-v1-compatibility field only (frozen
                # architecture section 19): elapsed gap size, by itself,
                # must NEVER make MarketDataValidity INVALID. This field is
                # retained only because it is already serialised in
                # provenance schema v1 -- it no longer gates anything.
                # Gap MEANING now belongs entirely to
                # marketdata.continuity.certify_continuity(), which
                # requires an actual TradingCalendar rather than a bare
                # day-count threshold.
                exceeding = overnight & (
                    deltas > pd.Timedelta(days=max_session_gap_days)
                )
                report.add(
                    ValidationIssue(
                        "LEGACY_MAX_SESSION_GAP_POLICY_NON_GATING",
                        Severity.WARNING,
                        f"{int(overnight.sum())} cross-day gap(s); largest spans "
                        f"{largest:.1f} calendar days. max_session_gap_days="
                        f"{max_session_gap_days} is set ({int(exceeding.sum())} "
                        "gap(s) would have exceeded it), but this field is "
                        "retained for schema-v1 compatibility only and no "
                        "longer gates MarketDataValidity -- elapsed gap size "
                        "alone can never make data INVALID. Use "
                        "marketdata.continuity.certify_continuity() with a "
                        "configured TradingCalendar to certify continuity.",
                        count=int(overnight.sum()),
                        samples=_samples(frame.loc[overnight, TS]),
                    )
                )
                legacy_gap_policy_issue_emitted = True

        # Unit 13B contract: a supplied legacy field must never be silently
        # ignored just because no cross-day gap happened to occur (or gap
        # detection did not even run). Emit the non-gating notice regardless.
        if max_session_gap_days is not None and not legacy_gap_policy_issue_emitted:
            report.add(
                ValidationIssue(
                    "LEGACY_MAX_SESSION_GAP_POLICY_NON_GATING",
                    Severity.WARNING,
                    f"max_session_gap_days={max_session_gap_days} is set, but no "
                    "cross-day gap was observed in this dataset. This field is "
                    "retained for schema-v1 compatibility only and does not "
                    "affect MarketDataValidity -- elapsed gap size alone can "
                    "never make data INVALID. Use "
                    "marketdata.continuity.certify_continuity() with a "
                    "configured TradingCalendar to certify continuity.",
                    count=1,
                )
            )
            legacy_gap_policy_issue_emitted = True
    if max_session_gap_days is not None and not legacy_gap_policy_issue_emitted:
        report.add(
            ValidationIssue(
                "LEGACY_MAX_SESSION_GAP_POLICY_NON_GATING",
                Severity.WARNING,
                f"max_session_gap_days={max_session_gap_days} is set, but gap "
                "detection was not evaluated (no expected_interval_minutes "
                "supplied). This field is retained for schema-v1 "
                "compatibility only and does not affect MarketDataValidity -- "
                "use marketdata.continuity.certify_continuity() with a "
                "configured TradingCalendar to certify continuity.",
                count=1,
            )
        )
        legacy_gap_policy_issue_emitted = True

    # --- session boundaries ------------------------------------------------
    if session_window is None:
        report.add(
            ValidationIssue(
                "SESSION_WINDOW_NOT_CHECKED",
                Severity.INFO,
                "No session window supplied, so candles were NOT checked against "
                "trading hours. NSE changed session boundaries on 2026-08-03 and "
                "the exact windows per segment are unconfirmed; supply "
                "session_window once the execution instrument and its segment "
                "are settled.",
                count=1,
            )
        )
    else:
        start, end = session_window
        wall = frame[TS].dt.time
        outside = (wall < start) | (wall > end)
        if outside.any():
            report.add(
                ValidationIssue(
                    "OUTSIDE_SESSION_WINDOW",
                    Severity.WARNING,
                    f"Candles fall outside the configured session "
                    f"{start.isoformat()}-{end.isoformat()} IST.",
                    count=int(outside.sum()),
                    samples=_samples(frame.loc[outside, TS]),
                )
            )

    # --- anomalous moves ---------------------------------------------------
    if len(frame) > 30:
        returns = frame[CLOSE].pct_change()
        std = returns.std()
        if std and std > 0 and np.isfinite(std):
            z = (returns - returns.mean()).abs() / std
            extreme = z > sigma_threshold
            if extreme.any():
                report.add(
                    ValidationIssue(
                        "EXTREME_RETURN",
                        Severity.WARNING,
                        f"Bar-to-bar returns beyond {sigma_threshold} sigma. These may "
                        "be genuine events or bad ticks; review before use.",
                        count=int(extreme.sum()),
                        samples=_samples(frame.loc[extreme, TS]),
                    )
                )

    return report
