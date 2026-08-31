"""Read-only FYERS historical candle adapter.

Scope is strictly historical data. This module has no order, position or funds
capability, by design (manager correction #14).

Chunking: the per-request range limit is not officially documented, so the
adapter splits requests conservatively and reports the coverage it actually
received. If the vendor silently truncates, the gap appears in the coverage
report rather than as quietly missing candles.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import pandas as pd

from brokers.base import (
    BrokerAuthError,
    BrokerDataError,
    BrokerRateLimitError,
    HistoricalDataProvider,
)
from brokers.fyers import endpoints as ep
from core.timeutils import to_api_date
from core.types import Resolution
from marketdata.schemas import TS, empty_ohlcv, from_fyers_candles, normalise

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Outcome of one /history request, for the coverage report."""

    range_from: str
    range_to: str
    rows: int
    ok: bool
    error: str | None = None


@dataclass
class FetchReport:
    """What was requested versus what actually arrived.

    Manager correction #12 requires reporting available, requested, downloaded
    and missing ranges rather than assuming the vendor served everything.
    """

    symbol: str
    resolution: str
    requested_from: str
    requested_to: str
    chunks: list[ChunkResult] = field(default_factory=list)
    total_rows: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    duplicate_rows_removed: int = 0
    conflicting_timestamps: int = 0

    @property
    def failed_chunks(self) -> list[ChunkResult]:
        return [c for c in self.chunks if not c.ok]

    @property
    def empty_chunks(self) -> list[ChunkResult]:
        return [c for c in self.chunks if c.ok and c.rows == 0]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "requested_from": self.requested_from,
            "requested_to": self.requested_to,
            "downloaded_first_ts": self.first_ts,
            "downloaded_last_ts": self.last_ts,
            "total_rows": self.total_rows,
            "chunks_requested": len(self.chunks),
            "chunks_failed": len(self.failed_chunks),
            "chunks_empty": len(self.empty_chunks),
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "conflicting_timestamps": self.conflicting_timestamps,
            # Consumed by the storage layer to decide whether this acquisition
            # may be represented as complete.
            "failed_chunk_detail": [
                {"from": c.range_from, "to": c.range_to, "error": c.error}
                for c in self.failed_chunks
            ],
            "chunk_detail": [
                {
                    "from": c.range_from,
                    "to": c.range_to,
                    "rows": c.rows,
                    "ok": c.ok,
                    "error": c.error,
                }
                for c in self.chunks
            ],
        }


class ProbeWindowStatus(str, Enum):
    """Classification of one probed window's OUTCOME, never a claim about
    what lies outside the window.

    - ``DATA``: the request completed successfully and returned >= 1 candle.
    - ``EMPTY_SUCCESS``: the request completed successfully and returned
      zero candles. This is a genuine observation (e.g. a holiday, or a
      period truly outside served history) -- but on its own it proves
      nothing about windows on either side of it.
    - ``ERROR``: the broker/data/rate-limit request failed. This is
      UNRESOLVED evidence, not evidence of absence -- it must never be
      treated as equivalent to ``EMPTY_SUCCESS``.
    - ``UNKNOWN``: the window was never successfully resolved/probed at all
      (reserved for future callers that construct a report without probing
      every window; the scan in this module always resolves every window it
      visits to one of the other three statuses).
    """

    DATA = "DATA"
    EMPTY_SUCCESS = "EMPTY_SUCCESS"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProbeWindow:
    """One probed window's outcome. Immutable -- a probe result must not be
    edited after the fact.
    """

    range_from: str
    range_to: str
    status: ProbeWindowStatus
    row_count: int
    earliest_ts: str | None = None
    latest_ts: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class HistoryDepthProbeReport:
    """Result of :meth:`FyersHistoricalData.probe_history_depth`.

    This is an OBSERVATION report, not a retention/availability
    certification. See :meth:`FyersHistoricalData.probe_history_depth`'s
    docstring for exactly what it does and does not establish.

    ``oldest_contiguous_empty_success_interval``: the oldest contiguous run
    of adjacent ``EMPTY_SUCCESS`` coarse windows directly older than
    (adjacent to) ``oldest_data_window``, if one exists -- narrowly an
    observed fact about probed windows, NEVER proof of no earlier data, a
    retention boundary, or a trading-calendar gap certification. Empty if
    no such run exists (including when ``oldest_data_window`` is ``None``).

    ``unresolved_intervals``: every probed window (coarse or subdivision)
    that came back ``ERROR`` or ``UNKNOWN`` -- evidence this scan could not
    resolve, never silently treated as absence.
    """

    symbol: str
    resolution: str
    search_horizon_start: str
    search_horizon_end: str
    coarse_window_days: int
    subdivision_resolution_days: int
    windows: tuple[ProbeWindow, ...]
    earliest_observed_candle: str | None
    earliest_observed_date: str | None
    oldest_data_window: ProbeWindow | None
    oldest_contiguous_empty_success_interval: tuple[ProbeWindow, ...]
    unresolved_intervals: tuple[ProbeWindow, ...]


class FyersHistoricalData(HistoricalDataProvider):
    """Fetches historical candles via ``FyersModel.history``.

    The ``client`` argument is any object exposing ``history(data: dict) -> dict``.
    In production this is a ``fyers_apiv3.fyersModel.FyersModel``; in tests it is
    a fake implementing the same one-method contract. Injecting it keeps the
    adapter testable without credentials or network access.
    """

    def __init__(self, client, *, request_pause_seconds: float = 0.25) -> None:
        self._client = client
        # Self-imposed throttle. The per-second limit is unverified, so we stay
        # well below the reported ceiling rather than probing it.
        self._pause = request_pause_seconds

    @property
    def source_name(self) -> str:
        return "fyers:history"

    # -- internals --------------------------------------------------------

    @staticmethod
    def _max_days(resolution: str) -> int:
        if resolution == Resolution.DAY.value:
            return ep.ASSUMED_MAX_DAYS_PER_REQUEST_DAILY
        return ep.ASSUMED_MAX_DAYS_PER_REQUEST_INTRADAY

    @staticmethod
    def _validate_resolution(resolution: str) -> None:
        allowed = {r.value for r in Resolution}
        if resolution not in allowed:
            raise ValueError(
                f"Resolution {resolution!r} is not in the SDK-documented set "
                f"{sorted(allowed)}. Do not pass undocumented resolutions."
            )

    def _parse_response(self, payload) -> pd.DataFrame:
        """Turn a /history response into a canonical frame, or raise."""
        if not isinstance(payload, dict):
            raise BrokerDataError(
                f"Expected a dict from /history, got {type(payload).__name__}."
            )

        status = payload.get(ep.STATUS_KEY)
        if status != ep.STATUS_OK:
            message = str(payload.get(ep.MESSAGE_KEY, "")).lower()
            code = payload.get("code")
            detail = f"status={status!r} code={code!r} message={payload.get(ep.MESSAGE_KEY)!r}"
            if "token" in message or "auth" in message or code in (-15, -16, -17):
                raise BrokerAuthError(
                    f"FYERS rejected the request as unauthenticated ({detail}). "
                    "The access token is short-lived and must be regenerated each "
                    "trading day."
                )
            if "rate" in message or "limit" in message or "too many" in message:
                raise BrokerRateLimitError(f"FYERS rate limit hit ({detail}).")
            raise BrokerDataError(f"FYERS /history returned an error ({detail}).")

        if ep.CANDLES_KEY not in payload:
            raise BrokerDataError(
                f"/history response has status 'ok' but no {ep.CANDLES_KEY!r} key. "
                f"Keys present: {sorted(payload)}. The response format may have "
                "changed; verify against current FYERS documentation."
            )
        return from_fyers_candles(payload[ep.CANDLES_KEY] or [])

    # -- public API -------------------------------------------------------

    def fetch_chunk(
        self, symbol: str, resolution: str, start: date, end: date
    ) -> pd.DataFrame:
        """Single /history request for a range within the per-request limit."""
        self._validate_resolution(resolution)
        request = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": ep.DATE_FORMAT_YMD,
            "range_from": to_api_date(start),
            "range_to": to_api_date(end),
            "cont_flag": 1,
        }
        logger.info(
            "fyers /history request symbol=%s resolution=%s from=%s to=%s",
            symbol,
            resolution,
            request["range_from"],
            request["range_to"],
        )
        payload = self._client.history(data=request)
        return self._parse_response(payload)

    def fetch_candles_with_report(
        self, symbol: str, resolution: str, start: date, end: date
    ) -> tuple[pd.DataFrame, FetchReport]:
        """Fetch a full range in chunks, returning data and a coverage report."""
        self._validate_resolution(resolution)
        if end < start:
            raise ValueError(f"end {end} is before start {start}")

        report = FetchReport(
            symbol=symbol,
            resolution=resolution,
            requested_from=to_api_date(start),
            requested_to=to_api_date(end),
        )

        max_days = self._max_days(resolution)
        frames: list[pd.DataFrame] = []
        cursor = start
        first = True

        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=max_days - 1), end)
            if not first and self._pause:
                time.sleep(self._pause)
            first = False
            try:
                chunk = self.fetch_chunk(symbol, resolution, cursor, chunk_end)
                frames.append(chunk)
                report.chunks.append(
                    ChunkResult(
                        to_api_date(cursor), to_api_date(chunk_end), len(chunk), True
                    )
                )
            except (BrokerDataError, BrokerRateLimitError) as exc:
                # Record and continue: one bad window should not discard an
                # otherwise good multi-year download. The gap is visible in the
                # report, and auth errors deliberately still abort.
                logger.error(
                    "chunk %s..%s failed: %s",
                    to_api_date(cursor),
                    to_api_date(chunk_end),
                    exc,
                )
                report.chunks.append(
                    ChunkResult(
                        to_api_date(cursor),
                        to_api_date(chunk_end),
                        0,
                        False,
                        str(exc),
                    )
                )
            cursor = chunk_end + timedelta(days=1)

        combined = (
            normalise(pd.concat(frames, ignore_index=True)) if frames else empty_ohlcv()
        )

        # Chunk boundaries can overlap: brokers commonly treat range_from and
        # range_to as inclusive, so the same candle may arrive in two chunks.
        # Two distinct cases, handled differently on purpose:
        #
        #   1. Byte-identical rows carry no information beyond the first copy,
        #      so collapsing them is lossless. Recorded, not silent.
        #   2. Rows sharing a timestamp but disagreeing on OHLCV are a genuine
        #      conflict. This code REFUSES to choose between them -- picking one
        #      would silently discard a contradictory observation of the same
        #      minute. They are left in place so the validator raises
        #      DUPLICATE_TIMESTAMPS, and the count is surfaced here.
        before = len(combined)
        combined = combined.drop_duplicates(keep="first").reset_index(drop=True)
        report.duplicate_rows_removed = before - len(combined)

        conflicts = combined[TS].duplicated(keep=False)
        report.conflicting_timestamps = int(combined.loc[conflicts, TS].nunique())
        if report.conflicting_timestamps:
            logger.error(
                "%d timestamp(s) appear more than once with DIFFERENT values across "
                "chunk boundaries; not resolved automatically",
                report.conflicting_timestamps,
            )

        report.total_rows = len(combined)
        if len(combined):
            report.first_ts = combined[TS].iloc[0].isoformat()
            report.last_ts = combined[TS].iloc[-1].isoformat()
        return combined, report

    def fetch_candles(
        self, symbol: str, resolution: str, start: date, end: date
    ) -> pd.DataFrame:
        frame, _ = self.fetch_candles_with_report(symbol, resolution, start, end)
        return frame

    def probe_history_depth(
        self,
        symbol: str,
        resolution: str,
        *,
        newest: date,
        oldest_to_try: date,
        coarse_window_days: int = 5,
        subdivision_resolution_days: int = 1,
    ) -> "HistoryDepthProbeReport":
        """Bounded backward-window scan of OBSERVED history.

        This method establishes, at most, two facts:

            A. the earliest candle actually observed within the probed horizon
            B. the earliest calendar date (IST) that candle falls on

        It does NOT and CANNOT establish a broker "retention boundary", a
        continuous-history boundary (that needs a trading calendar, which
        this method does not have), or that no data exists before/inside the
        horizon that this scan simply never asked about. If nothing was
        observed, ``earliest_observed_candle``/``earliest_observed_date`` are
        ``None`` -- that is silence, not proof of absence.

        Replaces the previous binary search, which assumed "did this narrow
        window return data?" was a MONOTONIC predicate across the horizon
        (i.e. that once you cross from "no data" to "data" walking backward
        in time, you never cross back). That assumption is false: a single
        holiday-shaped empty window between two windows that both have data
        makes a binary search converge on the WRONG boundary, silently
        skipping genuine older data on the other side of the gap. This
        method instead probes every coarse window in the horizon and never
        infers through an unresolved (ERROR) or empty result.

        Algorithm:

        1. Walk backward from ``newest`` in non-overlapping
           ``coarse_window_days``-sized windows down to ``oldest_to_try``
           (the last window is clipped to ``oldest_to_try``), probing every
           one -- never stopping early on an EMPTY_SUCCESS or ERROR result.
        2. Find the OLDEST coarse window classified DATA (if any). Only that
           one window is subdivided -- EMPTY_SUCCESS/ERROR windows are never
           subdivided, since neither proves anything about the data (or lack
           of it) inside them at finer granularity.
        3. Subdivide that window, backward, into non-overlapping
           ``subdivision_resolution_days``-sized windows, probing every one,
           to tighten the bracket around the earliest OBSERVED candle. The
           final ``earliest_observed_candle``/``earliest_observed_date`` are
           read from actual returned candle timestamps, never a requested
           window's start date.

        ``BrokerAuthError`` propagates immediately and aborts the whole scan
        (never downgraded to ERROR/EMPTY -- an expired token is not evidence
        about history depth). ``BrokerRateLimitError``/``BrokerDataError``
        become ``ProbeWindowStatus.ERROR`` windows and the scan continues.
        Any other exception propagates unchanged.
        """
        self._validate_resolution(resolution)
        if oldest_to_try > newest:
            raise ValueError(
                f"oldest_to_try ({oldest_to_try}) must be <= newest ({newest})."
            )
        if coarse_window_days <= 0:
            raise ValueError(
                f"coarse_window_days must be positive, got {coarse_window_days}."
            )
        if subdivision_resolution_days <= 0:
            raise ValueError(
                "subdivision_resolution_days must be positive, got "
                f"{subdivision_resolution_days}."
            )
        if subdivision_resolution_days > coarse_window_days:
            raise ValueError(
                f"subdivision_resolution_days ({subdivision_resolution_days}) "
                f"must be <= coarse_window_days ({coarse_window_days})."
            )

        first_request = True

        # --- 1. coarse backward scan, newest -> oldest, non-overlapping ---
        coarse_windows: list[ProbeWindow] = []
        window_end = newest
        while window_end >= oldest_to_try:
            window_start = max(
                window_end - timedelta(days=coarse_window_days - 1), oldest_to_try
            )
            probe_window, first_request = self._probe_one_window(
                symbol, resolution, window_start, window_end, first_request
            )
            coarse_windows.append(probe_window)
            window_end = window_start - timedelta(days=1)

        # --- 2. locate the OLDEST coarse window classified DATA -----------
        # coarse_windows is ordered newest -> oldest; the LAST DATA entry in
        # that order is the topologically oldest one.
        oldest_data_index: int | None = None
        for index, probed in enumerate(coarse_windows):
            if probed.status is ProbeWindowStatus.DATA:
                oldest_data_index = index
        oldest_data_window = (
            coarse_windows[oldest_data_index] if oldest_data_index is not None else None
        )

        # Contiguous EMPTY_SUCCESS run directly older than (adjacent to) the
        # oldest DATA window -- an observed fact only, never a retention or
        # trading-calendar claim. If oldest_data_window is None, or the
        # window immediately older is not EMPTY_SUCCESS, this is empty.
        empty_run: list[ProbeWindow] = []
        if oldest_data_index is not None:
            cursor = oldest_data_index + 1
            while (
                cursor < len(coarse_windows)
                and coarse_windows[cursor].status is ProbeWindowStatus.EMPTY_SUCCESS
            ):
                empty_run.append(coarse_windows[cursor])
                cursor += 1

        # --- 3. subdivide ONLY the oldest DATA window ---------------------
        subdivision_windows: list[ProbeWindow] = []
        if oldest_data_window is not None:
            subdivision_windows, first_request = self._subdivide_window(
                symbol,
                resolution,
                oldest_data_window,
                subdivision_resolution_days,
                first_request,
            )

        all_windows = tuple(coarse_windows) + tuple(subdivision_windows)

        # Earliest observed candle: the UNION of every successful DATA
        # observation relevant to the oldest region -- the coarse
        # oldest_data_window itself, PLUS every DATA subdivision window.
        # Subdivision may only TIGHTEN this value when it adds genuine new
        # information (a DATA sub-window with an even earlier real candle);
        # it must never ERASE a candle already genuinely observed in the
        # coarse request merely because a finer request covering that same
        # instant came back ERROR/EMPTY_SUCCESS. A failed finer probe
        # reduces precision, not evidence -- it is recorded in
        # unresolved_intervals, never used to invalidate a timestamp this
        # scan already actually saw. Both sources are genuine
        # returned-candle timestamps, never a requested window's start date.
        data_candidates = [w for w in subdivision_windows if w.status is ProbeWindowStatus.DATA]
        if oldest_data_window is not None:
            data_candidates = [oldest_data_window] + data_candidates

        earliest_observed_candle: str | None = None
        if data_candidates:
            earliest_observed_candle = min(
                w.earliest_ts for w in data_candidates if w.earliest_ts is not None
            )
        earliest_observed_date = (
            earliest_observed_candle[:10] if earliest_observed_candle is not None else None
        )

        unresolved_intervals = tuple(
            w for w in all_windows
            if w.status in (ProbeWindowStatus.ERROR, ProbeWindowStatus.UNKNOWN)
        )

        return HistoryDepthProbeReport(
            symbol=symbol,
            resolution=resolution,
            search_horizon_start=to_api_date(oldest_to_try),
            search_horizon_end=to_api_date(newest),
            coarse_window_days=coarse_window_days,
            subdivision_resolution_days=subdivision_resolution_days,
            windows=all_windows,
            earliest_observed_candle=earliest_observed_candle,
            earliest_observed_date=earliest_observed_date,
            oldest_data_window=oldest_data_window,
            oldest_contiguous_empty_success_interval=tuple(empty_run),
            unresolved_intervals=unresolved_intervals,
        )

    def _probe_one_window(
        self,
        symbol: str,
        resolution: str,
        start: date,
        end: date,
        first_request: bool,
    ) -> tuple["ProbeWindow", bool]:
        """Probe one window and classify it. Always makes exactly one
        request, so the returned ``bool`` is always ``False`` -- it is the
        UPDATED ``first_request`` flag for the caller's NEXT call, not a
        report of whether a request happened here (one always does). This
        lets the caller track ``first_request`` for pause timing across the
        whole scan, coarse + subdivision combined, without pausing before
        the very first request of the entire scan.
        """
        if not first_request and self._pause:
            time.sleep(self._pause)
        try:
            frame = self.fetch_chunk(symbol, resolution, start, end)
        except BrokerAuthError:
            # Never downgraded: an expired/invalid token tells us nothing
            # about history depth and must abort the whole scan.
            raise
        except (BrokerDataError, BrokerRateLimitError) as exc:
            logger.warning("probe %s..%s failed: %s", start, end, exc)
            return (
                ProbeWindow(
                    range_from=to_api_date(start),
                    range_to=to_api_date(end),
                    status=ProbeWindowStatus.ERROR,
                    row_count=0,
                    error=str(exc),
                ),
                False,
            )

        row_count = len(frame)
        if row_count == 0:
            return (
                ProbeWindow(
                    range_from=to_api_date(start),
                    range_to=to_api_date(end),
                    status=ProbeWindowStatus.EMPTY_SUCCESS,
                    row_count=0,
                ),
                False,
            )
        return (
            ProbeWindow(
                range_from=to_api_date(start),
                range_to=to_api_date(end),
                status=ProbeWindowStatus.DATA,
                row_count=row_count,
                earliest_ts=frame[TS].iloc[0].isoformat(),
                latest_ts=frame[TS].iloc[-1].isoformat(),
            ),
            False,
        )

    def _subdivide_window(
        self,
        symbol: str,
        resolution: str,
        coarse_window: "ProbeWindow",
        subdivision_resolution_days: int,
        first_request: bool,
    ) -> tuple[list["ProbeWindow"], bool]:
        """Backward, non-overlapping subdivision of exactly one DATA window,
        strictly within its own bounds, probing every sub-window (never
        stopping early) to tighten the bracket around the earliest observed
        candle.
        """
        window_start_bound = date.fromisoformat(coarse_window.range_from)
        window_end_bound = date.fromisoformat(coarse_window.range_to)

        sub_windows: list[ProbeWindow] = []
        cursor_end = window_end_bound
        while cursor_end >= window_start_bound:
            cursor_start = max(
                cursor_end - timedelta(days=subdivision_resolution_days - 1),
                window_start_bound,
            )
            probe_window, first_request = self._probe_one_window(
                symbol, resolution, cursor_start, cursor_end, first_request
            )
            sub_windows.append(probe_window)
            cursor_end = cursor_start - timedelta(days=1)

        return sub_windows, first_request

    def probe_earliest_available(
        self,
        symbol: str,
        resolution: str,
        *,
        newest: date,
        oldest_to_try: date,
        probe_window_days: int = 5,
    ) -> date | None:
        """DEPRECATED: use :meth:`probe_history_depth` instead.

        This name and its previous binary-search implementation incorrectly
        implied a certified "availability"/retention boundary, and the
        binary search itself silently mishandled non-monotonic evidence
        (a holiday-shaped empty window between two windows that both have
        data). This wrapper now delegates to :meth:`probe_history_depth` and
        returns ONLY ``earliest_observed_date`` (or ``None``) -- an
        OBSERVATION from the bounded backward scan, never a broker retention
        claim. Kept solely for ``scripts/download_data.py``'s existing call
        site; do not add new callers -- call :meth:`probe_history_depth`
        directly and use its structured report instead.
        """
        report = self.probe_history_depth(
            symbol,
            resolution,
            newest=newest,
            oldest_to_try=oldest_to_try,
            coarse_window_days=probe_window_days,
        )
        if report.earliest_observed_date is None:
            return None
        return date.fromisoformat(report.earliest_observed_date)
