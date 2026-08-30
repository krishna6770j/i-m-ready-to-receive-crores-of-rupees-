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

    def probe_earliest_available(
        self,
        symbol: str,
        resolution: str,
        *,
        newest: date,
        oldest_to_try: date,
        probe_window_days: int = 5,
    ) -> date | None:
        """Binary-search the earliest date that returns candles.

        Manager correction #12: do not assume a history depth. This determines
        empirically how far back the vendor actually serves data, using a small
        number of narrow probes rather than downloading everything first.

        Returns the start of the earliest probe window that produced data, or
        None if no window in the range did.
        """
        self._validate_resolution(resolution)
        lo, hi = oldest_to_try, newest
        earliest_hit: date | None = None

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            window_end = min(mid + timedelta(days=probe_window_days - 1), newest)
            try:
                got = len(self.fetch_chunk(symbol, resolution, mid, window_end))
            except (BrokerDataError, BrokerRateLimitError) as exc:
                logger.warning("probe at %s failed: %s", mid, exc)
                got = 0
            if got > 0:
                earliest_hit = mid
                hi = mid - timedelta(days=1)
            else:
                lo = mid + timedelta(days=1)
            if self._pause:
                time.sleep(self._pause)

        return earliest_hit
