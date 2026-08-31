"""FYERS adapter tests.

All tests use a fake client implementing ``history(data) -> dict``. No network
access and no credentials are involved, so these verify our parsing, chunking
and error handling -- NOT that the live API behaves as assumed. Live behaviour
remains unverified until credentials exist.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from brokers.base import BrokerAuthError, BrokerDataError, BrokerRateLimitError
from brokers.fyers import endpoints as ep
from brokers.fyers.historical import FyersHistoricalData
from marketdata.schemas import TS
from tests.conftest import FakeFyersClient, candles_payload


def provider(responses) -> tuple[FyersHistoricalData, FakeFyersClient]:
    client = FakeFyersClient(responses)
    return FyersHistoricalData(client, request_pause_seconds=0.0), client


# --- request construction ------------------------------------------------


def test_request_uses_sdk_documented_parameters():
    prov, client = provider(candles_payload(2))
    prov.fetch_chunk("NSE:NIFTY50-INDEX", "1", date(2026, 1, 1), date(2026, 1, 2))
    req = client.requests[0]
    assert set(req) == set(ep.HISTORY_PARAMS)
    assert req["symbol"] == "NSE:NIFTY50-INDEX"
    assert req["resolution"] == "1"
    assert req["date_format"] == ep.DATE_FORMAT_YMD
    assert req["range_from"] == "2026-01-01"
    assert req["range_to"] == "2026-01-02"
    assert req["cont_flag"] == 1


def test_undocumented_resolution_is_rejected():
    """'45' appears in community posts but not the SDK docstring."""
    prov, _ = provider(candles_payload())
    with pytest.raises(ValueError, match="not in the SDK-documented set"):
        prov.fetch_chunk("X", "45", date(2026, 1, 1), date(2026, 1, 2))


def test_source_name_is_recorded_for_provenance():
    prov, _ = provider(candles_payload())
    assert prov.source_name == "fyers:history"


# --- response parsing ----------------------------------------------------


def test_parses_successful_payload():
    prov, _ = provider(candles_payload(3))
    frame = prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))
    assert len(frame) == 3
    assert str(frame[TS].dtype.tz) == "Asia/Kolkata"


def test_empty_candles_yields_empty_frame():
    prov, _ = provider({"s": "ok", "candles": []})
    assert len(prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))) == 0


def test_auth_error_is_classified():
    prov, _ = provider({"s": "error", "code": -16, "message": "Invalid token"})
    with pytest.raises(BrokerAuthError, match="regenerated each"):
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))


def test_rate_limit_error_is_classified():
    prov, _ = provider({"s": "error", "code": 429, "message": "rate limit exceeded"})
    with pytest.raises(BrokerRateLimitError):
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))


def test_generic_error_is_classified():
    prov, _ = provider({"s": "error", "code": -99, "message": "something broke"})
    with pytest.raises(BrokerDataError):
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))


def test_ok_status_without_candles_key_raises():
    """A changed response shape must not be mistaken for empty data."""
    prov, _ = provider({"s": "ok"})
    with pytest.raises(BrokerDataError, match="no 'candles'"):
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))


def test_non_dict_response_raises():
    prov, _ = provider([["not", "a", "dict"]])
    with pytest.raises(BrokerDataError, match="Expected a dict"):
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))


# --- chunking ------------------------------------------------------------


def test_long_range_is_split_into_chunks():
    prov, client = provider(candles_payload(1))
    start, end = date(2026, 1, 1), date(2026, 12, 31)
    _, report = prov.fetch_candles_with_report("X", "1", start, end)
    expected = 4  # 365 days / 100-day windows
    assert len(client.requests) == expected
    assert len(report.chunks) == expected


def test_chunks_are_contiguous_and_cover_the_range():
    prov, client = provider(candles_payload(1))
    prov.fetch_candles_with_report("X", "1", date(2026, 1, 1), date(2026, 6, 30))
    froms = [r["range_from"] for r in client.requests]
    tos = [r["range_to"] for r in client.requests]
    assert froms[0] == "2026-01-01"
    assert tos[-1] == "2026-06-30"
    for previous_to, next_from in zip(tos, froms[1:]):
        assert date.fromisoformat(next_from) == date.fromisoformat(previous_to) + timedelta(
            days=1
        )


def test_short_range_uses_single_request():
    prov, client = provider(candles_payload(2))
    prov.fetch_candles_with_report("X", "1", date(2026, 1, 1), date(2026, 1, 5))
    assert len(client.requests) == 1


def test_daily_resolution_uses_larger_window():
    prov, client = provider(candles_payload(1))
    prov.fetch_candles_with_report("X", "1D", date(2026, 1, 1), date(2026, 12, 31))
    assert len(client.requests) == 1


def test_end_before_start_is_rejected():
    prov, _ = provider(candles_payload())
    with pytest.raises(ValueError, match="before start"):
        prov.fetch_candles_with_report("X", "1", date(2026, 6, 1), date(2026, 1, 1))


# --- coverage reporting (manager correction #12) -------------------------


def test_failed_chunk_is_reported_not_hidden():
    """One bad window must not discard the chunks that succeeded.

    Each good chunk returns a DISTINCT block of candles, as real chunks would.
    An earlier version of this test reused one payload for every chunk, so its
    row-count assertion silently depended on duplicates being retained.
    """
    base = 1767239100
    responses = [
        candles_payload(2, start_epoch=base),
        {"s": "error", "code": -99, "message": "server error"},
        candles_payload(2, start_epoch=base + 200 * 86400),
        candles_payload(2, start_epoch=base + 300 * 86400),
    ]
    prov, _ = provider(responses)
    frame, report = prov.fetch_candles_with_report(
        "X", "1", date(2026, 1, 1), date(2026, 12, 31)
    )
    assert len(report.failed_chunks) == 1
    assert "server error" in report.failed_chunks[0].error
    assert len(frame) == 6, "the three successful chunks must all survive"
    assert report.duplicate_rows_removed == 0


def test_auth_failure_aborts_rather_than_partially_downloading():
    """A dead token must stop the run, not produce a sparse dataset."""
    responses = [
        candles_payload(2),
        {"s": "error", "code": -16, "message": "Invalid token"},
    ]
    prov, _ = provider(responses)
    with pytest.raises(BrokerAuthError):
        prov.fetch_candles_with_report("X", "1", date(2026, 1, 1), date(2026, 12, 31))


def test_report_records_requested_versus_downloaded():
    prov, _ = provider(candles_payload(3))
    _, report = prov.fetch_candles_with_report(
        "X", "1", date(2026, 1, 1), date(2026, 1, 10)
    )
    payload = report.to_dict()
    assert payload["requested_from"] == "2026-01-01"
    assert payload["requested_to"] == "2026-01-10"
    assert payload["total_rows"] == 3
    assert payload["downloaded_first_ts"].startswith("2026-01-01T09:15")


def test_identical_candles_across_chunks_are_collapsed_losslessly():
    """Overlapping chunk boundaries must not inflate the dataset.

    Regression: four chunks returning the same five candles previously
    produced 20 rows with 15 duplicate timestamps.
    """
    prov, _ = provider(candles_payload(5))
    frame, report = prov.fetch_candles_with_report(
        "X", "1", date(2026, 1, 1), date(2026, 12, 31)
    )
    assert len(frame) == 5, "identical rows must collapse to one copy each"
    assert frame[TS].duplicated().sum() == 0
    assert report.duplicate_rows_removed == 15
    assert report.conflicting_timestamps == 0


def test_conflicting_candles_are_preserved_not_silently_resolved():
    """The adapter must never pick a winner between contradictory candles."""
    a = candles_payload(3)
    b = candles_payload(3)
    b["candles"][1][4] = 99999.0  # same timestamp, different close
    prov, _ = provider([a, b, a, b])
    frame, report = prov.fetch_candles_with_report(
        "X", "1", date(2026, 1, 1), date(2026, 12, 31)
    )
    assert report.conflicting_timestamps == 1
    conflicting = frame[frame[TS].duplicated(keep=False)]
    closes = set(conflicting["close"].tolist())
    assert 99999.0 in closes and len(closes) == 2, (
        "both contradictory observations must survive for the validator to flag"
    )


def test_conflicting_chunk_data_fails_validation():
    """End-to-end: a boundary conflict must make the dataset unusable."""
    from marketdata.validator import validate

    a = candles_payload(3)
    b = candles_payload(3)
    b["candles"][1][4] = 99999.0
    prov, _ = provider([a, b])
    frame, _ = prov.fetch_candles_with_report(
        "X", "1", date(2026, 1, 1), date(2026, 6, 30)
    )
    report = validate(frame, symbol="X", resolution="1", expected_interval_minutes=1)
    assert "DUPLICATE_TIMESTAMPS" in {i.code for i in report.issues}
    assert not report.is_usable


def test_dedup_counts_appear_in_fetch_report_dict():
    prov, _ = provider(candles_payload(5))
    _, report = prov.fetch_candles_with_report(
        "X", "1", date(2026, 1, 1), date(2026, 12, 31)
    )
    payload = report.to_dict()
    assert payload["duplicate_rows_removed"] == 15
    assert payload["conflicting_timestamps"] == 0


def test_empty_chunks_are_counted():
    prov, _ = provider({"s": "ok", "candles": []})
    _, report = prov.fetch_candles_with_report(
        "X", "1", date(2026, 1, 1), date(2026, 1, 5)
    )
    assert len(report.empty_chunks) == 1
    assert report.total_rows == 0


# --- history-depth probe -------------------------------------------------


def test_probe_returns_none_when_no_data_anywhere():
    prov, _ = provider({"s": "ok", "candles": []})
    got = prov.probe_earliest_available(
        "X", "1", newest=date(2026, 1, 31), oldest_to_try=date(2026, 1, 1)
    )
    assert got is None


def test_probe_finds_a_date_when_data_exists():
    prov, _ = provider(candles_payload(1))
    got = prov.probe_earliest_available(
        "X", "1", newest=date(2026, 1, 31), oldest_to_try=date(2026, 1, 1)
    )
    assert got is not None
    assert date(2026, 1, 1) <= got <= date(2026, 1, 31)


# ---------------------------------------------------------------------------
# probe_history_depth() -- bounded backward-window evidence scan
# ---------------------------------------------------------------------------

_BASE_EPOCH = 1767239100  # 2026-01-01 09:15:00 IST


def _epoch_for(d: date) -> int:
    return _BASE_EPOCH + (d - date(2026, 1, 1)).days * 86400


class ScriptedClient:
    """Fake FYERS client whose response depends on the REQUESTED range, via
    a caller-supplied ``classify(start, end) -> str`` callback. Records
    every request (and, for tests that care, timestamps between requests)
    so the scan's actual call pattern can be asserted on directly.
    """

    def __init__(self, classify):
        self._classify = classify
        self.requests: list[dict] = []

    def history(self, data=None):
        self.requests.append(dict(data or {}))
        start = date.fromisoformat(data["range_from"])
        end = date.fromisoformat(data["range_to"])
        kind = self._classify(start, end)
        if kind == "data":
            return candles_payload(1, start_epoch=_epoch_for(start))
        if kind == "empty":
            return {"s": "ok", "candles": []}
        if kind == "rate_limit":
            return {"s": "error", "message": "rate limit exceeded", "code": -99}
        if kind == "data_error":
            return {"s": "error", "message": "internal server error", "code": -50}
        if kind == "auth":
            return {"s": "error", "message": "invalid auth token", "code": -16}
        raise AssertionError(f"unhandled classify() result: {kind!r}")


def _req_ranges(client) -> list[tuple[str, str]]:
    return [(r["range_from"], r["range_to"]) for r in client.requests]


# --- all-one-status horizons ------------------------------------------------


def test_all_windows_data():
    from brokers.fyers.historical import ProbeWindowStatus

    client = ScriptedClient(lambda s, e: "data")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert all(w.status is ProbeWindowStatus.DATA for w in report.windows if w.range_from >= "2026-01-06")
    assert report.earliest_observed_date == "2026-01-01"
    assert report.unresolved_intervals == ()


def test_all_windows_empty_success():
    from brokers.fyers.historical import ProbeWindowStatus

    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert all(w.status is ProbeWindowStatus.EMPTY_SUCCESS for w in report.windows)
    assert report.earliest_observed_candle is None
    assert report.earliest_observed_date is None
    assert report.oldest_data_window is None
    assert report.unresolved_intervals == ()


def test_all_windows_error():
    from brokers.fyers.historical import ProbeWindowStatus

    client = ScriptedClient(lambda s, e: "data_error")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert all(w.status is ProbeWindowStatus.ERROR for w in report.windows)
    assert report.earliest_observed_date is None
    assert len(report.unresolved_intervals) == len(report.windows)


# --- non-monotonic patterns --------------------------------------------------


def test_data_empty_data_pattern_finds_older_data():
    """The exact counterexample that defeats the old binary search: DATA,
    then a holiday-shaped EMPTY_SUCCESS gap, then DATA again further back.
    The true earliest observation is in the OLDER data region; a binary
    search converges on the newer one instead (reproduced separately).
    """
    def classify(start, end):
        if start <= date(2026, 1, 5):
            return "data"
        if start >= date(2026, 1, 11):
            return "data"
        return "empty"

    client = ScriptedClient(classify)
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert report.earliest_observed_date == "2026-01-01"
    # Every coarse window in the horizon must have actually been probed --
    # no early stop on the first EMPTY_SUCCESS.
    assert len(report.windows) >= 3 + 5  # 3 coarse + up to 5 subdivision of Jan1-5


def test_data_error_data_pattern_still_finds_older_data():
    def classify(start, end):
        if start <= date(2026, 1, 5):
            return "data"
        if start >= date(2026, 1, 11):
            return "data"
        return "data_error"

    client = ScriptedClient(classify)
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert report.earliest_observed_date == "2026-01-01"
    from brokers.fyers.historical import ProbeWindowStatus

    assert any(w.status is ProbeWindowStatus.ERROR for w in report.windows)


def test_holiday_like_empty_window_between_valid_windows():
    from brokers.fyers.historical import ProbeWindowStatus

    def classify(start, end):
        if date(2026, 1, 6) <= start <= date(2026, 1, 10):
            return "empty"
        return "data"

    client = ScriptedClient(classify)
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    coarse = [w for w in report.windows if w.range_from in ("2026-01-01", "2026-01-06", "2026-01-11")]
    statuses = {w.range_from: w.status for w in coarse}
    assert statuses["2026-01-01"] is ProbeWindowStatus.DATA
    assert statuses["2026-01-06"] is ProbeWindowStatus.EMPTY_SUCCESS
    assert statuses["2026-01-11"] is ProbeWindowStatus.DATA
    assert report.earliest_observed_date == "2026-01-01"


def test_empty_data_empty_data_pattern():
    """EMPTY_SUCCESS, DATA, EMPTY_SUCCESS, DATA -- the report must show
    windows honestly rather than collapsing them into one cutoff."""
    from brokers.fyers.historical import ProbeWindowStatus

    # Coarse windows (5-day, newest->oldest) from Jan1..Jan20:
    # Jan16-20 EMPTY, Jan11-15 DATA, Jan6-10 EMPTY, Jan1-5 DATA
    def classify(start, end):
        if start in (date(2026, 1, 16), date(2026, 1, 6)):
            return "empty"
        return "data"

    client = ScriptedClient(classify)
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 20), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    coarse = {w.range_from: w.status for w in report.windows if w.range_from in
              ("2026-01-16", "2026-01-11", "2026-01-06", "2026-01-01")}
    assert coarse["2026-01-16"] is ProbeWindowStatus.EMPTY_SUCCESS
    assert coarse["2026-01-11"] is ProbeWindowStatus.DATA
    assert coarse["2026-01-06"] is ProbeWindowStatus.EMPTY_SUCCESS
    assert coarse["2026-01-01"] is ProbeWindowStatus.DATA
    assert report.earliest_observed_date == "2026-01-01"


# --- auth abort ---------------------------------------------------------


def test_auth_failure_aborts_immediately():
    client = ScriptedClient(lambda s, e: "auth")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(BrokerAuthError):
        prov.probe_history_depth(
            "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
        )
    # Aborted after the very first request -- never downgraded/continued.
    assert len(client.requests) == 1


def test_auth_failure_partway_through_still_aborts():
    def classify(start, end):
        if start == date(2026, 1, 11):
            return "auth"
        return "data"

    client = ScriptedClient(classify)
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(BrokerAuthError):
        prov.probe_history_depth(
            "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
        )


# --- error classification (never conflated with EMPTY_SUCCESS) ----------


def test_rate_limit_classified_as_error_not_empty():
    from brokers.fyers.historical import ProbeWindowStatus

    client = ScriptedClient(lambda s, e: "rate_limit")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert all(w.status is ProbeWindowStatus.ERROR for w in report.windows)
    assert all(w.status is not ProbeWindowStatus.EMPTY_SUCCESS for w in report.windows)
    assert all(w.error is not None for w in report.windows)


def test_data_error_classified_as_error_not_empty():
    from brokers.fyers.historical import ProbeWindowStatus

    client = ScriptedClient(lambda s, e: "data_error")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert all(w.status is ProbeWindowStatus.ERROR for w in report.windows)
    assert all(w.status is not ProbeWindowStatus.EMPTY_SUCCESS for w in report.windows)


def test_empty_success_is_distinct_status_from_error():
    from brokers.fyers.historical import ProbeWindowStatus

    assert ProbeWindowStatus.EMPTY_SUCCESS != ProbeWindowStatus.ERROR
    assert ProbeWindowStatus.EMPTY_SUCCESS.value != ProbeWindowStatus.ERROR.value


# --- horizon clipping / window construction ------------------------------


def test_oldest_and_newest_horizon_clipping():
    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 12), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    ranges = _req_ranges(client)
    assert ranges[0] == ("2026-01-08", "2026-01-12")  # newest anchor, full window
    assert ranges[-1] == ("2026-01-01", "2026-01-02")  # oldest window clipped
    assert report.search_horizon_start == "2026-01-01"
    assert report.search_horizon_end == "2026-01-12"


def test_windows_are_non_overlapping_and_walk_backward():
    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 31), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    ranges = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in _req_ranges(client)]
    for i in range(1, len(ranges)):
        prev_start = ranges[i - 1][0]
        this_end = ranges[i][1]
        assert this_end < prev_start  # strictly older, no overlap, no gap-skip
    assert ranges[0][1] == date(2026, 1, 31)
    assert ranges[-1][0] == date(2026, 1, 1)


def test_one_day_horizon():
    client = ScriptedClient(lambda s, e: "data")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 1), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    coarse_ranges = [w for w in report.windows if w.range_from == w.range_to or True]
    assert _req_ranges(client)[0] == ("2026-01-01", "2026-01-01")
    assert report.earliest_observed_date == "2026-01-01"


# --- input validation -----------------------------------------------------


def test_reversed_horizon_rejected():
    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(ValueError):
        prov.probe_history_depth(
            "X", "1", newest=date(2026, 1, 1), oldest_to_try=date(2026, 1, 15)
        )
    assert client.requests == []


@pytest.mark.parametrize("bad_days", [0, -1, -5])
def test_zero_or_negative_coarse_window_days_rejected(bad_days):
    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(ValueError):
        prov.probe_history_depth(
            "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1),
            coarse_window_days=bad_days,
        )


@pytest.mark.parametrize("bad_days", [0, -1, -5])
def test_zero_or_negative_subdivision_days_rejected(bad_days):
    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(ValueError):
        prov.probe_history_depth(
            "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1),
            coarse_window_days=5, subdivision_resolution_days=bad_days,
        )


def test_subdivision_larger_than_coarse_rejected():
    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(ValueError):
        prov.probe_history_depth(
            "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1),
            coarse_window_days=5, subdivision_resolution_days=10,
        )


# --- subdivision behaviour -------------------------------------------------


def test_only_oldest_data_coarse_window_is_subdivided():
    """Two coarse DATA windows exist; only the OLDER one may be subdivided."""
    client = ScriptedClient(lambda s, e: "data")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1),
        coarse_window_days=5, subdivision_resolution_days=1,
    )
    # oldest coarse DATA window is Jan1-5; its 5 daily sub-windows must all
    # be present. The newer coarse DATA window (Jan11-15) must NOT have been
    # subdivided (no daily sub-windows for it).
    subdivided_ranges = {w.range_from for w in report.windows if w.range_from == w.range_to}
    for day in ("2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"):
        assert day in subdivided_ranges
    for day in ("2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15"):
        assert day not in subdivided_ranges
    assert report.oldest_data_window.range_from == "2026-01-01"


def test_non_data_coarse_windows_are_never_subdivided():
    client = ScriptedClient(lambda s, e: "empty" if s >= date(2026, 1, 11) else "data")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1),
        coarse_window_days=5, subdivision_resolution_days=1,
    )
    # Jan11-15 was EMPTY_SUCCESS -- no daily sub-window requests for it.
    subdivided_ranges = {w.range_from for w in report.windows if w.range_from == w.range_to}
    for day in ("2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15"):
        assert day not in subdivided_ranges


def test_actual_candle_date_not_window_start_becomes_earliest_observed_date():
    """Critical example from the manager: a subdivided window of Jan1..Jan2
    whose first ACTUAL candle is Jan 2 09:15 must report earliest_observed_date
    == Jan 2, never Jan 1 (the requested window start)."""
    def classify(start, end):
        if start == date(2026, 1, 1) and end == date(2026, 1, 1):
            return "empty"  # no candle on Jan 1 itself
        return "data"

    client = ScriptedClient(classify)
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 2), oldest_to_try=date(2026, 1, 1),
        coarse_window_days=2, subdivision_resolution_days=1,
    )
    assert report.earliest_observed_date == "2026-01-02"
    assert report.earliest_observed_candle.startswith("2026-01-02")


# --- unresolved intervals ---------------------------------------------------


def test_unresolved_intervals_contain_error_windows():
    from brokers.fyers.historical import ProbeWindowStatus

    def classify(start, end):
        if start == date(2026, 1, 6):
            return "data_error"
        return "empty"

    client = ScriptedClient(classify)
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert len(report.unresolved_intervals) == 1
    assert report.unresolved_intervals[0].status is ProbeWindowStatus.ERROR
    assert report.unresolved_intervals[0].range_from == "2026-01-06"


# --- no-observation honesty ------------------------------------------------


def test_no_observed_data_produces_none_without_claiming_absence():
    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert report.earliest_observed_candle is None
    assert report.earliest_observed_date is None
    # The report itself makes no absence claim beyond the two None fields --
    # see test_report_has_no_unproven_certainty_fields for the field-shape
    # assertion that no stronger-sounding attribute exists at all.


# --- request count / pause behaviour ----------------------------------------


def test_request_count_is_bounded_and_deterministic():
    client = ScriptedClient(lambda s, e: "data")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 31), oldest_to_try=date(2026, 1, 1),
        coarse_window_days=5, subdivision_resolution_days=1,
    )
    first_count = len(client.requests)

    client2 = ScriptedClient(lambda s, e: "data")
    prov2 = FyersHistoricalData(client2, request_pause_seconds=0.0)
    prov2.probe_history_depth(
        "X", "1", newest=date(2026, 1, 31), oldest_to_try=date(2026, 1, 1),
        coarse_window_days=5, subdivision_resolution_days=1,
    )
    assert len(client2.requests) == first_count  # deterministic, same inputs
    # coarse: ceil(31/5) = 7 windows; oldest DATA coarse window subdivided
    # into up to 5 daily sub-windows -> bounded, not proportional to horizon
    # squared or unbounded.
    assert first_count <= 7 + 5


def test_pause_not_performed_before_first_request_and_occurs_between(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("brokers.fyers.historical.time.sleep", lambda s: sleeps.append(s))
    client = ScriptedClient(lambda s, e: "empty")
    prov = FyersHistoricalData(client, request_pause_seconds=0.5)
    prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    assert len(sleeps) == len(client.requests) - 1
    assert all(s == 0.5 for s in sleeps)


# --- report field-shape: no unproven certainty ------------------------------


def test_report_has_no_unproven_certainty_fields():
    from brokers.fyers.historical import HistoryDepthProbeReport

    forbidden = {"retention_date", "earliest_available", "history_start", "no_data_before"}
    field_names = {f for f in HistoryDepthProbeReport.__dataclass_fields__}
    assert forbidden.isdisjoint(field_names)
    # Also never exposed as a property/attribute on an actual instance.
    client = ScriptedClient(lambda s, e: "data")
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    report = prov.probe_history_depth(
        "X", "1", newest=date(2026, 1, 15), oldest_to_try=date(2026, 1, 1), coarse_window_days=5
    )
    for name in forbidden:
        assert not hasattr(report, name)
