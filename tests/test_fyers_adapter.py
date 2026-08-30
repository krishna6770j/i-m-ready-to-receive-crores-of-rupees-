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
