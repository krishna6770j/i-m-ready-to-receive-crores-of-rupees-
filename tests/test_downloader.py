"""End-to-end pipeline tests: fetch -> clean -> validate -> store.

Uses the fake FYERS client, so this exercises our orchestration only. It does
not verify live API behaviour.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from brokers.fyers.historical import FyersHistoricalData
from marketdata import store
from marketdata.downloader import download
from marketdata.schemas import CLOSE, HIGH, LOW, OPEN, TS
from tests.conftest import FakeFyersClient, candles_payload


def make_provider(responses):
    return FyersHistoricalData(FakeFyersClient(responses), request_pause_seconds=0.0)


def test_download_produces_data_report_and_manifest(tmp_store):
    provider = make_provider(candles_payload(5))
    outcome = download(
        provider,
        symbol="NSE:NIFTY50-INDEX",
        resolution="1",
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
        data_store_dir=tmp_store,
    )
    assert len(outcome.frame) == 5
    assert outcome.validation.is_usable
    assert outcome.manifest is not None
    assert outcome.path is not None and outcome.path.exists()
    assert outcome.fetch_report is not None


def test_downloaded_data_is_readable_and_intact(tmp_store):
    provider = make_provider(candles_payload(5))
    download(
        provider,
        symbol="NSE:NIFTY50-INDEX",
        resolution="1",
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
        data_store_dir=tmp_store,
    )
    assert store.verify_integrity(tmp_store, "NSE:NIFTY50-INDEX", "1")


def test_download_is_reproducible(tmp_store):
    """Same input twice -> identical normalised content hash (deliverable F)."""
    hashes = []
    for _ in range(2):
        provider = make_provider(candles_payload(20))
        outcome = download(
            provider,
            symbol="NSE:NIFTY50-INDEX",
            resolution="1",
            start=date(2026, 1, 1),
            end=date(2026, 1, 5),
            data_store_dir=tmp_store,
        )
        hashes.append(outcome.manifest.content_sha256)
    assert hashes[0] == hashes[1]


def test_bad_data_is_reported_not_silently_stored(tmp_store):
    """An impossible bar must surface as an ERROR in the report."""
    payload = candles_payload(5)
    payload["candles"][2] = [1767238620, 100.0, 50.0, 200.0, 120.0, 10]  # high<low
    provider = make_provider(payload)
    outcome = download(
        provider,
        symbol="X:Y",
        resolution="1",
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
        data_store_dir=tmp_store,
    )
    assert not outcome.validation.is_usable
    assert any("OHLC" in i.code for i in outcome.validation.errors)


def test_default_download_applies_no_cleaning_operations(tmp_store):
    provider = make_provider(candles_payload(5))
    outcome = download(
        provider,
        symbol="X:Y",
        resolution="1",
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
        data_store_dir=tmp_store,
    )
    assert outcome.cleaning.rows_removed == 0
    assert outcome.cleaning.operations == ["normalise: dtypes, column order, sort"]


def test_empty_result_stores_nothing(tmp_store):
    provider = make_provider({"s": "ok", "candles": []})
    outcome = download(
        provider,
        symbol="X:Y",
        resolution="1",
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
        data_store_dir=tmp_store,
    )
    assert outcome.path is None
    assert not outcome.validation.is_usable  # EMPTY_DATASET is an error


def test_summary_includes_coverage_and_quality(tmp_store):
    provider = make_provider(candles_payload(5))
    outcome = download(
        provider,
        symbol="X:Y",
        resolution="1",
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
        data_store_dir=tmp_store,
    )
    text = outcome.summary()
    assert "DATA QUALITY REPORT" in text
    assert "COVERAGE" in text
    assert "STORED" in text
