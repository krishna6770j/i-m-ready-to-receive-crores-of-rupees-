"""Canonical schema and FYERS payload parsing tests."""

from __future__ import annotations

import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.schemas import (
    CLOSE,
    HIGH,
    LOW,
    OHLCV_COLUMNS,
    OPEN,
    TS,
    VOLUME,
    SchemaError,
    assert_canonical,
    empty_ohlcv,
    from_fyers_candles,
    normalise,
)
from tests.conftest import candles_payload, make_ohlcv


def test_empty_frame_has_correct_dtypes():
    frame = empty_ohlcv()
    assert list(frame.columns) == list(OHLCV_COLUMNS)
    assert str(frame[TS].dtype.tz) == IST_NAME
    assert_canonical(frame)


def test_from_fyers_candles_parses_documented_row_order():
    """Row order is [epoch, open, high, low, close, volume]."""
    frame = from_fyers_candles(candles_payload(3)["candles"])
    assert len(frame) == 3
    assert frame[TS].iloc[0].isoformat() == "2026-01-01T09:15:00+05:30"
    assert frame[OPEN].iloc[0] == 24000.0
    assert frame[HIGH].iloc[0] == 24005.0
    assert frame[LOW].iloc[0] == 23995.0
    assert frame[CLOSE].iloc[0] == 24001.0
    assert frame[VOLUME].iloc[0] == 1000
    assert_canonical(frame)


def test_from_fyers_candles_handles_empty_list():
    assert len(from_fyers_candles([])) == 0


def test_from_fyers_candles_rejects_wrong_row_width():
    """A changed response shape must fail loudly, not mis-parse."""
    with pytest.raises(SchemaError, match="row width"):
        from_fyers_candles([[1767238500, 1.0, 2.0, 0.5]])


def test_normalise_rejects_naive_timestamps():
    frame = make_ohlcv(5)
    frame[TS] = frame[TS].dt.tz_localize(None)
    with pytest.raises(SchemaError, match="tz-aware"):
        normalise(frame)


def test_normalise_converts_other_timezone_to_ist():
    frame = make_ohlcv(5)
    frame[TS] = frame[TS].dt.tz_convert("UTC")
    out = normalise(frame)
    assert str(out[TS].dtype.tz) == IST_NAME


def test_normalise_sorts_unsorted_input():
    frame = make_ohlcv(10).sample(frac=1.0, random_state=1).reset_index(drop=True)
    out = normalise(frame)
    assert out[TS].is_monotonic_increasing


def test_normalise_rejects_missing_column():
    frame = make_ohlcv(5).drop(columns=[VOLUME])
    with pytest.raises(SchemaError, match="Missing required column"):
        normalise(frame)


def test_normalise_does_not_repair_bad_ohlc():
    """Normalisation is about containers, not correctness."""
    frame = make_ohlcv(5)
    frame.loc[2, HIGH] = 0.0
    out = normalise(frame)
    assert out.loc[2, HIGH] == 0.0


def test_assert_canonical_rejects_wrong_column_order():
    frame = make_ohlcv(5)[[VOLUME, TS, OPEN, HIGH, LOW, CLOSE]]
    with pytest.raises(SchemaError, match="Column mismatch"):
        assert_canonical(frame)
