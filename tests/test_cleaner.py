"""Cleaner tests: the default path must never alter data."""

from __future__ import annotations

import pandas as pd
import pytest

from marketdata import cleaner
from marketdata.schemas import CLOSE, TS
from tests.conftest import make_ohlcv


def test_default_clean_does_not_change_data(ohlcv):
    out, record = cleaner.clean(ohlcv)
    pd.testing.assert_frame_equal(out, ohlcv)
    assert record.rows_removed == 0


def test_default_clean_records_normalisation(ohlcv):
    _, record = cleaner.clean(ohlcv)
    assert any("normalise" in op for op in record.operations)


def test_drop_exact_duplicates_removes_identical_rows(ohlcv):
    doubled = pd.concat([ohlcv, ohlcv.iloc[[3]]], ignore_index=True)
    out, record = cleaner.clean(doubled, ["drop_exact_duplicate_rows"])
    assert len(out) == len(ohlcv)
    assert record.rows_removed == 1


def test_drop_exact_duplicates_keeps_conflicting_rows():
    """Same timestamp, different prices is a conflict, not a duplicate."""
    frame = make_ohlcv(10)
    conflicting = frame.iloc[[4]].copy()
    conflicting[CLOSE] = conflicting[CLOSE] + 10.0
    combined = pd.concat([frame, conflicting], ignore_index=True)
    out, _ = cleaner.clean(combined, ["drop_exact_duplicate_rows"])
    assert len(out) == len(frame) + 1


def test_conflicting_duplicate_removal_is_opt_in_and_lossy():
    frame = make_ohlcv(10)
    conflicting = frame.iloc[[4]].copy()
    conflicting[CLOSE] = conflicting[CLOSE] + 10.0
    combined = pd.concat([frame, conflicting], ignore_index=True)
    out, record = cleaner.clean(
        combined, ["drop_conflicting_duplicate_timestamps"]
    )
    assert len(out) == len(frame)
    assert any("LOSSY" in op for op in record.operations)


def test_unknown_operation_is_rejected(ohlcv):
    with pytest.raises(ValueError, match="Unknown cleaning operation"):
        cleaner.clean(ohlcv, ["fill_gaps"])


def test_no_gap_filling_operation_exists():
    """Gap filling invents prices; it must not be available at all."""
    for name in cleaner.AVAILABLE_OPERATIONS:
        assert "fill" not in name
        assert "interpolat" not in name
        assert "smooth" not in name


def test_record_serialises(ohlcv):
    _, record = cleaner.clean(ohlcv)
    payload = record.to_dict()
    assert payload["rows_before"] == len(ohlcv)
    assert payload["rows_after"] == len(ohlcv)
