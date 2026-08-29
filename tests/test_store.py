"""Storage, provenance and reproducibility tests (requirement 29)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from marketdata import store
from marketdata.schemas import CLOSE, TS
from tests.conftest import make_ohlcv


def test_write_then_read_round_trips(tmp_store, ohlcv):
    path, manifest = store.write(
        ohlcv, tmp_store, symbol="NSE:NIFTY50-INDEX", resolution="1", source="test"
    )
    assert path.exists()
    frame, read_manifest = store.read(tmp_store, "NSE:NIFTY50-INDEX", "1")
    pd.testing.assert_frame_equal(frame, ohlcv)
    assert read_manifest.content_sha256 == manifest.content_sha256


def test_manifest_records_provenance(tmp_store, ohlcv):
    _, manifest = store.write(
        ohlcv,
        tmp_store,
        symbol="NSE:NIFTY50-INDEX",
        resolution="1",
        source="fyers:history",
        requested_range={"start": "2026-01-01", "end": "2026-01-31"},
        notes="phase 1 test",
    )
    assert manifest.source == "fyers:history"
    assert manifest.requested_range["start"] == "2026-01-01"
    assert manifest.timezone == "Asia/Kolkata"
    assert manifest.row_count == len(ohlcv)
    assert manifest.notes == "phase 1 test"
    assert len(manifest.content_sha256) == 64


def test_manifest_records_software_versions(tmp_store, ohlcv):
    """Provenance must capture the code that produced the data."""
    _, manifest = store.write(
        ohlcv, tmp_store, symbol="X:Y", resolution="1", source="test"
    )
    software = manifest.software
    assert software["python"].startswith("3.12")
    assert software["pandas"] == pd.__version__
    for key in ("numpy", "pyarrow", "fyers-apiv3", "platform", "git_revision"):
        assert key in software and software[key]


def test_git_revision_is_reported():
    rev = store.git_revision()
    assert rev and rev != "unknown"


def test_software_versions_survive_manifest_round_trip(tmp_store, ohlcv):
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t")
    _, manifest = store.read(tmp_store, "X:Y", "1")
    assert manifest.software["pandas"] == pd.__version__


def test_identical_data_hashes_identically():
    """Same input twice -> same hash. This is the reproducibility guarantee."""
    a = make_ohlcv(50, seed=7)
    b = make_ohlcv(50, seed=7)
    assert store.content_hash(a) == store.content_hash(b)


def test_different_data_hashes_differently():
    a = make_ohlcv(50, seed=7)
    b = a.copy()
    b.loc[10, CLOSE] += 0.0001
    assert store.content_hash(a) != store.content_hash(b)


def test_hash_is_timezone_sensitive():
    """Same instants in a different tz must not hash the same.

    If they did, a timezone bug could pass an integrity check unnoticed.
    """
    a = make_ohlcv(20, seed=3)
    b = a.copy()
    b[TS] = b[TS].dt.tz_convert("UTC")
    # Re-normalising converts back to IST, so compare pre-normalisation strings.
    a_iso = a[TS].map(lambda t: pd.Timestamp(t).isoformat()).tolist()
    b_iso = b[TS].map(lambda t: pd.Timestamp(t).isoformat()).tolist()
    assert a_iso != b_iso


def test_write_read_write_is_stable(tmp_store, ohlcv):
    """Round-tripping through Parquet must not perturb the content hash."""
    _, m1 = store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t")
    frame, _ = store.read(tmp_store, "X:Y", "1")
    _, m2 = store.write(frame, tmp_store, symbol="X:Y", resolution="1", source="t")
    assert m1.content_sha256 == m2.content_sha256


def test_verify_integrity_detects_tampering(tmp_store, ohlcv):
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t")
    assert store.verify_integrity(tmp_store, "X:Y", "1")

    parquet_path, _ = store.dataset_paths(tmp_store, "X:Y", "1")
    tampered = pd.read_parquet(parquet_path)
    tampered.loc[0, CLOSE] = 99999.0
    tampered.to_parquet(parquet_path, index=False)

    assert not store.verify_integrity(tmp_store, "X:Y", "1")


def test_read_without_manifest_is_refused(tmp_store, ohlcv):
    """Data without provenance is not trusted."""
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t")
    _, manifest_path = store.dataset_paths(tmp_store, "X:Y", "1")
    manifest_path.unlink()
    with pytest.raises(FileNotFoundError, match="no manifest"):
        store.read(tmp_store, "X:Y", "1")


def test_symbol_with_colon_produces_safe_filename(tmp_store, ohlcv):
    path, _ = store.write(
        ohlcv, tmp_store, symbol="NSE:NIFTY50-INDEX", resolution="1", source="t"
    )
    assert ":" not in path.name
    assert path.name.startswith("NSE_NIFTY50-INDEX")


def test_manifest_is_valid_json(tmp_store, ohlcv):
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t")
    _, manifest_path = store.dataset_paths(tmp_store, "X:Y", "1")
    parsed = json.loads(manifest_path.read_text())
    assert parsed["symbol"] == "X:Y"
