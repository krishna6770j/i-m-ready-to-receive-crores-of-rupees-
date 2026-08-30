"""Storage, provenance and reproducibility tests (requirement 29)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from marketdata import store
from marketdata.schemas import CLOSE, HIGH, LOW, TS
from marketdata.schemas import normalise as store_normalise
from marketdata.validator import validate
from tests.conftest import make_ohlcv, validation_of


def test_write_then_read_round_trips(tmp_store, ohlcv):
    path, manifest = store.write(
        ohlcv, tmp_store, symbol="NSE:NIFTY50-INDEX", resolution="1",
        source="test", validation=validation_of(ohlcv)
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
        validation=validation_of(ohlcv),
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
        ohlcv, tmp_store, symbol="X:Y", resolution="1", source="test",
        validation=validation_of(ohlcv)
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
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
        validation=validation_of(ohlcv))
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


def test_hash_distinguishes_different_instants_with_same_wall_clock():
    """content_hash must encode the instant, not the wall-clock reading.

    Replaces an earlier test that compared two lists of ISO strings without
    ever calling content_hash -- it would have passed even if content_hash
    returned a constant, so it proved nothing about the hash.

    09:15 IST and 09:15 UTC are different moments in time. Their hashes must
    differ, or a timezone bug could pass an integrity check unnoticed.
    """
    ist = make_ohlcv(20, seed=3)
    utc_wall = ist.copy()
    # Same wall-clock numbers, relabelled UTC -> a genuinely different instant.
    utc_wall[TS] = utc_wall[TS].dt.tz_localize(None).dt.tz_localize("UTC")

    assert store.content_hash(ist) != store.content_hash(
        store_normalise(utc_wall)
    ), "hash must change when the underlying instants change"


def test_hash_is_stable_under_timezone_representation():
    """The same instants expressed in another tz must hash identically.

    normalise() converts to IST, so representation must not affect the hash --
    otherwise re-reading a dataset could spuriously fail integrity checks.
    """
    a = make_ohlcv(20, seed=4)
    b = a.copy()
    b[TS] = b[TS].dt.tz_convert("UTC")
    assert store.content_hash(a) == store.content_hash(store_normalise(b))


def test_write_read_write_is_stable(tmp_store, ohlcv):
    """Round-tripping through Parquet must not perturb the content hash."""
    _, m1 = store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
        validation=validation_of(ohlcv))
    frame, _ = store.read(tmp_store, "X:Y", "1")
    _, m2 = store.write(frame, tmp_store, symbol="X:Y", resolution="1", source="t",
                        validation=validation_of(frame))
    assert m1.content_sha256 == m2.content_sha256


def test_verify_integrity_detects_tampering(tmp_store, ohlcv):
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
        validation=validation_of(ohlcv))
    assert store.verify_integrity(tmp_store, "X:Y", "1")

    parquet_path, _ = store.dataset_paths(tmp_store, "X:Y", "1")
    tampered = pd.read_parquet(parquet_path)
    tampered.loc[0, CLOSE] = 99999.0
    tampered.to_parquet(parquet_path, index=False)

    assert not store.verify_integrity(tmp_store, "X:Y", "1")


def test_read_without_manifest_is_refused(tmp_store, ohlcv):
    """Data without provenance is not trusted."""
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
        validation=validation_of(ohlcv))
    _, manifest_path = store.dataset_paths(tmp_store, "X:Y", "1")
    manifest_path.unlink()
    with pytest.raises(FileNotFoundError, match="no manifest"):
        store.read(tmp_store, "X:Y", "1")


def test_symbol_with_colon_produces_safe_filename(tmp_store, ohlcv):
    path, _ = store.write(
        ohlcv, tmp_store, symbol="NSE:NIFTY50-INDEX", resolution="1",
        source="t", validation=validation_of(ohlcv)
    )
    assert ":" not in path.name
    assert path.name.startswith("NSE_NIFTY50-INDEX")


# --- persistence gate ----------------------------------------------------


def _invalid_frame():
    """A frame with an impossible bar: high < low."""
    frame = make_ohlcv(30)
    frame.loc[5, HIGH] = 100.0
    frame.loc[5, LOW] = 200.0
    return frame


def test_write_refuses_data_with_validation_errors(tmp_store):
    """THE critical gate: corrupt data must not become stored data."""
    bad = _invalid_frame()
    report = validate(bad, symbol="X", resolution="1", expected_interval_minutes=1)
    assert not report.is_usable, "fixture must actually be invalid"

    with pytest.raises(store.UnvalidatedDataError, match="Refusing to persist"):
        store.write(bad, tmp_store, symbol="X:Y", resolution="1", source="t",
                    validation=report)

    parquet_path, manifest_path = store.dataset_paths(tmp_store, "X:Y", "1")
    assert not parquet_path.exists(), "no Parquet file may be left behind"
    assert not manifest_path.exists(), "no manifest may be left behind"


def test_write_refuses_partial_acquisition(tmp_store, ohlcv):
    """A dataset with failed chunks must not be stored as complete."""
    fetch = {"failed_chunk_detail": [{"from": "2026-03-01", "to": "2026-06-08",
                                      "error": "server error"}]}
    with pytest.raises(store.IncompleteAcquisitionError, match="chunk"):
        store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
                    validation=validation_of(ohlcv), fetch=fetch)
    parquet_path, _ = store.dataset_paths(tmp_store, "X:Y", "1")
    assert not parquet_path.exists()


def test_forced_write_is_recorded_and_never_authoritative(tmp_store):
    """An override must be visible in provenance, not silent."""
    bad = _invalid_frame()
    report = validate(bad, symbol="X", resolution="1", expected_interval_minutes=1)
    _, manifest = store.write(bad, tmp_store, symbol="X:Y", resolution="1",
                              source="t", validation=report, force=True)
    assert manifest.forced is True
    assert manifest.validation_status == "invalid"
    assert manifest.validation_error_count > 0
    assert "OHLC_HIGH_BELOW_LOW" in manifest.validation_error_codes
    assert manifest.is_authoritative is False


def test_clean_complete_data_is_authoritative(tmp_store, ohlcv):
    _, manifest = store.write(
        ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
        validation=validation_of(ohlcv), fetch={"failed_chunk_detail": []},
    )
    assert manifest.validation_status == "valid"
    assert manifest.fetch_status == "complete"
    assert manifest.is_authoritative is True


def test_fetch_status_unknown_is_not_authoritative(tmp_store, ohlcv):
    """Absent acquisition evidence must not be read as success."""
    _, manifest = store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1",
                              source="t", validation=validation_of(ohlcv))
    assert manifest.fetch_status == "unknown"
    assert manifest.is_authoritative is False


def test_is_authoritative_survives_manifest_round_trip(tmp_store, ohlcv):
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
                validation=validation_of(ohlcv), fetch={"failed_chunk_detail": []})
    _, manifest = store.read(tmp_store, "X:Y", "1")
    assert manifest.is_authoritative is True


def test_is_authoritative_cannot_be_forged_by_editing_manifest(tmp_store, ohlcv):
    """It is derived on read, so hand-editing the JSON cannot fake it."""
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
                validation=validation_of(ohlcv), force=True)
    _, manifest_path = store.dataset_paths(tmp_store, "X:Y", "1")
    payload = json.loads(manifest_path.read_text())
    payload["is_authoritative"] = True          # attacker/mistake edits it
    manifest_path.write_text(json.dumps(payload))
    _, manifest = store.read(tmp_store, "X:Y", "1")
    assert manifest.is_authoritative is False, "derived, not trusted from disk"


def test_manifest_is_valid_json(tmp_store, ohlcv):
    store.write(ohlcv, tmp_store, symbol="X:Y", resolution="1", source="t",
        validation=validation_of(ohlcv))
    _, manifest_path = store.dataset_paths(tmp_store, "X:Y", "1")
    parsed = json.loads(manifest_path.read_text())
    assert parsed["symbol"] == "X:Y"
