"""Trusted/unverified dataset reader tests.

Frozen architecture section 12: ``read_trusted()``/``TrustedDataset`` and
``read_unverified()``/``UnverifiedDataset``.
"""

from __future__ import annotations

import json
import uuid

import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.dataset import ValidatedDataset, ValidationPolicy
from marketdata.evidence import ChunkResultSnapshot, FetchReportSnapshot
from marketdata.generation_store import write_generation
from marketdata.identity import DatasetIdentity
from marketdata.locator import CurrentPointer, safe_slug
from marketdata.provenance import Namespace, ProvenanceEnvelope
from marketdata.schemas import CLOSE, HIGH, LOW, OPEN, TS, VOLUME
from marketdata.trusted_reader import (
    TrustedReadError,
    UnverifiedReadError,
    read_trusted,
    read_unverified,
)

_TRUSTED_DIRNAME = "trusted_generations"
_FORCED_DIRNAME = "forced_generations"
_DATA_FILENAME = "data.parquet"
_MANIFEST_FILENAME = "manifest.json"
_CURRENT_FILENAME = "CURRENT"


def _identity(**overrides) -> DatasetIdentity:
    fields = {"source": "fyers:history", "symbol": "NIFTY", "resolution": "1"}
    fields.update(overrides)
    return DatasetIdentity(**fields)


def _frame_on(date_str: str = "2026-01-01", n: int = 3, *, base: float = 100.0) -> pd.DataFrame:
    ts0 = pd.Timestamp(f"{date_str} 09:15", tz=IST_NAME)
    rows = []
    for i in range(n):
        rows.append(
            {
                TS: ts0 + pd.Timedelta(minutes=i),
                OPEN: base + i,
                HIGH: base + i + 5,
                LOW: base + i - 5,
                CLOSE: base + i + 1,
                VOLUME: 1000 + i,
            }
        )
    return pd.DataFrame(rows)


def _dataset(date_str: str = "2026-01-01", n: int = 3, *, base: float = 100.0, **id_overrides) -> ValidatedDataset:
    return ValidatedDataset.build(_frame_on(date_str, n, base=base), identity=_identity(**id_overrides))


def _fetch_for(ds: ValidatedDataset, *, requested_from: str = "2026-01-01", requested_to: str = "2026-01-01") -> FetchReportSnapshot:
    frame = ds.frame
    return FetchReportSnapshot(
        symbol=ds.identity.symbol,
        resolution=ds.identity.resolution,
        requested_from=requested_from,
        requested_to=requested_to,
        chunks=(ChunkResultSnapshot(requested_from, requested_to, len(frame), True, None),),
        total_rows=len(frame),
        first_ts=frame[TS].iloc[0].isoformat(),
        last_ts=frame[TS].iloc[-1].isoformat(),
        duplicate_rows_removed=0,
        conflicting_timestamps=0,
    )


def _dataset_dir(root, identity: DatasetIdentity):
    return root / safe_slug(identity.source) / safe_slug(identity.symbol) / safe_slug(identity.resolution)


def _write_trusted(root, *, ds=None, fetch=None):
    """Build+write one normal TRUSTED generation via write_generation()
    (the real production write path, all its own gates included).
    """
    if ds is None:
        ds = _dataset()
    if fetch is None:
        fetch = _fetch_for(ds)
    env = ProvenanceEnvelope.build(ds, fetch=fetch)
    result = write_generation(ds, env, root)
    return ds, env, result.generation_dir


def _manual_write_trusted(root, ds: ValidatedDataset, env: ProvenanceEnvelope, *, frame_override: pd.DataFrame | None = None):
    """Write a generation DIRECTLY to disk, bypassing every gate
    ``generation_store.write_generation`` enforces -- used only to simulate
    an attacker with direct filesystem access, or a bug elsewhere that let
    bad data past the write-time gates. read_trusted()'s OWN independent
    re-verification must still catch whatever is wrong, defense-in-depth.
    """
    dataset_dir = _dataset_dir(root, ds.identity)
    generation_dir = dataset_dir / _TRUSTED_DIRNAME / str(env.generation_id)
    generation_dir.mkdir(parents=True)
    frame = frame_override if frame_override is not None else ds.frame
    frame.to_parquet(generation_dir / _DATA_FILENAME, index=False, engine="pyarrow", compression="snappy")
    (generation_dir / _MANIFEST_FILENAME).write_text(env.to_manifest_json(), encoding="utf-8")
    pointer = CurrentPointer(generation_id=env.generation_id, integrity_id=env.integrity_id)
    (dataset_dir / _CURRENT_FILENAME).write_text(pointer.to_json(), encoding="utf-8")
    return generation_dir


def _tamper_manifest(manifest_path, mutate):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def _current_path(root, identity):
    return _dataset_dir(root, identity) / _CURRENT_FILENAME


def _manifest_path(root, identity, generation_id):
    return _dataset_dir(root, identity) / _TRUSTED_DIRNAME / str(generation_id) / _MANIFEST_FILENAME


def _data_path(root, identity, generation_id):
    return _dataset_dir(root, identity) / _TRUSTED_DIRNAME / str(generation_id) / _DATA_FILENAME


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_read_trusted_happy_path(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    td = read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")
    assert td.identity == ds.identity
    assert td.data_digest == env.data_digest
    assert td.provenance_digest == env.provenance_digest
    assert td.integrity_id == env.integrity_id
    assert td.generation_id == env.generation_id
    assert td.market_data_validity.value == "VALID"
    assert td.acquisition_status.value == "REQUESTS_SUCCEEDED"
    assert len(td.frame) == 3
    assert not hasattr(td, "is_authoritative")


def test_read_unverified_happy_path(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    ud = read_unverified(
        tmp_path,
        source="fyers:history",
        symbol="NIFTY",
        resolution="1",
        namespace=Namespace.TRUSTED,
        generation_id=env.generation_id,
    )
    assert ud.identity == ds.identity
    assert ud.generation_id == env.generation_id
    assert ud.stored_data_digest == env.data_digest
    assert not hasattr(ud, "market_data_validity")
    assert not hasattr(ud, "acquisition_status")
    assert not hasattr(ud, "observed_data_coverage")
    assert not hasattr(ud, "requested_window_comparison")


# ---------------------------------------------------------------------------
# Required attack tests
# ---------------------------------------------------------------------------


def test_missing_current_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    _current_path(tmp_path, ds.identity).unlink()
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_malformed_current_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    _current_path(tmp_path, ds.identity).write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_current_duplicate_json_key_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _current_path(tmp_path, ds.identity)
    payload = json.loads(path.read_text(encoding="utf-8"))
    injected = (
        json.dumps(payload)[:-1]
        + f',"integrity_id":"{"0" * 64}"' + "}"
    )
    path.write_text(injected, encoding="utf-8")
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_missing_generation_rejected(tmp_path):
    ds, env, generation_dir = _write_trusted(tmp_path)
    import shutil

    shutil.rmtree(generation_dir)
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_missing_data_parquet_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    _data_path(tmp_path, ds.identity, env.generation_id).unlink()
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_missing_manifest_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    _manifest_path(tmp_path, ds.identity, env.generation_id).unlink()
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_manifest_unknown_field_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p.__setitem__("unexpected", "x"))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_manifest_missing_field_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p.pop("forced"))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_manifest_duplicate_key_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    text = path.read_text(encoding="utf-8")
    injected = text[:-1] + f',"forced":true' + "}"
    path.write_text(injected, encoding="utf-8")
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_wrong_schema_version_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p.__setitem__("provenance_schema_version", 2))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_caller_identity_mismatch_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p.__setitem__("symbol", "BANKNIFTY"))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_manifest_generation_id_mismatch_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    other_id = str(uuid.uuid4())
    _tamper_manifest(path, lambda p: p.__setitem__("generation_id", other_id))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_manifest_namespace_mismatch_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p.__setitem__("namespace", "FORCED"))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_consistent_forced_namespace_in_trusted_slot_rejected(tmp_path):
    # A genuinely-FORCED envelope (all digests internally consistent with
    # namespace=FORCED baked in from the start, not tampered post-hoc) is
    # manually placed as if it were a trusted generation -- simulating an
    # attacker/bug that copies a forced generation's files into
    # trusted_generations/ and repoints CURRENT at it. The namespace check
    # (step 14) must catch this on its own, independent of any digest
    # mismatch (there is none here).
    ds = _dataset()
    fetch = _fetch_for(ds)
    env = ProvenanceEnvelope.build(ds, forced=True, force_reason="test", fetch=fetch)
    assert env.namespace is Namespace.FORCED
    _manual_write_trusted(tmp_path, ds, env)
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_data_digest_field_mismatch_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p.__setitem__("data_digest", "0" * 64))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_changed_candle_on_disk_rejected(tmp_path):
    # Tamper volume, not price: an OHLC-breaking price edit would coincidentally
    # also get caught by MarketDataValidity re-validation, which would mask
    # whether the data_digest recomputation-and-compare is doing its own job.
    ds, env, _ = _write_trusted(tmp_path)
    data_path = _data_path(tmp_path, ds.identity, env.generation_id)
    frame = pd.read_parquet(data_path)
    frame.loc[0, VOLUME] = frame.loc[0, VOLUME] + 999
    frame.to_parquet(data_path, index=False, engine="pyarrow", compression="snappy")
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_noncanonical_data_on_disk_rejected(tmp_path):
    """A non-canonical Parquet file (rows out of ascending-timestamp order)
    must be REJECTED outright, never silently re-sorted/repaired by
    canonicalise() as a way to make it pass -- assert_canonical() is the
    only check read_trusted() ever runs against the loaded frame.
    """
    ds, env, _ = _write_trusted(tmp_path)
    data_path = _data_path(tmp_path, ds.identity, env.generation_id)
    frame = pd.read_parquet(data_path)
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    assert not reversed_frame[TS].is_monotonic_increasing
    reversed_frame.to_parquet(data_path, index=False, engine="pyarrow", compression="snappy")
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_provenance_fact_edited_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p["source_evidence"].__setitem__("row_count", 999))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_provenance_digest_field_edited_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p.__setitem__("provenance_digest", "0" * 64))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_pointer_integrity_id_edited_rejected(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    current_path = _current_path(tmp_path, ds.identity)
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    payload["integrity_id"] = "1" * 64
    current_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_invalid_ohlc_data_on_disk_rejected(tmp_path):
    # Negative price -> validate() ERROR -> MarketDataValidity.INVALID.
    # ProvenanceEnvelope.build() itself has no MarketDataValidity gate (that
    # gate lives only in generation_store's write path); building+writing
    # this directly bypasses that gate to simulate a bug/compromise letting
    # bad data reach trusted_generations. read_trusted's OWN re-validation
    # (using the manifest's stored ValidationPolicy) must still reject it.
    bad_frame = _frame_on()
    bad_frame.loc[0, LOW] = -50.0
    ds = ValidatedDataset.build(bad_frame, identity=_identity())
    assert ds.market_data_validity.value == "INVALID"
    fetch = _fetch_for(ds)
    env = ProvenanceEnvelope.build(ds, fetch=fetch)
    _manual_write_trusted(tmp_path, ds, env)
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_invalid_acquisition_evidence_on_disk_rejected(tmp_path):
    ds = _dataset()
    frame = ds.frame
    bad_fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1", requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", len(frame), True, None),),
        total_rows=999,  # row-arithmetic mismatch -- validate_fetch_evidence rejects
        first_ts=frame[TS].iloc[0].isoformat(), last_ts=frame[TS].iloc[-1].isoformat(),
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    env = ProvenanceEnvelope.build(ds, fetch=bad_fetch)
    _manual_write_trusted(tmp_path, ds, env)
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_requests_partial_rejected(tmp_path):
    ds = _dataset(n=3)
    frame = ds.frame
    partial_fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1", requested_from="2026-01-01", requested_to="2026-01-02",
        chunks=(
            ChunkResultSnapshot("2026-01-01", "2026-01-01", len(frame), True, None),
            ChunkResultSnapshot("2026-01-02", "2026-01-02", 0, False, "rate limited"),
        ),
        total_rows=len(frame), first_ts=frame[TS].iloc[0].isoformat(), last_ts=frame[TS].iloc[-1].isoformat(),
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    env = ProvenanceEnvelope.build(ds, fetch=partial_fetch)
    _manual_write_trusted(tmp_path, ds, env)
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_requests_unknown_rejected(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds)  # fetch=None -> REQUESTS_UNKNOWN
    _manual_write_trusted(tmp_path, ds, env)
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_forced_generation_unreachable_via_read_trusted(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, forced=True, force_reason="manual backfill")
    result = write_generation(ds, env, tmp_path)
    assert result.namespace is Namespace.FORCED
    assert result.current_updated is False
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")
    # But it IS reachable via the forensic doorway, with an explicit generation_id.
    ud = read_unverified(
        tmp_path, source="fyers:history", symbol="NIFTY", resolution="1",
        namespace=Namespace.FORCED, generation_id=env.generation_id,
    )
    assert ud.namespace is Namespace.FORCED


def test_trusted_dataset_frame_is_a_defensive_copy(tmp_path):
    _write_trusted(tmp_path)
    td = read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")
    first = td.frame
    first.loc[0, CLOSE] = 999999.0
    second = td.frame
    assert second.loc[0, CLOSE] != 999999.0


def test_unverified_dataset_frame_is_a_defensive_copy(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    ud = read_unverified(
        tmp_path, source="fyers:history", symbol="NIFTY", resolution="1",
        namespace=Namespace.TRUSTED, generation_id=env.generation_id,
    )
    first = ud.frame
    first.loc[0, CLOSE] = 999999.0
    second = ud.frame
    assert second.loc[0, CLOSE] != 999999.0


def test_old_valid_current_rollback_is_accepted_known_limitation(tmp_path):
    """Frozen architecture's explicitly accepted rollback limitation: if
    CURRENT is wholesale-replaced with an OLDER but completely valid trusted
    pointer, read_trusted() accepts that older generation. There is no
    generation-freshness detection anywhere in this codebase, and this test
    proves that limitation is real and accepted -- not "fixed" out of scope.
    """
    ds_a, env_a, _ = _write_trusted(tmp_path)
    current_path = _current_path(tmp_path, ds_a.identity)
    old_current_bytes = current_path.read_text(encoding="utf-8")

    ds_b = _dataset(base=200.0)
    fetch_b = _fetch_for(ds_b)
    env_b = ProvenanceEnvelope.build(ds_b, fetch=fetch_b)
    write_generation(ds_b, env_b, tmp_path)

    td_current = read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")
    assert td_current.generation_id == env_b.generation_id

    # Roll CURRENT back to the OLDER, still fully valid pointer.
    current_path.write_text(old_current_bytes, encoding="utf-8")

    td_rolled_back = read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")
    assert td_rolled_back.generation_id == env_a.generation_id


# ---------------------------------------------------------------------------
# Mutation checks (documented in the commit; exercised manually against the
# source during development -- see the manager report for this round).
# These tests are the ones each mutation is expected to break.
# ---------------------------------------------------------------------------


def test_mutation_target_trust_stored_data_digest(tmp_path):
    """If read_trusted ever trusted manifest.data_digest without
    recomputing it from the loaded frame, this test would stop catching a
    changed candle."""
    test_changed_candle_on_disk_rejected(tmp_path)


def test_mutation_target_trust_stored_provenance_digest(tmp_path):
    """If read_trusted ever trusted manifest.provenance_digest without
    recomputing it, this test would stop catching an edited provenance
    fact."""
    test_provenance_fact_edited_rejected(tmp_path)


def test_mutation_target_skip_validation_rerun(tmp_path):
    """If read_trusted ever skipped re-running validation, this test would
    stop catching invalid OHLC data written directly to disk."""
    test_invalid_ohlc_data_on_disk_rejected(tmp_path)


def test_mutation_target_skip_acquisition_validation(tmp_path):
    """If read_trusted ever skipped validate_fetch_evidence/cross-check,
    this test would stop catching internally-inconsistent acquisition
    evidence."""
    test_invalid_acquisition_evidence_on_disk_rejected(tmp_path)


def test_mutation_target_allow_forced_namespace(tmp_path):
    """If read_trusted ever allowed namespace != TRUSTED through, this test
    would stop catching a genuinely-consistent FORCED generation placed in
    the trusted slot."""
    test_consistent_forced_namespace_in_trusted_slot_rejected(tmp_path)


def test_mutation_target_canonicalise_before_checking(tmp_path):
    """If read_trusted ever called canonicalise() (a REPAIR step) on the
    loaded frame before/instead of assert_canonical(), this test would stop
    catching non-canonical data written directly to disk -- it would be
    silently re-sorted back into a passing frame instead of rejected."""
    test_noncanonical_data_on_disk_rejected(tmp_path)


# ---------------------------------------------------------------------------
# Unit 10 correction round: restored ProvenanceEnvelope.build() invariants,
# strict JSON constants, and wrapped ValidationPolicy errors, all exercised
# at the read_trusted() level.
# ---------------------------------------------------------------------------


def test_fetch_symbol_mismatch_rejected_by_trusted_reader(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p["fetch"].__setitem__("symbol", "SBIN"))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_fetch_resolution_mismatch_rejected_by_trusted_reader(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p["fetch"].__setitem__("resolution", "5"))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_namespace_trusted_with_forced_true_rejected_by_trusted_reader(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)

    def mutate(p):
        p["forced"] = True
        p["force_reason"] = "backfill"

    _tamper_manifest(path, mutate)
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_namespace_trusted_with_force_reason_present_rejected_by_trusted_reader(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p.__setitem__("force_reason", "not actually forced"))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_nonstandard_json_nan_in_manifest_rejected_by_trusted_reader(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sigma_literal = json.dumps(payload["validation_policy"]["sigma_threshold"])
    text = json.dumps(payload)
    injected = text.replace(sigma_literal, "NaN", 1)
    path.write_text(injected, encoding="utf-8")
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_invalid_validation_policy_cannot_escape_as_raw_value_error(tmp_path):
    ds, env, _ = _write_trusted(tmp_path)
    path = _manifest_path(tmp_path, ds.identity, env.generation_id)
    _tamper_manifest(path, lambda p: p["validation_policy"].__setitem__("sigma_threshold", 0))
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")


def test_trusted_reader_forced_defense_in_depth_isolated(tmp_path, monkeypatch):
    """Isolates the defense-in-depth forced/force_reason check in
    read_trusted() from BOTH the parser's own invariant rejection AND the
    digest-recomputation checks (forced/force_reason are themselves part of
    what provenance_digest is computed over, so simply flipping them on an
    already-parsed manifest would otherwise get caught by the digest
    mismatch instead, masking whether this specific check does its own
    job): monkeypatch ReconstructedManifest.from_manifest_json to return a
    manifest object whose namespace is TRUSTED (so the namespace check
    passes) but whose forced field is True, and ALSO monkeypatch its
    recompute_provenance_digest/recompute_integrity_id to trivially agree
    with its own stored digests -- a shape the real parser could never
    produce, simulating a parser regression. read_trusted()'s OWN explicit
    forced-must-be-False check must still catch it.
    """
    import dataclasses as _dc

    import marketdata.trusted_reader as trusted_reader_module

    ds, env, _ = _write_trusted(tmp_path)
    real_from_manifest_json = trusted_reader_module.ReconstructedManifest.from_manifest_json

    def patched(text):
        manifest = real_from_manifest_json(text)
        return _dc.replace(manifest, forced=True, force_reason="simulated parser regression")

    monkeypatch.setattr(
        trusted_reader_module.ReconstructedManifest, "from_manifest_json", staticmethod(patched)
    )
    monkeypatch.setattr(
        trusted_reader_module.ReconstructedManifest,
        "recompute_provenance_digest",
        lambda self: self.provenance_digest,
    )
    monkeypatch.setattr(
        trusted_reader_module.ReconstructedManifest,
        "recompute_integrity_id",
        lambda self: self.integrity_id,
    )
    with pytest.raises(TrustedReadError):
        read_trusted(tmp_path, source="fyers:history", symbol="NIFTY", resolution="1")
