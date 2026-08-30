"""Provenance envelope + generation integrity tests.

Frozen architecture section 6:

    data_digest        = SHA256(canonical encoding of identity + observations)
    provenance_digest  = SHA256(canonical encoding of the provenance envelope)
    generation_id      = uuid.uuid4()
    integrity_id       = SHA256(data_digest || provenance_digest)
"""

from __future__ import annotations

import dataclasses
import re
import uuid

import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.dataset import ValidatedDataset, ValidationPolicy
from marketdata.evidence import ChunkResultSnapshot, FetchReportSnapshot
from marketdata.identity import DatasetIdentity
from marketdata.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    Namespace,
    ProvenanceEnvelope,
)
from marketdata.schemas import (
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    TS,
    VOLUME,
)

_HEX64 = re.compile(r"^[a-f0-9]{64}$")


def _identity(**overrides) -> DatasetIdentity:
    fields = {"source": "fyers:history", "symbol": "NIFTY", "resolution": "1"}
    fields.update(overrides)
    return DatasetIdentity(**fields)


def _valid_frame(n: int = 5, *, base: float = 24000.0) -> pd.DataFrame:
    rows = []
    ts0 = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
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


def _dataset(*, validation_policy: ValidationPolicy | None = None, **identity_overrides) -> ValidatedDataset:
    kwargs = {}
    if validation_policy is not None:
        kwargs["validation_policy"] = validation_policy
    return ValidatedDataset.build(_valid_frame(), identity=_identity(**identity_overrides), **kwargs)


def _fetch_snapshot(**overrides) -> FetchReportSnapshot:
    fields = dict(
        symbol="NIFTY",
        resolution="1",
        requested_from="2026-01-01",
        requested_to="2026-01-05",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-05", 5, True, None),),
        total_rows=5,
        first_ts="2026-01-01T09:15:00+05:30",
        last_ts="2026-01-01T09:19:00+05:30",
        duplicate_rows_removed=0,
        conflicting_timestamps=0,
    )
    fields.update(overrides)
    return FetchReportSnapshot(**fields)


# --- schema version -----------------------------------------------------


def test_provenance_schema_version_is_frozen_at_1():
    assert PROVENANCE_SCHEMA_VERSION == 1


def test_build_signature_has_no_provenance_schema_version_parameter():
    import inspect

    params = inspect.signature(ProvenanceEnvelope.build).parameters
    assert "provenance_schema_version" not in params
    assert "market_data_schema_version" not in params


def test_envelope_exposes_both_schema_versions():
    env = ProvenanceEnvelope.build(_dataset())
    assert env.provenance_schema_version == 1
    assert env.market_data_schema_version == 1


# --- determinism ----------------------------------------------------------


def test_envelope_digest_is_deterministic_given_same_generation_id():
    ds = _dataset()
    gid = uuid.uuid4()
    env1 = ProvenanceEnvelope.build(ds, generation_id=gid)
    env2 = ProvenanceEnvelope.build(ds, generation_id=gid)
    assert env1.provenance_digest == env2.provenance_digest
    assert env1.integrity_id == env2.integrity_id


def test_provenance_digest_format():
    env = ProvenanceEnvelope.build(_dataset())
    assert _HEX64.match(env.provenance_digest)
    assert _HEX64.match(env.integrity_id)


# --- identity fields affect provenance digest ------------------------------


def test_different_symbol_changes_provenance_digest():
    gid = uuid.uuid4()
    d1 = ProvenanceEnvelope.build(_dataset(symbol="NIFTY"), generation_id=gid).provenance_digest
    d2 = ProvenanceEnvelope.build(_dataset(symbol="SBIN"), generation_id=gid).provenance_digest
    assert d1 != d2


def test_different_source_changes_provenance_digest():
    gid = uuid.uuid4()
    d1 = ProvenanceEnvelope.build(_dataset(source="fyers:history"), generation_id=gid).provenance_digest
    d2 = ProvenanceEnvelope.build(_dataset(source="other"), generation_id=gid).provenance_digest
    assert d1 != d2


def test_different_resolution_changes_provenance_digest():
    gid = uuid.uuid4()
    d1 = ProvenanceEnvelope.build(_dataset(resolution="1"), generation_id=gid).provenance_digest
    d2 = ProvenanceEnvelope.build(_dataset(resolution="5"), generation_id=gid).provenance_digest
    assert d1 != d2


# --- generation_id ----------------------------------------------------------


def test_generation_id_is_uuid4_when_generated():
    env = ProvenanceEnvelope.build(_dataset())
    assert isinstance(env.generation_id, uuid.UUID)
    assert env.generation_id.version == 4


def test_valid_uuid4_string_accepted():
    gid = uuid.uuid4()
    env = ProvenanceEnvelope.build(_dataset(), generation_id=str(gid))
    assert env.generation_id == gid


def test_malformed_uuid_rejected():
    with pytest.raises(ValueError):
        ProvenanceEnvelope.build(_dataset(), generation_id="not-a-uuid")


def test_non_uuid4_version_rejected():
    # A well-formed UUID, but version 1 (time-based), not version 4.
    v1 = uuid.uuid1()
    with pytest.raises(ValueError):
        ProvenanceEnvelope.build(_dataset(), generation_id=v1)


def test_wrong_type_generation_id_rejected():
    with pytest.raises(TypeError):
        ProvenanceEnvelope.build(_dataset(), generation_id=12345)


def test_generation_id_affects_provenance_digest():
    ds = _dataset()
    env1 = ProvenanceEnvelope.build(ds, generation_id=uuid.uuid4())
    env2 = ProvenanceEnvelope.build(ds, generation_id=uuid.uuid4())
    assert env1.provenance_digest != env2.provenance_digest


def test_two_generations_identical_data_still_differ_by_generation_id():
    # Same underlying ValidatedDataset (same data_digest, same
    # canonicalisation evidence) built into two envelopes with different
    # generation_ids -> different provenance_digest, purely from
    # generation_id.
    ds = _dataset()
    env1 = ProvenanceEnvelope.build(ds)
    env2 = ProvenanceEnvelope.build(ds)
    assert ds.digest == ds.digest  # trivially the same dataset
    assert env1.generation_id != env2.generation_id
    assert env1.provenance_digest != env2.provenance_digest


# --- namespace --------------------------------------------------------------


def test_namespace_trusted_by_default():
    env = ProvenanceEnvelope.build(_dataset())
    assert env.namespace is Namespace.TRUSTED
    assert env.forced is False
    assert env.force_reason is None


def test_namespace_forced_requires_reason():
    with pytest.raises(ValueError):
        ProvenanceEnvelope.build(_dataset(), forced=True)
    with pytest.raises(ValueError):
        ProvenanceEnvelope.build(_dataset(), forced=True, force_reason="")
    with pytest.raises(ValueError):
        ProvenanceEnvelope.build(_dataset(), forced=True, force_reason="   ")


def test_force_reason_without_forced_rejected():
    with pytest.raises(ValueError):
        ProvenanceEnvelope.build(_dataset(), forced=False, force_reason="some reason")


@pytest.mark.parametrize("bad_forced", [1, 0, "true", "", [], ["x"], None])
def test_forced_non_bool_rejected(bad_forced):
    with pytest.raises(TypeError):
        ProvenanceEnvelope.build(_dataset(), forced=bad_forced)


def test_forced_actual_bool_true_and_false_accepted():
    env_false = ProvenanceEnvelope.build(_dataset(), forced=False)
    assert env_false.forced is False
    env_true = ProvenanceEnvelope.build(
        _dataset(), forced=True, force_reason="backfill"
    )
    assert env_true.forced is True


def test_namespace_forced_with_reason():
    env = ProvenanceEnvelope.build(
        _dataset(), forced=True, force_reason="operator override for backfill"
    )
    assert env.namespace is Namespace.FORCED
    assert env.forced is True
    assert env.force_reason == "operator override for backfill"


def test_namespace_affects_provenance_digest():
    ds = _dataset()
    gid = uuid.uuid4()
    trusted = ProvenanceEnvelope.build(ds, generation_id=gid)
    forced = ProvenanceEnvelope.build(
        ds, generation_id=gid, forced=True, force_reason="backfill"
    )
    assert trusted.provenance_digest != forced.provenance_digest


def test_namespace_value_is_actually_present_in_the_encoded_bytes():
    # namespace is fully determined by `forced` (both change together), so
    # a hash-comparison test alone cannot prove namespace's OWN field is
    # encoded -- `forced`/`force_reason` differing would already change the
    # digest regardless. Inspecting the byte stream directly proves the
    # namespace value itself is actually part of what gets hashed.
    env = ProvenanceEnvelope.build(_dataset(), forced=True, force_reason="backfill")
    encoded = env._encode_envelope()
    assert b"FORCED" in encoded


# --- canonicalisation evidence ----------------------------------------------


def test_canonicalisation_evidence_affects_provenance_digest():
    gid = uuid.uuid4()
    # A dataset built from an out-of-order source carries a SOURCE_UNSORTED
    # transformation/anomaly that a naturally-sorted source does not.
    ts0 = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
    sorted_frame = pd.DataFrame(
        {
            TS: [ts0, ts0 + pd.Timedelta(minutes=1)],
            OPEN: [100.0, 101.0],
            HIGH: [105.0, 106.0],
            LOW: [95.0, 96.0],
            CLOSE: [101.0, 102.0],
            VOLUME: [1000, 1001],
        }
    )
    unsorted_frame = sorted_frame.iloc[::-1].reset_index(drop=True)

    identity = _identity()
    ds_sorted = ValidatedDataset.build(sorted_frame, identity=identity)
    ds_unsorted = ValidatedDataset.build(unsorted_frame, identity=identity)

    env_sorted = ProvenanceEnvelope.build(ds_sorted, generation_id=gid)
    env_unsorted = ProvenanceEnvelope.build(ds_unsorted, generation_id=gid)

    assert ds_sorted.digest == ds_unsorted.digest  # same canonical data_digest
    assert env_sorted.provenance_digest != env_unsorted.provenance_digest


def test_source_evidence_specifically_affects_provenance_digest():
    # Isolates source_evidence from transformations/source_anomalies: two
    # different (non-canonical) input column orderings both trigger the
    # SAME COLUMNS_REORDERED transformation (its description does not name
    # the specific source order) and produce the identical canonical frame
    # (same data_digest), but source_evidence.column_inventory records the
    # different raw input orders. If source_evidence were dropped from the
    # envelope encoding, these two would collide.
    ts0 = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
    base = {
        TS: [ts0],
        OPEN: [100.0],
        HIGH: [105.0],
        LOW: [95.0],
        CLOSE: [101.0],
        VOLUME: [1000],
    }
    order1 = pd.DataFrame(base)[[VOLUME, CLOSE, LOW, HIGH, OPEN, TS]]
    order2 = pd.DataFrame(base)[[TS, VOLUME, OPEN, CLOSE, HIGH, LOW]]

    identity = _identity()
    ds1 = ValidatedDataset.build(order1, identity=identity)
    ds2 = ValidatedDataset.build(order2, identity=identity)
    assert ds1.digest == ds2.digest
    assert ds1.transformations == ds2.transformations
    assert ds1.source_evidence.column_inventory != ds2.source_evidence.column_inventory

    gid = uuid.uuid4()
    env1 = ProvenanceEnvelope.build(ds1, generation_id=gid)
    env2 = ProvenanceEnvelope.build(ds2, generation_id=gid)
    assert env1.provenance_digest != env2.provenance_digest


# --- acquisition evidence -----------------------------------------------------


def test_acquisition_evidence_is_optional():
    env = ProvenanceEnvelope.build(_dataset())
    assert env.fetch is None


def test_acquisition_evidence_affects_provenance_digest():
    ds = _dataset()
    gid = uuid.uuid4()
    without_fetch = ProvenanceEnvelope.build(ds, generation_id=gid)
    with_fetch = ProvenanceEnvelope.build(ds, generation_id=gid, fetch=_fetch_snapshot())
    assert without_fetch.provenance_digest != with_fetch.provenance_digest


def test_different_fetch_evidence_changes_provenance_digest():
    ds = _dataset()
    gid = uuid.uuid4()
    fetch_a = _fetch_snapshot(total_rows=5)
    fetch_b = _fetch_snapshot(total_rows=999)
    env_a = ProvenanceEnvelope.build(ds, generation_id=gid, fetch=fetch_a)
    env_b = ProvenanceEnvelope.build(ds, generation_id=gid, fetch=fetch_b)
    assert env_a.provenance_digest != env_b.provenance_digest


def test_fake_fetch_object_rejected():
    class FakeFetch:
        symbol = "NIFTY"

    with pytest.raises(TypeError):
        ProvenanceEnvelope.build(_dataset(), fetch=FakeFetch())


def test_fetch_wrong_symbol_rejected():
    ds = _dataset(symbol="NIFTY")
    mismatched = _fetch_snapshot(symbol="SBIN", resolution="1")
    with pytest.raises(ValueError):
        ProvenanceEnvelope.build(ds, fetch=mismatched)


def test_fetch_wrong_resolution_rejected():
    ds = _dataset(resolution="1")
    mismatched = _fetch_snapshot(symbol="NIFTY", resolution="5")
    with pytest.raises(ValueError):
        ProvenanceEnvelope.build(ds, fetch=mismatched)


def test_fetch_matching_identity_accepted():
    ds = _dataset(symbol="NIFTY", resolution="1")
    matching = _fetch_snapshot(symbol="NIFTY", resolution="1")
    env = ProvenanceEnvelope.build(ds, fetch=matching)
    assert env.fetch is matching


# --- validation policy: bound as provenance/config evidence -----------------


def test_validation_policy_is_bound_to_the_envelope():
    policy = ValidationPolicy(max_session_gap_days=3.0)
    env = ProvenanceEnvelope.build(_dataset(validation_policy=policy))
    assert env.validation_policy == policy


def test_validation_result_is_not_part_of_the_envelope():
    # The RESULT (ValidationReportSnapshot / MarketDataValidity) is
    # data-derived and recomputed elsewhere; only the POLICY is bound here.
    env = ProvenanceEnvelope.build(_dataset())
    assert not hasattr(env, "validation")
    assert not hasattr(env, "market_data_validity")


def test_different_validation_policy_changes_provenance_digest_not_data_digest():
    ds_default = _dataset()
    ds_custom = _dataset(validation_policy=ValidationPolicy(max_session_gap_days=3.0))
    assert ds_default.digest == ds_custom.digest  # same underlying data

    gid = uuid.uuid4()
    env_default = ProvenanceEnvelope.build(ds_default, generation_id=gid)
    env_custom = ProvenanceEnvelope.build(ds_custom, generation_id=gid)

    assert env_default.data_digest == env_custom.data_digest
    assert env_default.provenance_digest != env_custom.provenance_digest
    assert env_default.integrity_id != env_custom.integrity_id


# --- mutable source evidence cannot change envelope after construction -----


def test_dataset_object_itself_is_already_immutable_so_envelope_stays_bound():
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds)
    original_digest = env.provenance_digest
    original_integrity = env.integrity_id

    # ValidatedDataset itself refuses attribute assignment (Unit 5); the
    # only mutation surface left is the caller's OWN raw frame, already
    # proven independent of ds in test_dataset.py. Confirm the envelope
    # built from ds is unaffected by anything reachable from here.
    with pytest.raises(AttributeError):
        ds.digest = "0" * 64

    assert env.provenance_digest == original_digest
    assert env.integrity_id == original_integrity


def test_fake_dataset_object_rejected():
    class FakeDataset:
        identity = _identity()
        digest = "0" * 64
        transformations = ()
        source_anomalies = ()
        source_evidence = None
        validation_policy = ValidationPolicy()

    with pytest.raises(TypeError):
        ProvenanceEnvelope.build(FakeDataset())


# --- DATA identity vs PROVENANCE identity: kept separate --------------------


def test_changed_candle_changes_data_digest_but_not_provenance_digest():
    identity = _identity()
    gid = uuid.uuid4()

    ds_original = ValidatedDataset.build(_valid_frame(), identity=identity)
    changed_frame = _valid_frame()
    changed_frame.iloc[0, changed_frame.columns.get_loc(OPEN)] = 99999.0
    ds_changed = ValidatedDataset.build(changed_frame, identity=identity)

    env_original = ProvenanceEnvelope.build(ds_original, generation_id=gid)
    env_changed = ProvenanceEnvelope.build(ds_changed, generation_id=gid)

    assert env_original.data_digest != env_changed.data_digest
    assert env_original.provenance_digest == env_changed.provenance_digest
    assert env_original.integrity_id != env_changed.integrity_id


def test_data_digest_not_present_in_encoded_envelope_bytes():
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds)
    encoded = env._encode_envelope()
    assert env.data_digest.encode("ascii") not in encoded


# --- integrity_id -------------------------------------------------------------


def test_integrity_id_changes_when_data_digest_changes():
    gid = uuid.uuid4()
    env1 = ProvenanceEnvelope.build(_dataset(symbol="NIFTY"), generation_id=gid)
    env2 = ProvenanceEnvelope.build(_dataset(symbol="SBIN"), generation_id=gid)
    assert env1.data_digest != env2.data_digest
    assert env1.integrity_id != env2.integrity_id


def test_integrity_id_changes_when_provenance_digest_changes():
    ds = _dataset()
    env1 = ProvenanceEnvelope.build(ds, generation_id=uuid.uuid4())
    env2 = ProvenanceEnvelope.build(ds, generation_id=uuid.uuid4())
    assert env1.data_digest == env2.data_digest  # same underlying dataset
    assert env1.provenance_digest != env2.provenance_digest
    assert env1.integrity_id != env2.integrity_id


def test_integrity_id_is_sha256_of_both_digests_bytes():
    import hashlib

    env = ProvenanceEnvelope.build(_dataset())
    expected = hashlib.sha256(
        bytes.fromhex(env.data_digest) + bytes.fromhex(env.provenance_digest)
    ).hexdigest()
    assert env.integrity_id == expected


# --- malformed digest rejected -----------------------------------------------


def test_malformed_data_digest_rejected():
    # A ValidatedDataset's own .digest is always well-formed hex
    # (dataset_digest() guarantees it) and a duck-typed FakeDataset is
    # already rejected on isinstance grounds (test_fake_dataset_object_rejected)
    # before its .digest is ever inspected -- so the only way to exercise
    # this specific guard is directly.
    from marketdata.provenance import _validate_sha256_hex

    with pytest.raises(ValueError):
        _validate_sha256_hex("not-a-valid-digest", "data_digest")
    with pytest.raises(ValueError):
        _validate_sha256_hex("a" * 63, "data_digest")  # too short
    with pytest.raises(ValueError):
        _validate_sha256_hex("A" * 64, "data_digest")  # uppercase rejected
    with pytest.raises(ValueError):
        _validate_sha256_hex(12345, "data_digest")  # not a string


# --- security: no credentials/tokens as envelope fields ----------------------


def test_no_secret_shaped_field_names_in_envelope():
    forbidden = ("token", "secret", "password", "credential", "auth_code", "api_key")
    field_names = set(ProvenanceEnvelope.__slots__)
    for name in field_names:
        lowered = name.lower()
        assert not any(bad in lowered for bad in forbidden), name


def test_software_snapshot_contains_no_obvious_secret_values():
    env = ProvenanceEnvelope.build(_dataset())
    for key, value in env.software.items():
        assert "token" not in str(value).lower()
        assert "secret" not in str(value).lower()


# --- object immutability ------------------------------------------------------


def test_direct_construction_is_blocked():
    with pytest.raises(TypeError):
        ProvenanceEnvelope()


def test_envelope_object_itself_is_immutable():
    env = ProvenanceEnvelope.build(_dataset())
    with pytest.raises(AttributeError):
        env.provenance_digest = "0" * 64
    with pytest.raises(AttributeError):
        env.new_attribute = 1


def test_envelope_cannot_be_pickled():
    import pickle

    env = ProvenanceEnvelope.build(_dataset())
    with pytest.raises(TypeError):
        pickle.dumps(env)


def test_software_mapping_is_read_only():
    env = ProvenanceEnvelope.build(_dataset())
    with pytest.raises(TypeError):
        env.software["python"] = "corrupted"


# ---------------------------------------------------------------------------
# ReconstructedManifest (Unit 10: strict manifest reconstruction/reverification)
# ---------------------------------------------------------------------------

import json

from marketdata.provenance import ManifestError, ReconstructedManifest


def _envelope_and_json(**dataset_kwargs):
    ds = _dataset(**dataset_kwargs)
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_snapshot())
    return env, env.to_manifest_json()


def test_reconstructed_manifest_round_trips_and_digests_match():
    env, manifest_json = _envelope_and_json()
    rm = ReconstructedManifest.from_manifest_json(manifest_json)
    assert rm.recompute_provenance_digest() == env.provenance_digest
    assert rm.recompute_integrity_id() == env.integrity_id
    assert rm.data_digest == env.data_digest
    assert rm.generation_id == env.generation_id
    assert rm.namespace == env.namespace
    assert rm.validation_policy == env.validation_policy
    assert rm.fetch == env.fetch
    assert rm.transformations == env.transformations
    assert rm.source_anomalies == env.source_anomalies
    assert rm.source_evidence == env.source_evidence
    assert dict(rm.software) == dict(env.software)


def test_reconstructed_manifest_no_fetch():
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds)
    rm = ReconstructedManifest.from_manifest_json(env.to_manifest_json())
    assert rm.fetch is None
    assert rm.recompute_provenance_digest() == env.provenance_digest


def test_malformed_json_rejected():
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json("{not valid json")


def test_top_level_duplicate_key_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    raw = manifest_json.rstrip("}")
    # Inject a duplicate of an existing key with a different value.
    tampered = raw + f',"forced":{str(not payload["forced"]).lower()}}}'
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(tampered)


def test_nested_duplicate_key_rejected():
    # Inject a duplicate key inside the nested "source_evidence" object,
    # via direct string surgery (json.dumps can never itself produce a
    # duplicate key, so this simulates a hand-tampered/corrupted file).
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    se_json = json.dumps(payload["source_evidence"], sort_keys=True)
    duplicated_se_json = se_json[:-1] + ',"row_count":999}'
    outer_json = json.dumps(payload, sort_keys=True)
    injected = outer_json.replace(se_json, duplicated_se_json, 1)
    assert injected != outer_json  # sanity: the replacement actually happened
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(injected)


def test_unknown_top_level_field_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["unexpected_field"] = "x"
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_missing_top_level_field_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    del payload["forced"]
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_wrong_provenance_schema_version_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["provenance_schema_version"] = 2
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_wrong_market_data_schema_version_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["market_data_schema_version"] = 2
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_bool_provenance_schema_version_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["provenance_schema_version"] = True
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_generation_id_not_uuid4_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["generation_id"] = str(uuid.uuid1())
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_namespace_invalid_value_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["namespace"] = "SOMETHING_ELSE"
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_forced_as_int_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["forced"] = 0
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_data_digest_not_hex_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["data_digest"] = "not-a-digest"
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_edited_provenance_fact_changes_recomputed_digest():
    env, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["source_evidence"]["row_count"] = 999
    tampered_json = json.dumps(payload)
    rm = ReconstructedManifest.from_manifest_json(tampered_json)
    # Structural parse succeeds (forensic inspection remains possible), but
    # the recomputed digest no longer matches the manifest's own stored claim.
    assert rm.recompute_provenance_digest() != payload["provenance_digest"]


def test_edited_provenance_digest_field_detected_by_recompute():
    env, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["provenance_digest"] = "0" * 64
    tampered_json = json.dumps(payload)
    rm = ReconstructedManifest.from_manifest_json(tampered_json)
    assert rm.recompute_provenance_digest() != rm.provenance_digest


def test_missing_chunk_field_rejected():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    del payload["fetch"]["chunks"][0]["error"]
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))


def test_chunk_ok_as_int_rejected_by_manifest_parser():
    _, manifest_json = _envelope_and_json()
    payload = json.loads(manifest_json)
    payload["fetch"]["chunks"][0]["ok"] = 1
    with pytest.raises(ManifestError):
        ReconstructedManifest.from_manifest_json(json.dumps(payload))
