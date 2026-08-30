"""ValidatedDataset tests.

Core invariants (frozen architecture defect #1, section 16, and manager
review of an earlier revision): a validation report -- and canonicalisation
evidence -- can never be supplied separately from the frame they describe.
ValidatedDataset.build() must canonicalise and validate internally, from the
exact raw frame it is given.
"""

from __future__ import annotations

import dataclasses
import datetime
import math

import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.dataset import (
    MarketDataValidity,
    TrustBlockerError,
    ValidatedDataset,
    ValidationPolicy,
)
from marketdata.evidence import ValidationReportSnapshot
from marketdata.identity import DatasetIdentity, dataset_digest
from marketdata.schemas import (
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    TS,
    VOLUME,
    canonicalise,
)
from marketdata.validator import ValidationReport


def _identity(**overrides) -> DatasetIdentity:
    fields = {"source": "fyers:history", "symbol": "NIFTY", "resolution": "1"}
    fields.update(overrides)
    return DatasetIdentity(**fields)


def _valid_rows(n: int = 5, *, start: str = "2026-01-01 09:15", base: float = 24000.0) -> pd.DataFrame:
    rows = []
    ts0 = pd.Timestamp(start, tz=IST_NAME)
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


def _invalid_ohlc_rows() -> pd.DataFrame:
    # high < low: an impossible bar. Structurally canonicalisable, but
    # market-data-invalid.
    return pd.DataFrame(
        {
            TS: [pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)],
            OPEN: [100.0],
            HIGH: [50.0],
            LOW: [200.0],
            CLOSE: [100.0],
            VOLUME: [1000],
        }
    )


def _conflicting_timestamp_rows() -> pd.DataFrame:
    # Two DISTINCT canonical observations sharing one timestamp -> a
    # BLOCKER-severity CANONICAL_CONFLICTING_TIMESTAMPS anomaly.
    t = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
    return pd.DataFrame(
        {
            TS: [t, t],
            OPEN: [100.0, 200.0],
            HIGH: [101.0, 201.0],
            LOW: [99.0, 199.0],
            CLOSE: [100.5, 200.5],
            VOLUME: [1000, 2000],
        }
    )


# --- Bug 1 regression: forged canonicalisation evidence ---------------------


def test_forged_canonicalisation_result_can_no_longer_be_supplied():
    # The old API accepted a CanonicalisationResult directly; build() no
    # longer has such a parameter at all, so this kind of forgery is
    # structurally impossible rather than merely discouraged.
    import inspect

    params = inspect.signature(ValidatedDataset.build).parameters
    assert "canonicalisation" not in params
    assert set(params) == {"raw_frame", "identity", "validation_policy"}


def test_forged_canonicalisation_result_rejected_if_attempted():
    frame_a = _valid_rows(n=3, start="2026-01-01 09:15", base=100.0)
    frame_b = _valid_rows(n=7, start="2026-02-01 09:15", base=900000.0)
    canon_a = canonicalise(frame_a)
    canon_b = canonicalise(frame_b)

    # There is no parameter to pass canon_b's evidence alongside canon_a's
    # frame at all -- attempting the old call shape is a TypeError.
    with pytest.raises(TypeError):
        ValidatedDataset.build(
            canon_a.frame,
            identity=_identity(),
            transformations=canon_b.transformations,
            source_anomalies=canon_b.source_anomalies,
            source=canon_b.source,
        )


# --- build canonicalises internally ------------------------------------------


def test_build_accepts_raw_dataframe_and_canonicalises_internally():
    raw = _valid_rows()
    ds = ValidatedDataset.build(raw, identity=_identity())
    # The bound frame is canonical (tz-aware IST, float64 prices, Int64
    # volume, sorted) even though `raw` itself was never canonicalised by
    # the caller.
    assert str(ds.frame[TS].dtype.tz) == IST_NAME
    assert ds.frame[OPEN].dtype == "float64"
    assert str(ds.frame[VOLUME].dtype) == "Int64"


def test_source_evidence_corresponds_to_the_raw_frame_supplied():
    raw = _valid_rows(n=4)
    ds = ValidatedDataset.build(raw, identity=_identity())
    assert ds.source_evidence.row_count == 4


def test_caller_raw_frame_mutation_after_build_changes_nothing():
    raw = _valid_rows()
    ds = ValidatedDataset.build(raw, identity=_identity())
    original_digest = ds.digest
    original_frame = ds.frame

    raw.iloc[0, raw.columns.get_loc(OPEN)] = -999.0

    assert ds.digest == original_digest
    pd.testing.assert_frame_equal(ds.frame, original_frame)


def test_adversarial_build_valid_then_mutate_caller_raw_frame_into_invalid():
    raw = _valid_rows()
    identity = _identity()
    ds = ValidatedDataset.build(raw, identity=identity)

    original_digest = ds.digest
    original_validity = ds.market_data_validity
    original_frame = ds.frame

    raw.iloc[0, raw.columns.get_loc(HIGH)] = -1.0  # would trip an OHLC error

    assert ds.digest == original_digest
    assert ds.market_data_validity == original_validity
    pd.testing.assert_frame_equal(ds.frame, original_frame)
    assert ds.validation.is_usable is True


# --- ValidationPolicy: self-validation and deep immutability ----------------


def test_fake_mutable_policy_object_is_rejected():
    # A duck-typed lookalike presenting the same attribute names is not an
    # actual ValidationPolicy, so build() must reject it outright rather
    # than store an object it never validated or froze.
    class FakePolicy:
        expected_interval_minutes = 1
        sigma_threshold = 10.0
        session_window = None
        max_session_gap_days = None

    with pytest.raises(TypeError):
        ValidatedDataset.build(
            _valid_rows(), identity=_identity(), validation_policy=FakePolicy()
        )


def test_session_window_list_cannot_leak_mutable_state():
    mutable_window = [datetime.time(9, 15), datetime.time(15, 30)]
    policy = ValidationPolicy(session_window=mutable_window)
    assert isinstance(policy.session_window, tuple)

    mutable_window.append(datetime.time(16, 0))  # mutate the caller's own list

    assert policy.session_window == (datetime.time(9, 15), datetime.time(15, 30))


def test_mutation_of_caller_owned_input_after_policy_creation_does_not_change_policy():
    mutable_window = [datetime.time(9, 15), datetime.time(15, 30)]
    policy = ValidationPolicy(session_window=mutable_window)
    ds = ValidatedDataset.build(
        _valid_rows(), identity=_identity(), validation_policy=policy
    )

    mutable_window[0] = datetime.time(0, 0)
    mutable_window.append("corrupted")

    assert ds.validation_policy.session_window == (
        datetime.time(9, 15),
        datetime.time(15, 30),
    )


def test_expected_interval_minutes_zero_rejected():
    with pytest.raises(ValueError):
        ValidationPolicy(expected_interval_minutes=0)


def test_expected_interval_minutes_negative_rejected():
    with pytest.raises(ValueError):
        ValidationPolicy(expected_interval_minutes=-1)


def test_expected_interval_minutes_non_integer_rejected():
    with pytest.raises(TypeError):
        ValidationPolicy(expected_interval_minutes=1.5)


def test_expected_interval_minutes_bool_rejected():
    with pytest.raises(TypeError):
        ValidationPolicy(expected_interval_minutes=True)


def test_sigma_threshold_zero_or_negative_rejected():
    with pytest.raises(ValueError):
        ValidationPolicy(sigma_threshold=0.0)
    with pytest.raises(ValueError):
        ValidationPolicy(sigma_threshold=-5.0)


def test_sigma_threshold_nan_or_inf_rejected():
    with pytest.raises(ValueError):
        ValidationPolicy(sigma_threshold=math.nan)
    with pytest.raises(ValueError):
        ValidationPolicy(sigma_threshold=math.inf)


def test_max_session_gap_days_zero_or_negative_rejected():
    with pytest.raises(ValueError):
        ValidationPolicy(max_session_gap_days=0.0)
    with pytest.raises(ValueError):
        ValidationPolicy(max_session_gap_days=-2.0)


def test_max_session_gap_days_nan_or_inf_rejected():
    with pytest.raises(ValueError):
        ValidationPolicy(max_session_gap_days=math.nan)
    with pytest.raises(ValueError):
        ValidationPolicy(max_session_gap_days=math.inf)


def test_malformed_session_window_rejected():
    with pytest.raises(TypeError):
        ValidationPolicy(session_window=(datetime.time(9, 15),))  # only one
    with pytest.raises(TypeError):
        ValidationPolicy(
            session_window=(datetime.time(9, 15), datetime.time(15, 30), datetime.time(16, 0))
        )  # three
    with pytest.raises(TypeError):
        ValidationPolicy(session_window=("09:15", "15:30"))  # not datetime.time
    with pytest.raises(TypeError):
        ValidationPolicy(session_window=42)  # not iterable of the right shape


def test_invalid_policy_values_fail_before_validator_calculations():
    # Construction itself must raise -- no validate() call, no build() call,
    # ever gets a chance to run against a malformed policy.
    with pytest.raises((ValueError, TypeError)):
        ValidationPolicy(expected_interval_minutes=-1)


def test_valid_policy_still_produces_expected_validation_behavior():
    policy = ValidationPolicy(
        expected_interval_minutes=1, sigma_threshold=8.0, max_session_gap_days=3.0
    )
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity(), validation_policy=policy)
    assert ds.market_data_validity is MarketDataValidity.VALID
    assert ds.validation_policy == policy


def test_validation_policy_is_frozen():
    policy = ValidationPolicy(max_session_gap_days=3.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_session_gap_days = 5.0


def test_validation_policy_values_are_exposed_exactly():
    policy = ValidationPolicy(
        expected_interval_minutes=1,
        sigma_threshold=8.0,
        session_window=None,
        max_session_gap_days=3.0,
    )
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity(), validation_policy=policy)
    assert ds.validation_policy == policy
    assert ds.validation_policy.expected_interval_minutes == 1
    assert ds.validation_policy.sigma_threshold == 8.0
    assert ds.validation_policy.max_session_gap_days == 3.0


def test_default_validation_policy_matches_validate_defaults():
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    assert ds.validation_policy == ValidationPolicy()


def test_different_max_session_gap_days_can_change_validity_same_digest():
    # A gap larger than a strict max_session_gap_days is an ERROR
    # (EXCESSIVE_DATA_GAP); the same gap under a lenient/unset policy is not
    # necessarily an ERROR. Same frame + identity -> same data_digest either
    # way, since the policy is not part of identity.
    ts0 = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
    raw = pd.DataFrame(
        {
            TS: [ts0, ts0 + pd.Timedelta(minutes=1), ts0 + pd.Timedelta(days=10)],
            OPEN: [100.0, 101.0, 102.0],
            HIGH: [105.0, 106.0, 107.0],
            LOW: [95.0, 96.0, 97.0],
            CLOSE: [101.0, 102.0, 103.0],
            VOLUME: [1000, 1001, 1002],
        }
    )
    identity = _identity()

    lenient = ValidatedDataset.build(
        raw,
        identity=identity,
        validation_policy=ValidationPolicy(expected_interval_minutes=1),
    )
    strict = ValidatedDataset.build(
        raw,
        identity=identity,
        validation_policy=ValidationPolicy(
            expected_interval_minutes=1, max_session_gap_days=1.0
        ),
    )

    assert lenient.digest == strict.digest
    assert lenient.market_data_validity == MarketDataValidity.VALID
    assert strict.market_data_validity == MarketDataValidity.INVALID
    assert "EXCESSIVE_DATA_GAP" in {i.code for i in strict.validation.errors}


# --- report generated internally, never accepted as a parameter ------------


def test_build_generates_validation_report_internally():
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    assert isinstance(ds.validation, ValidationReportSnapshot)
    assert ds.validation.symbol == "NIFTY"
    assert ds.validation.resolution == "1"


def test_build_signature_has_no_report_or_digest_parameter():
    import inspect

    params = inspect.signature(ValidatedDataset.build).parameters
    assert "report" not in params
    assert "validation" not in params
    assert "digest" not in params
    assert "schema_version" not in params


def test_build_rejects_a_caller_supplied_report_kwarg():
    fake_report = ValidationReport(
        symbol="SBIN",
        resolution="5",
        row_count=999,
        first_ts=None,
        last_ts=None,
        timezone="Asia/Kolkata",
    )
    with pytest.raises(TypeError):
        ValidatedDataset.build(_valid_rows(), identity=_identity(), report=fake_report)


def test_build_rejects_a_caller_supplied_digest_kwarg():
    with pytest.raises(TypeError):
        ValidatedDataset.build(_valid_rows(), identity=_identity(), digest="0" * 64)


def test_direct_construction_is_blocked():
    with pytest.raises(TypeError):
        ValidatedDataset()


# --- valid frame -> bound digest/report -------------------------------------


def test_valid_frame_builds_usable_dataset():
    identity = _identity()
    ds = ValidatedDataset.build(_valid_rows(), identity=identity)
    assert ds.market_data_validity is MarketDataValidity.VALID
    assert ds.validation.is_usable is True
    assert ds.digest == dataset_digest(identity, ds.frame)


# --- invalid OHLC frame: validated, possibly invalid ------------------------


def test_invalid_ohlc_frame_still_builds_with_error_evidence():
    ds = ValidatedDataset.build(_invalid_ohlc_rows(), identity=_identity())
    assert ds.market_data_validity is MarketDataValidity.INVALID
    assert ds.validation.is_usable is False
    codes = {i.code for i in ds.validation.errors}
    assert "OHLC_HIGH_BELOW_LOW" in codes
    assert len(ds.digest) == 64


def test_trust_blocker_anomaly_prevents_build():
    canon = canonicalise(_conflicting_timestamp_rows())
    codes = {a.code for a in canon.source_anomalies}
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" in codes
    with pytest.raises(TrustBlockerError):
        ValidatedDataset.build(_conflicting_timestamp_rows(), identity=_identity())


# --- mutation independence: returned frame -----------------------------------


def test_mutating_returned_frame_has_no_effect():
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    original_digest = ds.digest

    returned = ds.frame
    returned.iloc[0, returned.columns.get_loc(OPEN)] = -999.0

    assert ds.frame.iloc[0][OPEN] != -999.0
    assert ds.digest == original_digest


def test_returned_frame_remains_a_defensive_copy_each_access():
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    first = ds.frame
    second = ds.frame
    assert first is not second
    pd.testing.assert_frame_equal(first, second)


# --- digest / identity behaviour ---------------------------------------------


def test_digest_matches_internal_frame():
    identity = _identity()
    ds = ValidatedDataset.build(_valid_rows(), identity=identity)
    assert ds.digest == dataset_digest(identity, ds.frame)


def test_different_symbol_changes_digest():
    raw = _valid_rows()
    d1 = ValidatedDataset.build(raw, identity=_identity(symbol="NIFTY")).digest
    d2 = ValidatedDataset.build(raw, identity=_identity(symbol="SBIN")).digest
    assert d1 != d2


def test_different_source_changes_digest():
    raw = _valid_rows()
    d1 = ValidatedDataset.build(raw, identity=_identity(source="fyers:history")).digest
    d2 = ValidatedDataset.build(raw, identity=_identity(source="other")).digest
    assert d1 != d2


def test_different_resolution_changes_digest():
    raw = _valid_rows()
    d1 = ValidatedDataset.build(raw, identity=_identity(resolution="1")).digest
    d2 = ValidatedDataset.build(raw, identity=_identity(resolution="5")).digest
    assert d1 != d2


def test_build_is_deterministic():
    raw = _valid_rows()
    identity = _identity()
    ds1 = ValidatedDataset.build(raw, identity=identity)
    ds2 = ValidatedDataset.build(raw, identity=identity)
    assert ds1.digest == ds2.digest


# --- immutability of evidence and object -------------------------------------


def test_validation_snapshot_is_immutable():
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ds.validation.symbol = "SBIN"
    assert isinstance(ds.validation.issues, tuple)


def test_canonicalisation_evidence_is_immutable_tuples():
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    assert isinstance(ds.transformations, tuple)
    assert isinstance(ds.source_anomalies, tuple)
    with pytest.raises(AttributeError):
        ds.transformations.append(object())


def test_dataset_object_itself_is_immutable():
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    with pytest.raises(AttributeError):
        ds.digest = "0" * 64
    with pytest.raises(AttributeError):
        ds.new_attribute = 1


def test_dataset_cannot_be_pickled():
    import pickle

    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    with pytest.raises(TypeError):
        pickle.dumps(ds)


# --- pandas is not claimed immutable -----------------------------------------


def test_returned_frame_is_an_ordinary_mutable_dataframe():
    ds = ValidatedDataset.build(_valid_rows(), identity=_identity())
    returned = ds.frame
    # No exception: pandas DataFrames are never claimed immutable here.
    returned.iloc[0, returned.columns.get_loc(OPEN)] = 12345.0
    assert returned.iloc[0][OPEN] == 12345.0
