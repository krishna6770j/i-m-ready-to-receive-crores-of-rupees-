"""ValidatedDataset tests.

Core invariant (frozen architecture defect #1, section 16): a validation
report can never be supplied separately from the frame it claims to
validate. ValidatedDataset.build() must generate the report itself, from
the exact canonical frame it ends up holding.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.dataset import MarketDataValidity, TrustBlockerError, ValidatedDataset
from marketdata.evidence import ValidationReportSnapshot
from marketdata.identity import DatasetIdentity, dataset_digest
from marketdata.schemas import (
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    TS,
    VOLUME,
    SchemaError,
    canonicalise,
)
from marketdata.validator import Severity, ValidationIssue, ValidationReport


def _identity(**overrides) -> DatasetIdentity:
    fields = {"source": "fyers:history", "symbol": "NIFTY", "resolution": "1"}
    fields.update(overrides)
    return DatasetIdentity(**fields)


def _valid_rows(n: int = 5) -> pd.DataFrame:
    base = 24000.0
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


def _invalid_ohlc_rows() -> pd.DataFrame:
    # high < low: an impossible bar. Structurally canonical (right dtypes,
    # sorted), but market-data-invalid.
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


# --- report generated internally, never accepted as a parameter ------------


def test_build_generates_validation_report_internally():
    canon = canonicalise(_valid_rows())
    ds = ValidatedDataset.build(canon, identity=_identity())
    assert isinstance(ds.validation, ValidationReportSnapshot)
    assert ds.validation.symbol == "NIFTY"
    assert ds.validation.resolution == "1"


def test_build_signature_has_no_report_parameter():
    import inspect

    params = inspect.signature(ValidatedDataset.build).parameters
    assert "report" not in params
    assert "validation" not in params
    assert "digest" not in params
    assert "schema_version" not in params


def test_build_rejects_a_caller_supplied_report_kwarg():
    canon = canonicalise(_valid_rows())
    fake_report = ValidationReport(
        symbol="SBIN",
        resolution="5",
        row_count=999,
        first_ts=None,
        last_ts=None,
        timezone="Asia/Kolkata",
    )
    with pytest.raises(TypeError):
        ValidatedDataset.build(canon, identity=_identity(), report=fake_report)


def test_direct_construction_is_blocked():
    with pytest.raises(TypeError):
        ValidatedDataset()


# --- valid frame -> bound digest/report -------------------------------------


def test_valid_frame_builds_usable_dataset():
    canon = canonicalise(_valid_rows())
    identity = _identity()
    ds = ValidatedDataset.build(canon, identity=identity)
    assert ds.market_data_validity is MarketDataValidity.VALID
    assert ds.validation.is_usable is True
    assert ds.digest == dataset_digest(identity, canon.frame)


# --- invalid OHLC frame: validated, possibly invalid ------------------------


def test_invalid_ohlc_frame_still_builds_with_error_evidence():
    canon = canonicalise(_invalid_ohlc_rows())
    ds = ValidatedDataset.build(canon, identity=_identity())
    assert ds.market_data_validity is MarketDataValidity.INVALID
    assert ds.validation.is_usable is False
    codes = {i.code for i in ds.validation.errors}
    assert "OHLC_HIGH_BELOW_LOW" in codes
    # Still produces a real digest -- it is a validated (invalid) dataset,
    # not a refused build.
    assert len(ds.digest) == 64


def test_trust_blocker_anomaly_prevents_build():
    canon = canonicalise(_conflicting_timestamp_rows())
    codes = {a.code for a in canon.source_anomalies}
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" in codes
    with pytest.raises(TrustBlockerError):
        ValidatedDataset.build(canon, identity=_identity())


def test_noncanonical_frame_in_canonicalisation_result_rejected():
    # Defence in depth: a hand-crafted, counterfeit CanonicalisationResult
    # whose .frame is not actually canonical must still be rejected.
    import marketdata.schemas as schemas_module

    canon = canonicalise(_valid_rows())
    bogus_frame = canon.frame.iloc[::-1].reset_index(drop=True)  # now unsorted
    bogus = schemas_module.CanonicalisationResult(
        frame=bogus_frame,
        transformations=canon.transformations,
        source_anomalies=canon.source_anomalies,
        source=canon.source,
    )
    with pytest.raises(SchemaError):
        ValidatedDataset.build(bogus, identity=_identity())


# --- mutation independence ---------------------------------------------------


def test_mutating_source_canonicalisation_frame_after_build_has_no_effect():
    canon = canonicalise(_valid_rows())
    identity = _identity()
    ds = ValidatedDataset.build(canon, identity=identity)
    original_digest = ds.digest
    original_frame = ds.frame

    # Mutate the CanonicalisationResult's own frame reference, which the
    # caller still holds, into something structurally different.
    canon.frame.iloc[0, canon.frame.columns.get_loc(OPEN)] = -999.0

    assert ds.digest == original_digest
    pd.testing.assert_frame_equal(ds.frame, original_frame)


def test_mutating_returned_frame_has_no_effect():
    canon = canonicalise(_valid_rows())
    ds = ValidatedDataset.build(canon, identity=_identity())
    original_digest = ds.digest

    returned = ds.frame
    returned.iloc[0, returned.columns.get_loc(OPEN)] = -999.0

    assert ds.frame.iloc[0][OPEN] != -999.0
    assert ds.digest == original_digest


def test_adversarial_build_valid_then_mutate_caller_frame_into_invalid():
    raw = _valid_rows()
    canon = canonicalise(raw)
    identity = _identity()
    ds = ValidatedDataset.build(canon, identity=identity)

    original_digest = ds.digest
    original_validity = ds.market_data_validity
    original_frame = ds.frame

    # Corrupt the caller's own canonicalisation result into OHLC-invalid data.
    col = canon.frame.columns.get_loc(HIGH)
    canon.frame.iloc[0, col] = -1.0  # would trip OHLC_HIGH_TOO_LOW / negative price

    assert ds.digest == original_digest
    assert ds.market_data_validity == original_validity
    pd.testing.assert_frame_equal(ds.frame, original_frame)
    assert ds.validation.is_usable is True  # unchanged from the original valid build


# --- digest / identity behaviour ---------------------------------------------


def test_digest_matches_internal_frame():
    canon = canonicalise(_valid_rows())
    identity = _identity()
    ds = ValidatedDataset.build(canon, identity=identity)
    assert ds.digest == dataset_digest(identity, ds.frame)


def test_different_symbol_changes_digest():
    canon = canonicalise(_valid_rows())
    d1 = ValidatedDataset.build(canon, identity=_identity(symbol="NIFTY")).digest
    d2 = ValidatedDataset.build(canon, identity=_identity(symbol="SBIN")).digest
    assert d1 != d2


def test_different_source_changes_digest():
    canon = canonicalise(_valid_rows())
    d1 = ValidatedDataset.build(canon, identity=_identity(source="fyers:history")).digest
    d2 = ValidatedDataset.build(canon, identity=_identity(source="other")).digest
    assert d1 != d2


def test_different_resolution_changes_digest():
    canon = canonicalise(_valid_rows())
    d1 = ValidatedDataset.build(canon, identity=_identity(resolution="1")).digest
    d2 = ValidatedDataset.build(canon, identity=_identity(resolution="5")).digest
    assert d1 != d2


def test_build_is_deterministic():
    canon = canonicalise(_valid_rows())
    identity = _identity()
    ds1 = ValidatedDataset.build(canon, identity=identity)
    ds2 = ValidatedDataset.build(canon, identity=identity)
    assert ds1.digest == ds2.digest


# --- immutability of evidence and object -------------------------------------


def test_validation_snapshot_is_immutable():
    ds = ValidatedDataset.build(canonicalise(_valid_rows()), identity=_identity())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ds.validation.symbol = "SBIN"
    assert isinstance(ds.validation.issues, tuple)


def test_canonicalisation_evidence_is_immutable_tuples():
    ds = ValidatedDataset.build(canonicalise(_valid_rows()), identity=_identity())
    assert isinstance(ds.transformations, tuple)
    assert isinstance(ds.source_anomalies, tuple)
    with pytest.raises(AttributeError):
        ds.transformations.append(object())


def test_dataset_object_itself_is_immutable():
    ds = ValidatedDataset.build(canonicalise(_valid_rows()), identity=_identity())
    with pytest.raises(AttributeError):
        ds.digest = "0" * 64
    with pytest.raises(AttributeError):
        ds.new_attribute = 1


def test_dataset_cannot_be_pickled():
    import pickle

    ds = ValidatedDataset.build(canonicalise(_valid_rows()), identity=_identity())
    with pytest.raises(TypeError):
        pickle.dumps(ds)


# --- pandas is not claimed immutable -----------------------------------------


def test_returned_frame_is_an_ordinary_mutable_dataframe():
    ds = ValidatedDataset.build(canonicalise(_valid_rows()), identity=_identity())
    returned = ds.frame
    # No exception: pandas DataFrames are never claimed immutable here.
    returned.iloc[0, returned.columns.get_loc(OPEN)] = 12345.0
    assert returned.iloc[0][OPEN] == 12345.0
    # But the object's own bound frame is unaffected (already covered above).
