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
    AnomalySeverity,
    CanonicalisationResult,
    SchemaError,
    assert_canonical,
    canonicalise,
    canonicalise_fyers_candles,
    empty_ohlcv,
    from_fyers_candles,
    normalise,
)
from tests.conftest import candles_payload, make_ohlcv


# --- helpers for the canonicalisation-evidence tests ---------------------


def anomaly_codes(result: CanonicalisationResult) -> set[str]:
    return {a.code for a in result.source_anomalies}


def transformation_codes(result: CanonicalisationResult) -> set[str]:
    return {t.code for t in result.transformations}


def anomaly(result: CanonicalisationResult, code: str):
    return next(a for a in result.source_anomalies if a.code == code)


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


# --- normalisation contract: value preservation --------------------------


def test_normalise_preserves_every_price_exactly():
    """Exact equality, not tolerance: normalisation must not alter values."""
    frame = make_ohlcv(200, seed=11)
    out = normalise(frame.copy())
    for col in (OPEN, HIGH, LOW, CLOSE):
        assert out[col].tolist() == frame[col].tolist()


def test_normalise_preserves_volume_exactly():
    frame = make_ohlcv(200, seed=12)
    out = normalise(frame.copy())
    assert out[VOLUME].tolist() == frame[VOLUME].tolist()


def test_normalise_does_not_fabricate_missing_volume():
    """Regression: volume was filled with 0, asserting 'no trades occurred'."""
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("Int64")
    frame.loc[2, VOLUME] = pd.NA
    out = normalise(frame)
    assert pd.isna(out.loc[2, VOLUME]), "missing volume must stay missing"
    assert (out[VOLUME] == 0).sum() == 0, "no volume may have been filled with 0"


def test_normalise_preserves_already_missing_prices():
    """Genuine source missingness passes through for the validator to report."""
    frame = make_ohlcv(5)
    frame.loc[3, CLOSE] = float("nan")
    out = normalise(frame)
    assert pd.isna(out.loc[3, CLOSE])


def test_normalise_refuses_to_coerce_unparseable_price():
    """Regression: 'bad' silently became NaN, disguising a source defect."""
    frame = make_ohlcv(5)
    frame[OPEN] = frame[OPEN].astype(object)
    frame.loc[1, OPEN] = "bad"
    with pytest.raises(SchemaError, match="not parseable as numeric"):
        normalise(frame)


def test_normalise_refuses_to_coerce_unparseable_volume():
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype(object)
    frame.loc[1, VOLUME] = "lots"
    with pytest.raises(SchemaError, match="not parseable as numeric"):
        normalise(frame)


def test_coercion_error_names_the_offending_value():
    """The error must identify what was wrong, not just that something was."""
    frame = make_ohlcv(5)
    frame[HIGH] = frame[HIGH].astype(object)
    frame.loc[2, HIGH] = "N/A"
    with pytest.raises(SchemaError, match="N/A"):
        normalise(frame)


def test_volume_dtype_is_nullable_int64():
    out = normalise(make_ohlcv(5))
    assert str(out[VOLUME].dtype) == "Int64"


def test_assert_canonical_rejects_wrong_column_order():
    frame = make_ohlcv(5)[[VOLUME, TS, OPEN, HIGH, LOW, CLOSE]]
    with pytest.raises(SchemaError, match="Column mismatch"):
        assert_canonical(frame)


# =========================================================================
# Unit 2 — canonicalisation contract and source evidence
#
# The defect these cover: normalise() sorted the source before the validator
# could observe it, and silently deleted every non-canonical column. Testing
# only the OUTPUT (as test_normalise_sorts_unsorted_input does) cannot detect
# either, because the output looks identical whether or not the source was
# anomalous.
# =========================================================================


def _unsorted_frame(n: int = 6) -> pd.DataFrame:
    frame = make_ohlcv(n, seed=21)
    order = list(range(n))
    order[1], order[3] = order[3], order[1]
    return frame.iloc[order].reset_index(drop=True)


# --- extra / unsupported columns -----------------------------------------


def test_extra_column_is_rejected_and_named():
    """Regression: extras were silently deleted by frame.loc[:, OHLCV_COLUMNS]."""
    frame = make_ohlcv(5)
    frame["mystery"] = 1
    with pytest.raises(SchemaError, match="mystery"):
        canonicalise(frame)


def test_multiple_extra_columns_all_named():
    frame = make_ohlcv(5)
    frame["oi"] = 1
    frame["vwap"] = 2.0
    with pytest.raises(SchemaError) as exc:
        canonicalise(frame)
    assert "oi" in str(exc.value) and "vwap" in str(exc.value)


def test_extra_column_rejected_even_when_ohlcv_is_perfect():
    """Adversarial B: one unknown column poisons an otherwise flawless frame."""
    frame = make_ohlcv(50, seed=7)
    frame["broker_extra"] = 0
    with pytest.raises(SchemaError, match="broker_extra"):
        canonicalise(frame)


def test_missing_column_still_rejected():
    with pytest.raises(SchemaError, match="Missing required column"):
        canonicalise(make_ohlcv(5).drop(columns=[VOLUME]))


# --- source column inventory and row count -------------------------------


def test_source_column_inventory_is_recorded():
    result = canonicalise(make_ohlcv(5)[[VOLUME, TS, OPEN, HIGH, LOW, CLOSE]])
    assert result.source.column_inventory == (VOLUME, TS, OPEN, HIGH, LOW, CLOSE)


def test_source_row_count_is_recorded():
    result = canonicalise(make_ohlcv(17))
    assert result.source.row_count == 17


# --- source ordering ------------------------------------------------------


def test_unsorted_source_yields_sorted_frame_AND_records_the_anomaly():
    """Both halves matter: output sorted, and evidence the source was not."""
    result = canonicalise(_unsorted_frame())
    assert result.frame[TS].is_monotonic_increasing, "canonical output must be sorted"
    assert result.source.timestamps_sorted is False
    assert "SOURCE_UNSORTED" in anomaly_codes(result)
    assert "ROWS_SORTED" in transformation_codes(result)
    assert anomaly(result, "SOURCE_UNSORTED").severity is AnomalySeverity.INFO


def test_sorted_source_records_no_ordering_anomaly():
    result = canonicalise(make_ohlcv(10))
    assert result.source.timestamps_sorted is True
    assert "SOURCE_UNSORTED" not in anomaly_codes(result)
    assert "ROWS_SORTED" not in transformation_codes(result)


def test_inversion_count_is_precisely_defined():
    """Adjacent descending pairs — unambiguous even with duplicate timestamps."""
    result = canonicalise(_unsorted_frame())
    assert result.source.descending_adjacent_pairs == 2


def test_sorted_source_has_zero_inversions():
    assert canonicalise(make_ohlcv(10)).source.descending_adjacent_pairs == 0


# --- timezone -------------------------------------------------------------


def test_timezone_conversion_is_recorded():
    frame = make_ohlcv(5)
    frame[TS] = frame[TS].dt.tz_convert("UTC")
    result = canonicalise(frame)
    assert "TIMEZONE_CONVERTED" in transformation_codes(result)


def test_no_timezone_record_when_already_ist():
    assert "TIMEZONE_CONVERTED" not in transformation_codes(canonicalise(make_ohlcv(5)))


def test_timezone_conversion_preserves_the_exact_instant():
    ist = make_ohlcv(5, seed=31)
    utc = ist.copy()
    utc[TS] = utc[TS].dt.tz_convert("UTC")
    assert canonicalise(utc).frame[TS].tolist() == canonicalise(ist).frame[TS].tolist()


def test_naive_timestamps_still_rejected():
    frame = make_ohlcv(5)
    frame[TS] = frame[TS].dt.tz_localize(None)
    with pytest.raises(SchemaError, match="tz-aware"):
        canonicalise(frame)


# --- duplicates survive this layer ---------------------------------------


def test_exact_duplicate_rows_survive_and_are_recorded():
    frame = make_ohlcv(10)
    doubled = pd.concat([frame, frame.iloc[[4]]], ignore_index=True)
    result = canonicalise(doubled)
    assert len(result.frame) == 11, "row multiplicity must be preserved"
    assert result.source.exact_duplicate_row_count == 1
    assert "SOURCE_EXACT_DUPLICATE_ROWS" in anomaly_codes(result)
    assert anomaly(result, "SOURCE_EXACT_DUPLICATE_ROWS").severity is AnomalySeverity.WARNING


def test_duplicate_evidence_is_not_described_as_byte_identical():
    """pandas .duplicated() establishes value-equality, not byte-identity.

    Regression: the anomaly description previously claimed rows were
    'byte-identical', which is a stronger and inaccurate claim.
    """
    frame = make_ohlcv(10)
    doubled = pd.concat([frame, frame.iloc[[4]]], ignore_index=True)
    result = canonicalise(doubled)
    description = anomaly(result, "SOURCE_EXACT_DUPLICATE_ROWS").description
    assert "byte-identical" not in description.lower()


# --- source-level evidence must not be computed from converted values ----


def test_source_evidence_reflects_source_representation_not_canonical_value():
    """Load-bearing principle: SOURCE EVIDENCE != POST-CONVERSION EVIDENCE.

    Two rows share a timestamp. Row 1's open is the int 1; row 2's open is the
    string "1". As RECEIVED these are different representations -- not an
    exact duplicate -- even though both parse to the identical canonical
    float 1.0. Source-level conflict detection must see two distinct
    observations (a conflict), not silently treat them as one, and the
    canonical output must still show the converged numeric value.
    """
    base = make_ohlcv(1, seed=61).iloc[0]
    row1 = base.copy()
    row1[OPEN] = 1
    row2 = base.copy()
    row2[OPEN] = "1"
    frame = pd.DataFrame([row1, row2]).reset_index(drop=True)
    frame[OPEN] = frame[OPEN].astype(object)  # preserve the mixed source types

    result = canonicalise(frame)
    assert len(result.frame) == 2, "both observations must survive"

    # int(1) and "1" are NOT equal as received -- this is not an exact
    # source-level duplicate.
    assert "SOURCE_EXACT_DUPLICATE_ROWS" not in anomaly_codes(result)

    # But the SOURCE did present two rows sharing one timestamp -- that
    # structural fact is true regardless of how the values later resolve.
    assert "SOURCE_DUPLICATE_TIMESTAMPS" in anomaly_codes(result)
    assert result.source.duplicate_timestamp_row_count == 2

    # Once losslessly converted, both canonical values agree (1.0 == 1.0),
    # so there is NO market-value conflict. A representation difference that
    # converges to the same canonical value must not be reported as a
    # BLOCKER: doing so would treat ordinary lossless parsing as a data
    # integrity failure.
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" not in anomaly_codes(result)
    assert result.frame[OPEN].tolist() == [1.0, 1.0]


def _rows_sharing_one_timestamp(*close_values: float) -> pd.DataFrame:
    """Build a frame where every row shares one timestamp.

    Rows with equal ``close`` are the "same observation, repeated"; rows with
    different ``close`` are "distinct observations at that timestamp". Every
    other field is held constant so ``close`` alone determines row identity.
    """
    base = make_ohlcv(1, seed=53).iloc[0]
    rows = []
    for close in close_values:
        row = base.copy()
        row[CLOSE] = close
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


# --- full conflict matrix (manager-specified cases A-F) ------------------
#
# Regression: the original algorithm compared
#     group.duplicated(keep=False).sum() < len(group)
# which is wrong whenever every row in the group has SOME matching partner,
# even if there are two or more distinct partner-groups. Case D (A,A,B,B) is
# the exact counterexample: every row is "duplicated" by its own twin, so the
# old check saw sum()==len(group) and concluded "no conflict" -- reproduced
# and confirmed against the unpatched implementation before this fix.
#
# The correct test is: how many DISTINCT observations share this timestamp?
# More than one -> conflict, regardless of how many copies of each exist.


def test_conflict_matrix_case_A_two_identical_is_duplicate_not_conflict():
    result = canonicalise(_rows_sharing_one_timestamp(100.0, 100.0))
    assert len(result.frame) == 2, "both rows must survive"
    assert "SOURCE_EXACT_DUPLICATE_ROWS" in anomaly_codes(result)
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" not in anomaly_codes(result)


def test_conflict_matrix_case_B_two_distinct_is_conflict():
    result = canonicalise(_rows_sharing_one_timestamp(100.0, 200.0))
    assert len(result.frame) == 2
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" in anomaly_codes(result)
    assert anomaly(result, "CANONICAL_CONFLICTING_TIMESTAMPS").severity is AnomalySeverity.BLOCKER


def test_conflict_matrix_case_C_AAB_is_conflict():
    result = canonicalise(_rows_sharing_one_timestamp(100.0, 100.0, 200.0))
    assert len(result.frame) == 3
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" in anomaly_codes(result)


def test_conflict_matrix_case_D_AABB_is_conflict():
    """THE regression case: every row has a matching twin, but two distinct
    observations (100.0 and 200.0) share the timestamp."""
    result = canonicalise(_rows_sharing_one_timestamp(100.0, 100.0, 200.0, 200.0))
    assert len(result.frame) == 4, "all four rows must survive"
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" in anomaly_codes(result)
    assert anomaly(result, "CANONICAL_CONFLICTING_TIMESTAMPS").severity is AnomalySeverity.BLOCKER


def test_conflict_matrix_case_E_four_identical_is_duplicate_not_conflict():
    result = canonicalise(_rows_sharing_one_timestamp(100.0, 100.0, 100.0, 100.0))
    assert len(result.frame) == 4
    assert "SOURCE_EXACT_DUPLICATE_ROWS" in anomaly_codes(result)
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" not in anomaly_codes(result)


def test_conflict_matrix_case_F_conflict_in_one_of_two_timestamp_groups():
    """T1 has AA (no conflict); T2 has B,C (conflict). Overall: conflict."""
    t1 = _rows_sharing_one_timestamp(100.0, 100.0)
    t2 = _rows_sharing_one_timestamp(300.0, 400.0)
    t2[TS] = t2[TS] + pd.Timedelta(minutes=1)
    combined = pd.concat([t1, t2], ignore_index=True)
    result = canonicalise(combined)
    assert len(result.frame) == 4
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" in anomaly_codes(result)


def test_no_duplicate_anomaly_on_clean_input():
    codes = anomaly_codes(canonicalise(make_ohlcv(10)))
    assert "SOURCE_EXACT_DUPLICATE_ROWS" not in codes
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" not in codes


# --- value preservation ---------------------------------------------------


def test_malformed_numeric_still_rejected():
    frame = make_ohlcv(5)
    frame[OPEN] = frame[OPEN].astype(object)
    frame.loc[1, OPEN] = "bad"
    with pytest.raises(SchemaError, match="not parseable as numeric"):
        canonicalise(frame)


def test_genuine_na_volume_is_preserved():
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("Int64")
    frame.loc[2, VOLUME] = pd.NA
    result = canonicalise(frame)
    assert pd.isna(result.frame.loc[2, VOLUME])
    assert (result.frame[VOLUME] == 0).sum() == 0


def test_fractional_volume_raises_schema_error_not_pandas_typeerror():
    """Baseline surfaced a raw pandas TypeError at this boundary."""
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("float64")
    frame.loc[1, VOLUME] = 250.7
    with pytest.raises(SchemaError, match="whole number"):
        canonicalise(frame)


def test_fractional_volume_is_never_rounded_or_truncated():
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("float64")
    frame.loc[1, VOLUME] = 250.7
    with pytest.raises(SchemaError):
        canonicalise(frame)
    assert frame.loc[1, VOLUME] == 250.7, "caller's frame must be untouched"


def test_integral_float_volume_is_accepted_losslessly():
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("float64")
    result = canonicalise(frame)
    assert result.frame[VOLUME].tolist() == frame[VOLUME].astype("int64").tolist()


# --- duplicate-timestamp structural evidence (distinct from exact-duplicate
# rows and distinct from canonical conflict) -------------------------------


def test_duplicate_timestamp_row_count_definition():
    """Locks the definition: rows whose timestamp is shared, regardless of
    whether the rest of the row matches. Both A,A and A,B produce the same
    count for the same group size -- multiplicity of VALUE agreement is a
    separate question from multiplicity of TIMESTAMP sharing."""
    same_values = canonicalise(_rows_sharing_one_timestamp(100.0, 100.0))
    different_values = canonicalise(_rows_sharing_one_timestamp(100.0, 200.0))
    assert same_values.source.duplicate_timestamp_row_count == 2
    assert different_values.source.duplicate_timestamp_row_count == 2


def test_duplicate_timestamp_row_count_zero_when_all_timestamps_unique():
    result = canonicalise(make_ohlcv(10))
    assert result.source.duplicate_timestamp_row_count == 0
    assert "SOURCE_DUPLICATE_TIMESTAMPS" not in anomaly_codes(result)


def test_exact_duplicate_implies_duplicate_timestamp_but_not_reverse():
    """A,A is both an exact duplicate AND a duplicate timestamp. A,B (distinct
    values) is a duplicate timestamp WITHOUT being an exact duplicate."""
    exact = canonicalise(_rows_sharing_one_timestamp(100.0, 100.0))
    assert "SOURCE_EXACT_DUPLICATE_ROWS" in anomaly_codes(exact)
    assert "SOURCE_DUPLICATE_TIMESTAMPS" in anomaly_codes(exact)

    distinct = canonicalise(_rows_sharing_one_timestamp(100.0, 200.0))
    assert "SOURCE_EXACT_DUPLICATE_ROWS" not in anomaly_codes(distinct)
    assert "SOURCE_DUPLICATE_TIMESTAMPS" in anomaly_codes(distinct)


# --- lossless dtype/representation conversion evidence (manager matrix) --
#
# Frozen architecture section 14: a lossless dtype conversion is a NORMAL
# TRANSFORMATION, not a defect. It must be recorded when it actually happens,
# and NOT recorded when the source already had the canonical dtype.


def test_dtype_matrix_A_canonical_float64_price_records_nothing():
    result = canonicalise(make_ohlcv(5))  # already float64 prices
    codes = transformation_codes(result)
    assert "DTYPE_CONVERTED_OPEN" not in codes
    assert "DTYPE_CONVERTED_HIGH" not in codes
    assert "DTYPE_CONVERTED_LOW" not in codes
    assert "DTYPE_CONVERTED_CLOSE" not in codes


def test_dtype_matrix_B_numeric_string_price_is_recorded():
    frame = make_ohlcv(5)
    frame[OPEN] = frame[OPEN].astype(object)
    frame.loc[0, OPEN] = "24000.5"  # numeric string, still object dtype overall
    result = canonicalise(frame)
    assert "DTYPE_CONVERTED_OPEN" in transformation_codes(result)
    assert result.frame[OPEN].iloc[0] == 24000.5


def test_dtype_matrix_C_canonical_Int64_volume_records_nothing():
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("Int64")
    result = canonicalise(frame)
    assert "DTYPE_CONVERTED_VOLUME" not in transformation_codes(result)


def test_dtype_matrix_D_integral_float64_volume_is_recorded():
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("float64")
    result = canonicalise(frame)
    assert "DTYPE_CONVERTED_VOLUME" in transformation_codes(result)
    description = next(
        t.description for t in result.transformations
        if t.code == "DTYPE_CONVERTED_VOLUME"
    )
    assert "float64" in description and "Int64" in description


def test_dtype_matrix_E_fractional_volume_raises_no_result():
    """SchemaError, no partial CanonicalisationResult -- distinct from a
    dtype TRANSFORMATION, which only ever describes a SUCCESSFUL conversion."""
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("float64")
    frame.loc[1, VOLUME] = 250.7
    with pytest.raises(SchemaError, match="whole number"):
        canonicalise(frame)


def test_dtype_matrix_F_malformed_numeric_string_raises_no_result():
    frame = make_ohlcv(5)
    frame[OPEN] = frame[OPEN].astype(object)
    frame.loc[1, OPEN] = "not-a-number"
    with pytest.raises(SchemaError, match="not parseable as numeric"):
        canonicalise(frame)


def test_dtype_conversion_is_not_described_as_a_value_change():
    """Section 11: a lossless representation conversion must not be
    describable as a semantic market-value modification."""
    frame = make_ohlcv(5)
    frame[VOLUME] = frame[VOLUME].astype("float64")
    result = canonicalise(frame)
    description = next(
        t.description for t in result.transformations
        if t.code == "DTYPE_CONVERTED_VOLUME"
    )
    assert "value change" not in description.lower() or "not a value change" in description.lower()
    assert "representation" in description.lower()


# --- defensive copy and immutability of evidence -------------------------


def test_canonicalise_does_not_mutate_the_callers_frame():
    frame = make_ohlcv(5)[[VOLUME, TS, OPEN, HIGH, LOW, CLOSE]]
    before = list(frame.columns)
    canonicalise(frame)
    assert list(frame.columns) == before, "input column order must be untouched"


def test_result_is_unaffected_by_later_mutation_of_the_input():
    """Adversarial D: the result owns a defensive copy."""
    frame = make_ohlcv(5)
    result = canonicalise(frame)
    original = result.frame[CLOSE].tolist()
    frame.loc[0, CLOSE] = 99999.0
    assert result.frame[CLOSE].tolist() == original


def test_evidence_collections_are_immutable():
    result = canonicalise(_unsorted_frame())
    assert isinstance(result.transformations, tuple)
    assert isinstance(result.source_anomalies, tuple)
    assert isinstance(result.source.column_inventory, tuple)
    with pytest.raises(AttributeError):
        result.source_anomalies.append("forged")


def test_result_record_is_frozen():
    result = canonicalise(make_ohlcv(5))
    with pytest.raises(Exception):
        result.source_anomalies = ()


# --- FYERS positional contract -------------------------------------------


def test_fyers_width_seven_is_rejected():
    """No x_unknown_7 invention: an unnamed positional field has no meaning."""
    rows = [[1767239100, 1.0, 2.0, 0.5, 1.5, 10, 999]]
    with pytest.raises(SchemaError, match="row width"):
        canonicalise_fyers_candles(rows)


def test_fyers_width_five_is_rejected():
    with pytest.raises(SchemaError, match="row width"):
        canonicalise_fyers_candles([[1767239100, 1.0, 2.0, 0.5, 1.5]])


def test_fyers_canonicalisation_returns_evidence():
    result = canonicalise_fyers_candles(candles_payload(3)["candles"])
    assert isinstance(result, CanonicalisationResult)
    assert result.source.row_count == 3
    assert result.frame[TS].is_monotonic_increasing


def test_fyers_unsorted_payload_records_the_anomaly():
    """The production parser previously erased this evidence entirely."""
    rows = candles_payload(4)["candles"]
    rows[1], rows[3] = rows[3], rows[1]
    result = canonicalise_fyers_candles(rows)
    assert result.source.timestamps_sorted is False
    assert "SOURCE_UNSORTED" in anomaly_codes(result)
    assert result.frame[TS].is_monotonic_increasing


def test_fyers_records_the_epoch_to_ist_transformation():
    """Regression: by the time canonicalise() ran, ts was already IST, so the
    epoch-seconds -> Asia/Kolkata conversion left no transformation evidence.
    """
    result = canonicalise_fyers_candles(candles_payload(3)["candles"])
    codes = transformation_codes(result)
    assert "FYERS_EPOCH_TO_IST" in codes
    epoch_transform = next(
        t for t in result.transformations if t.code == "FYERS_EPOCH_TO_IST"
    )
    assert "epoch" in epoch_transform.description.lower()
    assert "Asia/Kolkata" in epoch_transform.description


def test_fyers_epoch_transformation_and_unsorted_evidence_coexist():
    """Both facts about the FYERS ingestion must be visible together: the raw
    payload needed sorting, AND the adapter performed the epoch conversion."""
    rows = candles_payload(4)["candles"]
    rows[1], rows[3] = rows[3], rows[1]
    result = canonicalise_fyers_candles(rows)
    codes = transformation_codes(result)
    assert "FYERS_EPOCH_TO_IST" in codes
    assert "ROWS_SORTED" in codes
    assert result.source.row_count == 4
    assert len(result.frame) == 4, "no observations may be removed"


def test_fyers_empty_payload_has_no_epoch_transformation():
    """Nothing was actually converted, so no transformation is claimed."""
    result = canonicalise_fyers_candles([])
    assert transformation_codes(result) == set()


def test_fyers_empty_payload_source_evidence_is_fully_honest():
    """Section 13: for [], every fact must be the vacuous/absent case, not a
    default that happens to look like 'nothing wrong'."""
    result = canonicalise_fyers_candles([])
    assert result.source.row_count == 0
    assert result.source.timestamps_sorted is True
    assert result.source.descending_adjacent_pairs == 0
    assert result.source.exact_duplicate_row_count == 0
    assert result.source.duplicate_timestamp_row_count == 0
    assert anomaly_codes(result) == set()
    assert transformation_codes(result) == set()


# --- FYERS adapter-input evidence: measured on RAW epoch integers, BEFORE
# epoch->Timestamp conversion (section 6/7) ---------------------------------


def test_fyers_raw_epoch_order_is_measured_before_conversion():
    """epoch3, epoch1, epoch2 (out of order) -> SOURCE_UNSORTED recorded,
    canonical output ascending IST, all rows survive."""
    base = candles_payload(3)["candles"]  # three ascending epochs
    shuffled = [base[2], base[0], base[1]]  # epoch3, epoch1, epoch2
    result = canonicalise_fyers_candles(shuffled)
    assert result.source.row_count == 3
    assert result.source.timestamps_sorted is False
    assert "SOURCE_UNSORTED" in anomaly_codes(result)
    assert result.frame[TS].is_monotonic_increasing
    assert len(result.frame) == 3, "no observations may be removed"
    # The canonical output's actual epoch order, once sorted, must match the
    # original ascending sequence -- proving the sort used the RIGHT ordering
    # key, not an artifact of shuffling.
    assert result.frame[TS].tolist() == canonicalise_fyers_candles(base).frame[TS].tolist()


def test_fyers_raw_epoch_order_matches_generic_canonicalise_inversion_count():
    """The adapter-level inversion count must agree with what a direct
    generic canonicalise() call would find on the equivalent IST-converted
    frame -- proving the two measurement paths are consistent."""
    base = candles_payload(4)["candles"]
    shuffled = [base[1], base[3], base[0], base[2]]
    result = canonicalise_fyers_candles(shuffled)
    assert result.source.descending_adjacent_pairs > 0
    # Same shuffle, measured generically on the already-converted frame.
    import pandas as _pd

    from core.timeutils import epoch_series_to_ist as _to_ist

    frame = _pd.DataFrame(shuffled, columns=list(OHLCV_COLUMNS))
    frame[TS] = _to_ist(frame[TS])
    generic = canonicalise(frame)
    assert result.source.descending_adjacent_pairs == generic.source.descending_adjacent_pairs


def test_fyers_raw_duplicate_epoch_is_measured_before_conversion():
    row = candles_payload(1)["candles"][0]
    duplicate_epoch_different_close = list(row)
    duplicate_epoch_different_close[4] = duplicate_epoch_different_close[4] + 10.0
    result = canonicalise_fyers_candles([row, duplicate_epoch_different_close])
    assert result.source.duplicate_timestamp_row_count == 2
    assert "SOURCE_DUPLICATE_TIMESTAMPS" in anomaly_codes(result)
    assert len(result.frame) == 2, "both observations must survive"


def test_fyers_raw_exact_duplicate_row_is_measured_before_conversion():
    row = candles_payload(1)["candles"][0]
    result = canonicalise_fyers_candles([row, list(row)])
    assert result.source.exact_duplicate_row_count == 1
    assert "SOURCE_EXACT_DUPLICATE_ROWS" in anomaly_codes(result)


def test_fyers_exact_duplicate_candle_is_a_duplicate_not_a_conflict():
    row = candles_payload(1)["candles"][0]
    result = canonicalise_fyers_candles([row, list(row)])
    assert len(result.frame) == 2
    assert "SOURCE_EXACT_DUPLICATE_ROWS" in anomaly_codes(result)
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" not in anomaly_codes(result)


def test_fyers_same_epoch_different_close_is_a_conflict_not_a_duplicate():
    row_a = candles_payload(1)["candles"][0]
    row_b = list(row_a)
    row_b[4] = row_b[4] + 500.0  # same epoch (index 0), different close
    result = canonicalise_fyers_candles([row_a, row_b])
    assert len(result.frame) == 2, "both observations must survive"
    assert "CANONICAL_CONFLICTING_TIMESTAMPS" in anomaly_codes(result)
    assert "SOURCE_EXACT_DUPLICATE_ROWS" not in anomaly_codes(result)


def test_fyers_column_inventory_is_the_fixed_adapter_mapping():
    """FYERS positional rows carry no field names at all -- 'ts', 'open', etc.
    are the ADAPTER's fixed interpretation of position 0, 1, 2..., not names
    physically present in the payload. This mapping is therefore constant
    regardless of payload size, including an empty payload: an empty list did
    not "have" these columns any more than a non-empty one did, since neither
    ever carried column labels in the first place.
    """
    empty_result = canonicalise_fyers_candles([])
    populated_result = canonicalise_fyers_candles(candles_payload(3)["candles"])
    assert empty_result.source.column_inventory == OHLCV_COLUMNS
    assert populated_result.source.column_inventory == OHLCV_COLUMNS


# --- adversarial combinations --------------------------------------------


def test_adversarial_unsorted_plus_duplicate_plus_missing_volume():
    """Adversarial A: three defects at once, none may mask another."""
    frame = make_ohlcv(8, seed=41)
    frame[VOLUME] = frame[VOLUME].astype("Int64")
    frame.loc[3, VOLUME] = pd.NA
    frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    order = list(range(len(frame)))
    order[0], order[2] = order[2], order[0]
    scrambled = frame.iloc[order].reset_index(drop=True)

    result = canonicalise(scrambled)
    assert result.frame[TS].is_monotonic_increasing
    assert len(result.frame) == 9, "duplicate multiplicity preserved"
    assert result.frame[VOLUME].isna().sum() == 1, "missing volume preserved"
    assert (result.frame[VOLUME] == 0).sum() == 0, "no fabricated zero"
    assert result.source.timestamps_sorted is False
    assert "SOURCE_UNSORTED" in anomaly_codes(result)
    assert "SOURCE_EXACT_DUPLICATE_ROWS" in anomaly_codes(result)


def test_adversarial_fractional_volume_plus_unsorted_fails_atomically():
    """Adversarial C: must raise, not return a partially-canonicalised result."""
    frame = _unsorted_frame(6)
    frame[VOLUME] = frame[VOLUME].astype("float64")
    frame.loc[2, VOLUME] = 3.5
    with pytest.raises(SchemaError, match="whole number"):
        canonicalise(frame)


# --- transitional normalise() wrapper ------------------------------------


def test_normalise_remains_frame_returning_for_existing_callers():
    out = normalise(make_ohlcv(5))
    assert isinstance(out, pd.DataFrame)
    assert_canonical(out)


def test_normalise_enforces_the_same_extra_column_rejection():
    frame = make_ohlcv(5)
    frame["mystery"] = 1
    with pytest.raises(SchemaError, match="mystery"):
        normalise(frame)
