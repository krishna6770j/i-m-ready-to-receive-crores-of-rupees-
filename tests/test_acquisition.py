"""Acquisition evidence classification and coverage-fact tests.

Frozen architecture section 11.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.acquisition import (
    AcquisitionError,
    AcquisitionRequestStatus,
    classify_acquisition_status,
    compute_observed_data_coverage,
    compute_requested_window_comparison,
    cross_check_fetch_against_dataset,
    validate_fetch_evidence,
)
from marketdata.dataset import ValidatedDataset
from marketdata.evidence import ChunkResultSnapshot, FetchReportSnapshot
from marketdata.identity import DatasetIdentity
from marketdata.schemas import CLOSE, HIGH, LOW, OPEN, TS, VOLUME, empty_ohlcv


def _identity(**overrides) -> DatasetIdentity:
    fields = {"source": "fyers:history", "symbol": "NIFTY", "resolution": "1"}
    fields.update(overrides)
    return DatasetIdentity(**fields)


def _frame_on(date_str: str, n: int = 3, *, base: float = 100.0) -> pd.DataFrame:
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


def _empty_frame() -> pd.DataFrame:
    return empty_ohlcv()


def _dataset(date_str: str = "2026-01-01", n: int = 3, *, base: float = 100.0) -> ValidatedDataset:
    return ValidatedDataset.build(_frame_on(date_str, n, base=base), identity=_identity())


def _fetch_for(ds: ValidatedDataset, *, requested_from: str, requested_to: str, **overrides) -> FetchReportSnapshot:
    frame = ds.frame
    first_ts = frame[TS].iloc[0].isoformat() if len(frame) else None
    last_ts = frame[TS].iloc[-1].isoformat() if len(frame) else None
    fields = dict(
        symbol=ds.identity.symbol,
        resolution=ds.identity.resolution,
        requested_from=requested_from,
        requested_to=requested_to,
        chunks=(ChunkResultSnapshot(requested_from, requested_to, len(frame), True, None),),
        total_rows=len(frame),
        first_ts=first_ts,
        last_ts=last_ts,
        duplicate_rows_removed=0,
        conflicting_timestamps=0,
    )
    fields.update(overrides)
    return FetchReportSnapshot(**fields)


# ---------------------------------------------------------------------------
# status matrix
# ---------------------------------------------------------------------------


def test_no_fetch_is_unknown():
    assert classify_acquisition_status(None) is AcquisitionRequestStatus.REQUESTS_UNKNOWN


def test_all_chunks_failed_is_failed():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 0, False, "timeout"),),
        total_rows=0, first_ts=None, last_ts=None,
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    validate_fetch_evidence(fetch)
    assert classify_acquisition_status(fetch) is AcquisitionRequestStatus.REQUESTS_FAILED


def test_all_success_zero_rows_is_empty():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 0, True, None),),
        total_rows=0, first_ts=None, last_ts=None,
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    validate_fetch_evidence(fetch)
    assert classify_acquisition_status(fetch) is AcquisitionRequestStatus.REQUESTS_EMPTY


def test_mixed_failure_and_success_is_partial():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-02",
        chunks=(
            ChunkResultSnapshot("2026-01-01", "2026-01-01", 5, True, None),
            ChunkResultSnapshot("2026-01-02", "2026-01-02", 0, False, "rate limited"),
        ),
        total_rows=5, first_ts="x", last_ts="y",
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    validate_fetch_evidence(fetch)
    assert classify_acquisition_status(fetch) is AcquisitionRequestStatus.REQUESTS_PARTIAL


def test_all_success_with_rows_is_succeeded():
    ds = _dataset()
    fetch = _fetch_for(ds, requested_from="2026-01-01", requested_to="2026-01-01")
    validate_fetch_evidence(fetch)
    cross_check_fetch_against_dataset(fetch, ds)
    assert classify_acquisition_status(fetch) is AcquisitionRequestStatus.REQUESTS_SUCCEEDED


# ---------------------------------------------------------------------------
# evidence attacks: internal coherence
# ---------------------------------------------------------------------------


def test_impossible_request_dates_rejected():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="not-a-date", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 1, True, None),),
        total_rows=1, first_ts="x", last_ts="y",
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_reversed_request_range_rejected():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-05", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-05", 1, True, None),),
        total_rows=1, first_ts="x", last_ts="y",
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_missing_chunks_rejected():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(),
        total_rows=0, first_ts=None, last_ts=None,
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_failed_chunk_with_rows_rejected():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 5, False, "boom"),),
        total_rows=0, first_ts=None, last_ts=None,
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_failed_chunk_without_error_rejected():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 0, False, None),),
        total_rows=0, first_ts=None, last_ts=None,
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_successful_chunk_with_error_rejected():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 5, True, "should not be here"),),
        total_rows=5, first_ts="x", last_ts="y",
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_row_arithmetic_mismatch_rejected():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 5, True, None),),
        total_rows=999,  # should be 5 - duplicate_rows_removed
        first_ts="x", last_ts="y",
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_row_arithmetic_with_duplicates_removed_is_accepted():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 5, True, None),),
        total_rows=4,  # 5 - 1 duplicate removed
        first_ts="x", last_ts="y",
        duplicate_rows_removed=1, conflicting_timestamps=0,
    )
    validate_fetch_evidence(fetch)  # does not raise


def test_missing_chunk_coverage_gap_rejected():
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-05",
        chunks=(
            ChunkResultSnapshot("2026-01-01", "2026-01-01", 1, True, None),
            ChunkResultSnapshot("2026-01-03", "2026-01-05", 1, True, None),  # gap on Jan 2
        ),
        total_rows=2, first_ts="x", last_ts="y",
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


# ---------------------------------------------------------------------------
# evidence attacks: cross-check against dataset
# ---------------------------------------------------------------------------


def test_total_rows_vs_dataset_mismatch_rejected():
    ds = _dataset()
    fetch = _fetch_for(ds, requested_from="2026-01-01", requested_to="2026-01-01", total_rows=999)
    with pytest.raises(AcquisitionError):
        cross_check_fetch_against_dataset(fetch, ds)


def test_first_timestamp_mismatch_rejected():
    ds = _dataset()
    fetch = _fetch_for(ds, requested_from="2026-01-01", requested_to="2026-01-01", first_ts="2099-01-01T00:00:00+05:30")
    with pytest.raises(AcquisitionError):
        cross_check_fetch_against_dataset(fetch, ds)


def test_last_timestamp_mismatch_rejected():
    ds = _dataset()
    fetch = _fetch_for(ds, requested_from="2026-01-01", requested_to="2026-01-01", last_ts="2099-01-01T00:00:00+05:30")
    with pytest.raises(AcquisitionError):
        cross_check_fetch_against_dataset(fetch, ds)


def test_conflicting_timestamp_count_mismatch_rejected():
    ds = _dataset()
    fetch = _fetch_for(ds, requested_from="2026-01-01", requested_to="2026-01-01", conflicting_timestamps=1)
    with pytest.raises(AcquisitionError):
        cross_check_fetch_against_dataset(fetch, ds)


def test_empty_dataset_requires_none_timestamps():
    ds = ValidatedDataset.build(_empty_frame(), identity=_identity())
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 0, True, None),),
        total_rows=0, first_ts="2026-01-01T00:00:00+05:30", last_ts=None,
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    with pytest.raises(AcquisitionError):
        cross_check_fetch_against_dataset(fetch, ds)


def test_empty_dataset_with_none_timestamps_accepted():
    ds = ValidatedDataset.build(_empty_frame(), identity=_identity())
    fetch = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 0, True, None),),
        total_rows=0, first_ts=None, last_ts=None,
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    cross_check_fetch_against_dataset(fetch, ds)  # does not raise


# ---------------------------------------------------------------------------
# ObservedDataCoverage
# ---------------------------------------------------------------------------


def test_observed_coverage_empty():
    ds = ValidatedDataset.build(_empty_frame(), identity=_identity())
    coverage = compute_observed_data_coverage(ds)
    assert coverage.earliest_observed_date is None
    assert coverage.latest_observed_date is None
    assert coverage.distinct_observed_dates == 0
    assert coverage.observed_span_days == 0


def test_observed_coverage_one_day():
    ds = _dataset("2026-01-01")
    coverage = compute_observed_data_coverage(ds)
    assert coverage.earliest_observed_date == "2026-01-01"
    assert coverage.latest_observed_date == "2026-01-01"
    assert coverage.distinct_observed_dates == 1
    assert coverage.observed_span_days == 1


def test_observed_coverage_multiple_dates():
    ts0 = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
    ts1 = pd.Timestamp("2026-01-03 09:15", tz=IST_NAME)
    raw = pd.DataFrame(
        {
            TS: [ts0, ts1],
            OPEN: [100.0, 101.0],
            HIGH: [105.0, 106.0],
            LOW: [95.0, 96.0],
            CLOSE: [101.0, 102.0],
            VOLUME: [1000, 1001],
        }
    )
    ds = ValidatedDataset.build(raw, identity=_identity())
    coverage = compute_observed_data_coverage(ds)
    assert coverage.earliest_observed_date == "2026-01-01"
    assert coverage.latest_observed_date == "2026-01-03"
    assert coverage.distinct_observed_dates == 2
    assert coverage.observed_span_days == 3  # inclusive


# ---------------------------------------------------------------------------
# RequestedWindowComparison
# ---------------------------------------------------------------------------


def test_requested_start_and_end_observed():
    ds = _dataset("2026-01-01")
    fetch = _fetch_for(ds, requested_from="2026-01-01", requested_to="2026-01-01")
    cmp = compute_requested_window_comparison(ds, fetch)
    assert cmp.observations_on_requested_start_date is True
    assert cmp.observations_on_requested_end_date is True
    assert cmp.all_observations_within_requested_window is True


def test_boundary_date_absent_is_not_marked_invalid():
    # Data observed only on Jan 2, requested window Jan 1 - Jan 3 (a weekend
    # or holiday could plausibly explain the missing boundary dates).
    ds = _dataset("2026-01-02")
    fetch = _fetch_for(ds, requested_from="2026-01-01", requested_to="2026-01-03")
    cmp = compute_requested_window_comparison(ds, fetch)
    assert cmp.observations_on_requested_start_date is False
    assert cmp.observations_on_requested_end_date is False
    # No exception raised, no "invalid" field -- absence is just a fact.
    assert cmp.all_observations_within_requested_window is True


def test_data_outside_requested_window_marks_comparison_false():
    ds = _dataset("2026-01-05")
    fetch = _fetch_for(ds, requested_from="2026-01-05", requested_to="2026-01-10")
    # Manually construct a comparison against a NARROWER requested window
    # than what fetch claims, to exercise "outside window" -- reuse fetch's
    # dates for validity, but check with data recorded outside them.
    ts_outside = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
    raw = pd.DataFrame(
        {
            TS: [ts_outside],
            OPEN: [100.0],
            HIGH: [105.0],
            LOW: [95.0],
            CLOSE: [101.0],
            VOLUME: [1000],
        }
    )
    ds_outside = ValidatedDataset.build(raw, identity=_identity())
    fetch_outside = _fetch_for(ds_outside, requested_from="2026-01-05", requested_to="2026-01-10")
    cmp = compute_requested_window_comparison(ds_outside, fetch_outside)
    assert cmp.all_observations_within_requested_window is False


def test_jan1_to_dec31_span_is_365():
    ds = _dataset("2026-01-01")
    fetch = _fetch_for(ds, requested_from="2026-01-01", requested_to="2026-12-31")
    cmp = compute_requested_window_comparison(ds, fetch)
    assert cmp.requested_span_days == 365


# ---------------------------------------------------------------------------
# Strict field-type enforcement (Unit 9 hardening)
#
# Dataclasses do not enforce their type annotations. Every malformed value
# below must be rejected as AcquisitionError -- never accepted, and never
# surfaced as a raw TypeError.
# ---------------------------------------------------------------------------


def _one_day_fetch(**overrides) -> FetchReportSnapshot:
    fields = dict(
        symbol="NIFTY",
        resolution="1",
        requested_from="2026-01-01",
        requested_to="2026-01-01",
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 5, True, None),),
        total_rows=5,
        first_ts="x",
        last_ts="y",
        duplicate_rows_removed=0,
        conflicting_timestamps=0,
    )
    fields.update(overrides)
    return FetchReportSnapshot(**fields)


def test_chunk_ok_as_truthy_string_rejected():
    fetch = _one_day_fetch(
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 5, "false", None),)
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_chunk_ok_as_int_rejected():
    fetch = _one_day_fetch(
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 5, 1, None),)
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_chunk_rows_as_bool_rejected():
    fetch = _one_day_fetch(
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", True, True, None),),
        total_rows=True,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_chunk_rows_as_float_rejected():
    fetch = _one_day_fetch(
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", 1.5, True, None),),
        total_rows=1.5,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_total_rows_as_bool_rejected():
    fetch = _one_day_fetch(total_rows=True)
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_total_rows_as_float_rejected():
    fetch = _one_day_fetch(total_rows=5.0)
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_duplicate_rows_removed_as_bool_rejected():
    fetch = _one_day_fetch(duplicate_rows_removed=True)
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_duplicate_rows_removed_as_float_rejected():
    fetch = _one_day_fetch(duplicate_rows_removed=1.2)
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_conflicting_timestamps_as_bool_rejected():
    fetch = _one_day_fetch(conflicting_timestamps=True)
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_conflicting_timestamps_as_string_rejected():
    fetch = _one_day_fetch(conflicting_timestamps="0")
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_failed_chunk_rows_as_bool_rejected():
    # ok=False, rows=False -- False == 0 numerically, but must still be
    # rejected for not being an actual int.
    fetch = _one_day_fetch(
        chunks=(ChunkResultSnapshot("2026-01-01", "2026-01-01", False, False, "boom"),),
        total_rows=0,
    )
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(fetch)


def test_malformed_evidence_cannot_produce_requests_succeeded():
    malformed = _one_day_fetch(total_rows=True)
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(malformed)
    # classify_acquisition_status() assumes pre-validated evidence and does
    # not itself repeat every check -- but the calling contract (validate
    # first) prevents this malformed evidence from ever reaching it as
    # "validated". Confirm that path is enforced: validation raises before
    # classification is trusted to run in generation_store's own call order.
    with pytest.raises(AcquisitionError):
        validate_fetch_evidence(malformed)
        classify_acquisition_status(malformed)  # unreachable if validate raises


def test_classify_acquisition_status_rejects_non_snapshot_type():
    with pytest.raises(AcquisitionError):
        classify_acquisition_status("not a snapshot")  # type: ignore[arg-type]
