"""Acquisition evidence classification and coverage facts, per the frozen
architecture (docs/architecture/phase1-trust-hardening.md, section 11).

Three independent things, kept separate exactly as section 11 requires:

- ``AcquisitionRequestStatus`` -- **only** whether broker requests returned
  without error. ``REQUESTS_SUCCEEDED`` says nothing about requested-range
  coverage, interior completeness, trading days, or bar density. No
  consumer may infer completeness from this enum.
- ``ObservedDataCoverage`` -- derived from the canonical candles ALONE.
  Never looks at the requested range.
- ``RequestedWindowComparison`` -- derived from candles PLUS integrity-
  bound acquisition evidence (``FetchReportSnapshot.requested_from/to``).
  Only computable when fetch evidence exists.

Before any of these are trusted, ``validate_fetch_evidence()`` checks that a
supplied ``FetchReportSnapshot`` is internally coherent (well-formed dates,
consistent chunk/row bookkeeping), and
``cross_check_fetch_against_dataset()`` checks that it actually describes
the ``ValidatedDataset`` it is bound to (matching row count, first/last
timestamp, conflicting-timestamp count) -- the same "evidence must not be
supplied separately from what it describes" principle every earlier unit in
this branch has enforced for its own layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

import pandas as pd

from marketdata.dataset import ValidatedDataset
from marketdata.evidence import ChunkResultSnapshot, FetchReportSnapshot
from marketdata.schemas import TS


class AcquisitionError(ValueError):
    """Raised when fetch evidence is internally incoherent, or does not
    actually describe the dataset it is bound to."""


class AcquisitionRequestStatus(str, Enum):
    """Section 11.1. **Only** whether broker requests returned without
    error -- ``REQUESTS_SUCCEEDED`` certifies nothing about coverage,
    continuity, sessions, or density.
    """

    REQUESTS_FAILED = "REQUESTS_FAILED"
    REQUESTS_EMPTY = "REQUESTS_EMPTY"
    REQUESTS_PARTIAL = "REQUESTS_PARTIAL"
    REQUESTS_SUCCEEDED = "REQUESTS_SUCCEEDED"
    REQUESTS_UNKNOWN = "REQUESTS_UNKNOWN"


# --- date parsing -------------------------------------------------------------


def _parse_date(value: object, field_name: str) -> date:
    if not isinstance(value, str):
        raise AcquisitionError(
            f"{field_name} must be a YYYY-MM-DD string, got {value!r} "
            f"({type(value).__name__})"
        )
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise AcquisitionError(
            f"{field_name} must be a valid YYYY-MM-DD date, got {value!r}"
        ) from exc


# --- internal fetch-evidence validation --------------------------------------


def _validate_chunk(chunk: ChunkResultSnapshot, index: int) -> tuple[date, date]:
    if not isinstance(chunk, ChunkResultSnapshot):
        raise AcquisitionError(
            f"chunks[{index}] must be an actual ChunkResultSnapshot, got "
            f"{type(chunk).__name__}"
        )
    range_from = _parse_date(chunk.range_from, f"chunks[{index}].range_from")
    range_to = _parse_date(chunk.range_to, f"chunks[{index}].range_to")
    if range_from > range_to:
        raise AcquisitionError(
            f"chunks[{index}].range_from ({chunk.range_from}) is after "
            f"range_to ({chunk.range_to})."
        )
    if chunk.ok:
        if chunk.rows < 0:
            raise AcquisitionError(
                f"chunks[{index}] is ok=True but rows is negative ({chunk.rows})."
            )
        if chunk.error is not None:
            raise AcquisitionError(
                f"chunks[{index}] is ok=True but carries an error message "
                f"({chunk.error!r}); a successful chunk must have error=None."
            )
    else:
        if chunk.rows != 0:
            raise AcquisitionError(
                f"chunks[{index}] failed (ok=False) but rows is {chunk.rows}, "
                "expected 0."
            )
        if not (isinstance(chunk.error, str) and chunk.error.strip()):
            raise AcquisitionError(
                f"chunks[{index}] failed (ok=False) but has no non-empty "
                f"error message, got {chunk.error!r}."
            )
    return range_from, range_to


def validate_fetch_evidence(fetch: FetchReportSnapshot) -> None:
    """Raise ``AcquisitionError`` unless ``fetch`` is internally coherent.

    Checks: ``requested_from``/``requested_to`` are valid, non-reversed
    dates; at least one chunk exists; every chunk's own range is valid
    (``range_from <= range_to``) and each chunk is internally consistent
    (a failed chunk has zero rows and a non-empty error; a successful chunk
    has no error and non-negative rows); the chunk sequence, sorted by
    ``range_from``, covers ``requested_from..requested_to`` contiguously
    with no gap and no overlap; ``total_rows``/``duplicate_rows_removed``/
    ``conflicting_timestamps`` are all non-negative; and the row arithmetic
    ``sum(rows for successful chunks) - duplicate_rows_removed ==
    total_rows`` holds exactly (NOT ``sum(chunk.rows) == total_rows`` --
    cross-chunk exact duplicates are legitimately removed and recorded
    separately, per the existing FetchReport contract).
    """
    if not isinstance(fetch, FetchReportSnapshot):
        raise AcquisitionError(
            f"fetch must be an actual FetchReportSnapshot, got {type(fetch).__name__}"
        )

    requested_from = _parse_date(fetch.requested_from, "requested_from")
    requested_to = _parse_date(fetch.requested_to, "requested_to")
    if requested_from > requested_to:
        raise AcquisitionError(
            f"requested_from ({fetch.requested_from}) is after requested_to "
            f"({fetch.requested_to})."
        )

    if len(fetch.chunks) == 0:
        raise AcquisitionError(
            "fetch evidence has no chunks; at least one chunk is required "
            "when fetch evidence is supplied at all."
        )

    ranges = [_validate_chunk(c, i) for i, c in enumerate(fetch.chunks)]

    # Chunk sequence must cover the requested window contiguously: sorted by
    # range_from, the first chunk starts at requested_from, each next chunk
    # starts exactly one day after the previous one ends, and the last
    # chunk ends at requested_to. This matches how this project's own
    # chunker (brokers/fyers/historical.py) partitions a request into
    # non-overlapping, contiguous windows.
    ordered = sorted(ranges, key=lambda pair: pair[0])
    if ordered[0][0] != requested_from:
        raise AcquisitionError(
            f"Chunk coverage does not start at requested_from: first chunk "
            f"starts {ordered[0][0].isoformat()}, requested_from is "
            f"{fetch.requested_from}."
        )
    if ordered[-1][1] != requested_to:
        raise AcquisitionError(
            f"Chunk coverage does not end at requested_to: last chunk ends "
            f"{ordered[-1][1].isoformat()}, requested_to is {fetch.requested_to}."
        )
    for i in range(1, len(ordered)):
        previous_end = ordered[i - 1][1]
        this_start = ordered[i][0]
        expected_start = previous_end + pd.Timedelta(days=1).to_pytimedelta()
        if this_start != expected_start:
            raise AcquisitionError(
                "Chunk coverage has a gap or overlap: chunk ending "
                f"{previous_end.isoformat()} is followed by a chunk starting "
                f"{this_start.isoformat()}, expected {expected_start.isoformat()}."
            )

    if fetch.total_rows < 0:
        raise AcquisitionError(f"total_rows must be >= 0, got {fetch.total_rows}.")
    if fetch.duplicate_rows_removed < 0:
        raise AcquisitionError(
            f"duplicate_rows_removed must be >= 0, got {fetch.duplicate_rows_removed}."
        )
    if fetch.conflicting_timestamps < 0:
        raise AcquisitionError(
            f"conflicting_timestamps must be >= 0, got {fetch.conflicting_timestamps}."
        )

    successful_rows = sum(c.rows for c in fetch.chunks if c.ok)
    expected_total = successful_rows - fetch.duplicate_rows_removed
    if expected_total != fetch.total_rows:
        raise AcquisitionError(
            "Row arithmetic mismatch: sum(successful chunk rows) "
            f"({successful_rows}) - duplicate_rows_removed "
            f"({fetch.duplicate_rows_removed}) = {expected_total}, but "
            f"total_rows is {fetch.total_rows}."
        )


def cross_check_fetch_against_dataset(
    fetch: FetchReportSnapshot, dataset: ValidatedDataset
) -> None:
    """Raise ``AcquisitionError`` unless ``fetch`` actually describes
    ``dataset``: ``fetch.total_rows == len(dataset.frame)``; for a
    non-empty frame, ``fetch.first_ts``/``fetch.last_ts`` match the actual
    first/last canonical timestamp exactly (ISO-8601); for an empty frame,
    both are ``None``; and ``fetch.conflicting_timestamps`` matches a fresh
    recount of duplicate-timestamp groups on the canonical frame.

    Deliberately does NOT attempt to reconstruct ``duplicate_rows_removed``
    from the final candles -- that fact describes an event during
    acquisition (cross-chunk merging) that cannot be recovered from the
    final, already-deduplicated data.
    """
    frame = dataset.frame
    if fetch.total_rows != len(frame):
        raise AcquisitionError(
            f"fetch.total_rows ({fetch.total_rows}) does not match "
            f"len(dataset.frame) ({len(frame)})."
        )

    if len(frame) == 0:
        if fetch.first_ts is not None or fetch.last_ts is not None:
            raise AcquisitionError(
                "fetch.first_ts/last_ts must both be None for an empty "
                f"dataset, got first_ts={fetch.first_ts!r} last_ts={fetch.last_ts!r}."
            )
    else:
        actual_first = frame[TS].iloc[0].isoformat()
        actual_last = frame[TS].iloc[-1].isoformat()
        if fetch.first_ts != actual_first:
            raise AcquisitionError(
                f"fetch.first_ts ({fetch.first_ts!r}) does not match the "
                f"actual first canonical timestamp ({actual_first!r})."
            )
        if fetch.last_ts != actual_last:
            raise AcquisitionError(
                f"fetch.last_ts ({fetch.last_ts!r}) does not match the "
                f"actual last canonical timestamp ({actual_last!r})."
            )

    if len(frame) == 0:
        conflicting_count = 0
    else:
        dup_mask = frame[TS].duplicated(keep=False)
        conflicting_count = int(frame.loc[dup_mask, TS].nunique())
    if conflicting_count != fetch.conflicting_timestamps:
        raise AcquisitionError(
            f"fetch.conflicting_timestamps ({fetch.conflicting_timestamps}) "
            "does not match a fresh recount from the canonical frame "
            f"({conflicting_count})."
        )


# --- status classification -----------------------------------------------------


def classify_acquisition_status(
    fetch: FetchReportSnapshot | None,
) -> AcquisitionRequestStatus:
    """Section 11.1's five states. Does NOT validate ``fetch`` itself --
    call :func:`validate_fetch_evidence` first; this function assumes
    already-coherent evidence.
    """
    if fetch is None:
        return AcquisitionRequestStatus.REQUESTS_UNKNOWN
    all_ok = all(c.ok for c in fetch.chunks)
    any_ok = any(c.ok for c in fetch.chunks)
    if not any_ok:
        return AcquisitionRequestStatus.REQUESTS_FAILED
    if not all_ok:
        return AcquisitionRequestStatus.REQUESTS_PARTIAL
    if fetch.total_rows == 0:
        return AcquisitionRequestStatus.REQUESTS_EMPTY
    return AcquisitionRequestStatus.REQUESTS_SUCCEEDED


# --- ObservedDataCoverage: candles only ---------------------------------------


@dataclass(frozen=True, slots=True)
class ObservedDataCoverage:
    """Section 11.2. Derived from the canonical candles ALONE -- never the
    requested range.
    """

    earliest_observed_date: str | None
    latest_observed_date: str | None
    distinct_observed_dates: int
    observed_span_days: int


def compute_observed_data_coverage(dataset: ValidatedDataset) -> ObservedDataCoverage:
    frame = dataset.frame
    if len(frame) == 0:
        return ObservedDataCoverage(
            earliest_observed_date=None,
            latest_observed_date=None,
            distinct_observed_dates=0,
            observed_span_days=0,
        )
    observed_dates = frame[TS].dt.date
    earliest = observed_dates.min()
    latest = observed_dates.max()
    distinct = int(observed_dates.nunique())
    # Inclusive calendar days, matching RequestedWindowComparison's own
    # requested_span_days convention (section 6 example: Jan 1 - Dec 31 =>
    # 365) for consistency within this module.
    span = (latest - earliest).days + 1
    return ObservedDataCoverage(
        earliest_observed_date=earliest.isoformat(),
        latest_observed_date=latest.isoformat(),
        distinct_observed_dates=distinct,
        observed_span_days=span,
    )


# --- RequestedWindowComparison: candles + integrity-bound provenance ---------


@dataclass(frozen=True, slots=True)
class RequestedWindowComparison:
    """Section 11.3. Only computable when fetch evidence exists --
    ``requested_from``/``requested_to`` are integrity-bound acquisition
    facts, not derivable from candles alone.

    Absence of observations on a requested boundary date is NOT graded as
    deficient here (section 11.3): that date may simply not have been a
    trading day. No calendar is invented to decide that either way.
    """

    requested_from: str
    requested_to: str
    requested_span_days: int
    observations_on_requested_start_date: bool
    observations_on_requested_end_date: bool
    all_observations_within_requested_window: bool
    observed_distinct_dates_ratio: float


def compute_requested_window_comparison(
    dataset: ValidatedDataset, fetch: FetchReportSnapshot
) -> RequestedWindowComparison:
    requested_from = _parse_date(fetch.requested_from, "requested_from")
    requested_to = _parse_date(fetch.requested_to, "requested_to")
    requested_span_days = (requested_to - requested_from).days + 1

    frame = dataset.frame
    observed_dates = set(frame[TS].dt.date) if len(frame) else set()

    on_start = requested_from in observed_dates
    on_end = requested_to in observed_dates
    all_within = all(requested_from <= d <= requested_to for d in observed_dates)
    distinct_count = len(observed_dates)
    ratio = distinct_count / requested_span_days if requested_span_days > 0 else 0.0

    return RequestedWindowComparison(
        requested_from=fetch.requested_from,
        requested_to=fetch.requested_to,
        requested_span_days=requested_span_days,
        observations_on_requested_start_date=on_start,
        observations_on_requested_end_date=on_end,
        all_observations_within_requested_window=all_within,
        observed_distinct_dates_ratio=ratio,
    )
