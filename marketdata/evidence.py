"""Immutable evidence snapshots.

Frozen architecture, section 16, item 3: baseline ``ValidationReport``,
``FetchReport`` and ``DatasetManifest`` are mutable ``@dataclass``es -- that
must change, because a future ``ValidatedDataset`` (Unit 5, not implemented
here) must never retain evidence that can be edited after
validation/acquisition/persistence.

This module does exactly one thing: given an existing mutable report object,
produce a frozen, defensively-copied snapshot of the same values. It does
not change how ``validate()``, ``fetch_candles_with_report()`` or
``store.write()`` run, and it does not touch ``pandas`` DataFrames -- those
remain outside scope (frozen architecture section 16 states plainly that
pandas cannot be made immutable; this module does not attempt to).

Nothing here invents a fact that was not already present on the source
report. If a future consumer needs a fact these reports do not carry, that
is a scope decision for a later, explicitly authorised unit -- not something
to paper over here by fabricating a field.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from brokers.fyers.historical import ChunkResult, FetchReport
from marketdata.store import DatasetManifest
from marketdata.validator import Severity, ValidationIssue, ValidationReport


def _freeze(value: Any) -> Any:
    """Recursively convert ``dict``/``list`` into immutable equivalents.

    Builds entirely new containers (dict comprehension, generator-to-tuple)
    rather than wrapping the originals in place, so mutating the SOURCE
    object after this call never reaches the frozen copy. ``MappingProxyType``
    alone would not be enough: it blocks writes through the *same* dict
    object, but the manifest fields this is applied to (``requested_range``,
    ``cleaning``, ``software``, ``failed_chunks``) are plain caller-supplied
    dicts/lists, and only a fresh copy at every level severs that reference.
    """
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationReportSnapshot:
    """Immutable snapshot of a ``marketdata.validator.ValidationReport``.

    ``issues`` is a ``tuple`` of ``ValidationIssue`` -- already a frozen
    dataclass with a ``tuple`` ``samples`` field in the source module, so no
    further per-issue copying is needed; only the mutable *list* holding
    them needs to be replaced with an immutable container.

    ``errors``/``warnings``/``is_usable`` are properties, not stored fields:
    they are recomputed from ``issues`` on every access rather than cached at
    snapshot time, matching the frozen architecture's general principle
    (section 4) that data-derived facts are recomputed, not stored as a
    second, potentially-diverging assertion.
    """

    symbol: str
    resolution: str
    row_count: int
    first_ts: str | None
    last_ts: str | None
    timezone: str
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    @property
    def is_usable(self) -> bool:
        return not self.errors


def snapshot_validation_report(report: ValidationReport) -> ValidationReportSnapshot:
    """Defensively copy a ``ValidationReport`` into an immutable snapshot.

    Mutating ``report`` (e.g. ``report.add(...)``) after this call never
    affects the returned snapshot: ``tuple(report.issues)`` materialises a
    new tuple from the list's current contents, decoupled from the list
    object itself.
    """
    return ValidationReportSnapshot(
        symbol=report.symbol,
        resolution=report.resolution,
        row_count=report.row_count,
        first_ts=report.first_ts,
        last_ts=report.last_ts,
        timezone=report.timezone,
        issues=tuple(report.issues),
    )


# ---------------------------------------------------------------------------
# FetchReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkResultSnapshot:
    """Immutable snapshot of one ``brokers.fyers.historical.ChunkResult``."""

    range_from: str
    range_to: str
    rows: int
    ok: bool
    error: str | None


def snapshot_chunk_result(chunk: ChunkResult) -> ChunkResultSnapshot:
    return ChunkResultSnapshot(
        range_from=chunk.range_from,
        range_to=chunk.range_to,
        rows=chunk.rows,
        ok=chunk.ok,
        error=chunk.error,
    )


@dataclass(frozen=True, slots=True)
class FetchReportSnapshot:
    """Immutable snapshot of a ``brokers.fyers.historical.FetchReport``."""

    symbol: str
    resolution: str
    requested_from: str
    requested_to: str
    chunks: tuple[ChunkResultSnapshot, ...]
    total_rows: int
    first_ts: str | None
    last_ts: str | None
    duplicate_rows_removed: int
    conflicting_timestamps: int

    @property
    def failed_chunks(self) -> tuple[ChunkResultSnapshot, ...]:
        return tuple(c for c in self.chunks if not c.ok)

    @property
    def empty_chunks(self) -> tuple[ChunkResultSnapshot, ...]:
        return tuple(c for c in self.chunks if c.ok and c.rows == 0)


def snapshot_fetch_report(report: FetchReport) -> FetchReportSnapshot:
    """Defensively copy a ``FetchReport`` into an immutable snapshot.

    Mutating ``report.chunks`` (e.g. ``report.chunks.append(...)``) after
    this call never affects the returned snapshot, for the same reason as
    ``snapshot_validation_report``: a new tuple is built from the list's
    current contents at call time.
    """
    return FetchReportSnapshot(
        symbol=report.symbol,
        resolution=report.resolution,
        requested_from=report.requested_from,
        requested_to=report.requested_to,
        chunks=tuple(snapshot_chunk_result(c) for c in report.chunks),
        total_rows=report.total_rows,
        first_ts=report.first_ts,
        last_ts=report.last_ts,
        duplicate_rows_removed=report.duplicate_rows_removed,
        conflicting_timestamps=report.conflicting_timestamps,
    )


# ---------------------------------------------------------------------------
# DatasetManifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManifestSnapshot:
    """Immutable snapshot of a ``marketdata.store.DatasetManifest``.

    ``failed_chunks``, ``requested_range``, ``cleaning`` and ``software`` are
    plain caller-supplied ``dict``/``list`` structures on the source
    manifest (JSON-serialisable provenance detail, not domain types) --
    frozen recursively via ``_freeze`` into ``tuple``/``MappingProxyType``
    so no nested mutable container survives into the snapshot.
    """

    symbol: str
    resolution: str
    source: str
    row_count: int
    first_ts: str | None
    last_ts: str | None
    timezone: str
    content_sha256: str
    fetched_at_utc: str
    validation_status: str
    validation_error_count: int
    validation_warning_count: int
    validation_error_codes: tuple[str, ...]
    fetch_status: str
    failed_chunks: tuple
    requested_range: MappingProxyType
    cleaning: MappingProxyType
    software: MappingProxyType
    forced: bool
    notes: str

    @property
    def is_authoritative(self) -> bool:
        return (
            self.validation_status == "valid"
            and self.fetch_status == "complete"
            and not self.forced
        )


def snapshot_manifest(manifest: DatasetManifest) -> ManifestSnapshot:
    """Defensively copy a ``DatasetManifest`` into an immutable snapshot.

    Every ``dict``/``list``-valued field is recursively frozen via
    ``_freeze``, which builds new containers rather than wrapping the
    originals -- mutating ``manifest.requested_range`` (or any nested value
    within it) after this call never affects the returned snapshot.
    """
    return ManifestSnapshot(
        symbol=manifest.symbol,
        resolution=manifest.resolution,
        source=manifest.source,
        row_count=manifest.row_count,
        first_ts=manifest.first_ts,
        last_ts=manifest.last_ts,
        timezone=manifest.timezone,
        content_sha256=manifest.content_sha256,
        fetched_at_utc=manifest.fetched_at_utc,
        validation_status=manifest.validation_status,
        validation_error_count=manifest.validation_error_count,
        validation_warning_count=manifest.validation_warning_count,
        validation_error_codes=tuple(manifest.validation_error_codes),
        fetch_status=manifest.fetch_status,
        failed_chunks=_freeze(manifest.failed_chunks),
        requested_range=_freeze(manifest.requested_range),
        cleaning=_freeze(manifest.cleaning),
        software=_freeze(manifest.software),
        forced=manifest.forced,
        notes=manifest.notes,
    )
