"""Immutable evidence snapshot tests.

Frozen architecture section 16, item 3: ValidationReport, FetchReport and
DatasetManifest are mutable baseline dataclasses that must become immutable
snapshots before any future ValidatedDataset can rely on them.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pytest

from brokers.fyers.historical import ChunkResult, FetchReport
from marketdata.evidence import (
    ChunkResultSnapshot,
    FetchReportSnapshot,
    ManifestSnapshot,
    ValidationReportSnapshot,
    snapshot_chunk_result,
    snapshot_fetch_report,
    snapshot_manifest,
    snapshot_validation_report,
)
from marketdata.store import DatasetManifest
from marketdata.validator import Severity, ValidationIssue, ValidationReport

# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


def _validation_report() -> ValidationReport:
    report = ValidationReport(
        symbol="NIFTY",
        resolution="1",
        row_count=100,
        first_ts="2026-01-01T09:15:00+05:30",
        last_ts="2026-01-01T10:55:00+05:30",
        timezone="Asia/Kolkata",
    )
    report.add(
        ValidationIssue(
            "WITHIN_DAY_GAPS", Severity.WARNING, "gaps found", count=2, samples=("a", "b")
        )
    )
    return report


def test_validation_snapshot_matches_source_at_creation():
    report = _validation_report()
    snap = snapshot_validation_report(report)
    assert snap.symbol == report.symbol
    assert snap.resolution == report.resolution
    assert snap.row_count == report.row_count
    assert snap.first_ts == report.first_ts
    assert snap.last_ts == report.last_ts
    assert snap.timezone == report.timezone
    assert tuple(snap.issues) == tuple(report.issues)


def test_validation_snapshot_unaffected_by_later_report_mutation():
    report = _validation_report()
    snap = snapshot_validation_report(report)
    before = snap.issues

    report.add(
        ValidationIssue("OHLC_HIGH_TOO_LOW", Severity.ERROR, "bad bar", count=1)
    )

    assert snap.issues == before
    assert len(snap.issues) == 1
    assert len(report.issues) == 2


def test_validation_snapshot_issues_is_tuple_and_unappendable():
    snap = snapshot_validation_report(_validation_report())
    assert isinstance(snap.issues, tuple)
    with pytest.raises(AttributeError):
        snap.issues.append(
            ValidationIssue("X", Severity.INFO, "x", count=1)
        )


def test_validation_snapshot_fields_are_frozen():
    snap = snapshot_validation_report(_validation_report())
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.symbol = "SBIN"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.issues = ()


def test_validation_snapshot_creation_is_deterministic():
    report = _validation_report()
    snap1 = snapshot_validation_report(report)
    snap2 = snapshot_validation_report(report)
    assert snap1 == snap2


def test_validation_snapshot_derived_properties():
    report = _validation_report()
    report.add(ValidationIssue("BAD", Severity.ERROR, "bad", count=1))
    snap = snapshot_validation_report(report)
    assert len(snap.errors) == 1
    assert len(snap.warnings) == 1
    assert snap.is_usable is False


def test_validation_snapshot_usable_when_no_errors():
    snap = snapshot_validation_report(_validation_report())
    assert snap.is_usable is True


# ---------------------------------------------------------------------------
# FetchReport
# ---------------------------------------------------------------------------


def _fetch_report() -> FetchReport:
    report = FetchReport(
        symbol="NIFTY",
        resolution="1",
        requested_from="2026-01-01",
        requested_to="2026-01-05",
    )
    report.chunks.append(ChunkResult("2026-01-01", "2026-01-03", 100, True))
    report.total_rows = 100
    report.first_ts = "2026-01-01T09:15:00+05:30"
    report.last_ts = "2026-01-03T15:29:00+05:30"
    return report


def test_chunk_snapshot_matches_source():
    chunk = ChunkResult("2026-01-01", "2026-01-03", 100, True)
    snap = snapshot_chunk_result(chunk)
    assert snap.range_from == chunk.range_from
    assert snap.range_to == chunk.range_to
    assert snap.rows == chunk.rows
    assert snap.ok == chunk.ok
    assert snap.error == chunk.error


def test_fetch_snapshot_matches_source_at_creation():
    report = _fetch_report()
    snap = snapshot_fetch_report(report)
    assert snap.symbol == report.symbol
    assert snap.resolution == report.resolution
    assert snap.requested_from == report.requested_from
    assert snap.requested_to == report.requested_to
    assert snap.total_rows == report.total_rows
    assert snap.first_ts == report.first_ts
    assert snap.last_ts == report.last_ts
    assert snap.duplicate_rows_removed == report.duplicate_rows_removed
    assert snap.conflicting_timestamps == report.conflicting_timestamps
    assert len(snap.chunks) == len(report.chunks)
    assert snap.chunks[0].range_from == report.chunks[0].range_from


def test_fetch_snapshot_unaffected_by_later_chunk_list_mutation():
    report = _fetch_report()
    snap = snapshot_fetch_report(report)
    before = snap.chunks

    report.chunks.append(
        ChunkResult("2026-01-04", "2026-01-05", 0, False, "timeout")
    )

    assert snap.chunks == before
    assert len(snap.chunks) == 1
    assert len(report.chunks) == 2


def test_fetch_snapshot_chunks_is_tuple_and_unappendable():
    snap = snapshot_fetch_report(_fetch_report())
    assert isinstance(snap.chunks, tuple)
    assert isinstance(snap.chunks[0], ChunkResultSnapshot)
    with pytest.raises(AttributeError):
        snap.chunks.append(ChunkResultSnapshot("x", "y", 0, True, None))


def test_fetch_snapshot_fields_are_frozen():
    snap = snapshot_fetch_report(_fetch_report())
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.symbol = "SBIN"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.chunks = ()


def test_chunk_snapshot_fields_are_frozen():
    snap = snapshot_chunk_result(ChunkResult("a", "b", 1, True))
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.ok = False


def test_fetch_snapshot_creation_is_deterministic():
    report = _fetch_report()
    snap1 = snapshot_fetch_report(report)
    snap2 = snapshot_fetch_report(report)
    assert snap1 == snap2


def test_fetch_snapshot_derived_failed_and_empty_chunks():
    report = _fetch_report()
    report.chunks.append(ChunkResult("2026-01-04", "2026-01-04", 0, False, "boom"))
    report.chunks.append(ChunkResult("2026-01-05", "2026-01-05", 0, True))
    snap = snapshot_fetch_report(report)
    assert len(snap.failed_chunks) == 1
    assert snap.failed_chunks[0].error == "boom"
    assert len(snap.empty_chunks) == 1


# ---------------------------------------------------------------------------
# DatasetManifest
# ---------------------------------------------------------------------------


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        symbol="NIFTY",
        resolution="1",
        source="fyers:history",
        row_count=100,
        first_ts="2026-01-01T09:15:00+05:30",
        last_ts="2026-01-01T10:55:00+05:30",
        timezone="Asia/Kolkata",
        content_sha256="a" * 64,
        fetched_at_utc="2026-01-01T00:00:00+00:00",
        validation_status="valid",
        validation_error_count=0,
        validation_warning_count=1,
        validation_error_codes=[],
        fetch_status="complete",
        failed_chunks=[{"from": "2026-01-01", "to": "2026-01-02", "error": "boom"}],
        requested_range={"from": "2026-01-01", "to": "2026-01-05"},
        cleaning={"dedup": True},
        software={"python": "3.12.14"},
        forced=False,
        notes="",
    )


def test_manifest_snapshot_matches_source_at_creation():
    manifest = _manifest()
    snap = snapshot_manifest(manifest)
    assert snap.symbol == manifest.symbol
    assert snap.resolution == manifest.resolution
    assert snap.source == manifest.source
    assert snap.row_count == manifest.row_count
    assert snap.content_sha256 == manifest.content_sha256
    assert snap.fetched_at_utc == manifest.fetched_at_utc
    assert snap.validation_status == manifest.validation_status
    assert snap.validation_error_count == manifest.validation_error_count
    assert snap.validation_warning_count == manifest.validation_warning_count
    assert tuple(snap.validation_error_codes) == tuple(manifest.validation_error_codes)
    assert snap.fetch_status == manifest.fetch_status
    assert snap.forced == manifest.forced
    assert snap.notes == manifest.notes
    assert dict(snap.requested_range) == manifest.requested_range
    assert dict(snap.cleaning) == manifest.cleaning
    assert dict(snap.software) == manifest.software
    assert [dict(c) for c in snap.failed_chunks] == manifest.failed_chunks


def test_manifest_snapshot_unaffected_by_later_list_mutation():
    manifest = _manifest()
    snap = snapshot_manifest(manifest)
    before = snap.failed_chunks

    manifest.failed_chunks.append({"from": "2026-01-03", "to": "2026-01-04"})

    assert snap.failed_chunks == before
    assert len(snap.failed_chunks) == 1
    assert len(manifest.failed_chunks) == 2


def test_manifest_snapshot_unaffected_by_later_dict_mutation():
    manifest = _manifest()
    snap = snapshot_manifest(manifest)
    before = dict(snap.requested_range)

    manifest.requested_range["extra"] = "value"
    manifest.cleaning["another"] = True

    assert dict(snap.requested_range) == before
    assert "extra" not in snap.requested_range
    assert "another" not in snap.cleaning


def test_manifest_snapshot_unaffected_by_nested_dict_mutation():
    manifest = _manifest()
    manifest.cleaning["nested"] = {"a": 1}
    snap = snapshot_manifest(manifest)

    manifest.cleaning["nested"]["a"] = 999

    assert snap.cleaning["nested"]["a"] == 1


def test_manifest_snapshot_nested_collections_are_immutable():
    snap = snapshot_manifest(_manifest())
    assert isinstance(snap.failed_chunks, tuple)
    assert isinstance(snap.failed_chunks[0], MappingProxyType)
    assert isinstance(snap.requested_range, MappingProxyType)
    assert isinstance(snap.cleaning, MappingProxyType)
    assert isinstance(snap.software, MappingProxyType)
    assert isinstance(snap.validation_error_codes, tuple)

    with pytest.raises(TypeError):
        snap.requested_range["from"] = "changed"
    with pytest.raises(AttributeError):
        snap.failed_chunks.append({})


def test_manifest_snapshot_fields_are_frozen():
    snap = snapshot_manifest(_manifest())
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.symbol = "SBIN"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.requested_range = MappingProxyType({})


def test_manifest_snapshot_creation_is_deterministic():
    manifest = _manifest()
    snap1 = snapshot_manifest(manifest)
    snap2 = snapshot_manifest(manifest)
    assert snap1 == snap2


def test_manifest_snapshot_is_authoritative_property():
    manifest = _manifest()
    snap = snapshot_manifest(manifest)
    assert snap.is_authoritative == manifest.is_authoritative is True

    forced_manifest = _manifest()
    forced_manifest.forced = True
    assert snapshot_manifest(forced_manifest).is_authoritative is False


# ---------------------------------------------------------------------------
# Security: no secret/token values introduced
# ---------------------------------------------------------------------------


def test_no_snapshot_introduces_fields_beyond_the_source_report():
    # Snapshots are pure defensive copies: none of them should ever add a
    # field name that isn't already present on the corresponding source
    # dataclass -- in particular nothing resembling a credential.
    validation_fields = {f.name for f in dataclasses.fields(ValidationReport)}
    snapshot_fields = {f.name for f in dataclasses.fields(ValidationReportSnapshot)}
    assert snapshot_fields == validation_fields

    fetch_fields = {f.name for f in dataclasses.fields(FetchReport)}
    fetch_snapshot_fields = {f.name for f in dataclasses.fields(FetchReportSnapshot)}
    assert fetch_snapshot_fields == fetch_fields

    manifest_fields = {f.name for f in dataclasses.fields(DatasetManifest)}
    manifest_snapshot_fields = {f.name for f in dataclasses.fields(ManifestSnapshot)}
    assert manifest_snapshot_fields == manifest_fields


def test_no_secret_shaped_field_names_anywhere_in_evidence():
    forbidden = ("token", "secret", "password", "credential", "auth_code", "api_key")
    all_field_names = (
        {f.name for f in dataclasses.fields(ValidationReportSnapshot)}
        | {f.name for f in dataclasses.fields(ChunkResultSnapshot)}
        | {f.name for f in dataclasses.fields(FetchReportSnapshot)}
        | {f.name for f in dataclasses.fields(ManifestSnapshot)}
    )
    for name in all_field_names:
        lowered = name.lower()
        assert not any(bad in lowered for bad in forbidden), name
