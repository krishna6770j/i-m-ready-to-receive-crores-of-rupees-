"""Orchestrates fetch -> clean -> validate -> store with full provenance.

Deliberate ordering: normalise, then validate, then store. Validation runs on
exactly the frame that gets written, so a stored dataset's quality report
describes its actual contents rather than an earlier version of them.

Nothing here repairs data. The default cleaning operation list is empty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from brokers.base import HistoricalDataProvider
from brokers.fyers.historical import FetchReport, FyersHistoricalData
from core.types import Resolution
from marketdata import cleaner, store
from marketdata.validator import ValidationReport, validate

logger = logging.getLogger(__name__)


@dataclass
class DownloadOutcome:
    """Everything produced by one download, for reporting."""

    frame: pd.DataFrame
    fetch_report: FetchReport | None
    validation: ValidationReport
    cleaning: cleaner.CleaningRecord
    manifest: store.DatasetManifest | None
    path: Path | None

    def summary(self) -> str:
        lines = ["", self.validation.to_text()]
        if self.fetch_report is not None:
            fr = self.fetch_report
            lines += [
                "",
                "COVERAGE (requested vs downloaded)",
                "-" * 72,
                f"  requested : {fr.requested_from} .. {fr.requested_to}",
                f"  downloaded: {fr.first_ts} .. {fr.last_ts}",
                f"  rows      : {fr.total_rows}",
                f"  chunks    : {len(fr.chunks)} requested, "
                f"{len(fr.failed_chunks)} failed, {len(fr.empty_chunks)} empty",
            ]
            for chunk in fr.failed_chunks:
                lines.append(
                    f"    FAILED {chunk.range_from}..{chunk.range_to}: {chunk.error}"
                )
            lines.append("-" * 72)
        if self.path is not None:
            lines += [
                "",
                f"STORED: {self.path}",
                f"  sha256: {self.manifest.content_sha256 if self.manifest else '?'}",
            ]
        return "\n".join(lines)


def download(
    provider: HistoricalDataProvider,
    *,
    symbol: str,
    resolution: str,
    start: date,
    end: date,
    data_store_dir: Path,
    cleaning_operations: list[str] | None = None,
    persist: bool = True,
    notes: str = "",
) -> DownloadOutcome:
    """Fetch, clean, validate and optionally store one dataset."""
    fetch_report: FetchReport | None = None
    if isinstance(provider, FyersHistoricalData):
        frame, fetch_report = provider.fetch_candles_with_report(
            symbol, resolution, start, end
        )
    else:
        frame = provider.fetch_candles(symbol, resolution, start, end)

    frame, cleaning_record = cleaner.clean(frame, cleaning_operations)

    interval = Resolution(resolution).minutes
    validation = validate(
        frame,
        symbol=symbol,
        resolution=resolution,
        expected_interval_minutes=interval,
    )

    path: Path | None = None
    manifest: store.DatasetManifest | None = None
    if persist and len(frame):
        path, manifest = store.write(
            frame,
            data_store_dir,
            symbol=symbol,
            resolution=resolution,
            source=provider.source_name,
            requested_range={
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            cleaning=cleaning_record.to_dict(),
            notes=notes,
        )
        logger.info("stored %d rows -> %s", len(frame), path)
    elif persist:
        logger.warning("nothing stored: fetch returned zero rows")

    return DownloadOutcome(
        frame=frame,
        fetch_report=fetch_report,
        validation=validation,
        cleaning=cleaning_record,
        manifest=manifest,
        path=path,
    )
