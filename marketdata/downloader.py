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
    refusal: str | None = None

    @property
    def persisted(self) -> bool:
        return self.path is not None

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
        if self.path is not None and self.manifest is not None:
            lines += [
                "",
                f"STORED: {self.path}",
                f"  sha256          : {self.manifest.content_sha256}",
                f"  validation      : {self.manifest.validation_status}",
                f"  fetch           : {self.manifest.fetch_status}",
                f"  AUTHORITATIVE   : {self.manifest.is_authoritative}",
            ]
        else:
            lines += [
                "",
                "NOT STORED",
                f"  reason: {self.refusal or 'persist disabled'}",
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
    force: bool = False,
) -> DownloadOutcome:
    """Fetch, clean, validate and conditionally store one dataset.

    Storage is GATED on validation and acquisition completeness. A dataset with
    ERROR-severity defects, or one whose acquisition had failed chunks, is not
    written unless ``force=True``, and a forced write is permanently marked
    non-authoritative in its manifest.
    """
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
    refusal: str | None = None

    if persist and len(frame):
        try:
            path, manifest = store.write(
                frame,
                data_store_dir,
                symbol=symbol,
                resolution=resolution,
                source=provider.source_name,
                validation=validation,
                fetch=fetch_report.to_dict() if fetch_report else None,
                requested_range={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                cleaning=cleaning_record.to_dict(),
                notes=notes,
                force=force,
            )
            logger.info("stored %d rows -> %s", len(frame), path)
        except (store.UnvalidatedDataError, store.IncompleteAcquisitionError) as exc:
            # Not re-raised: the caller still needs the validation report and
            # coverage detail to understand WHY nothing was written.
            refusal = str(exc)
            logger.error("refused to persist: %s", exc)
    elif persist:
        refusal = "fetch returned zero rows; nothing to store"
        logger.warning(refusal)

    return DownloadOutcome(
        frame=frame,
        fetch_report=fetch_report,
        validation=validation,
        cleaning=cleaning_record,
        manifest=manifest,
        path=path,
        refusal=refusal,
    )
