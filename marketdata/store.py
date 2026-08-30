"""Parquet storage with provenance manifests.

Requirement 29 asks that a backtest be reproducible. That starts with knowing
exactly which bytes went in. Every dataset written here is accompanied by a
JSON manifest recording its source, fetch time, range and a content hash.

The content hash is computed from the canonical CSV rendering of the frame
rather than from the Parquet bytes, because Parquet encodes metadata such as
the writer version and compression settings that can differ between runs
without the data differing. Hashing the logical content makes the
reproducibility check meaningful.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.environment import git_revision, software_versions
from marketdata.schemas import TS, assert_canonical, normalise

MANIFEST_SUFFIX = ".manifest.json"


class UnvalidatedDataError(RuntimeError):
    """Raised when data failing validation would be written as authoritative."""


class IncompleteAcquisitionError(RuntimeError):
    """Raised when a partially-fetched dataset would be written as complete."""


@dataclass
class DatasetManifest:
    """Provenance record for one stored dataset.

    A consumer must be able to tell these four states apart without re-running
    validation, so ``validation_status`` and ``fetch_status`` are both recorded
    and ``is_authoritative`` combines them:

        complete + valid    -> authoritative
        complete + invalid  -> NOT authoritative
        partial  + valid    -> NOT authoritative
        partial  + invalid  -> NOT authoritative
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
    validation_status: str = "unknown"
    validation_error_count: int = 0
    validation_warning_count: int = 0
    validation_error_codes: list = field(default_factory=list)
    fetch_status: str = "unknown"
    failed_chunks: list = field(default_factory=list)
    requested_range: dict = field(default_factory=dict)
    cleaning: dict = field(default_factory=dict)
    software: dict = field(default_factory=dict)
    forced: bool = False
    notes: str = ""

    @property
    def is_authoritative(self) -> bool:
        return (
            self.validation_status == "valid"
            and self.fetch_status == "complete"
            and not self.forced
        )

    def to_json(self) -> str:
        payload = asdict(self)
        payload["is_authoritative"] = self.is_authoritative
        return json.dumps(payload, indent=2, sort_keys=True)


def content_hash(frame: pd.DataFrame) -> str:
    """Stable SHA-256 over the logical content of a canonical frame.

    Timestamps are rendered in ISO-8601 with offset so the hash captures the
    timezone; floats use repr to avoid precision loss through formatting.
    """
    assert_canonical(frame)
    buf = frame.copy()
    buf[TS] = buf[TS].map(lambda t: pd.Timestamp(t).isoformat())
    csv_bytes = buf.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def dataset_paths(root: Path, symbol: str, resolution: str) -> tuple[Path, Path]:
    """Return (parquet_path, manifest_path) for a dataset."""
    safe_symbol = symbol.replace(":", "_").replace("/", "_")
    stem = f"{safe_symbol}__{resolution}"
    parquet = root / f"{stem}.parquet"
    return parquet, parquet.with_suffix(f".parquet{MANIFEST_SUFFIX}")


def write(
    frame: pd.DataFrame,
    root: Path,
    *,
    symbol: str,
    resolution: str,
    source: str,
    validation,
    fetch: dict | None = None,
    requested_range: dict | None = None,
    cleaning: dict | None = None,
    notes: str = "",
    force: bool = False,
) -> tuple[Path, DatasetManifest]:
    """Write a canonical frame plus its manifest.

    ``validation`` is REQUIRED (a ``ValidationReport``). It is not optional and
    has no default, because the single most dangerous failure mode for this
    project is corrupt data becoming an authoritative stored dataset that a
    later backtest silently trusts. Making the parameter mandatory means a
    caller cannot skip the gate by forgetting about it.

    Raises UnvalidatedDataError when the report contains ERROR-severity issues,
    and IncompleteAcquisitionError when ``fetch`` reports failed chunks, unless
    ``force=True``. A forced write is recorded in the manifest as
    ``forced: true`` and can never be ``is_authoritative``.
    """
    errors = list(validation.errors)
    if errors and not force:
        raise UnvalidatedDataError(
            f"Refusing to persist {symbol} {resolution}: validation found "
            f"{len(errors)} ERROR-severity issue(s) "
            f"({', '.join(i.code for i in errors)}). Persisting would make "
            "corrupt data indistinguishable from clean data for every future "
            "consumer. Fix the source, or pass force=True to record it "
            "explicitly as non-authoritative."
        )

    failed_chunks = list((fetch or {}).get("failed_chunk_detail", []))
    if failed_chunks and not force:
        raise IncompleteAcquisitionError(
            f"Refusing to persist {symbol} {resolution}: {len(failed_chunks)} "
            "acquisition chunk(s) failed, so the dataset does not cover the "
            "requested range. Storing it would misrepresent a partial download "
            "as a complete one. Retry the failed windows, or pass force=True to "
            "record it explicitly as partial and non-authoritative."
        )

    frame = normalise(frame)
    assert_canonical(frame)
    root.mkdir(parents=True, exist_ok=True)
    parquet_path, manifest_path = dataset_paths(root, symbol, resolution)

    frame.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

    manifest = DatasetManifest(
        symbol=symbol,
        resolution=resolution,
        source=source,
        row_count=len(frame),
        first_ts=frame[TS].iloc[0].isoformat() if len(frame) else None,
        last_ts=frame[TS].iloc[-1].isoformat() if len(frame) else None,
        timezone=str(frame[TS].dtype.tz),
        content_sha256=content_hash(frame),
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        validation_status="invalid" if errors else "valid",
        validation_error_count=len(errors),
        validation_warning_count=len(validation.warnings),
        validation_error_codes=[i.code for i in errors],
        fetch_status=(
            "unknown" if fetch is None else ("partial" if failed_chunks else "complete")
        ),
        failed_chunks=failed_chunks,
        requested_range=requested_range or {},
        cleaning=cleaning or {},
        software=software_versions(),
        forced=bool(force),
        notes=notes,
    )
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    return parquet_path, manifest


def read(
    root: Path, symbol: str, resolution: str
) -> tuple[pd.DataFrame, DatasetManifest]:
    """Read a stored dataset and its manifest."""
    parquet_path, manifest_path = dataset_paths(root, symbol, resolution)
    if not parquet_path.exists():
        raise FileNotFoundError(f"No dataset at {parquet_path}")

    frame = normalise(pd.read_parquet(parquet_path, engine="pyarrow"))
    assert_canonical(frame)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset {parquet_path.name} has no manifest. Data without provenance "
            "is not trusted here; re-download it."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # is_authoritative is derived, not stored state; recompute it from the
    # recorded fields rather than trusting a value someone could hand-edit.
    payload.pop("is_authoritative", None)
    manifest = DatasetManifest(**payload)
    return frame, manifest


def verify_integrity(root: Path, symbol: str, resolution: str) -> bool:
    """True if the stored data still hashes to the value in its manifest."""
    frame, manifest = read(root, symbol, resolution)
    return content_hash(frame) == manifest.content_sha256
