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

from marketdata.schemas import TS, assert_canonical, normalise

MANIFEST_SUFFIX = ".manifest.json"


@dataclass
class DatasetManifest:
    """Provenance record for one stored dataset."""

    symbol: str
    resolution: str
    source: str
    row_count: int
    first_ts: str | None
    last_ts: str | None
    timezone: str
    content_sha256: str
    fetched_at_utc: str
    requested_range: dict = field(default_factory=dict)
    cleaning: dict = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


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
    requested_range: dict | None = None,
    cleaning: dict | None = None,
    notes: str = "",
) -> tuple[Path, DatasetManifest]:
    """Write a canonical frame plus its manifest. Returns (path, manifest)."""
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
        requested_range=requested_range or {},
        cleaning=cleaning or {},
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
    manifest = DatasetManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    return frame, manifest


def verify_integrity(root: Path, symbol: str, resolution: str) -> bool:
    """True if the stored data still hashes to the value in its manifest."""
    frame, manifest = read(root, symbol, resolution)
    return content_hash(frame) == manifest.content_sha256
