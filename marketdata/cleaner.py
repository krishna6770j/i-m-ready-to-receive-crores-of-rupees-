"""Explicit, logged data cleaning.

Nothing here runs automatically. Each operation must be requested by name, and
each returns a record of exactly what it changed. The validator finds problems;
a human decides which repairs are justified; this module applies only those.

Deliberately NOT provided:
  * gap filling / forward filling -- invents prices that never traded, which
    corrupts indicators and lets a backtest fill at a fabricated level.
  * outlier smoothing -- hides the bad ticks we most need to see.
  * OHLC "correction" -- an impossible bar is a data integrity failure to be
    investigated with the vendor, not patched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from marketdata.schemas import TS, normalise

logger = logging.getLogger(__name__)


@dataclass
class CleaningRecord:
    """Audit trail of what cleaning actually did."""

    operations: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    def to_dict(self) -> dict:
        return {
            "operations": list(self.operations),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_removed,
        }


def drop_exact_duplicate_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Remove rows that are byte-identical across every column.

    Safe because such rows carry no information beyond the first copy. This is
    distinct from rows that share a timestamp but disagree on prices, which is
    a genuine conflict and is NOT resolved here.
    """
    before = len(frame)
    out = frame.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(out)
    return out, f"drop_exact_duplicate_rows: removed {removed} identical row(s)"


def drop_conflicting_duplicate_timestamps(
    frame: pd.DataFrame, *, keep: str = "first"
) -> tuple[pd.DataFrame, str]:
    """Resolve rows sharing a timestamp but disagreeing on values.

    This is a lossy choice and is never applied by default. Prefer
    re-downloading the affected range. Exposed only so that a documented,
    deliberate decision is possible when a vendor genuinely returns conflicting
    duplicates and re-fetching does not help.
    """
    before = len(frame)
    out = frame.drop_duplicates(subset=[TS], keep=keep).reset_index(drop=True)
    removed = before - len(out)
    return out, (
        f"drop_conflicting_duplicate_timestamps(keep={keep!r}): "
        f"removed {removed} row(s) -- LOSSY, values were discarded"
    )


def sort_by_timestamp(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Sort ascending by timestamp, preserving relative order of ties."""
    out = frame.sort_values(TS, kind="stable").reset_index(drop=True)
    return out, "sort_by_timestamp: ascending, stable"


AVAILABLE_OPERATIONS = {
    "drop_exact_duplicate_rows": drop_exact_duplicate_rows,
    "drop_conflicting_duplicate_timestamps": drop_conflicting_duplicate_timestamps,
    "sort_by_timestamp": sort_by_timestamp,
}


def clean(
    frame: pd.DataFrame, operations: list[str] | None = None
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Apply the named operations in order, recording each one.

    With ``operations=None`` this normalises dtypes and does nothing else, so
    the default path never alters data.
    """
    record = CleaningRecord(rows_before=len(frame))
    out = normalise(frame)
    record.operations.append("normalise: dtypes, column order, sort")

    for name in operations or []:
        try:
            func = AVAILABLE_OPERATIONS[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown cleaning operation {name!r}. "
                f"Available: {sorted(AVAILABLE_OPERATIONS)}"
            ) from exc
        out, description = func(out)
        record.operations.append(description)
        logger.info("cleaning: %s", description)

    record.rows_after = len(out)
    return out, record
