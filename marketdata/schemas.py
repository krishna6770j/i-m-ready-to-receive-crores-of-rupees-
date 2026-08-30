"""Canonical OHLCV schema and normalisation.

One schema is used everywhere, so that a DataFrame from the FYERS adapter, from
a Parquet file, or from a test fixture are indistinguishable downstream. Any
divergence is a bug that surfaces here rather than deep inside a backtest.

Canonical form:
    ts      datetime64[ns, Asia/Kolkata]  candle OPEN time, tz-aware
    open    float64
    high    float64
    low     float64
    close   float64
    volume  int64

Rows are sorted by ``ts`` ascending and the index is a clean RangeIndex.
"""

from __future__ import annotations

import pandas as pd

from core.timeutils import IST_NAME, epoch_series_to_ist

TS = "ts"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"

OHLCV_COLUMNS: tuple[str, ...] = (TS, OPEN, HIGH, LOW, CLOSE, VOLUME)
PRICE_COLUMNS: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE)


class SchemaError(ValueError):
    """Raised when a frame does not conform to the canonical OHLCV schema."""


def empty_ohlcv() -> pd.DataFrame:
    """An empty frame with correct dtypes, for safe concatenation."""
    return pd.DataFrame(
        {
            TS: pd.Series([], dtype=f"datetime64[ns, {IST_NAME}]"),
            OPEN: pd.Series([], dtype="float64"),
            HIGH: pd.Series([], dtype="float64"),
            LOW: pd.Series([], dtype="float64"),
            CLOSE: pd.Series([], dtype="float64"),
            VOLUME: pd.Series([], dtype="Int64"),
        }
    )


def from_fyers_candles(candles: list[list]) -> pd.DataFrame:
    """Convert a FYERS ``candles`` array into the canonical schema.

    The FYERS history response carries rows of
    ``[epoch_seconds, open, high, low, close, volume]``. This ordering is the
    documented convention for the endpoint; because the response body could not
    be retrieved from the docs site during Phase 0 research, the row shape is
    validated defensively here and a precise error is raised on mismatch rather
    than mis-parsing silently.
    """
    if not candles:
        return empty_ohlcv()

    widths = {len(row) for row in candles}
    if widths != {6}:
        raise SchemaError(
            "Unexpected FYERS candle row width(s): "
            f"{sorted(widths)}. Expected every row to have 6 fields in the order "
            "[epoch_seconds, open, high, low, close, volume]. The response format "
            "may have changed; verify against current FYERS documentation before "
            "adjusting this parser."
        )

    frame = pd.DataFrame(candles, columns=list(OHLCV_COLUMNS))
    frame[TS] = epoch_series_to_ist(frame[TS])
    return normalise(frame)


def _to_numeric_strict(series: pd.Series, column: str) -> pd.Series:
    """Convert to numeric WITHOUT destroying evidence of bad source values.

    ``pd.to_numeric(errors="coerce")`` turns an unparseable value into NaN,
    which is indistinguishable from a value the source genuinely reported as
    missing. That silently converts a data-integrity failure into ordinary
    missingness. Here, a value that was present but unparseable raises instead,
    so the defect surfaces at the boundary with the offending values named.

    Values that were ALREADY null pass through as null: real missingness is
    preserved for the validator to report.
    """
    was_null = series.isna()
    converted = pd.to_numeric(series, errors="coerce")
    destroyed = converted.isna() & ~was_null
    if destroyed.any():
        positions = [int(i) for i in range(len(series)) if bool(destroyed.iloc[i])]
        samples = [repr(series.iloc[i]) for i in positions[:5]]
        raise SchemaError(
            f"Column {column!r}: {len(positions)} value(s) present in the source "
            f"but not parseable as numeric, e.g. {', '.join(samples)}. "
            "Refusing to coerce them to NaN, which would disguise a source "
            "defect as ordinary missing data. Investigate the source."
        )
    return converted


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Make the container canonical WITHOUT altering any market value.

    Permitted here: column selection and ordering, dtype conversion that
    preserves values, timezone conversion that preserves the instant, and
    deterministic stable sorting by timestamp.

    Explicitly NOT permitted: replacing missing values, fabricating volume,
    interpolating prices, dropping rows, or coercing unparseable source values
    into NaN. Duplicates, gaps and OHLC violations are left intact for the
    validator to find and report.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"Missing required column(s): {missing}. Required: {list(OHLCV_COLUMNS)}"
        )

    out = frame.loc[:, list(OHLCV_COLUMNS)].copy()

    if not isinstance(out[TS].dtype, pd.DatetimeTZDtype):
        raise SchemaError(
            f"Column '{TS}' must be tz-aware datetime, got dtype {out[TS].dtype}. "
            "Naive timestamps are rejected: this project never assumes a timezone."
        )
    if str(out[TS].dtype.tz) != IST_NAME:
        out[TS] = out[TS].dt.tz_convert(IST_NAME)

    for col in PRICE_COLUMNS:
        out[col] = _to_numeric_strict(out[col], col).astype("float64")
    # Volume uses pandas' nullable Int64 so that MISSING volume stays missing.
    # Filling it with 0 would fabricate a market fact ("no trades occurred")
    # that the source never asserted; the validator reports missingness instead.
    out[VOLUME] = _to_numeric_strict(out[VOLUME], VOLUME).astype("Int64")

    out = out.sort_values(TS, kind="stable").reset_index(drop=True)
    return out


def assert_canonical(frame: pd.DataFrame) -> None:
    """Raise SchemaError unless ``frame`` is exactly canonical."""
    if list(frame.columns) != list(OHLCV_COLUMNS):
        raise SchemaError(
            f"Column mismatch. Expected {list(OHLCV_COLUMNS)}, got {list(frame.columns)}"
        )
    if not isinstance(frame[TS].dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"'{TS}' must be tz-aware datetime, got {frame[TS].dtype}")
    if str(frame[TS].dtype.tz) != IST_NAME:
        raise SchemaError(f"'{TS}' must be {IST_NAME}, got {frame[TS].dtype.tz}")
    for col in PRICE_COLUMNS:
        if frame[col].dtype != "float64":
            raise SchemaError(f"'{col}' must be float64, got {frame[col].dtype}")
    if str(frame[VOLUME].dtype) != "Int64":
        raise SchemaError(
            f"'{VOLUME}' must be nullable Int64 (so missing volume stays "
            f"missing rather than being fabricated as 0), got {frame[VOLUME].dtype}"
        )
    if not frame[TS].is_monotonic_increasing:
        raise SchemaError(f"'{TS}' must be sorted ascending")
