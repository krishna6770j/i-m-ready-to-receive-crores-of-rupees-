"""Canonical OHLCV schema, canonicalisation and normalisation.

One schema is used everywhere, so that a DataFrame from the FYERS adapter, from
a Parquet file, or from a test fixture are indistinguishable downstream. Any
divergence is a bug that surfaces here rather than deep inside a backtest.

Canonical form:
    ts      datetime64[ns, Asia/Kolkata]  candle OPEN time, tz-aware
    open    float64
    high    float64
    low     float64
    close   float64
    volume  Int64 (nullable)  -- missing volume stays missing, never becomes 0

Rows are sorted by ``ts`` ascending and the index is a clean RangeIndex.

Two API layers exist here, per the frozen architecture
(docs/architecture/phase1-trust-hardening.md, section 14):

  ``canonicalise()`` / ``canonicalise_fyers_candles()``
      Return a ``CanonicalisationResult``: the canonical frame PLUS evidence of
      what the source required to become canonical (was it already sorted?
      did it carry unsupported columns? were there duplicate timestamps?).
      This is the API new code should use.

  ``normalise()`` / ``from_fyers_candles()``
      Transitional, frame-only wrappers kept for existing callers
      (marketdata/cleaner.py, marketdata/store.py,
      brokers/fyers/historical.py). They discard the evidence
      ``canonicalise()`` computes. Migrating those callers to consume
      ``CanonicalisationResult`` directly is a separate, explicitly scoped
      unit -- not this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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

_CANONICAL_COLUMN_SET = frozenset(OHLCV_COLUMNS)


class SchemaError(ValueError):
    """Raised when a frame does not conform to the canonical OHLCV schema."""


class AnomalySeverity(str, Enum):
    """How seriously to treat something canonicalisation observed in the source.

    None of these severities make canonicalise() raise on their own -- a
    BLOCKER anomaly is still recorded and returned, never resolved by picking
    a winner between contradictory observations. BLOCKER means a later stage
    (ValidatedDataset, not implemented in this unit) is expected to refuse to
    treat the result as authoritative.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    BLOCKER = "BLOCKER"


@dataclass(frozen=True)
class CanonicalisationTransformation:
    """A deterministic, value-preserving change canonicalisation performed."""

    code: str
    description: str


@dataclass(frozen=True)
class CanonicalisationAnomaly:
    """Something the SOURCE delivered that a well-formed feed would not.

    Recording this is not repair. The frame is still canonicalised the same
    way regardless; the fact that the source needed help is preserved rather
    than erased.
    """

    code: str
    severity: AnomalySeverity
    description: str


@dataclass(frozen=True)
class SourceEvidence:
    """Facts about the input AS RECEIVED, measured before any canonical
    transformation (timezone conversion, numeric coercion, column reordering
    or sorting). If the source used a different type or representation for
    two values that later convert to the same canonical number, that
    difference is what this evidence reflects -- not the post-conversion
    result. See ``canonicalise()`` for where the line is drawn.

    ``descending_adjacent_pairs`` is defined precisely so it means the same
    thing regardless of duplicate timestamps: the count of adjacent row pairs
    (i, i+1) in the INPUT row order where timestamp[i] > timestamp[i+1]. Equal
    timestamps are never counted as a descent, so this is well-defined even
    when duplicate timestamps are present.

    ``exact_duplicate_row_count`` counts rows whose values, AS RECEIVED
    (before any type conversion), are equal to an earlier row's values under
    the source DataFrame's own dtypes. This is value-equality at the source
    representation, not byte-identity of any underlying storage -- two rows
    that later convert to the same canonical number but arrived in different
    forms (e.g. the int ``1`` and the string ``"1"``) are NOT counted as
    duplicates here, because they were not equal as received.
    """

    row_count: int
    column_inventory: tuple[str, ...]
    timestamps_sorted: bool
    descending_adjacent_pairs: int
    exact_duplicate_row_count: int


@dataclass(frozen=True)
class CanonicalisationResult:
    """Canonical frame plus the evidence of how it got that way.

    ``frame`` is a defensive copy: canonicalise() never aliases the caller's
    DataFrame in either direction. Mutating ``frame`` after the call does not
    affect anything the caller still holds, and mutating the caller's original
    frame afterwards does not affect this result.

    This does NOT make the DataFrame immutable -- pandas offers no such
    guarantee, and nothing in this module claims otherwise. It only ensures
    this function does not hand out or retain a shared reference.
    """

    frame: pd.DataFrame
    transformations: tuple[CanonicalisationTransformation, ...]
    source_anomalies: tuple[CanonicalisationAnomaly, ...]
    source: SourceEvidence


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


def _is_fractional(value: object) -> bool:
    return pd.notna(value) and not float(value).is_integer()


def _to_whole_number_strict(series: pd.Series, column: str) -> pd.Series:
    """Convert to nullable Int64, requiring every present value be whole.

    A fractional volume (e.g. 250.7) is a source defect, not data to round or
    truncate. Rounding it would silently alter what the source reported;
    letting pandas raise its own ``TypeError`` on the eventual
    ``.astype("Int64")`` would leak an implementation detail instead of a
    diagnosed ``SchemaError``. Both are avoided by checking first.
    """
    converted = _to_numeric_strict(series, column)
    fractional_mask = converted.map(_is_fractional)
    if fractional_mask.any():
        offending = converted[fractional_mask]
        samples = offending.head(5).tolist()
        raise SchemaError(
            f"Column {column!r}: {int(fractional_mask.sum())} value(s) are not "
            f"whole numbers, e.g. {samples}. Volume is a count of units traded; "
            "refusing to round or truncate a fractional value."
        )
    return converted.astype("Int64")


def canonicalise(frame: pd.DataFrame) -> CanonicalisationResult:
    """Canonicalise ``frame``, returning the result AND the evidence.

    Permitted: column selection/ordering (when the input already carries
    exactly the canonical columns, in any order), dtype conversion that
    preserves values exactly, timezone conversion that preserves the instant,
    and deterministic stable sorting by timestamp.

    Forbidden, and enforced here: silently dropping unknown columns, replacing
    missing values, fabricating volume, interpolating prices, dropping rows,
    coercing unparseable source values into NaN, or resolving a duplicate or
    conflicting timestamp by choosing between the candidates. All of that is
    left to a later validation/cleaning stage; this function only makes the
    container canonical and records exactly what it had to do to get there.

    Raises SchemaError for: an unsupported/unmapped column present in the
    source (there is currently no schema mapping for any additional field --
    see the frozen architecture, section 15); a missing required column; a
    naive or otherwise malformed timestamp column; a present-but-unparseable
    numeric value; or a fractional volume. None of these produce a partial
    result -- the function raises before constructing one.
    """
    column_inventory = tuple(frame.columns)

    extra = [c for c in frame.columns if c not in _CANONICAL_COLUMN_SET]
    if extra:
        raise SchemaError(
            f"Unsupported column(s) present in the source: {extra}. There is "
            "no schema mapping for additional fields yet -- a new field must "
            "be explicitly declared and versioned before it can be preserved, "
            "rather than silently dropped or invented a name. Canonical "
            f"columns are: {list(OHLCV_COLUMNS)}."
        )

    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"Missing required column(s): {missing}. Required: {list(OHLCV_COLUMNS)}"
        )

    row_count = len(frame)
    transformations: list[CanonicalisationTransformation] = []
    anomalies: list[CanonicalisationAnomaly] = []

    if list(frame.columns) != list(OHLCV_COLUMNS):
        transformations.append(
            CanonicalisationTransformation(
                "COLUMNS_REORDERED",
                "Source columns were present but not in canonical order; reordered.",
            )
        )

    # Defensive copy: canonicalise() must never alias the caller's frame in
    # either direction. Everything below reads from / writes to `out`, never
    # back into `frame`.
    out = frame.loc[:, list(OHLCV_COLUMNS)].copy()

    if not isinstance(out[TS].dtype, pd.DatetimeTZDtype):
        raise SchemaError(
            f"Column '{TS}' must be tz-aware datetime, got dtype {out[TS].dtype}. "
            "Naive timestamps are rejected: this project never assumes a timezone."
        )

    # --- SOURCE-level evidence -------------------------------------------
    # Everything in this block is measured on `out` EXACTLY as received: no
    # timezone conversion, no numeric coercion has happened yet. This is what
    # makes SourceEvidence actually describe the source (manager correction):
    # a value that only becomes equal to another after later conversion must
    # not be reported as an equal/duplicate observation here.
    ts_series = out[TS]
    if row_count > 1:
        descending_adjacent_pairs = int((ts_series.diff().dt.total_seconds() < 0).sum())
    else:
        descending_adjacent_pairs = 0
    timestamps_sorted = descending_adjacent_pairs == 0

    if not timestamps_sorted:
        anomalies.append(
            CanonicalisationAnomaly(
                "SOURCE_UNSORTED",
                AnomalySeverity.INFO,
                f"Source rows were not in ascending timestamp order "
                f"({descending_adjacent_pairs} descending adjacent pair(s)). "
                "The output is stable-sorted; this anomaly records that the "
                "source required it.",
            )
        )

    exact_duplicate_row_count = int(out.duplicated(keep="first").sum())
    if exact_duplicate_row_count:
        anomalies.append(
            CanonicalisationAnomaly(
                "SOURCE_EXACT_DUPLICATE_ROWS",
                AnomalySeverity.WARNING,
                f"{exact_duplicate_row_count} row(s) are equal to an earlier "
                "row's values as received (before any type conversion). "
                "Not removed here.",
            )
        )

    ts_dup_mask = out[TS].duplicated(keep=False)
    if ts_dup_mask.any():
        conflicting = False
        for _, group in out.loc[ts_dup_mask].groupby(TS):
            # Correct rule: how many DISTINCT observations share this
            # timestamp? More than one is a conflict, regardless of how many
            # copies of each exist. The previous check --
            # "duplicated(keep=False).sum() < len(group)" -- was wrong
            # whenever every row had SOME matching partner, e.g. two copies of
            # A plus two copies of B (A,A,B,B): every row is "duplicated" by
            # its own twin, sum()==len(group)==4, and the old check concluded
            # "no conflict" despite two distinct observations being present.
            distinct_observations = len(group) - int(
                group.duplicated(keep="first").sum()
            )
            if distinct_observations > 1:
                conflicting = True
                break
        if conflicting:
            anomalies.append(
                CanonicalisationAnomaly(
                    "SOURCE_CONFLICTING_TIMESTAMPS",
                    AnomalySeverity.BLOCKER,
                    "Two or more DISTINCT observations share a timestamp. "
                    "Canonicalisation does not choose between contradictory "
                    "observations; all are preserved.",
                )
            )
    # --- end SOURCE-level evidence ----------------------------------------

    if str(out[TS].dtype.tz) != IST_NAME:
        source_tz = out[TS].dtype.tz
        out[TS] = out[TS].dt.tz_convert(IST_NAME)
        transformations.append(
            CanonicalisationTransformation(
                "TIMEZONE_CONVERTED",
                f"Converted timestamps from {source_tz} to {IST_NAME}, "
                "preserving the instant.",
            )
        )

    for col in PRICE_COLUMNS:
        out[col] = _to_numeric_strict(out[col], col).astype("float64")
    out[VOLUME] = _to_whole_number_strict(out[VOLUME], VOLUME)

    if not timestamps_sorted:
        transformations.append(
            CanonicalisationTransformation(
                "ROWS_SORTED",
                "Rows were stable-sorted by timestamp ascending.",
            )
        )
    out = out.sort_values(TS, kind="stable").reset_index(drop=True)

    source = SourceEvidence(
        row_count=row_count,
        column_inventory=column_inventory,
        timestamps_sorted=timestamps_sorted,
        descending_adjacent_pairs=descending_adjacent_pairs,
        exact_duplicate_row_count=exact_duplicate_row_count,
    )

    return CanonicalisationResult(
        frame=out,
        transformations=tuple(transformations),
        source_anomalies=tuple(anomalies),
        source=source,
    )


def canonicalise_fyers_candles(candles: list[list]) -> CanonicalisationResult:
    """FYERS ``/history`` positional payload -> ``CanonicalisationResult``.

    Row shape is exactly ``[epoch_seconds, open, high, low, close, volume]``,
    the documented convention for the endpoint. This is
    ADAPTER-CONTRACT-VALID: it matches the SDK docstring and community
    documentation consulted while writing this parser. It has never been
    checked against a live response, so it remains LIVE-BROKER-UNVERIFIED.

    Any row width other than 6 is rejected outright. An unnamed extra
    positional field has no safe interpretation and is never invented as a
    synthetic column (e.g. ``x_unknown_7``) -- see the frozen architecture,
    section 15.

    ``SourceEvidence.column_inventory`` for this function is always
    ``OHLCV_COLUMNS``, for empty and non-empty payloads alike. This is
    deliberate, not an oversight: a raw FYERS row is a bare positional array
    with no field names at all, so "ts", "open", etc. are never columns
    physically present in the source -- they are the fixed mapping THIS
    ADAPTER imposes on every position-0..5 payload it accepts, independent of
    how many rows arrive. An empty list did not "have" these columns any more
    than a populated one did, since neither ever carried labels.
    """
    if not candles:
        return CanonicalisationResult(
            frame=empty_ohlcv(),
            transformations=(),
            source_anomalies=(),
            source=SourceEvidence(
                row_count=0,
                column_inventory=OHLCV_COLUMNS,
                timestamps_sorted=True,
                descending_adjacent_pairs=0,
                exact_duplicate_row_count=0,
            ),
        )

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

    result = canonicalise(frame)
    # canonicalise() sees `frame[TS]` already tz-aware IST (epoch_series_to_ist
    # converted it above), so its own TIMEZONE_CONVERTED check never fires and
    # the epoch->IST conversion that DID happen would otherwise leave no
    # transformation evidence at all. Recorded here, at the point where the
    # conversion is actually known to have occurred.
    epoch_transformation = CanonicalisationTransformation(
        "FYERS_EPOCH_TO_IST",
        "Interpreted the source 'ts' field as Unix epoch seconds (UTC by "
        "definition), then converted through UTC to Asia/Kolkata, preserving "
        "the instant. This is the FYERS positional-payload adapter contract "
        "(ADAPTER-CONTRACT-VALID; not yet verified against a live response).",
    )
    return CanonicalisationResult(
        frame=result.frame,
        transformations=(epoch_transformation, *result.transformations),
        source_anomalies=result.source_anomalies,
        source=result.source,
    )


def from_fyers_candles(candles: list[list]) -> pd.DataFrame:
    """Transitional, frame-only wrapper. Prefer ``canonicalise_fyers_candles``.

    KNOWN LIMITATION: ``brokers/fyers/historical.py``, the production
    ingestion path, still calls this function and therefore still discards
    the source-anomaly and transformation evidence ``canonicalise_fyers_candles``
    computes. Migrating that caller to consume ``CanonicalisationResult``
    directly is a separate, explicitly scoped unit -- not this one.
    """
    return canonicalise_fyers_candles(candles).frame


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Transitional, frame-only wrapper. Prefer ``canonicalise``.

    KNOWN LIMITATION: ``marketdata/cleaner.py`` and ``marketdata/store.py``
    still call this function and therefore still discard the evidence
    ``canonicalise`` computes. Migrating those callers to consume
    ``CanonicalisationResult`` directly is a separate, explicitly scoped unit
    -- not this one.
    """
    return canonicalise(frame).frame


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
