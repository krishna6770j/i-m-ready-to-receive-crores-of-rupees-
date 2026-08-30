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

THREE DISTINCT CONCEPTS, kept separate throughout this module (frozen
architecture section 14 requires this and does not allow collapsing them):

  SOURCE structural facts (row order, which rows/timestamps repeat, AS
      RECEIVED) -- always recorded, never gate anything by themselves.

  Lossless representation conversion ("1" -> 1.0, object -> float64,
      float64 -> Int64) -- a NORMAL TRANSFORMATION. It is recorded as
      provenance, but it is not a defect and never a BLOCKER.

  CANONICAL market-value conflict -- after all lossless conversions, two
      observations at one timestamp disagree on an actual OHLCV value. This,
      and only this, is a TRUST BLOCKER. A representation difference that
      converges to the same canonical value is NOT this.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum

import numpy as np
import pandas as pd

from core.timeutils import IST_NAME, epoch_series_to_ist

# Every |int| at or below this round-trips through float64 exactly (53
# mantissa bits). Beyond it, converting to float64 can silently change the
# value -- see _convert_numeric_exact().
_MAX_EXACT_FLOAT64_INT = 2**53

# Frozen contract (docs/architecture/phase1-trust-hardening.md, section 8.0).
# Identifies the canonical market-data representation below. A new value is
# required only when a change would alter what the canonical dataset means
# (new field, changed timestamp/numeric/missing-value semantics) -- never for
# refactors, tests, or storage changes. Never a caller-supplied parameter.
MARKET_DATA_SCHEMA_VERSION = 1

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
    """How seriously to treat something canonicalisation observed.

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
    """A deterministic, value-preserving change canonicalisation performed.

    This includes lossless representation conversions (e.g. the string "1"
    parsed to the float 1.0, or a float64 column with only whole-number values
    converted to Int64). A representation change is NOT a value change: the
    frozen architecture's "any value changed => BLOCKER" rule means semantic
    market-value modification -- rounding, clipping, replacing, filling,
    picking a winner between conflicting observations, or other arithmetic
    modification. Losslessly re-expressing the same value in a different type
    is what this class records, and it is always a NORMAL TRANSFORMATION.
    """

    code: str
    description: str


@dataclass(frozen=True)
class CanonicalisationAnomaly:
    """Something worth recording about the source or the canonical result.

    Recording this is not repair. The frame is still canonicalised the same
    way regardless; the fact is preserved rather than erased.
    """

    code: str
    severity: AnomalySeverity
    description: str


@dataclass(frozen=True)
class SourceEvidence:
    """Structural facts about the input, measured before any lossy-looking
    (but still lossless) canonical transformation -- timezone conversion,
    numeric/dtype coercion, column reordering, or sorting.

    For ``canonicalise(frame)``: these facts describe the DataFrame exactly as
    presented to this function -- its own dtypes, its own representation of
    each value -- before this function touches anything. If the caller already
    normalised or coerced values before calling this function, that is outside
    what this evidence can see; it describes what canonicalise() received, not
    an earlier stage the caller may have performed.

    For ``canonicalise_fyers_candles(candles)``: these facts describe the
    adapter-provided Python positional observations (the ``list[list]``) to
    the extent representable from that structure -- epoch integers, raw
    numeric literals -- before this function performs ANY conversion,
    including the epoch-to-Timestamp step. There is no claim here about raw
    HTTP transport bytes; only Python objects are ever received.

    ``descending_adjacent_pairs`` is defined precisely so it means the same
    thing regardless of duplicate timestamps: the count of adjacent row pairs
    (i, i+1) in the INPUT row order where timestamp[i] > timestamp[i+1]. Equal
    timestamps are never counted as a descent, so this is well-defined even
    when duplicate timestamps are present.

    ``exact_duplicate_row_count`` counts rows whose values, AS RECEIVED
    (before any type conversion), are equal to an earlier row's values. This
    is value-equality at the source representation, not byte-identity of any
    underlying storage -- two rows that later convert to the same canonical
    number but arrived in different forms (e.g. the int ``1`` and the string
    ``"1"``) are NOT counted as duplicates here, because they were not equal
    as received.

    ``duplicate_timestamp_row_count`` is a DIFFERENT, coarser fact: the count
    of rows whose timestamp (as received) is shared by at least one other row,
    REGARDLESS of whether the rest of the row's values agree. Two rows with
    the same timestamp but different representations of an equivalent value
    (the int/string example above) still count here, because the SOURCE
    genuinely presented two rows for one instant -- that structural fact is
    real even when it turns out not to be a market-value conflict once
    canonicalised. This is what distinguishes "the source had a duplicate
    timestamp" from "the source had two exact duplicate rows": the former is
    always true when the latter is, but not the reverse.
    """

    row_count: int
    column_inventory: tuple[str, ...]
    timestamps_sorted: bool
    descending_adjacent_pairs: int
    exact_duplicate_row_count: int
    duplicate_timestamp_row_count: int


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

    ``source_anomalies`` mixes facts detected at the source-structural level
    (e.g. ``SOURCE_UNSORTED``) with facts that can only be established after
    canonical conversion (``CANONICAL_CONFLICTING_TIMESTAMPS``). Each
    anomaly's own code and description say which kind it is; the severity does
    not depend on which measurement stage produced it.
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


def _is_bool_like(value: object) -> bool:
    """True for a Python ``bool`` or a numpy boolean scalar.

    ``bool`` is a subclass of ``int`` in Python -- that is a language
    implementation detail, not a valid market-data representation. It is
    rejected explicitly rather than silently converted (``True`` -> ``1.0``).
    Both a native pandas ``bool`` column and the nullable ``"boolean"``
    extension dtype must be caught: elements of the latter come back as
    ``numpy.bool_``, which is NOT a subclass of Python ``bool``.
    """
    return isinstance(value, (bool, np.bool_))


def _parse_source_number(value: object) -> int | float:
    """Parse ONE present, non-null, non-boolean source value.

    Returns a Python ``int`` if the source represents an exact whole number
    (magnitude not yet checked against float64's exact range -- the caller
    does that), or a Python ``float`` if the source has a genuine fractional
    part, is already a binary float, or is +/-inf.

    Raises ``ValueError`` if the value cannot be parsed as a number at all;
    the caller batches offenders into one ``SchemaError``.

    String parsing goes through ``decimal.Decimal`` rather than a naive
    ``int()``-then-``float()`` fallback, specifically so that a string like
    ``"9007199254740993.0"`` -- a whole number that merely carries a
    redundant decimal point -- is still recognised as whole and exactness-
    checked, instead of silently falling into the always-accepted float
    branch and escaping the range check entirely.
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # Already a binary float: whatever precision it carries was fixed
        # before this function ever saw it. Preserving that value exactly --
        # not re-deriving it from a decimal -- is what "lossless" means here;
        # there is nothing further to check.
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            d = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError("not parseable as numeric") from exc
        if d.is_nan():
            # A literal "nan" string is not how this schema represents
            # missingness (that is a real null, caught before this function
            # is ever called); treat it as an unparseable value instead of
            # silently accepting it as missing.
            raise ValueError("not parseable as numeric")
        if d.is_infinite():
            return float(d)
        if d == d.to_integral_value():
            return int(d)
        return float(text)
    raise ValueError(f"unsupported type {type(value).__name__}")


def _convert_numeric_exact(
    series: pd.Series, column: str, *, whole_numbers_only: bool
) -> pd.Series:
    """Convert to float64, REJECTING any conversion that would silently
    change the numeric value, rather than recording a lossy conversion as a
    lossless ``DTYPE_CONVERTED`` transformation.

    Policy, precise and non-fuzzy (no epsilon/tolerance is used anywhere):

    * Boolean values are rejected outright -- see ``_is_bool_like``.
    * A value that were ALREADY null passes through as null: real
      missingness is preserved for the validator to report.
    * A value present but unparseable as any number raises, naming the
      offending value, instead of being coerced to NaN (which would disguise
      a source defect as ordinary missingness).
    * A value representing an exact WHOLE NUMBER (a Python/numpy int, or a
      numeric string/float with no fractional part) must round-trip through
      float64 exactly. A magnitude beyond +/-2**53 is REJECTED: silently
      returning a different integer would be a value change, not a
      representation change (e.g. 9007199254740993 -> 9007199254740992).
    * A value with a genuine fractional part (e.g. the string ``"24000.13"``,
      or an existing float) is accepted via ordinary decimal-to-binary
      conversion. This is NOT treated as lossy: almost no non-power-of-two
      decimal fraction is exactly representable in binary floating point, so
      rejecting every such value would make this schema unusable for real
      decimal prices. This is a representation fact about IEEE-754, not the
      semantic value modification (rounding, clipping, replacing, filling)
      the frozen architecture's BLOCKER rule targets.
    * +/-inf is accepted WHEN THE TARGET IS FLOAT64 (price columns): float64
      represents it exactly, so nothing is lost by preserving it. A later
      validation stage (unchanged by this module) is responsible for
      rejecting a non-finite price as market data.

    When ``whole_numbers_only`` is True (used for volume, ahead of the
    ``.astype("Int64")`` cast the caller performs), a value with a genuine
    fractional part is rejected instead of accepted, with the exact same
    wording as before this correction so existing callers matching on
    "whole number" continue to work. Infinity is ALSO rejected in this case,
    for the same reason: Int64 has no representation for it, and accepting
    it here would only defer the failure to the caller's later cast, which
    raises a raw ``OverflowError`` instead of a diagnosed ``SchemaError``.

    Volume's ultimate target dtype (Int64) can exactly hold integers far
    larger than float64's +/-2**53 range, but this function applies the same
    float64-exact threshold to volume as to prices, deliberately: no
    realistic trading volume approaches that magnitude, and a dedicated,
    larger Int64-specific threshold would add complexity this project does
    not currently need. If that ever changes, it is a separate, explicitly
    scoped correction.
    """
    was_null = series.isna()

    bool_positions = [
        i for i in range(len(series))
        if not bool(was_null.iloc[i]) and _is_bool_like(series.iloc[i])
    ]
    if bool_positions:
        samples = [repr(series.iloc[i]) for i in bool_positions[:5]]
        raise SchemaError(
            f"Column {column!r}: {len(bool_positions)} value(s) are boolean, "
            f"e.g. {', '.join(samples)}. bool is a subclass of int as a "
            "Python implementation detail, not a valid market-data "
            "representation; refusing to silently convert True/False to a "
            "number."
        )

    unparseable: list[object] = []
    imprecise: list[object] = []
    fractional: list[object] = []
    values: list[float] = [float("nan")] * len(series)

    for i in range(len(series)):
        if bool(was_null.iloc[i]):
            continue
        raw = series.iloc[i]
        try:
            numeric = _parse_source_number(raw)
        except ValueError:
            unparseable.append(raw)
            continue
        if isinstance(numeric, int):
            if abs(numeric) > _MAX_EXACT_FLOAT64_INT:
                imprecise.append(raw)
                continue
            values[i] = float(numeric)
        else:
            if math.isinf(numeric):
                if whole_numbers_only:
                    # Int64 (volume's eventual target, via the caller's
                    # .astype("Int64")) has no representation for infinity.
                    # Accepting it here would leak a raw pandas
                    # OverflowError from that later cast instead of a
                    # diagnosed SchemaError -- reproduced directly before
                    # this fix. Infinity is treated the same as any other
                    # value with no finite whole-number representation.
                    fractional.append(raw)
                    continue
                values[i] = numeric
                continue
            if whole_numbers_only and not numeric.is_integer():
                fractional.append(raw)
                continue
            values[i] = numeric

    if unparseable:
        samples = [repr(v) for v in unparseable[:5]]
        raise SchemaError(
            f"Column {column!r}: {len(unparseable)} value(s) present in the source "
            f"but not parseable as numeric, e.g. {', '.join(samples)}. "
            "Refusing to coerce them to NaN, which would disguise a source "
            "defect as ordinary missing data. Investigate the source."
        )
    if imprecise:
        samples = [repr(v) for v in imprecise[:5]]
        raise SchemaError(
            f"Column {column!r}: {len(imprecise)} whole-number value(s) exceed "
            f"float64's exact-integer range (+/-{_MAX_EXACT_FLOAT64_INT}), e.g. "
            f"{', '.join(samples)}. Converting would silently change the value "
            "(e.g. 9007199254740993 -> 9007199254740992); refusing rather than "
            "recording a lossy conversion as lossless."
        )
    if fractional:
        samples = [repr(v) for v in fractional[:5]]
        raise SchemaError(
            f"Column {column!r}: {len(fractional)} value(s) are not whole "
            f"numbers, e.g. {samples}. Volume is a count of units traded; "
            "refusing to round or truncate a fractional value."
        )

    result = pd.Series(values, index=series.index, dtype="float64")
    result[was_null] = float("nan")
    return result


def _dtype_transformation_code(column: str) -> str:
    return f"DTYPE_CONVERTED_{column.upper()}"


def canonicalise(frame: pd.DataFrame) -> CanonicalisationResult:
    """Canonicalise ``frame``, returning the result AND the evidence.

    Permitted: column selection/ordering (when the input already carries
    exactly the canonical columns, in any order), dtype/representation
    conversion that preserves the value exactly (a NORMAL TRANSFORMATION,
    recorded but never a defect), timezone conversion that preserves the
    instant, and deterministic stable sorting by timestamp.

    Forbidden, and enforced here: silently dropping unknown columns, replacing
    missing values, fabricating volume, interpolating prices, dropping rows,
    coercing unparseable source values into NaN, or resolving a market-value
    conflict by choosing between the candidates. All of that is left to a
    later validation/cleaning stage; this function only makes the container
    canonical and records exactly what it had to do to get there.

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

    # --- SOURCE structural evidence ---------------------------------------
    # Everything in this block is measured on `out` EXACTLY as received: no
    # timezone conversion, no numeric/dtype coercion has happened yet. These
    # are STRUCTURAL facts (ordering, row/timestamp repetition) -- they never
    # depend on whether values later turn out to be canonically equal.
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

    duplicate_timestamp_row_count = int(out[TS].duplicated(keep=False).sum())
    if duplicate_timestamp_row_count:
        anomalies.append(
            CanonicalisationAnomaly(
                "SOURCE_DUPLICATE_TIMESTAMPS",
                AnomalySeverity.INFO,
                f"{duplicate_timestamp_row_count} row(s) share a timestamp "
                "with at least one other row, as received. This records the "
                "structural fact only; it does not say whether the shared "
                "rows agree on OHLCV values -- see "
                "CANONICAL_CONFLICTING_TIMESTAMPS for that.",
            )
        )
    # --- end SOURCE structural evidence -----------------------------------

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

    # --- lossless dtype/representation conversion -------------------------
    # Frozen architecture section 14: a lossless dtype conversion is a NORMAL
    # TRANSFORMATION, not a defect, and must be recorded when it actually
    # happens -- and NOT recorded when the source already had the canonical
    # dtype, so this stays useful signal rather than noise on every call.
    for col in PRICE_COLUMNS:
        source_dtype = out[col].dtype
        out[col] = _convert_numeric_exact(out[col], col, whole_numbers_only=False)
        if str(source_dtype) != "float64":
            transformations.append(
                CanonicalisationTransformation(
                    _dtype_transformation_code(col),
                    f"Column {col!r}: losslessly converted from {source_dtype} "
                    "to float64. This is a representation change, not a value "
                    "change -- the numeric value is preserved exactly.",
                )
            )

    volume_source_dtype = out[VOLUME].dtype
    out[VOLUME] = _convert_numeric_exact(
        out[VOLUME], VOLUME, whole_numbers_only=True
    ).astype("Int64")
    if str(volume_source_dtype) != "Int64":
        transformations.append(
            CanonicalisationTransformation(
                _dtype_transformation_code(VOLUME),
                f"Column {VOLUME!r}: losslessly converted from "
                f"{volume_source_dtype} to Int64. This is a representation "
                "change, not a value change -- every value was already a "
                "whole number.",
            )
        )
    # --- end lossless dtype/representation conversion ----------------------

    # --- CANONICAL market-value conflict -----------------------------------
    # Deliberately measured HERE, after timezone conversion and numeric/dtype
    # coercion: this must compare CANONICAL OHLCV values, not source
    # representations. Two rows that differ only in how they represented an
    # identical value (the int 1 vs the string "1") are NOT a conflict once
    # both have been losslessly converted to the same float64 -- they are
    # only a SOURCE_DUPLICATE_TIMESTAMPS fact, recorded above. This is the
    # manager-mandated separation of "source representation differs" from
    # "canonical market value differs".
    ts_dup_mask = out[TS].duplicated(keep=False)
    if ts_dup_mask.any():
        conflicting = False
        for _, group in out.loc[ts_dup_mask].groupby(TS):
            # How many DISTINCT canonical observations share this timestamp?
            # More than one is a conflict, regardless of how many copies of
            # each exist. (Regression guard: "duplicated(keep=False).sum() <
            # len(group)" is WRONG here -- it fails on A,A,B,B, where every
            # row has a matching twin so sum()==len(group)==4 despite two
            # distinct observations being present.)
            distinct_observations = len(group) - int(
                group.duplicated(keep="first").sum()
            )
            if distinct_observations > 1:
                conflicting = True
                break
        if conflicting:
            anomalies.append(
                CanonicalisationAnomaly(
                    "CANONICAL_CONFLICTING_TIMESTAMPS",
                    AnomalySeverity.BLOCKER,
                    "Two or more DISTINCT canonical OHLCV observations share a "
                    "timestamp, after all lossless conversions. Canonicalisation "
                    "does not choose between contradictory observations; all "
                    "are preserved.",
                )
            )
    # --- end CANONICAL market-value conflict --------------------------------

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
        duplicate_timestamp_row_count=duplicate_timestamp_row_count,
    )

    return CanonicalisationResult(
        frame=out,
        transformations=tuple(transformations),
        source_anomalies=tuple(anomalies),
        source=source,
    )


def _validate_epoch(value: object) -> int:
    """Validate and return a single FYERS row's raw epoch value as an exact
    Python ``int``. Raises ``SchemaError`` -- never ``TypeError``,
    ``ValueError`` from pandas internals, or any other implementation
    exception -- for anything outside the accepted adapter contract.

    Accepted contract: an integer count of whole seconds since the Unix
    epoch (matching the FYERS SDK docstring's "epoch value" description).
    Rejected: booleans; non-finite values (NaN, +/-inf); fractional seconds
    (no rounding/truncation policy has been decided, and no observed payload
    has ever needed one); non-numeric strings; and any type this project has
    no defined interpretation for (e.g. a nested list). Validating and
    normalising every epoch to a plain ``int`` HERE, before anything else
    touches it, is what prevents a malformed value from later reaching a raw
    Python comparison (``a > b``) or pandas conversion and leaking a
    ``TypeError`` instead of a diagnosed ``SchemaError``.
    """
    if _is_bool_like(value):
        raise SchemaError(
            f"FYERS epoch value {value!r} is boolean; a bool is not a valid "
            "epoch timestamp."
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            raise SchemaError(
                f"FYERS epoch value {value!r} is non-finite; an epoch must be "
                "a finite whole number of seconds."
            )
        if not f.is_integer():
            raise SchemaError(
                f"FYERS epoch value {value!r} is not a whole number of "
                "seconds; fractional epoch seconds are not part of the "
                "accepted adapter contract."
            )
        return int(f)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise SchemaError(
                f"FYERS epoch value {value!r} is not parseable as an integer."
            ) from exc
    raise SchemaError(
        f"FYERS epoch value {value!r} has unsupported type "
        f"{type(value).__name__}; expected an integer count of seconds."
    )


def _validate_fyers_epochs(candles: list[list]) -> list[list]:
    """Validate every row's epoch (position 0), returning a NEW list of rows
    with the epoch replaced by its validated, normalised ``int``. All other
    positions are passed through unchanged.

    Every offending row is collected and reported together in one
    ``SchemaError`` (consistent with this module's other batched-diagnostic
    style), rather than raising on the first bad row and hiding the rest.
    """
    validated: list[list] = []
    errors: list[str] = []
    for row in candles:
        try:
            epoch = _validate_epoch(row[0])
        except SchemaError as exc:
            errors.append(str(exc))
            continue
        validated.append([epoch, *row[1:]])

    if errors:
        raise SchemaError(
            f"{len(errors)} FYERS row(s) have an invalid epoch value. "
            + " | ".join(errors[:5])
        )
    return validated


def _adapter_source_evidence(
    candles: list[list],
) -> tuple[SourceEvidence, tuple[CanonicalisationAnomaly, ...]]:
    """Structural evidence for a raw FYERS positional payload, computed
    strictly BEFORE epoch-to-Timestamp conversion or DataFrame construction.
    Operates on the Python ``list[list]`` directly.

    ``candles`` here must already have passed ``_validate_fyers_epochs`` --
    this function assumes every ``row[0]`` is a plain validated ``int`` and
    performs no further validation itself, so that ordering comparisons and
    duplicate counting never encounter a value that could raise.

    This is intentionally a separate, simpler implementation from the
    pandas-vectorised logic in ``canonicalise()``: it measures a genuinely
    different, EARLIER boundary (the raw adapter input), not the DataFrame
    canonicalise() eventually receives. Only Python objects are available
    here -- there is no claim about raw HTTP transport bytes.
    """
    row_count = len(candles)
    epochs = [row[0] for row in candles]

    descending_adjacent_pairs = sum(
        1 for a, b in zip(epochs, epochs[1:]) if a > b
    )
    timestamps_sorted = descending_adjacent_pairs == 0

    epoch_counts = Counter(epochs)
    duplicate_timestamp_row_count = sum(c for c in epoch_counts.values() if c > 1)

    seen: set[tuple] = set()
    exact_duplicate_row_count = 0
    for row in candles:
        key = tuple(row)
        if key in seen:
            exact_duplicate_row_count += 1
        else:
            seen.add(key)

    anomalies: list[CanonicalisationAnomaly] = []
    if not timestamps_sorted:
        anomalies.append(
            CanonicalisationAnomaly(
                "SOURCE_UNSORTED",
                AnomalySeverity.INFO,
                f"Adapter input rows were not in ascending epoch order "
                f"({descending_adjacent_pairs} descending adjacent pair(s)), "
                "measured on the raw epoch values before any conversion. The "
                "canonical output is stable-sorted.",
            )
        )
    if exact_duplicate_row_count:
        anomalies.append(
            CanonicalisationAnomaly(
                "SOURCE_EXACT_DUPLICATE_ROWS",
                AnomalySeverity.WARNING,
                f"{exact_duplicate_row_count} adapter row(s) are exact repeats "
                "(identical positional values, as received) of an earlier row.",
            )
        )
    if duplicate_timestamp_row_count:
        anomalies.append(
            CanonicalisationAnomaly(
                "SOURCE_DUPLICATE_TIMESTAMPS",
                AnomalySeverity.INFO,
                f"{duplicate_timestamp_row_count} adapter row(s) share a raw "
                "epoch value with at least one other row. Structural fact "
                "only; see CANONICAL_CONFLICTING_TIMESTAMPS for whether the "
                "canonical OHLCV values actually disagree.",
            )
        )

    source = SourceEvidence(
        row_count=row_count,
        column_inventory=OHLCV_COLUMNS,
        timestamps_sorted=timestamps_sorted,
        descending_adjacent_pairs=descending_adjacent_pairs,
        exact_duplicate_row_count=exact_duplicate_row_count,
        duplicate_timestamp_row_count=duplicate_timestamp_row_count,
    )
    return source, tuple(anomalies)


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

    The returned ``source`` evidence is measured on the RAW ``candles`` list
    -- before epoch-to-Timestamp conversion, before any DataFrame exists --
    via ``_adapter_source_evidence()``. It is NOT the ``source`` that the
    inner ``canonicalise()`` call computes internally, because by the time
    that call runs, the timestamp has already been epoch-converted to IST;
    treating that as "the source" would describe the payload one stage later
    than it actually arrived. Only ``canonicalise()``'s CANONICAL-level
    finding (``CANONICAL_CONFLICTING_TIMESTAMPS``, if present) is carried
    through, since that fact is genuinely about the final canonical values
    and is unaffected by which stage computed it.
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
                duplicate_timestamp_row_count=0,
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

    # Validate and normalise every epoch to a plain int BEFORE anything else
    # touches it (ordering comparisons, duplicate counting, DataFrame
    # construction). This is what prevents a malformed epoch -- a string, a
    # bool, NaN, a nested list -- from reaching a raw Python `a > b`
    # comparison or a pandas conversion and leaking a TypeError/ValueError
    # instead of a diagnosed SchemaError.
    candles = _validate_fyers_epochs(candles)

    adapter_source, adapter_anomalies = _adapter_source_evidence(candles)

    frame = pd.DataFrame(candles, columns=list(OHLCV_COLUMNS))
    frame[TS] = epoch_series_to_ist(frame[TS])

    inner = canonicalise(frame)

    epoch_transformation = CanonicalisationTransformation(
        "FYERS_EPOCH_TO_IST",
        "Interpreted the source 'ts' field as Unix epoch seconds (UTC by "
        "definition), then converted through UTC to Asia/Kolkata, preserving "
        "the instant. This is the FYERS positional-payload adapter contract "
        "(ADAPTER-CONTRACT-VALID; not yet verified against a live response).",
    )

    # Keep only the CANONICAL-level finding from the inner call. Its own
    # SOURCE-level anomalies (SOURCE_UNSORTED / SOURCE_EXACT_DUPLICATE_ROWS /
    # SOURCE_DUPLICATE_TIMESTAMPS) describe the post-epoch-conversion frame,
    # which is not the true adapter source for this function -- those are
    # replaced by `adapter_anomalies`, computed above on the raw payload.
    canonical_level_anomalies = tuple(
        a for a in inner.source_anomalies if a.code == "CANONICAL_CONFLICTING_TIMESTAMPS"
    )

    return CanonicalisationResult(
        frame=inner.frame,
        transformations=(epoch_transformation, *inner.transformations),
        source_anomalies=adapter_anomalies + canonical_level_anomalies,
        source=adapter_source,
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
