"""Exact canonical dataset identity + digest.

Per the frozen architecture (docs/architecture/phase1-trust-hardening.md,
sections 8.0-8.2), dataset identity binds:

    (MARKET_DATA_SCHEMA_VERSION, source, symbol, resolution,
     canonical observation sequence)

into one SHA-256 digest, over a typed, length-prefixed binary encoding --
never CSV, JSON floats, ``repr()``/``str()``, or delimiter-joined text, all of
which can either lose precision or let one field's content bleed into the
next.

This module computes ONLY that digest. It does not know about Parquet bytes,
request ranges, fetch/validation results, forced state, git revisions, or any
other provenance fact -- those belong to a later provenance-envelope unit
(architecture section 6).
"""

from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from dataclasses import dataclass

import pandas as pd

from marketdata.schemas import (
    MARKET_DATA_SCHEMA_VERSION,
    OHLCV_COLUMNS,
    PRICE_COLUMNS,
    TS,
    VOLUME,
    assert_canonical,
)

# Architecture section 8.2 type tags. Every encoded field is:
#   type_tag (1 byte) || length (8 bytes, big-endian unsigned) || payload
# Fixed-width types (F64/I64/TS) still carry their length prefix, and the
# zero-payload types (NA/NAN/POSINF/NEGINF) carry a zero length -- uniform
# framing, so no special-cased reader is required and no two distinct field
# sequences can ever produce the same byte stream.
_TAG_STR = 0x01
_TAG_F64 = 0x02
_TAG_I64 = 0x03
_TAG_NA = 0x04
_TAG_NAN = 0x05
_TAG_POSINF = 0x06
_TAG_NEGINF = 0x07
_TAG_TS = 0x08


class DatasetIdentityError(ValueError):
    """Raised when dataset identity metadata is invalid."""


def _validate_identity_text(field_name: str, value: object) -> str:
    """Validate one identity-metadata field and return its NFC-normalised form.

    Per architecture section 9, identity metadata (source/symbol/resolution)
    uses the TEXT_NFC policy: NFC normalisation is applied because these are
    our own identifiers, where semantic equality is what we want -- unlike a
    preserved source string column (TEXT_EXACT), none of which exist in
    schema v1.

    NUL (``\\x00``) is rejected outright (manager decision): a broker
    source/symbol/resolution should never need one, and refusing it keeps
    identity text safe to print in logs and error messages without the
    length-framing already making it structurally safe.
    """
    if not isinstance(value, str):
        raise DatasetIdentityError(
            f"{field_name!r} must be a str, got {type(value).__name__}"
        )
    if value == "":
        raise DatasetIdentityError(f"{field_name!r} must not be empty")
    if "\x00" in value:
        raise DatasetIdentityError(
            f"{field_name!r} must not contain a NUL character"
        )
    return unicodedata.normalize("NFC", value)


@dataclass(frozen=True)
class DatasetIdentity:
    """Logical identity metadata for a canonical market-data dataset.

    Deliberately excludes ``schema_version``: ``MARKET_DATA_SCHEMA_VERSION``
    is a frozen global constant (docs/architecture/phase1-trust-hardening.md,
    section 8.0) pulled directly from ``marketdata.schemas`` by
    ``dataset_digest()``, never a value a caller supplies or overrides --
    passing ``schema_version`` to this constructor raises ``TypeError``
    (there is no such field).

    This is logical identity only: no filesystem path/slug behaviour, no
    observations, no credentials.
    """

    source: str
    symbol: str
    resolution: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _validate_identity_text("source", self.source))
        object.__setattr__(self, "symbol", _validate_identity_text("symbol", self.symbol))
        object.__setattr__(
            self, "resolution", _validate_identity_text("resolution", self.resolution)
        )


def _encode_field(tag: int, payload: bytes) -> bytes:
    return bytes([tag]) + struct.pack(">Q", len(payload)) + payload


def _encode_str(value: str) -> bytes:
    return _encode_field(_TAG_STR, value.encode("utf-8"))


def _encode_i64(value: int) -> bytes:
    return _encode_field(_TAG_I64, struct.pack(">q", value))


def _encode_na() -> bytes:
    return _encode_field(_TAG_NA, b"")


def _encode_f64(value: float) -> bytes:
    """Encode one canonical price value.

    NaN and +/-Inf get their own zero-length tags (architecture section 8.2)
    so they can never collide with a finite value's 8-byte payload, and so
    that NaN's arbitrary in-memory payload bits never leak into the digest --
    every NaN, regardless of which of the many IEEE-754 bit patterns it
    happens to carry, encodes identically.

    ``-0.0`` is normalised to ``+0.0`` before encoding (architecture section
    8.2 / section 10): IEEE signed zero is not a market distinction, and a
    zero price is invalid data anyway -- this digest still exists for
    invalid data (section 12), so the two must not silently diverge on sign
    alone.
    """
    if math.isnan(value):
        return _encode_field(_TAG_NAN, b"")
    if math.isinf(value):
        return _encode_field(_TAG_POSINF if value > 0 else _TAG_NEGINF, b"")
    if value == 0.0:
        value = 0.0  # collapse -0.0 to +0.0; the float literal is always +0.0
    return _encode_field(_TAG_F64, struct.pack(">d", value))


def _encode_ts_ns(epoch_ns: int) -> bytes:
    """Encode a canonical timestamp as signed Unix nanoseconds (UTC).

    ``epoch_ns`` is pandas' internal UTC-based nanosecond integer -- the same
    for a given instant regardless of which tz is attached for display, which
    is exactly what makes a UTC-sourced and an IST-sourced representation of
    the same instant hash identically once both have been through
    ``canonicalise()``.
    """
    return _encode_field(_TAG_TS, struct.pack(">q", epoch_ns))


def _encode_observation_values(
    price_columns: dict[str, "np.ndarray"], volume_series: pd.Series, i: int
) -> bytes:
    """Encode one observation's non-timestamp canonical fields (open, high,
    low, close, volume) using the exact same field encoders as everywhere
    else in this module.

    This is the single source of truth for "observation bytes": architecture
    section 8.3 requires that equal-timestamp observations be tie-broken by
    sorting their own canonical encoded bytes, and those must be the SAME
    bytes that end up in the digest -- otherwise the sort key and the hashed
    payload could silently diverge. Returning one concatenated ``bytes``
    keeps that guarantee structural rather than relying on two call sites
    staying in sync by convention.
    """
    chunks = [_encode_f64(float(price_columns[col][i])) for col in PRICE_COLUMNS]
    vol = volume_series.iloc[i]
    if pd.isna(vol):
        chunks.append(_encode_na())
    else:
        chunks.append(_encode_i64(int(vol)))
    return b"".join(chunks)


def _encode_dataset(identity: DatasetIdentity, frame: pd.DataFrame) -> bytes:
    """Build the exact byte stream that ``dataset_digest`` hashes.

    Requires ``frame`` to already be canonical (``assert_canonical``) --
    this primitive never sorts, repairs, or otherwise canonicalises its
    input; that is ``canonicalise()``'s job (marketdata/schemas.py). The
    frame itself is never mutated or reordered.

    Timestamp remains the primary key (architecture section 8.3): rows are
    walked in the frame's own ascending-timestamp order, exactly as before.
    Within one CONTIGUOUS run of rows sharing an identical timestamp --
    ``assert_canonical`` guarantees timestamps are non-decreasing, so equal
    timestamps are always contiguous -- source arrival order is provenance,
    not identity (section 8.3), so those rows are re-ordered for hashing
    purposes only (the DataFrame itself is untouched) by sorting their own
    canonical encoded observation bytes lexicographically. Every occurrence
    is kept: this is a sort, never a deduplication, so ``T:A`` and ``T:A,A``
    still differ. A timestamp with exactly one row takes the trivial
    single-element path, which byte-for-byte matches this function's
    pre-section-8.3 encoding -- ordinary unique-timestamp datasets are
    therefore unaffected.
    """
    assert_canonical(frame)

    parts: list[bytes] = [
        _encode_i64(MARKET_DATA_SCHEMA_VERSION),
        _encode_str(identity.source),
        _encode_str(identity.symbol),
        _encode_str(identity.resolution),
        _encode_i64(len(OHLCV_COLUMNS)),
    ]
    for column_name in OHLCV_COLUMNS:
        parts.append(_encode_str(column_name))

    row_count = len(frame)
    parts.append(_encode_i64(row_count))

    ts_ns = frame[TS].to_numpy(dtype="datetime64[ns]").view("int64")
    price_columns = {col: frame[col].to_numpy(dtype="float64") for col in PRICE_COLUMNS}
    volume_series = frame[VOLUME]

    observation_bytes = [
        _encode_observation_values(price_columns, volume_series, i)
        for i in range(row_count)
    ]

    i = 0
    while i < row_count:
        j = i
        while j + 1 < row_count and ts_ns[j + 1] == ts_ns[i]:
            j += 1
        ts_field = _encode_ts_ns(int(ts_ns[i]))
        if j == i:
            parts.append(ts_field)
            parts.append(observation_bytes[i])
        else:
            # Tie-break ONLY on already-canonical encoded bytes (never
            # repr()/format()/tuple comparison): Python's default bytes
            # ordering is exact lexicographic byte comparison, which is
            # deterministic even where NaN has no total order under `<`.
            for obs in sorted(observation_bytes[i : j + 1]):
                parts.append(ts_field)
                parts.append(obs)
        i = j + 1

    return b"".join(parts)


def dataset_digest(identity: DatasetIdentity, frame: pd.DataFrame) -> str:
    """SHA-256 of the canonical typed encoding of ``identity`` + ``frame``.

    Returns a lowercase 64-character hex digest. Raises ``SchemaError`` if
    ``frame`` is not already canonical (see ``marketdata.schemas.canonicalise``
    to produce one).
    """
    return hashlib.sha256(_encode_dataset(identity, frame)).hexdigest()
