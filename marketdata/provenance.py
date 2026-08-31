"""Provenance envelope + generation integrity, per the frozen architecture
(docs/architecture/phase1-trust-hardening.md, section 6):

    data_digest        = SHA256( canonical encoding of identity + observations )
    provenance_digest  = SHA256( canonical encoding of the provenance envelope )
    generation_id      = uuid.uuid4()
    integrity_id       = SHA256( data_digest || provenance_digest )

``data_digest`` already exists (``marketdata.identity.dataset_digest``, Unit
3). This module implements the other three, plus the envelope itself.

**No storage writes here.** Section 7 (generation identity binds to
location) and section 13 (storage layout, pointer format, atomic write) are
explicitly out of scope for this unit -- this module produces the envelope
and its digests only; nothing here touches a filesystem.

**What the envelope binds**, per section 6's list, reusing a
``ValidatedDataset`` (Unit 5) as the single source of truth for identity and
canonicalisation evidence -- the same reason ``ValidatedDataset.build()``
itself takes a raw frame rather than separately-suppliable evidence:

- ``provenance_schema_version`` (this module's own constant, never
  caller-overridable)
- ``market_data_schema_version`` (``marketdata.schemas.MARKET_DATA_SCHEMA_VERSION``)
- identity: ``source``, ``symbol``, ``resolution`` (``dataset.identity``)
- ``generation_id`` (fresh ``uuid.uuid4()`` unless one is supplied and
  validated) and ``namespace`` (section 7; derived from ``forced``, since no
  filesystem structure exists yet to derive it from independently)
- canonicalisation snapshot (section 14): ``dataset.transformations``,
  ``dataset.source_anomalies``, ``dataset.source_evidence``
- ``dataset.validation_policy`` (``marketdata.dataset.ValidationPolicy``) --
  per manager review of an earlier revision (now reflected in section 6's
  amended text): a validation POLICY is a provenance/config fact about HOW
  data was checked, not a data-derived fact -- unlike ``MarketDataValidity``
  itself (see below), which is recomputed and therefore still NOT bound.
  Two envelopes for the same data + same generation but a different policy
  must therefore share ``data_digest`` while producing a different
  ``provenance_digest``/``integrity_id``.
- acquisition snapshot (section 11): an optional
  ``marketdata.evidence.FetchReportSnapshot`` -- optional because section
  11.1 explicitly allows ``REQUESTS_UNKNOWN`` ("no acquisition evidence
  (fixtures, manual frames)"); this unit does not yet compute
  ``AcquisitionRequestStatus`` from it (that is Unit 9 in the frozen
  architecture's implementation sequence, section 28) -- it binds the raw,
  already-immutable evidence snapshot only. **Consistency-checked**: when
  supplied, ``fetch.symbol``/``fetch.resolution`` must match
  ``dataset.identity`` exactly, or ``build()`` raises -- an earlier revision
  accepted acquisition evidence for a completely different symbol/
  resolution than the dataset it was bound to. This is an identity
  consistency check only; coverage/chunk completeness is explicitly Unit
  9's job, not this one.
- operator declarations (section 10): ``forced``, ``force_reason``
  (required non-empty when ``forced=True``; ``forced`` must be an actual
  ``bool`` -- ``1``, ``"true"``, or any other truthy-but-not-``bool`` value
  is rejected outright rather than coerced)
- environment snapshot: ``core.environment.software_versions()`` (the
  ACTUAL environment only; moved here from ``marketdata.store`` -- see
  below). Section 22's ``environment_expected_digest`` /
  ``ReproducibilityCertification`` machinery is NOT implemented here --
  there is no lock-file-digest concept anywhere in this codebase yet, and
  inventing one to fill this field would be exactly the kind of placeholder
  fact the manager's directive prohibits. This is a deliberately incomplete
  prerequisite, not an oversight; see the module's test/commit notes.

``data_digest`` (``dataset.digest``) is exposed as ``.data_digest`` and
still feeds ``integrity_id``, but is deliberately NOT part of
``_encode_envelope()``/``provenance_digest`` -- an earlier revision included
it there too, conflating DATA identity with PROVENANCE identity: changing
one candle value then changed provenance_digest despite nothing about HOW
the data was acquired/checked/handled actually differing. Keeping them
separate means two generations with identical provenance facts but
different underlying candles get the same ``provenance_digest`` and a
different ``integrity_id`` (via ``data_digest`` alone), which is the
correct signal for "provenance process unchanged, data changed."

**Deliberately NOT bound**: the validation RESULT
(``ValidatedDataset.validation`` / ``MarketDataValidity``). Section 6's
envelope bullet list does not name it, and sections 2/4 classify
``MarketDataValidity`` as a data-derived fact that is RECOMPUTED from stored
data at every use (not integrity-bound provenance the way acquisition or
canonicalisation evidence is, since those describe an external process that
cannot be reconstructed from the final data alone). Binding the RESULT here
would be inventing a requirement the frozen text does not state; the POLICY
(above) is different and is bound.

**No dependency on ``marketdata.store``.** An earlier revision imported
``software_versions`` from ``marketdata.store``, which would create an
import cycle the day a storage unit needs to import ``marketdata.provenance``
(to persist an envelope). The environment/git-version snapshot now lives in
``core.environment`` -- a module below both ``marketdata.store`` and
``marketdata.provenance`` -- moved verbatim, not redesigned; ``store.py``
imports it from there too and its own tests are unaffected.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from datetime import time as _time
from enum import Enum
from types import MappingProxyType
from typing import Any

from core.environment import software_versions
from marketdata.dataset import ValidatedDataset, ValidationPolicy
from marketdata.evidence import ChunkResultSnapshot, FetchReportSnapshot
from marketdata.identity import DatasetIdentity, DatasetIdentityError
from marketdata.schemas import (
    MARKET_DATA_SCHEMA_VERSION,
    AnomalySeverity,
    CanonicalisationAnomaly,
    CanonicalisationTransformation,
    SourceEvidence,
)

# Frozen contract (section 8.0's sibling for provenance, section 6).
# Versioned independently of MARKET_DATA_SCHEMA_VERSION -- acquisition
# fields are expected to evolve on their own timeline. Never a caller
# parameter.
PROVENANCE_SCHEMA_VERSION = 1

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Namespace(str, Enum):
    """Section 7/10. Only these two frozen values exist. Filesystem
    namespace directories (``trusted_generations`` / ``forced_generations``)
    are explicitly out of scope for this unit -- this enum captures the
    OPERATIONAL state only, derived from ``forced`` (section 10: "Force is
    an operational provenance state").
    """

    TRUSTED = "TRUSTED"
    FORCED = "FORCED"


def _validate_sha256_hex(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex "
            f"digest, got {value!r}"
        )
    return value


def _coerce_generation_id(value: object) -> uuid.UUID:
    """``None`` generates a fresh ``uuid.uuid4()`` (section 6/13). A
    supplied value must be an actual version-4 UUID -- a ``uuid.UUID``
    instance or a well-formed UUID string; anything else, or a wrong
    version, is rejected outright.
    """
    if value is None:
        return uuid.uuid4()
    if isinstance(value, uuid.UUID):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError(
                f"generation_id must be a valid UUID4 string, got {value!r}"
            ) from exc
    else:
        raise TypeError(
            "generation_id must be a uuid.UUID, a UUID string, or None, got "
            f"{type(value).__name__}"
        )
    if parsed.version != 4:
        raise ValueError(
            f"generation_id must be a version-4 UUID, got version "
            f"{parsed.version} ({parsed})"
        )
    return parsed


# --- canonical typed encoding (no delimiters, no json.dumps) ----------------
#
# Same design as marketdata/identity.py's field framing (architecture
# section 8.2), generalised to a small recursive encoder so every evidence
# shape bound here (frozen dataclasses, tuples, mappings, enums, UUIDs) goes
# through ONE deterministic path rather than a hand-written serialiser per
# type -- every field is:
#
#     type_tag (1 byte) || length (8 bytes, big-endian unsigned) || payload
#
# Field NAMES are encoded alongside values (both for the top-level envelope
# and for every nested dataclass, via dataclasses.fields() in their
# unchanging class-body declaration order), so the encoding is self-
# describing and stable-field-order by construction, not by convention.

_TAG_STR = 0x01
_TAG_I64 = 0x02
_TAG_BOOL = 0x03
_TAG_NA = 0x04
_TAG_SEQ = 0x05
_TAG_UUID = 0x06
_TAG_F64 = 0x07
_TAG_NAN = 0x08
_TAG_POSINF = 0x09
_TAG_NEGINF = 0x0A


def _encode_field(tag: int, payload: bytes) -> bytes:
    return bytes([tag]) + struct.pack(">Q", len(payload)) + payload


def _encode_str(value: str) -> bytes:
    return _encode_field(_TAG_STR, value.encode("utf-8"))


def _encode_i64(value: int) -> bytes:
    return _encode_field(_TAG_I64, struct.pack(">q", value))


def _encode_value(value: Any) -> bytes:
    """Recursively encode one Python value into the typed framing above.

    Unicode policy: identity metadata (source/symbol/resolution) is already
    NFC-normalised by ``DatasetIdentity`` itself (architecture section 9)
    before it ever reaches this function; every other string encoded here
    (git revision, error messages, force_reason, ...) is TEXT_EXACT -- byte-
    exact UTF-8, never normalised -- because none of it is identity
    metadata this project owns the semantics of.

    ``bool`` is checked before ``int`` (``bool`` is an ``int`` subclass);
    ``str``-backed ``Enum`` members are checked as plain ``str`` deliberately
    -- every enum used in this envelope (``Namespace``, ``AnomalySeverity``)
    subclasses ``str``, so a member already equals its own ``.value`` and
    encodes identically either way; no separate Enum branch is needed.
    """
    if value is None:
        return _encode_field(_TAG_NA, b"")
    if isinstance(value, bool):
        return _encode_field(_TAG_BOOL, bytes([1 if value else 0]))
    if isinstance(value, uuid.UUID):
        return _encode_field(_TAG_UUID, value.bytes)
    if isinstance(value, _time):
        # ValidationPolicy.session_window elements. ISO-8601 ("HH:MM:SS[.ffffff]")
        # is a lossless, unambiguous text representation of a bare time-of-day.
        return _encode_str(value.isoformat())
    if isinstance(value, int):
        return _encode_i64(value)
    if isinstance(value, float):
        # ValidationPolicy.sigma_threshold/max_session_gap_days. Same
        # normalisation policy as marketdata/identity.py's price encoding:
        # NaN/+-Inf get dedicated zero-length tags (ValidationPolicy itself
        # already forbids them, but this encoder is reused generically, so
        # it does not silently rely on that upstream guarantee), and -0.0
        # collapses to +0.0.
        if value != value:  # NaN
            return _encode_field(_TAG_NAN, b"")
        if value == float("inf"):
            return _encode_field(_TAG_POSINF, b"")
        if value == float("-inf"):
            return _encode_field(_TAG_NEGINF, b"")
        if value == 0.0:
            value = 0.0
        return _encode_field(_TAG_F64, struct.pack(">d", value))
    if isinstance(value, str):
        return _encode_str(value)
    if isinstance(value, (tuple, list)):
        parts = [_encode_i64(len(value))]
        parts.extend(_encode_value(v) for v in value)
        return _encode_field(_TAG_SEQ, b"".join(parts))
    if isinstance(value, Mapping):
        # Deterministic regardless of the mapping's own iteration order.
        parts = [_encode_i64(len(value))]
        for key in sorted(value.keys()):
            parts.append(_encode_str(str(key)))
            parts.append(_encode_value(value[key]))
        return _encode_field(_TAG_SEQ, b"".join(parts))
    if is_dataclass(value) and not isinstance(value, type):
        field_list = dataclass_fields(value)
        parts = [_encode_i64(len(field_list))]
        for f in field_list:
            parts.append(_encode_str(f.name))
            parts.append(_encode_value(getattr(value, f.name)))
        return _encode_field(_TAG_SEQ, b"".join(parts))
    raise TypeError(
        f"Cannot encode value of type {type(value).__name__!r} into the "
        "provenance envelope; no unambiguous typed representation is "
        "defined for it."
    )


def _to_jsonable(value: Any) -> Any:
    """Recursively convert one Python value into a JSON-safe structure.

    Same shape-coverage as ``_encode_value`` (dataclasses via
    ``dataclasses.fields()`` in their unchanging declaration order,
    tuples/lists, mappings sorted by key, enums, UUIDs, ``datetime.time``),
    but targeting JSON primitives instead of binary framing -- this is what
    ``ProvenanceEnvelope.to_manifest_dict()``/``to_manifest_json()`` use to
    persist a generation's provenance envelope losslessly enough for a
    later unit to reconstruct and re-verify it, per manager direction for
    Unit 8 (generation storage): "manifest must contain the fields required
    to recompute provenance_digest/integrity_id/...". Field NAMES are
    included for every dataclass, matching ``_encode_value``'s self-
    describing structure.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _time):
        return value.isoformat()
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {
            str(k): _to_jsonable(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclass_fields(value)}
    raise TypeError(
        f"Cannot convert value of type {type(value).__name__!r} into a "
        "JSON-safe manifest value; no representation is defined for it."
    )


def _encode_provenance_fields(
    *,
    provenance_schema_version: int,
    market_data_schema_version: int,
    source: str,
    symbol: str,
    resolution: str,
    generation_id: uuid.UUID,
    namespace: "Namespace",
    transformations: tuple,
    source_anomalies: tuple,
    source_evidence: SourceEvidence,
    validation_policy: ValidationPolicy,
    fetch: FetchReportSnapshot | None,
    forced: bool,
    force_reason: str | None,
    software: MappingProxyType,
) -> bytes:
    """The exact byte stream a provenance envelope's ``provenance_digest``
    hashes -- shared between ``ProvenanceEnvelope._encode_envelope`` (build
    time, live values) and ``ReconstructedManifest.recompute_provenance_digest``
    (read time, values reconstructed from a persisted manifest). A single
    encoding path means the reader can never silently drift from what
    ``build()`` actually hashed.

    Deliberately excludes ``data_digest`` -- see the module docstring's
    "DATA identity vs PROVENANCE identity" note.
    """
    ordered_fields = (
        ("provenance_schema_version", provenance_schema_version),
        ("market_data_schema_version", market_data_schema_version),
        ("source", source),
        ("symbol", symbol),
        ("resolution", resolution),
        ("generation_id", generation_id),
        ("namespace", namespace),
        ("transformations", transformations),
        ("source_anomalies", source_anomalies),
        ("source_evidence", source_evidence),
        ("validation_policy", validation_policy),
        ("fetch", fetch),
        ("forced", forced),
        ("force_reason", force_reason),
        ("software", software),
    )
    parts = [_encode_i64(len(ordered_fields))]
    for name, value in ordered_fields:
        parts.append(_encode_str(name))
        parts.append(_encode_value(value))
    return b"".join(parts)


class ProvenanceEnvelope:
    """Immutable provenance envelope bound to one ``ValidatedDataset``.

    Construct ONLY via :meth:`build` -- mirrors ``ValidatedDataset``'s own
    hand-rolled ``__slots__`` class shape (direct instantiation and
    attribute assignment both raise) for the same reason: a plain
    dataclass's public constructor would let a caller assemble an
    internally-inconsistent envelope (e.g. a ``data_digest`` bound to a
    ``ValidatedDataset`` it was never actually computed from) directly.
    """

    __slots__ = (
        "_provenance_schema_version",
        "_market_data_schema_version",
        "_source",
        "_symbol",
        "_resolution",
        "_generation_id",
        "_namespace",
        "_data_digest",
        "_transformations",
        "_source_anomalies",
        "_source_evidence",
        "_validation_policy",
        "_fetch",
        "_forced",
        "_force_reason",
        "_software",
        "_provenance_digest",
        "_integrity_id",
    )

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "ProvenanceEnvelope cannot be constructed directly; use "
            "ProvenanceEnvelope.build(dataset, ...) instead."
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ProvenanceEnvelope is immutable.")

    def __reduce__(self):
        raise TypeError(
            "ProvenanceEnvelope cannot be pickled: a revived instance would "
            "carry evidence that was never re-verified in this process."
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ProvenanceEnvelope(generation_id={self._generation_id!s}, "
            f"namespace={self._namespace.value}, "
            f"integrity_id={self._integrity_id!r})"
        )

    @classmethod
    def build(
        cls,
        dataset: ValidatedDataset,
        *,
        forced: bool = False,
        force_reason: str | None = None,
        fetch: FetchReportSnapshot | None = None,
        generation_id: uuid.UUID | str | None = None,
    ) -> "ProvenanceEnvelope":
        """Build a ``ProvenanceEnvelope`` bound to ``dataset``.

        ``dataset`` must be an actual ``ValidatedDataset`` instance (no
        duck-typed lookalike -- same rejection ``ValidatedDataset.build()``
        already applies to ``validation_policy``). Its ``.identity``,
        ``.digest``, ``.transformations``, ``.source_anomalies`` and
        ``.source_evidence`` are the ONLY source of canonicalisation/data
        evidence bound here; there is no separate parameter through which
        mismatched evidence could be supplied.

        ``forced=True`` requires a non-empty ``force_reason`` (section 10).
        ``namespace`` is derived from ``forced`` (``FORCED`` vs
        ``TRUSTED``) -- there is no separate ``namespace`` parameter,
        because an independently-supplied boolean/enum here would be
        exactly the "editable state" section 10 warns against; the
        filesystem-structural binding that makes namespace non-editable in
        the full design is explicitly future work (section 7/13, not this
        unit).

        ``fetch``, if given, must be an actual ``FetchReportSnapshot`` whose
        ``symbol``/``resolution`` match ``dataset.identity`` exactly --
        acquisition evidence for a different instrument/resolution than the
        dataset it is being bound to is rejected as a provenance
        consistency error. This is an identity check only: chunk/coverage
        completeness is Unit 9's job, not this one.

        ``forced`` must be an actual ``bool`` -- ``1``, ``0``, ``"true"``,
        ``[]`` and other truthy-but-not-``bool`` values are all rejected
        rather than silently coerced via ``bool(forced)``.

        ``generation_id`` defaults to a fresh ``uuid.uuid4()``; if supplied,
        it is validated as an actual version-4 UUID.
        """
        if not isinstance(dataset, ValidatedDataset):
            raise TypeError(
                "dataset must be an actual ValidatedDataset instance, got "
                f"{type(dataset).__name__}."
            )
        if fetch is not None and not isinstance(fetch, FetchReportSnapshot):
            raise TypeError(
                "fetch must be None or an actual FetchReportSnapshot "
                f"instance, got {type(fetch).__name__}."
            )
        if fetch is not None and (
            fetch.symbol != dataset.identity.symbol
            or fetch.resolution != dataset.identity.resolution
        ):
            raise ValueError(
                "Refusing to build a ProvenanceEnvelope: fetch evidence "
                f"describes symbol={fetch.symbol!r} resolution="
                f"{fetch.resolution!r}, but dataset.identity is symbol="
                f"{dataset.identity.symbol!r} resolution="
                f"{dataset.identity.resolution!r}. Acquisition evidence "
                "must describe the exact instrument/resolution it is bound to."
            )
        if not isinstance(forced, bool):
            raise TypeError(
                f"forced must be an actual bool, got {forced!r} "
                f"({type(forced).__name__}). Truthy-but-not-bool values "
                "(1, \"true\", [...], ...) are rejected rather than "
                "silently coerced."
            )
        if forced and not (isinstance(force_reason, str) and force_reason.strip()):
            raise ValueError(
                "forced=True requires a non-empty force_reason (frozen "
                "architecture section 10)."
            )
        if not forced and force_reason is not None:
            raise ValueError(
                "force_reason must be None when forced=False -- a reason "
                "with no force is not a coherent operator declaration."
            )

        resolved_generation_id = _coerce_generation_id(generation_id)
        namespace = Namespace.FORCED if forced else Namespace.TRUSTED
        data_digest = _validate_sha256_hex(dataset.digest, "dataset.digest")
        software = MappingProxyType(dict(software_versions()))

        self = object.__new__(cls)
        object.__setattr__(
            self, "_provenance_schema_version", PROVENANCE_SCHEMA_VERSION
        )
        object.__setattr__(
            self, "_market_data_schema_version", MARKET_DATA_SCHEMA_VERSION
        )
        object.__setattr__(self, "_source", dataset.identity.source)
        object.__setattr__(self, "_symbol", dataset.identity.symbol)
        object.__setattr__(self, "_resolution", dataset.identity.resolution)
        object.__setattr__(self, "_generation_id", resolved_generation_id)
        object.__setattr__(self, "_namespace", namespace)
        object.__setattr__(self, "_data_digest", data_digest)
        object.__setattr__(self, "_transformations", dataset.transformations)
        object.__setattr__(self, "_source_anomalies", dataset.source_anomalies)
        object.__setattr__(self, "_source_evidence", dataset.source_evidence)
        object.__setattr__(self, "_validation_policy", dataset.validation_policy)
        object.__setattr__(self, "_fetch", fetch)
        object.__setattr__(self, "_forced", forced)
        object.__setattr__(self, "_force_reason", force_reason)
        object.__setattr__(self, "_software", software)

        provenance_digest = hashlib.sha256(self._encode_envelope()).hexdigest()
        object.__setattr__(self, "_provenance_digest", provenance_digest)

        integrity_id = hashlib.sha256(
            bytes.fromhex(data_digest) + bytes.fromhex(provenance_digest)
        ).hexdigest()
        object.__setattr__(self, "_integrity_id", integrity_id)

        return self

    def _encode_envelope(self) -> bytes:
        """The exact byte stream ``provenance_digest`` hashes.

        Deliberately excludes ``data_digest`` -- see the module docstring's
        "DATA identity vs PROVENANCE identity" note. ``data_digest`` is
        still bound into ``integrity_id`` separately (raw bytes, in
        ``build()``), so tampering with it is still detected; it just no
        longer changes ``provenance_digest`` itself.
        """
        return _encode_provenance_fields(
            provenance_schema_version=self._provenance_schema_version,
            market_data_schema_version=self._market_data_schema_version,
            source=self._source,
            symbol=self._symbol,
            resolution=self._resolution,
            generation_id=self._generation_id,
            namespace=self._namespace,
            transformations=self._transformations,
            source_anomalies=self._source_anomalies,
            source_evidence=self._source_evidence,
            validation_policy=self._validation_policy,
            fetch=self._fetch,
            forced=self._forced,
            force_reason=self._force_reason,
            software=self._software,
        )

    # -- public, read-only surface -----------------------------------------

    @property
    def provenance_schema_version(self) -> int:
        return self._provenance_schema_version

    @property
    def market_data_schema_version(self) -> int:
        return self._market_data_schema_version

    @property
    def source(self) -> str:
        return self._source

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def resolution(self) -> str:
        return self._resolution

    @property
    def generation_id(self) -> uuid.UUID:
        return self._generation_id

    @property
    def namespace(self) -> Namespace:
        return self._namespace

    @property
    def data_digest(self) -> str:
        return self._data_digest

    @property
    def transformations(self):
        return self._transformations

    @property
    def source_anomalies(self):
        return self._source_anomalies

    @property
    def source_evidence(self):
        return self._source_evidence

    @property
    def validation_policy(self) -> ValidationPolicy:
        return self._validation_policy

    @property
    def fetch(self) -> FetchReportSnapshot | None:
        return self._fetch

    @property
    def forced(self) -> bool:
        return self._forced

    @property
    def force_reason(self) -> str | None:
        return self._force_reason

    @property
    def software(self) -> MappingProxyType:
        return self._software

    @property
    def provenance_digest(self) -> str:
        return self._provenance_digest

    @property
    def integrity_id(self) -> str:
        return self._integrity_id

    # -- manifest persistence (Unit 8: marketdata/generation_store.py) -----

    def to_manifest_dict(self) -> dict:
        """A JSON-safe ``dict`` capturing every field this envelope binds,
        losslessly enough for a later unit to reconstruct and re-verify
        ``provenance_digest``/``integrity_id`` from the persisted manifest
        alone. Includes the digests themselves (for direct cross-check
        without recomputation) as well as every field ``provenance_digest``
        was actually computed from.

        This is NOT the same encoding ``_encode_envelope()`` produces
        (that one is binary, framed, and excludes ``data_digest`` on
        purpose -- see the module docstring); this is a separate,
        JSON-oriented serialisation whose only job is faithful persistence.
        """
        return {
            "provenance_schema_version": self._provenance_schema_version,
            "market_data_schema_version": self._market_data_schema_version,
            "source": self._source,
            "symbol": self._symbol,
            "resolution": self._resolution,
            "generation_id": str(self._generation_id),
            "namespace": self._namespace.value,
            "data_digest": self._data_digest,
            "transformations": [_to_jsonable(t) for t in self._transformations],
            "source_anomalies": [_to_jsonable(a) for a in self._source_anomalies],
            "source_evidence": _to_jsonable(self._source_evidence),
            "validation_policy": _to_jsonable(self._validation_policy),
            "fetch": _to_jsonable(self._fetch) if self._fetch is not None else None,
            "forced": self._forced,
            "force_reason": self._force_reason,
            "software": _to_jsonable(self._software),
            "provenance_digest": self._provenance_digest,
            "integrity_id": self._integrity_id,
        }

    def to_manifest_json(self) -> str:
        """Canonical JSON of :meth:`to_manifest_dict`: sorted keys, compact
        separators, deterministic for a fixed envelope.
        """
        return json.dumps(self.to_manifest_dict(), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Strict manifest reconstruction (frozen architecture section 12: read_trusted
# needs to reconstruct and re-verify a persisted manifest, not merely
# serialise one).
#
# ``ReconstructedManifest.from_manifest_json()`` performs STRUCTURAL
# validation only -- malformed JSON, duplicate keys (at any nesting level),
# unknown/missing fields, and wrong field types are all rejected, and every
# nested evidence shape (transformations, source_anomalies, source_evidence,
# ValidationPolicy, FetchReportSnapshot/ChunkResultSnapshot) is reconstructed
# into the exact same immutable types ``ProvenanceEnvelope`` bound at build
# time. It deliberately does NOT compare the manifest's own
# ``provenance_digest``/``integrity_id`` fields against a recomputation --
# that is a TRUST decision belonging to the caller
# (``marketdata.trusted_reader``), via ``recompute_provenance_digest()``/
# ``recompute_integrity_id()``. Separating the two means a forensic/
# unverified read can still inspect a manifest whose stored digests do not
# match its own contents, instead of being unable to parse it at all.
# ---------------------------------------------------------------------------


class ManifestError(ValueError):
    """Raised when a persisted manifest is malformed or internally
    inconsistent (wrong shape, unknown/missing field, wrong type) -- never
    raised for a digest MISMATCH, which is a trust judgement the caller
    makes via :meth:`ReconstructedManifest.recompute_provenance_digest` /
    :meth:`ReconstructedManifest.recompute_integrity_id`.
    """


_MANIFEST_FIELDS = frozenset(
    {
        "provenance_schema_version",
        "market_data_schema_version",
        "source",
        "symbol",
        "resolution",
        "generation_id",
        "namespace",
        "data_digest",
        "transformations",
        "source_anomalies",
        "source_evidence",
        "validation_policy",
        "fetch",
        "forced",
        "force_reason",
        "software",
        "provenance_digest",
        "integrity_id",
    }
)
_TRANSFORMATION_FIELDS = frozenset({"code", "description"})
_ANOMALY_FIELDS = frozenset({"code", "severity", "description"})
_SOURCE_EVIDENCE_FIELDS = frozenset(
    {
        "row_count",
        "column_inventory",
        "timestamps_sorted",
        "descending_adjacent_pairs",
        "exact_duplicate_row_count",
        "duplicate_timestamp_row_count",
    }
)
_VALIDATION_POLICY_FIELDS = frozenset(
    {"expected_interval_minutes", "sigma_threshold", "session_window", "max_session_gap_days"}
)
_CHUNK_FIELDS = frozenset({"range_from", "range_to", "rows", "ok", "error"})
_FETCH_FIELDS = frozenset(
    {
        "symbol",
        "resolution",
        "requested_from",
        "requested_to",
        "chunks",
        "total_rows",
        "first_ts",
        "last_ts",
        "duplicate_rows_removed",
        "conflicting_timestamps",
    }
)


def _manifest_require_sha256_hex(value: object, field_name: str) -> str:
    try:
        return _validate_sha256_hex(value, field_name)
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc


def _reject_json_constants(token: str) -> float:
    """``parse_constant`` for ``json.loads``: raises on ``NaN``/``Infinity``/
    ``-Infinity`` -- Python's ``json`` module accepts these as a
    non-standard extension, but they are NOT valid JSON, and a numeric
    field silently receiving ``float('nan')`` from a manifest is exactly
    the kind of "malformed persisted data accepted anyway" defect this
    parser exists to close. Applied recursively by ``json.loads`` at every
    nesting level, so a NaN/Infinity buried inside a nested object (e.g.
    ``validation_policy.sigma_threshold``) is caught identically to one at
    the top level.
    """
    raise ManifestError(
        f"manifest JSON contains the non-standard constant {token!r}; "
        "NaN/Infinity/-Infinity are not valid JSON."
    )


def _reject_duplicate_manifest_keys(pairs: list) -> dict:
    """``object_pairs_hook`` for ``json.loads``: raises on any repeated key
    at any object level (applied recursively by ``json.loads`` to every
    nested object -- transformations, anomalies, source_evidence,
    validation_policy, fetch, and each fetch chunk), instead of ``dict()``'s
    default silent last-value-wins behaviour.
    """
    seen: set[str] = set()
    result: dict = {}
    for key, value in pairs:
        if key in seen:
            raise ManifestError(f"manifest JSON has a duplicate key: {key!r}")
        seen.add(key)
        result[key] = value
    return result


def _manifest_require_exact_keys(payload: object, required: frozenset, label: str) -> dict:
    if type(payload) is not dict:
        raise ManifestError(f"{label} must be a JSON object, got {type(payload).__name__}")
    actual = set(payload.keys())
    unknown = actual - required
    missing = required - actual
    if unknown:
        raise ManifestError(f"{label} has unknown field(s): {sorted(unknown)}")
    if missing:
        raise ManifestError(f"{label} is missing field(s): {sorted(missing)}")
    return payload


def _manifest_require_type(value: object, expected_type: type, field_name: str):
    if type(value) is not expected_type:
        raise ManifestError(
            f"manifest field {field_name!r} must be {expected_type.__name__}, got "
            f"{value!r} ({type(value).__name__})"
        )
    return value


def _manifest_require_nonneg_int(value: object, field_name: str) -> int:
    _manifest_require_type(value, int, field_name)
    if value < 0:
        raise ManifestError(f"manifest field {field_name!r} must be >= 0, got {value}")
    return value


def _manifest_require_str_or_none(value: object, field_name: str):
    if value is not None and type(value) is not str:
        raise ManifestError(
            f"manifest field {field_name!r} must be str or null, got "
            f"{value!r} ({type(value).__name__})"
        )
    return value


def _require_canonical_manifest_identity(source: str, symbol: str, resolution: str) -> None:
    """Restore ``DatasetIdentity``'s own construction invariants (str,
    non-empty, no NUL, NFC-normalised) on the manifest's persisted
    source/symbol/resolution -- ``ReconstructedManifest`` previously only
    checked ``type(...) is str``, which allowed manifest identity states
    ``ProvenanceEnvelope.build()`` could never have produced (empty string,
    embedded NUL, non-NFC text).

    Deliberately does NOT normalise the persisted value to make it pass --
    a manifest's identity fields must ALREADY be canonical NFC, exactly as
    ``DatasetIdentity.__post_init__`` would have left them at build time;
    silently normalising here would accept a manifest whose bytes never
    actually matched what ``build()`` wrote.
    """
    try:
        identity = DatasetIdentity(source=source, symbol=symbol, resolution=resolution)
    except DatasetIdentityError as exc:
        raise ManifestError(f"manifest identity is invalid: {exc}") from exc
    if identity.source != source or identity.symbol != symbol or identity.resolution != resolution:
        raise ManifestError(
            "manifest identity is not already NFC-normalised -- a "
            "persisted manifest's source/symbol/resolution must already be "
            "canonical NFC text, exactly as DatasetIdentity would have left "
            "it at build time; this parser never normalises it for you."
        )


def _parse_transformation(payload: object, index: int) -> CanonicalisationTransformation:
    payload = _manifest_require_exact_keys(payload, _TRANSFORMATION_FIELDS, f"transformations[{index}]")
    code = _manifest_require_type(payload["code"], str, f"transformations[{index}].code")
    description = _manifest_require_type(
        payload["description"], str, f"transformations[{index}].description"
    )
    return CanonicalisationTransformation(code=code, description=description)


def _parse_anomaly(payload: object, index: int) -> CanonicalisationAnomaly:
    payload = _manifest_require_exact_keys(payload, _ANOMALY_FIELDS, f"source_anomalies[{index}]")
    code = _manifest_require_type(payload["code"], str, f"source_anomalies[{index}].code")
    severity_raw = payload["severity"]
    valid_severities = {s.value for s in AnomalySeverity}
    if type(severity_raw) is not str or severity_raw not in valid_severities:
        raise ManifestError(
            f"source_anomalies[{index}].severity must be one of "
            f"{sorted(valid_severities)}, got {severity_raw!r}"
        )
    description = _manifest_require_type(
        payload["description"], str, f"source_anomalies[{index}].description"
    )
    return CanonicalisationAnomaly(code=code, severity=AnomalySeverity(severity_raw), description=description)


def _parse_source_evidence(payload: object) -> SourceEvidence:
    payload = _manifest_require_exact_keys(payload, _SOURCE_EVIDENCE_FIELDS, "source_evidence")
    row_count = _manifest_require_nonneg_int(payload["row_count"], "source_evidence.row_count")
    column_inventory_raw = payload["column_inventory"]
    if type(column_inventory_raw) is not list or not all(type(c) is str for c in column_inventory_raw):
        raise ManifestError("source_evidence.column_inventory must be a list of str")
    timestamps_sorted = _manifest_require_type(
        payload["timestamps_sorted"], bool, "source_evidence.timestamps_sorted"
    )
    descending = _manifest_require_nonneg_int(
        payload["descending_adjacent_pairs"], "source_evidence.descending_adjacent_pairs"
    )
    exact_dupes = _manifest_require_nonneg_int(
        payload["exact_duplicate_row_count"], "source_evidence.exact_duplicate_row_count"
    )
    ts_dupes = _manifest_require_nonneg_int(
        payload["duplicate_timestamp_row_count"], "source_evidence.duplicate_timestamp_row_count"
    )

    # Basic invariants real canonicalisation output always satisfies (frozen
    # architecture section 14/8.3) -- a manifest violating one of these is
    # not a coherent SourceEvidence at all, regardless of whether its
    # individual fields each pass their own type/non-negativity check.
    # Deliberately does NOT invent any stronger claim that cannot be proven
    # from these six fields alone (e.g. nothing here reconstructs actual
    # source ordering or representation-level duplicates).
    max_descents = max(row_count - 1, 0)
    if descending > max_descents:
        raise ManifestError(
            f"source_evidence.descending_adjacent_pairs ({descending}) "
            f"exceeds the maximum possible for row_count ({row_count}): "
            f"{max_descents}."
        )
    if exact_dupes > row_count:
        raise ManifestError(
            f"source_evidence.exact_duplicate_row_count ({exact_dupes}) "
            f"exceeds row_count ({row_count})."
        )
    if ts_dupes > row_count:
        raise ManifestError(
            f"source_evidence.duplicate_timestamp_row_count ({ts_dupes}) "
            f"exceeds row_count ({row_count})."
        )
    if exact_dupes > ts_dupes:
        raise ManifestError(
            f"source_evidence.exact_duplicate_row_count ({exact_dupes}) "
            f"exceeds duplicate_timestamp_row_count ({ts_dupes}) -- every "
            "exact duplicate is necessarily also a duplicate timestamp."
        )
    if timestamps_sorted != (descending == 0):
        raise ManifestError(
            f"source_evidence.timestamps_sorted ({timestamps_sorted}) is "
            f"inconsistent with descending_adjacent_pairs ({descending}): "
            "timestamps_sorted must be True if and only if "
            "descending_adjacent_pairs == 0."
        )

    return SourceEvidence(
        row_count=row_count,
        column_inventory=tuple(column_inventory_raw),
        timestamps_sorted=timestamps_sorted,
        descending_adjacent_pairs=descending,
        exact_duplicate_row_count=exact_dupes,
        duplicate_timestamp_row_count=ts_dupes,
    )


def _parse_validation_policy(payload: object) -> ValidationPolicy:
    payload = _manifest_require_exact_keys(payload, _VALIDATION_POLICY_FIELDS, "validation_policy")

    expected_interval = payload["expected_interval_minutes"]
    if expected_interval is not None:
        _manifest_require_type(expected_interval, int, "validation_policy.expected_interval_minutes")

    sigma = payload["sigma_threshold"]
    if type(sigma) not in (int, float):
        raise ManifestError(
            f"validation_policy.sigma_threshold must be a number, got {sigma!r} "
            f"({type(sigma).__name__})"
        )

    session_window_raw = payload["session_window"]
    session_window = None
    if session_window_raw is not None:
        if (
            type(session_window_raw) is not list
            or len(session_window_raw) != 2
            or not all(type(t) is str for t in session_window_raw)
        ):
            raise ManifestError(
                "validation_policy.session_window must be null or a list of "
                f"exactly two ISO time strings, got {session_window_raw!r}"
            )
        try:
            session_window = tuple(_time.fromisoformat(t) for t in session_window_raw)
        except ValueError as exc:
            raise ManifestError(
                f"validation_policy.session_window contains an invalid ISO time: {exc}"
            ) from exc

    max_gap = payload["max_session_gap_days"]
    if max_gap is not None and type(max_gap) not in (int, float):
        raise ManifestError(
            f"validation_policy.max_session_gap_days must be null or a number, "
            f"got {max_gap!r} ({type(max_gap).__name__})"
        )

    # ValidationPolicy.__post_init__ performs its own type/bounds validation
    # and normalisation (int/float coercion, positivity, session_window
    # re-tupling) -- reused here rather than duplicated. Its failures
    # (ValueError/TypeError) are NOT this parser's domain error, so they are
    # caught and re-raised as ManifestError -- manifest parsing has exactly
    # ONE error boundary; a raw ValueError/TypeError escaping here would be
    # a second, inconsistent one.
    try:
        return ValidationPolicy(
            expected_interval_minutes=expected_interval,
            sigma_threshold=sigma,
            session_window=session_window,
            max_session_gap_days=max_gap,
        )
    except (ValueError, TypeError) as exc:
        raise ManifestError(f"validation_policy is invalid: {exc}") from exc


def _parse_chunk(payload: object, index: int) -> ChunkResultSnapshot:
    payload = _manifest_require_exact_keys(payload, _CHUNK_FIELDS, f"fetch.chunks[{index}]")
    range_from = _manifest_require_type(payload["range_from"], str, f"fetch.chunks[{index}].range_from")
    range_to = _manifest_require_type(payload["range_to"], str, f"fetch.chunks[{index}].range_to")
    rows = _manifest_require_nonneg_int(payload["rows"], f"fetch.chunks[{index}].rows")
    ok = _manifest_require_type(payload["ok"], bool, f"fetch.chunks[{index}].ok")
    error = _manifest_require_str_or_none(payload["error"], f"fetch.chunks[{index}].error")
    return ChunkResultSnapshot(range_from=range_from, range_to=range_to, rows=rows, ok=ok, error=error)


def _parse_fetch(payload: object) -> FetchReportSnapshot:
    payload = _manifest_require_exact_keys(payload, _FETCH_FIELDS, "fetch")
    symbol = _manifest_require_type(payload["symbol"], str, "fetch.symbol")
    resolution = _manifest_require_type(payload["resolution"], str, "fetch.resolution")
    requested_from = _manifest_require_type(payload["requested_from"], str, "fetch.requested_from")
    requested_to = _manifest_require_type(payload["requested_to"], str, "fetch.requested_to")

    chunks_raw = payload["chunks"]
    if type(chunks_raw) is not list:
        raise ManifestError("fetch.chunks must be a list")
    chunks = tuple(_parse_chunk(c, i) for i, c in enumerate(chunks_raw))

    total_rows = _manifest_require_nonneg_int(payload["total_rows"], "fetch.total_rows")
    first_ts = _manifest_require_str_or_none(payload["first_ts"], "fetch.first_ts")
    last_ts = _manifest_require_str_or_none(payload["last_ts"], "fetch.last_ts")
    dup_removed = _manifest_require_nonneg_int(
        payload["duplicate_rows_removed"], "fetch.duplicate_rows_removed"
    )
    conflicting = _manifest_require_nonneg_int(
        payload["conflicting_timestamps"], "fetch.conflicting_timestamps"
    )
    return FetchReportSnapshot(
        symbol=symbol,
        resolution=resolution,
        requested_from=requested_from,
        requested_to=requested_to,
        chunks=chunks,
        total_rows=total_rows,
        first_ts=first_ts,
        last_ts=last_ts,
        duplicate_rows_removed=dup_removed,
        conflicting_timestamps=conflicting,
    )


@dataclass(frozen=True, slots=True)
class ReconstructedManifest:
    """A persisted generation manifest, strictly re-parsed and reconstructed
    into the exact same immutable evidence types ``ProvenanceEnvelope`` bound
    at build time. See the module section header above for what
    :meth:`from_manifest_json` does and does not check.
    """

    provenance_schema_version: int
    market_data_schema_version: int
    source: str
    symbol: str
    resolution: str
    generation_id: uuid.UUID
    namespace: Namespace
    data_digest: str
    transformations: tuple
    source_anomalies: tuple
    source_evidence: SourceEvidence
    validation_policy: ValidationPolicy
    fetch: FetchReportSnapshot | None
    forced: bool
    force_reason: str | None
    software: MappingProxyType
    provenance_digest: str
    integrity_id: str

    def recompute_provenance_digest(self) -> str:
        """Recomputed using the SOFTWARE VALUES STORED in this manifest
        (``self.software``), never the live process environment --
        ``ProvenanceEnvelope.build()`` captures the CURRENT environment,
        which would be the wrong thing to hash when re-verifying an
        already-persisted manifest.
        """
        encoded = _encode_provenance_fields(
            provenance_schema_version=self.provenance_schema_version,
            market_data_schema_version=self.market_data_schema_version,
            source=self.source,
            symbol=self.symbol,
            resolution=self.resolution,
            generation_id=self.generation_id,
            namespace=self.namespace,
            transformations=self.transformations,
            source_anomalies=self.source_anomalies,
            source_evidence=self.source_evidence,
            validation_policy=self.validation_policy,
            fetch=self.fetch,
            forced=self.forced,
            force_reason=self.force_reason,
            software=self.software,
        )
        return hashlib.sha256(encoded).hexdigest()

    def recompute_integrity_id(self) -> str:
        """``SHA256(data_digest || recompute_provenance_digest())`` -- uses
        ``self.data_digest`` (the manifest's OWN stored field) combined with
        the FRESHLY recomputed provenance digest, never the manifest's own
        stored ``provenance_digest``.
        """
        provenance_digest = self.recompute_provenance_digest()
        return hashlib.sha256(
            bytes.fromhex(self.data_digest) + bytes.fromhex(provenance_digest)
        ).hexdigest()

    @classmethod
    def from_manifest_json(cls, text: str) -> "ReconstructedManifest":
        """Strict structural parse. Raises ``ManifestError`` for: malformed
        JSON; a duplicate key at any nesting level; any unknown or missing
        field (top-level or nested); a wrong ``provenance_schema_version``/
        ``market_data_schema_version``; any wrong field type (bools are
        never accepted where an int is required, and vice versa); a
        malformed UUID/namespace/digest.

        Does NOT compare ``provenance_digest``/``integrity_id`` against a
        recomputation -- see :meth:`recompute_provenance_digest`.
        """
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_manifest_keys,
                parse_constant=_reject_json_constants,
            )
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc

        payload = _manifest_require_exact_keys(payload, _MANIFEST_FIELDS, "manifest")

        provenance_schema_version = payload["provenance_schema_version"]
        if (
            type(provenance_schema_version) is not int
            or provenance_schema_version != PROVENANCE_SCHEMA_VERSION
        ):
            raise ManifestError(
                "manifest.provenance_schema_version must be exactly "
                f"{PROVENANCE_SCHEMA_VERSION} (as an int), got {provenance_schema_version!r}"
            )
        market_data_schema_version = payload["market_data_schema_version"]
        if (
            type(market_data_schema_version) is not int
            or market_data_schema_version != MARKET_DATA_SCHEMA_VERSION
        ):
            raise ManifestError(
                "manifest.market_data_schema_version must be exactly "
                f"{MARKET_DATA_SCHEMA_VERSION} (as an int), got {market_data_schema_version!r}"
            )

        source = _manifest_require_type(payload["source"], str, "source")
        symbol = _manifest_require_type(payload["symbol"], str, "symbol")
        resolution = _manifest_require_type(payload["resolution"], str, "resolution")
        _require_canonical_manifest_identity(source, symbol, resolution)

        generation_id_raw = payload["generation_id"]
        if type(generation_id_raw) is not str:
            raise ManifestError(
                f"manifest.generation_id must be a str, got {type(generation_id_raw).__name__}"
            )
        try:
            generation_id = uuid.UUID(generation_id_raw)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ManifestError(
                f"manifest.generation_id must be a valid UUID, got {generation_id_raw!r}"
            ) from exc
        if generation_id.version != 4:
            raise ManifestError(
                f"manifest.generation_id must be a version-4 UUID, got version "
                f"{generation_id.version} ({generation_id})"
            )

        namespace_raw = payload["namespace"]
        valid_namespaces = {n.value for n in Namespace}
        if type(namespace_raw) is not str or namespace_raw not in valid_namespaces:
            raise ManifestError(
                f"manifest.namespace must be one of {sorted(valid_namespaces)}, "
                f"got {namespace_raw!r}"
            )
        namespace = Namespace(namespace_raw)

        data_digest = _manifest_require_sha256_hex(payload["data_digest"], "manifest.data_digest")

        transformations_raw = payload["transformations"]
        if type(transformations_raw) is not list:
            raise ManifestError("manifest.transformations must be a list")
        transformations = tuple(
            _parse_transformation(t, i) for i, t in enumerate(transformations_raw)
        )

        anomalies_raw = payload["source_anomalies"]
        if type(anomalies_raw) is not list:
            raise ManifestError("manifest.source_anomalies must be a list")
        source_anomalies = tuple(_parse_anomaly(a, i) for i, a in enumerate(anomalies_raw))

        source_evidence = _parse_source_evidence(payload["source_evidence"])
        validation_policy = _parse_validation_policy(payload["validation_policy"])

        fetch_raw = payload["fetch"]
        fetch = None if fetch_raw is None else _parse_fetch(fetch_raw)

        forced = _manifest_require_type(payload["forced"], bool, "forced")
        force_reason = _manifest_require_str_or_none(payload["force_reason"], "force_reason")

        software_raw = payload["software"]
        if type(software_raw) is not dict or not all(
            type(k) is str and type(v) is str for k, v in software_raw.items()
        ):
            raise ManifestError("manifest.software must be an object of str -> str")
        software = MappingProxyType(dict(software_raw))

        provenance_digest = _manifest_require_sha256_hex(payload["provenance_digest"], "manifest.provenance_digest")
        integrity_id = _manifest_require_sha256_hex(payload["integrity_id"], "manifest.integrity_id")

        # Restore ProvenanceEnvelope.build()'s own invariants -- these are
        # internally IMPOSSIBLE states under build(), so a manifest claiming
        # one is not a coherent persisted envelope at all, regardless of
        # whether its digests happen to be self-consistent.
        if fetch is not None and (fetch.symbol != symbol or fetch.resolution != resolution):
            raise ManifestError(
                f"manifest.fetch describes symbol={fetch.symbol!r} "
                f"resolution={fetch.resolution!r}, but manifest identity is "
                f"symbol={symbol!r} resolution={resolution!r}."
            )
        if namespace is Namespace.TRUSTED and forced is not False:
            raise ManifestError(
                "manifest.namespace is TRUSTED but manifest.forced is not "
                "False -- these are mutually exclusive (namespace is "
                "derived from forced at build time)."
            )
        if namespace is Namespace.FORCED and forced is not True:
            raise ManifestError(
                "manifest.namespace is FORCED but manifest.forced is not "
                "True -- these are mutually exclusive (namespace is "
                "derived from forced at build time)."
            )
        if forced and not (isinstance(force_reason, str) and force_reason.strip()):
            raise ManifestError(
                "manifest.forced is True but force_reason is missing or "
                "blank (frozen architecture section 10 requires a "
                "non-empty reason)."
            )
        if not forced and force_reason is not None:
            raise ManifestError(
                "manifest.forced is False but force_reason is not null -- "
                "a reason with no force is not a coherent build() output."
            )

        return cls(
            provenance_schema_version=provenance_schema_version,
            market_data_schema_version=market_data_schema_version,
            source=source,
            symbol=symbol,
            resolution=resolution,
            generation_id=generation_id,
            namespace=namespace,
            data_digest=data_digest,
            transformations=transformations,
            source_anomalies=source_anomalies,
            source_evidence=source_evidence,
            validation_policy=validation_policy,
            fetch=fetch,
            forced=forced,
            force_reason=force_reason,
            software=software,
            provenance_digest=provenance_digest,
            integrity_id=integrity_id,
        )
