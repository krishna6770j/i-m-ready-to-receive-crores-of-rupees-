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
- ``data_digest`` (``dataset.digest``) -- not explicitly listed among
  section 6's envelope bullets, but binding it here too means a tampered
  ``data_digest`` breaks BOTH ``provenance_digest`` and ``integrity_id``,
  which is strictly stronger than the minimum the architecture text requires
- canonicalisation snapshot (section 14): ``dataset.transformations``,
  ``dataset.source_anomalies``, ``dataset.source_evidence``
- acquisition snapshot (section 11): an optional
  ``marketdata.evidence.FetchReportSnapshot`` -- optional because section
  11.1 explicitly allows ``REQUESTS_UNKNOWN`` ("no acquisition evidence
  (fixtures, manual frames)"); this unit does not yet compute
  ``AcquisitionRequestStatus`` from it (that is Unit 9 in the frozen
  architecture's implementation sequence, section 28) -- it binds the raw,
  already-immutable evidence snapshot only
- operator declarations (section 10): ``forced``, ``force_reason``
  (required non-empty when ``forced=True``)
- environment snapshot: ``marketdata.store.software_versions()`` (the
  ACTUAL environment only). Section 22's ``environment_expected_digest`` /
  ``ReproducibilityCertification`` machinery is NOT implemented here --
  there is no lock-file-digest concept anywhere in this codebase yet, and
  inventing one to fill this field would be exactly the kind of placeholder
  fact the manager's directive prohibits. This is a deliberately incomplete
  prerequisite, not an oversight; see the module's test/commit notes.

**Deliberately NOT bound**: validation policy/evidence
(``marketdata.dataset.ValidationPolicy`` / ``.validation``). Section 6's
envelope bullet list does not name it, and sections 2/4 classify
``MarketDataValidity`` as a data-derived fact that is RECOMPUTED from stored
data at every use (not integrity-bound provenance the way acquisition or
canonicalisation evidence is, since those describe an external process that
cannot be reconstructed from the final data alone). Binding it here would be
inventing a requirement the frozen text does not state.
"""

from __future__ import annotations

import hashlib
import re
import struct
import uuid
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from marketdata.dataset import ValidatedDataset
from marketdata.evidence import FetchReportSnapshot
from marketdata.schemas import MARKET_DATA_SCHEMA_VERSION
from marketdata.store import software_versions

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
    if isinstance(value, int):
        return _encode_i64(value)
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

        ``fetch``, if given, must be an actual ``FetchReportSnapshot``.

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
        object.__setattr__(self, "_fetch", fetch)
        object.__setattr__(self, "_forced", bool(forced))
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
        """The exact byte stream ``provenance_digest`` hashes."""
        ordered_fields = (
            ("provenance_schema_version", self._provenance_schema_version),
            ("market_data_schema_version", self._market_data_schema_version),
            ("source", self._source),
            ("symbol", self._symbol),
            ("resolution", self._resolution),
            ("generation_id", self._generation_id),
            ("namespace", self._namespace),
            ("data_digest", self._data_digest),
            ("transformations", self._transformations),
            ("source_anomalies", self._source_anomalies),
            ("source_evidence", self._source_evidence),
            ("fetch", self._fetch),
            ("forced", self._forced),
            ("force_reason", self._force_reason),
            ("software", self._software),
        )
        parts = [_encode_i64(len(ordered_fields))]
        for name, value in ordered_fields:
            parts.append(_encode_str(name))
            parts.append(_encode_value(value))
        return b"".join(parts)

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
