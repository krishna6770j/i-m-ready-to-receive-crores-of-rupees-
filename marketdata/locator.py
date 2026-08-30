"""Safe dataset locator (identifier slugs) and the ``CURRENT`` pointer
format, per the frozen architecture
(docs/architecture/phase1-trust-hardening.md, sections 13.1 and 13.2).

**No filesystem I/O anywhere in this module.** It produces pure values --
slugs, relative path components, and a pointer value object with
serialise/parse methods returning/accepting plain strings. Directory
creation and reading/writing an actual ``CURRENT`` file are explicitly out
of scope (Unit 8, not this one).

Section 13.1: baseline builds paths as
``symbol.replace(":", "_").replace("/", "_")``, so a ``..`` component would
traverse. The safe mapping here makes the raw identifier's UTF-8 text NEVER
a path component: ``sanitised_prefix`` is drawn from a restricted alphabet
that structurally excludes ``.``, so a ``.``/``..`` component is impossible
BY CONSTRUCTION rather than by a rejection rule that could be forgotten,
and ``digest_suffix`` makes the mapping collision-resistant even when two
different identifiers sanitise to the same prefix. The slug is one-way:
decoding is never required (the original identifier lives in the envelope
and the identity digest, §8.1) and this module does not attempt it.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

# --- §13.1: safe identifier slug --------------------------------------------

_ALLOWED_PREFIX_CHAR = re.compile(r"[A-Za-z0-9_-]")
_PREFIX_MAX_LEN = 32
_SUFFIX_HEX_LEN = 16  # hex characters => 8 raw SHA-256 bytes


class LocatorError(ValueError):
    """Raised for an invalid identifier or malformed CURRENT pointer."""


def _sanitised_prefix(identifier: str) -> str:
    """Every character outside ``[A-Za-z0-9_-]`` becomes ``_``.

    ``.`` is deliberately excluded from the allowed alphabet (frozen
    architecture section 13.1): a restricted alphabet that cannot produce
    ``.`` makes a ``.``/``..`` path component impossible by construction,
    rather than relying on a rejection rule elsewhere that could be
    forgotten. Operates per-character on the Python ``str`` (not raw UTF-8
    bytes), so a single non-ASCII character becomes exactly one ``_``
    rather than several (one per UTF-8 byte it happens to encode to).
    """
    sanitised = "".join(
        ch if _ALLOWED_PREFIX_CHAR.match(ch) else "_" for ch in identifier
    )
    return sanitised[:_PREFIX_MAX_LEN]


def _digest_suffix(identifier: str) -> str:
    """First 16 hex characters (8 raw bytes) of ``SHA256(utf8(identifier))``."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:_SUFFIX_HEX_LEN]


def safe_slug(identifier: str) -> str:
    """``sanitised_prefix + "-" + digest_suffix`` for one raw identifier.

    Properties, each satisfied by construction (frozen architecture section
    13.1): the raw identifier is never itself a path component; no ``/`` or
    ``\\`` can appear (excluded from the allowed alphabet, becomes ``_``);
    no ``.``/``..`` component is possible; deterministic; collision-
    resistant even when two different identifiers sanitise to the same
    prefix (their digest suffixes still differ). Maximum length: 32
    (prefix) + 1 (``-``) + 16 (suffix) = 49 characters.
    """
    if not isinstance(identifier, str) or identifier == "":
        raise LocatorError(
            f"identifier must be a non-empty str, got {identifier!r}"
        )
    return f"{_sanitised_prefix(identifier)}-{_digest_suffix(identifier)}"


def dataset_relative_path(*, source: str, symbol: str, resolution: str) -> PurePosixPath:
    """Pure, safe relative path components for one dataset's location.

    Returns a ``PurePosixPath`` of three slugs (``source/symbol/resolution``)
    -- a location value only. Does not create any directory, does not touch
    a filesystem, and is not itself a claim of identity (frozen architecture
    section 13.1: "the slug is a locator only and is never treated as
    identity").
    """
    return PurePosixPath(safe_slug(source), safe_slug(symbol), safe_slug(resolution))


# --- §13.2: CURRENT pointer format -------------------------------------------

POINTER_VERSION = 1

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_POINTER_FIELDS = frozenset({"pointer_version", "generation_id", "integrity_id"})


def _validate_integrity_id(value: object) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise LocatorError(
            "integrity_id must be a lowercase 64-character SHA-256 hex "
            f"digest, got {value!r}"
        )
    return value


def _validate_generation_id(value: object) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError, TypeError) as exc:
            raise LocatorError(
                f"generation_id must be a valid UUID4 string, got {value!r}"
            ) from exc
    else:
        raise LocatorError(
            "generation_id must be a uuid.UUID or a UUID string, got "
            f"{type(value).__name__}"
        )
    if parsed.version != 4:
        raise LocatorError(
            f"generation_id must be a version-4 UUID, got version "
            f"{parsed.version} ({parsed})"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class CurrentPointer:
    """The ``CURRENT`` pointer's exact fields (frozen architecture section
    13.2): ``pointer_version`` (always ``1``, never a constructor
    parameter -- it is a property, not a dataclass field, so
    ``CurrentPointer(pointer_version=2, ...)`` raises ``TypeError`` before
    any validation logic even runs), ``generation_id`` (a version-4 UUID),
    ``integrity_id`` (a lowercase 64-character SHA-256 hex digest).

    No path-shaped material can exist inside this structure: every field is
    either the fixed integer ``1``, a UUID's canonical string form, or a
    pure hex string -- none of which can contain ``/`` or ``..``.
    """

    generation_id: uuid.UUID
    integrity_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generation_id", _validate_generation_id(self.generation_id)
        )
        object.__setattr__(
            self, "integrity_id", _validate_integrity_id(self.integrity_id)
        )

    @property
    def pointer_version(self) -> int:
        return POINTER_VERSION

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, no incidental whitespace, UTF-8 text.

        Deterministic for a fixed ``(generation_id, integrity_id)`` pair --
        calling this repeatedly on the same instance always yields the
        identical string.
        """
        payload = {
            "pointer_version": self.pointer_version,
            "generation_id": str(self.generation_id),
            "integrity_id": self.integrity_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "CurrentPointer":
        """Strict parse: malformed JSON, a non-object payload, any unknown
        field, any missing field, or an unsupported ``pointer_version`` all
        raise ``LocatorError`` rather than silently accepting or ignoring
        the problem.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LocatorError(f"CURRENT pointer is not valid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise LocatorError(
                f"CURRENT pointer JSON must be an object, got {type(payload).__name__}"
            )

        actual_fields = set(payload.keys())
        unknown = actual_fields - _REQUIRED_POINTER_FIELDS
        missing = _REQUIRED_POINTER_FIELDS - actual_fields
        if unknown:
            raise LocatorError(
                f"CURRENT pointer has unknown field(s): {sorted(unknown)}"
            )
        if missing:
            raise LocatorError(
                f"CURRENT pointer is missing field(s): {sorted(missing)}"
            )

        version = payload["pointer_version"]
        if version != POINTER_VERSION:
            raise LocatorError(
                f"Unsupported pointer_version {version!r}; expected "
                f"{POINTER_VERSION}"
            )

        return cls(
            generation_id=payload["generation_id"],
            integrity_id=payload["integrity_id"],
        )
