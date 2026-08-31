"""Sanitized broker diagnostics.

``BrokerDiagnostic`` is the ONLY thing a broker adapter's errors are allowed
to carry. A raw broker payload/message may exist transiently, in memory, for
classification (e.g. "does this message mention 'token'?"), but it must never
be logged, stored on an exception, printed, or reachable via ``repr``/``str``
of anything that survives the parsing call. Unknown payload fields never flow
through automatically -- only an explicit allowlist of structured fields is
kept, and even an allowlisted field name is rejected at construction if its
value is not one of the safe scalar types defined below. This boundary fails
CLOSED: a malformed value raises ``TypeError`` rather than being silently
stringified or smuggled through, because these fields are programmer-facing
diagnostic construction, not a place to dump raw payload data.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from core.secrets import scrub

# ALL ASCII control characters -- 0x00..0x1F and 0x7F -- including tab, LF and
# CR. A diagnostic message must be single-line: allowing \n/\r/\t through
# would let a broker-controlled string inject fake log lines or multi-line
# noise into what is meant to be one line of operator-facing text.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# Collapses whatever whitespace is left (including the spaces that replaced
# control characters above) into single spaces, deterministically, so
# "error\n\n token" becomes "error token" rather than "error   token" or an
# accidental multi-line string.
_WHITESPACE_RUN = re.compile(r"\s+")

_MAX_MESSAGE_LENGTH = 500

# The only structured fields a diagnostic may carry. Anything else supplied
# is silently dropped -- an unknown raw payload key must never automatically
# flow into a diagnostic just because it happened to be present.
ALLOWED_STRUCTURED_FIELDS = frozenset(
    {"status", "code", "symbol", "resolution", "range_from", "range_to"}
)

# Exact supported scalar types per structured field. Anything outside these
# (dict, list, tuple, set, a custom object) is rejected with TypeError rather
# than stringified -- str(obj) could invoke an attacker/broker-controlled
# __str__/__repr__ that embeds arbitrary text, including a secret.
_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "status": (str, type(None)),
    "code": (int, str, type(None)),
    "symbol": (str,),
    "resolution": (str,),
    "range_from": (str,),
    "range_to": (str,),
}


class BrokerDiagnosticStatus(str, Enum):
    AUTH_ERROR = "AUTH_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    DATA_ERROR = "DATA_ERROR"


def _sanitize_text(text: str) -> str:
    """Scrub, THEN normalize, THEN scrub again.

    A registered secret can itself contain control characters (a multiline
    or tab/CR-containing token). Normalizing control characters to spaces
    first would mangle that literal before ``scrub()`` ever sees it, so the
    exact registered string no longer appears in the text and never matches
    -- the secret leaks in plain sight. Scrubbing first catches the exact
    literal while it's still intact. The second scrub afterwards is defence
    in depth: it catches anything that only became a registered-secret match
    as a side effect of normalization (e.g. a secret that was split across a
    control character and became contiguous once that character turned into
    a space).
    """
    text = scrub(text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE_RUN.sub(" ", text).strip()
    text = scrub(text)
    if len(text) > _MAX_MESSAGE_LENGTH:
        text = text[:_MAX_MESSAGE_LENGTH] + "...[truncated]"
    return text


def _validate_code(value: object, *, where: str) -> int | str | None:
    """``code`` may be int, str or None. ``bool`` is explicitly rejected even
    though ``isinstance(True, int)`` is true in Python -- a diagnostic code
    that happens to be a bool is almost certainly a programming mistake, not
    a genuine broker error code, and letting it through silently would hide
    that mistake.
    """
    if isinstance(value, bool):
        raise TypeError(f"{where} must not be bool")
    if not isinstance(value, (int, str, type(None))):
        raise TypeError(
            f"{where} must be int, str or None, got {type(value).__name__}"
        )
    return value


def _validate_structured_value(key: str, value: object) -> object:
    if key == "code":
        return _validate_code(value, where="structured field 'code'")
    allowed = _FIELD_TYPES[key]
    if not isinstance(value, allowed):
        allowed_names = " | ".join(t.__name__ for t in allowed)
        raise TypeError(
            f"structured field {key!r} must be {allowed_names}, "
            f"got {type(value).__name__}"
        )
    return value


def _sanitize_fields(fields: object) -> MappingProxyType:
    if not isinstance(fields, Mapping):
        raise TypeError(
            f"sanitized_structured_fields must be a mapping, got "
            f"{type(fields).__name__}"
        )
    cleaned = {}
    for key, value in dict(fields).items():
        if key not in ALLOWED_STRUCTURED_FIELDS:
            # Unknown field name: dropped, not an error -- a raw payload
            # commonly carries fields we never asked for.
            continue
        value = _validate_structured_value(key, value)
        if isinstance(value, str):
            value = _sanitize_text(value)
        cleaned[key] = value
    return MappingProxyType(cleaned)


@dataclass(frozen=True)
class BrokerDiagnostic:
    """Immutable, pre-sanitized broker error information.

    Construction sanitizes ``sanitized_message`` and
    ``sanitized_structured_fields`` unconditionally -- callers do not need to
    (and must not rely on having to) scrub before constructing one. Every
    field is also type-checked with no duck typing or truthiness: a
    malformed ``BrokerDiagnostic`` fails at construction (``TypeError``)
    rather than being built and carrying unsafe material.
    """

    status: BrokerDiagnosticStatus
    code: str | int | None
    sanitized_message: str
    sanitized_structured_fields: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.status, BrokerDiagnosticStatus):
            raise TypeError(
                "status must be a BrokerDiagnosticStatus, got "
                f"{type(self.status).__name__}"
            )
        validated_code = _validate_code(self.code, where="code")
        if isinstance(validated_code, str):
            # A str code is free text from the same untrusted source as the
            # message, and __repr__ renders it -- it must be scrubbed and
            # normalized exactly like sanitized_message, or a registered
            # secret passed as `code=` would leak straight through repr().
            validated_code = _sanitize_text(validated_code)
        object.__setattr__(self, "code", validated_code)
        if not isinstance(self.sanitized_message, str):
            raise TypeError(
                "sanitized_message must be a str, got "
                f"{type(self.sanitized_message).__name__}"
            )
        object.__setattr__(
            self, "sanitized_message", _sanitize_text(self.sanitized_message)
        )
        object.__setattr__(
            self,
            "sanitized_structured_fields",
            _sanitize_fields(self.sanitized_structured_fields),
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            "BrokerDiagnostic("
            f"status={self.status}, code={self.code!r}, "
            f"sanitized_message={self.sanitized_message!r}, "
            f"sanitized_structured_fields={dict(self.sanitized_structured_fields)!r})"
        )
