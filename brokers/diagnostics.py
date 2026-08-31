"""Sanitized broker diagnostics.

``BrokerDiagnostic`` is the ONLY thing a broker adapter's errors are allowed
to carry. A raw broker payload/message may exist transiently, in memory, for
classification (e.g. "does this message mention 'token'?"), but it must never
be logged, stored on an exception, printed, or reachable via ``repr``/``str``
of anything that survives the parsing call. Unknown payload fields never flow
through automatically -- only an explicit allowlist of structured fields is
kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from core.secrets import scrub

# Strips ASCII control characters (other than the ones dataclasses.replace
# etc. would never see anyway) so a broker payload cannot smuggle terminal
# escapes or embedded nulls into a log line via a diagnostic message.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_MAX_MESSAGE_LENGTH = 500

# The only structured fields a diagnostic may carry. Anything else supplied
# is silently dropped -- an unknown raw payload key must never automatically
# flow into a diagnostic just because it happened to be present.
ALLOWED_STRUCTURED_FIELDS = frozenset(
    {"status", "code", "symbol", "resolution", "range_from", "range_to"}
)


class BrokerDiagnosticStatus(str, Enum):
    AUTH_ERROR = "AUTH_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    DATA_ERROR = "DATA_ERROR"


def _sanitize_text(text: str) -> str:
    text = _CONTROL_CHARS.sub(" ", text)
    text = scrub(text)
    if len(text) > _MAX_MESSAGE_LENGTH:
        text = text[:_MAX_MESSAGE_LENGTH] + "...[truncated]"
    return text


def _sanitize_fields(fields: dict) -> MappingProxyType:
    cleaned = {}
    for key, value in dict(fields).items():
        if key not in ALLOWED_STRUCTURED_FIELDS:
            continue
        if isinstance(value, str):
            value = _sanitize_text(value)
        cleaned[key] = value
    return MappingProxyType(cleaned)


@dataclass(frozen=True)
class BrokerDiagnostic:
    """Immutable, pre-sanitized broker error information.

    Construction sanitizes ``sanitized_message`` and
    ``sanitized_structured_fields`` unconditionally -- callers do not need to
    (and must not rely on having to) scrub before constructing one.
    """

    status: BrokerDiagnosticStatus
    code: str | int | None
    sanitized_message: str
    sanitized_structured_fields: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
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
