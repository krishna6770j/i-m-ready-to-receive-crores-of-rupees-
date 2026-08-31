"""Loads instrument definitions from config/instruments.yaml.

Keeps the signal instrument and the execution candidates in separate
collections so that code cannot accidentally iterate "all instruments" and
treat the index as tradable.

Configuration validation is strict and FAILS CLOSED (architecture section
23): every field is required, optional-nullable, or optional with an
explicit documented default -- never a truthiness-based guess. In
particular this module never turns a malformed value into a plausible
default (the confirmed defect this replaces was
``int(lot_size) if lot_size else 1``, which silently turned ``0`` and ``""``
into ``1``). A present-but-invalid value always raises
:class:`InstrumentConfigError` naming the file, section, entry and field; it
is never silently repaired or skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.types import InstrumentKind, InstrumentRole
from instruments.instrument import Instrument

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "instruments.yaml"


class _DuplicateYamlKeyError(yaml.YAMLError):
    """A mapping in the YAML document repeats a key.

    PyYAML's default mapping construction silently keeps the LAST occurrence
    of a duplicate key and discards earlier ones -- including an earlier,
    genuinely invalid value someone was trying to "fix" with a second line.
    That loss happens during YAML construction, before our schema validator
    ever sees the document, so a duplicate key can make malformed source
    text disappear entirely. This must be caught here, not after
    ``safe_load`` returns, when the information is already gone.
    """


class _StrictSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that rejects duplicate mapping keys at EVERY mapping
    level (top-level document, each signal/candidate entry, and any nested
    mapping introduced later). Otherwise identical to ``yaml.safe_load``:
    same safety characteristics, no arbitrary object construction -- only
    ``construct_mapping`` is overridden.
    """

    def construct_mapping(self, node, deep=False):
        mapping: dict = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                mark = key_node.start_mark
                raise _DuplicateYamlKeyError(
                    f"duplicate key {key!r} at line {mark.line + 1}, "
                    f"column {mark.column + 1}"
                )
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

SCHEMA_VERSION = 1

_TOP_LEVEL_KEYS = frozenset({"schema_version", "signal_instruments", "execution_candidates"})

_SIGNAL_ENTRY_KEYS = frozenset(
    {"symbol", "name", "kind", "role", "lot_size", "tick_size", "exchange", "notes"}
)
_CANDIDATE_ENTRY_KEYS = _SIGNAL_ENTRY_KEYS | {"verified"}


class InstrumentConfigError(ValueError):
    """Raised for any structural or value defect in the instrument config.

    Every raise site includes the config path, section, entry reference and
    the offending field/value, so the operator can find and fix the exact
    line without needing a raw YAML dump.
    """


def _error(path: Path, section: str, ref: str, message: str) -> None:
    raise InstrumentConfigError(f"{path}: {section}{ref}: {message}")


def _require_identity_str(entry: dict, field: str, path: Path, section: str, ref: str) -> str:
    """Validate a REQUIRED identity field (symbol/kind/role/exchange).

    Identity fields must arrive already canonical: present, an actual
    ``str`` (no int/bool/list coercion), non-empty, and with no
    leading/trailing whitespace. Silently stripping and accepting the
    changed value would let a malformed config look correct after the fact;
    this rejects it instead so the source file gets fixed.
    """
    if field not in entry:
        _error(path, section, ref, f"{field!r} is required")
    value = entry[field]
    if value is None:
        _error(path, section, ref, f"{field!r} is required, got null")
    if not isinstance(value, str):
        _error(
            path, section, ref,
            f"{field!r} must be a string, got {type(value).__name__}: {value!r}",
        )
    if value.strip() == "":
        _error(path, section, ref, f"{field!r} must be non-empty")
    if value != value.strip():
        _error(
            path, section, ref,
            f"{field!r} must not have leading/trailing whitespace, got {value!r}",
        )
    return value


def _require_or_none_symbol(
    entry: dict, path: Path, section: str, ref: str, *, allow_none: bool
) -> str | None:
    """Validate ``symbol``, which may be null ONLY for an unverified
    execution candidate (architecture section 23: the current Options
    candidate intentionally has no symbol yet). A verified candidate or any
    signal instrument must have a real symbol.
    """
    if "symbol" not in entry:
        _error(path, section, ref, "'symbol' is required")
    value = entry["symbol"]
    if value is None:
        if allow_none:
            return None
        _error(
            path, section, ref,
            "'symbol' must not be null for a verified candidate",
        )
    if not isinstance(value, str):
        _error(
            path, section, ref,
            f"'symbol' must be a string, got {type(value).__name__}: {value!r}",
        )
    if value.strip() == "":
        _error(path, section, ref, "'symbol' must be non-empty")
    if value != value.strip():
        _error(
            path, section, ref,
            f"'symbol' must not have leading/trailing whitespace, got {value!r}",
        )
    return value


def _validate_name(entry: dict, symbol: str | None, path: Path, section: str, ref: str) -> str:
    """Optional, documented default: ``symbol`` if it exists, else
    ``"unnamed"``. Preserves the value verbatim when present -- no
    whitespace policy is imposed here since none is architecturally frozen
    for this field.
    """
    if "name" not in entry or entry["name"] is None:
        return symbol if symbol is not None else "unnamed"
    value = entry["name"]
    if not isinstance(value, str):
        _error(
            path, section, ref,
            f"'name' must be a string, got {type(value).__name__}: {value!r}",
        )
    return value


def _validate_notes(entry: dict, path: Path, section: str, ref: str) -> str:
    """Optional, documented default ``""``. Whitespace-stripped: this is
    pre-existing, documented behaviour for free-text notes (unlike identity
    fields, where stripping would silently change meaning).
    """
    if "notes" not in entry or entry["notes"] is None:
        return ""
    value = entry["notes"]
    if not isinstance(value, str):
        _error(
            path, section, ref,
            f"'notes' must be a string, got {type(value).__name__}: {value!r}",
        )
    return value.strip()


def _validate_verified(entry: dict, path: Path, section: str, ref: str) -> bool:
    """Optional, documented default ``False``. Must be an EXACT bool when
    present -- this specifically fixes the confirmed defect
    ``bool(entry.get("verified", False))``, under which
    ``bool("false") == True``. ``None`` is rejected only when the key is
    explicitly present with a null value; an absent key uses the default.
    """
    if "verified" not in entry:
        return False
    value = entry["verified"]
    if value is None:
        _error(path, section, ref, "'verified' must not be null")
    if not isinstance(value, bool):
        _error(
            path, section, ref,
            f"'verified' must be a bool, got {type(value).__name__}: {value!r}",
        )
    return value


def _validate_positive_int(
    value: object, field: str, path: Path, section: str, ref: str, *, allow_none: bool
) -> int | None:
    """The confirmed defect this replaces:
    ``int(lot_size) if lot_size else 1``, which silently turned ``0`` and
    ``""`` into ``1``. No coercion of any kind: the value must already be an
    actual ``int`` (``type(value) is int`` -- excludes ``bool``, since
    ``isinstance(True, int)`` is true in Python but a bool is never a
    legitimate lot size), and must be >= 1.
    """
    if value is None:
        if allow_none:
            return None
        _error(path, section, ref, f"{field!r} is required")
    if isinstance(value, bool):
        _error(path, section, ref, f"{field!r} must be an int, got bool: {value!r}")
    if type(value) is not int:
        _error(
            path, section, ref,
            f"{field!r} must be an int, got {type(value).__name__}: {value!r}",
        )
    if value < 1:
        _error(path, section, ref, f"{field!r} must be a positive integer, got {value}")
    return value


def _validate_tick_size(
    value: object, path: Path, section: str, ref: str, *, required: bool
) -> float | None:
    """Nullable while unverified; required and must be positive+finite for a
    verified tradable candidate. No coercion: bool, zero, negative, NaN,
    +/-inf and numeric strings are all rejected.
    """
    if value is None:
        if required:
            _error(
                path, section, ref,
                "'tick_size' is required for a verified tradable candidate",
            )
        return None
    if isinstance(value, bool):
        _error(path, section, ref, f"'tick_size' must be numeric, got bool: {value!r}")
    if not isinstance(value, (int, float)):
        _error(
            path, section, ref,
            f"'tick_size' must be numeric, got {type(value).__name__}: {value!r}",
        )
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        _error(path, section, ref, f"'tick_size' must be finite, got {value!r}")
    if numeric <= 0:
        _error(path, section, ref, f"'tick_size' must be positive, got {value!r}")
    return numeric


def _parse_enum(enum_cls, raw_value: str, field: str, path: Path, section: str, ref: str):
    try:
        return enum_cls(raw_value)
    except ValueError:
        allowed = ", ".join(sorted(e.value for e in enum_cls))
        _error(
            path, section, ref,
            f"{field!r} must be one of [{allowed}], got {raw_value!r}",
        )


def _construct_instrument(
    *, symbol: str, name: str, kind, role, lot_size: int, tick_size: float | None,
    exchange: str, notes: str, path: Path, section: str, ref: str,
) -> Instrument:
    try:
        return Instrument(
            symbol=symbol, name=name, kind=kind, role=role,
            lot_size=lot_size, tick_size=tick_size, exchange=exchange, notes=notes,
        )
    except ValueError as exc:
        _error(path, section, ref, str(exc))


@dataclass(frozen=True)
class ExecutionCandidate:
    """A candidate execution instrument that has not been selected or verified.

    Wrapping the Instrument rather than subclassing it keeps "is this verified"
    separate from "what is this", and makes it impossible to pass an unverified
    candidate where a verified Instrument is required without unwrapping it
    explicitly.

    A ``verified=True`` candidate is ALWAYS constructed with
    ``instrument is not None`` -- the loader rejects the whole config before
    producing a coherent-but-incomplete ``ExecutionCandidate(verified=True,
    instrument=None)``, which must be impossible to represent.
    """

    instrument: Instrument | None
    verified: bool
    name: str
    notes: str

    def require_verified(self) -> Instrument:
        """Return the instrument, or explain why it cannot be used yet."""
        if not self.verified or self.instrument is None:
            raise ValueError(
                f"Execution candidate {self.name!r} is not verified and cannot be "
                "used for sizing, costing or backtesting. Its lot size, tick size, "
                "margin, liquidity and cost model must be confirmed against live "
                "broker data first. No execution instrument has been selected."
            )
        return self.instrument


@dataclass(frozen=True)
class InstrumentRegistry:
    signal_instruments: dict[str, Instrument]
    execution_candidates: tuple[ExecutionCandidate, ...]

    def signal(self, symbol: str) -> Instrument:
        try:
            return self.signal_instruments[symbol]
        except KeyError as exc:
            known = ", ".join(sorted(self.signal_instruments)) or "(none)"
            raise KeyError(
                f"No signal instrument {symbol!r}. Known: {known}"
            ) from exc

    @property
    def selected_execution_instrument(self) -> Instrument:
        """Always raises: selection is an open research decision."""
        raise NotImplementedError(
            "No execution instrument has been selected. This is an open research "
            "decision that depends on cost, liquidity, margin and granularity "
            "analysis not yet performed. The system holds "
            f"{len(self.execution_candidates)} candidates for evaluation."
        )


def _validate_signal_entry(entry: object, path: Path, index: int) -> Instrument:
    section = "signal_instruments"
    ref = f"[{index}]"
    if not isinstance(entry, dict):
        _error(path, section, ref, f"entry must be a mapping, got {type(entry).__name__}")

    unknown = set(entry) - _SIGNAL_ENTRY_KEYS
    if unknown:
        _error(path, section, ref, f"unknown field(s): {sorted(unknown)}")

    symbol = _require_identity_str(entry, "symbol", path, section, ref)
    kind_raw = _require_identity_str(entry, "kind", path, section, ref)
    role_raw = _require_identity_str(entry, "role", path, section, ref)
    exchange = _require_identity_str(entry, "exchange", path, section, ref)

    if role_raw != InstrumentRole.SIGNAL.value:
        _error(
            path, section, ref,
            f"'role' must be {InstrumentRole.SIGNAL.value!r} in {section}, got {role_raw!r}",
        )

    kind = _parse_enum(InstrumentKind, kind_raw, "kind", path, section, ref)
    role = _parse_enum(InstrumentRole, role_raw, "role", path, section, ref)

    name = _validate_name(entry, symbol, path, section, ref)
    notes = _validate_notes(entry, path, section, ref)

    lot_size = _validate_positive_int(
        entry.get("lot_size"), "lot_size", path, section, ref, allow_none=False
    )
    tick_size = _validate_tick_size(entry.get("tick_size"), path, section, ref, required=False)

    return _construct_instrument(
        symbol=symbol, name=name, kind=kind, role=role, lot_size=lot_size,
        tick_size=tick_size, exchange=exchange, notes=notes,
        path=path, section=section, ref=ref,
    )


def _validate_candidate_entry(
    entry: object, path: Path, index: int
) -> tuple[str | None, ExecutionCandidate]:
    section = "execution_candidates"
    ref = f"[{index}]"
    if not isinstance(entry, dict):
        _error(path, section, ref, f"entry must be a mapping, got {type(entry).__name__}")

    unknown = set(entry) - _CANDIDATE_ENTRY_KEYS
    if unknown:
        _error(path, section, ref, f"unknown field(s): {sorted(unknown)}")

    verified = _validate_verified(entry, path, section, ref)

    symbol = _require_or_none_symbol(entry, path, section, ref, allow_none=not verified)
    kind_raw = _require_identity_str(entry, "kind", path, section, ref)
    role_raw = _require_identity_str(entry, "role", path, section, ref)
    exchange = _require_identity_str(entry, "exchange", path, section, ref)

    if role_raw != InstrumentRole.EXECUTION.value:
        _error(
            path, section, ref,
            f"'role' must be {InstrumentRole.EXECUTION.value!r} in {section}, got {role_raw!r}",
        )

    kind = _parse_enum(InstrumentKind, kind_raw, "kind", path, section, ref)
    role = _parse_enum(InstrumentRole, role_raw, "role", path, section, ref)

    name = _validate_name(entry, symbol, path, section, ref)
    notes = _validate_notes(entry, path, section, ref)

    lot_size = _validate_positive_int(
        entry.get("lot_size"), "lot_size", path, section, ref, allow_none=not verified
    )
    tick_size = _validate_tick_size(entry.get("tick_size"), path, section, ref, required=verified)

    instrument: Instrument | None = None
    if verified:
        # All required fields are already guaranteed valid above (symbol
        # non-null, lot_size positive int, tick_size positive finite) -- a
        # verified candidate is therefore ALWAYS constructible. If any were
        # missing/invalid, the calls above already raised and aborted the
        # whole config load; ExecutionCandidate(verified=True,
        # instrument=None) is not a state this function can produce.
        instrument = _construct_instrument(
            symbol=symbol, name=name, kind=kind, role=role, lot_size=lot_size,
            tick_size=tick_size, exchange=exchange, notes=notes,
            path=path, section=section, ref=ref,
        )
    elif symbol is not None and lot_size is not None:
        # Unverified but enough is known to represent it as a real
        # Instrument (still unusable for sizing/costing without verified=True).
        instrument = _construct_instrument(
            symbol=symbol, name=name, kind=kind, role=role, lot_size=lot_size,
            tick_size=tick_size, exchange=exchange, notes=notes,
            path=path, section=section, ref=ref,
        )
    # else: genuinely underspecified (e.g. symbol=None, or lot_size=None).
    # Kept as candidate metadata only -- never silently skipped from the
    # registry, just represented with instrument=None.

    return symbol, ExecutionCandidate(
        instrument=instrument, verified=verified, name=name, notes=notes
    )


def _register_symbol(seen: dict[str, str], symbol: str, ref: str, path: Path) -> None:
    """Reject duplicate non-null symbols across the ENTIRE config (signal vs
    signal, candidate vs candidate, signal vs candidate). Multiple ``None``
    symbols among unverified candidates are explicitly NOT a duplicate --
    ``None`` carries no instrument identity to collide on.
    """
    if symbol in seen:
        raise InstrumentConfigError(
            f"{path}: duplicate symbol {symbol!r} in {ref} "
            f"(already used in {seen[symbol]})"
        )
    seen[symbol] = ref


def load_registry(path: str | Path | None = None) -> InstrumentRegistry:
    """Read and validate the instrument configuration. Fails closed: any
    structural or value defect raises :class:`InstrumentConfigError` for the
    WHOLE file rather than silently repairing or skipping the bad entry.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as fh:
        try:
            raw = yaml.load(fh, Loader=_StrictSafeLoader)
        except _DuplicateYamlKeyError as exc:
            raise InstrumentConfigError(f"{config_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise InstrumentConfigError(
                f"{config_path}: could not parse YAML ({type(exc).__name__})"
            ) from exc

    if not isinstance(raw, dict):
        raise InstrumentConfigError(
            f"{config_path}: top-level document must be a mapping, got "
            f"{type(raw).__name__}"
        )

    unknown_top = set(raw) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise InstrumentConfigError(
            f"{config_path}: unknown top-level key(s): {sorted(unknown_top)}"
        )

    if "schema_version" not in raw:
        raise InstrumentConfigError(f"{config_path}: 'schema_version' is required")
    version = raw["schema_version"]
    if type(version) is not int:
        raise InstrumentConfigError(
            f"{config_path}: 'schema_version' must be an int, got "
            f"{type(version).__name__}: {version!r}"
        )
    if version != SCHEMA_VERSION:
        raise InstrumentConfigError(
            f"{config_path}: unsupported schema_version {version!r}; "
            f"this loader supports {SCHEMA_VERSION!r} only"
        )

    if "signal_instruments" not in raw:
        raise InstrumentConfigError(f"{config_path}: 'signal_instruments' is required")
    if "execution_candidates" not in raw:
        raise InstrumentConfigError(f"{config_path}: 'execution_candidates' is required")

    signal_raw = raw["signal_instruments"]
    candidate_raw = raw["execution_candidates"]
    if not isinstance(signal_raw, list):
        raise InstrumentConfigError(
            f"{config_path}: 'signal_instruments' must be a list, got "
            f"{type(signal_raw).__name__}"
        )
    if not isinstance(candidate_raw, list):
        raise InstrumentConfigError(
            f"{config_path}: 'execution_candidates' must be a list, got "
            f"{type(candidate_raw).__name__}"
        )

    seen_symbols: dict[str, str] = {}

    signals: dict[str, Instrument] = {}
    for index, entry in enumerate(signal_raw):
        instrument = _validate_signal_entry(entry, config_path, index)
        _register_symbol(seen_symbols, instrument.symbol, f"signal_instruments[{index}]", config_path)
        signals[instrument.symbol] = instrument

    candidates: list[ExecutionCandidate] = []
    for index, entry in enumerate(candidate_raw):
        symbol, candidate = _validate_candidate_entry(entry, config_path, index)
        if symbol is not None:
            _register_symbol(seen_symbols, symbol, f"execution_candidates[{index}]", config_path)
        candidates.append(candidate)

    return InstrumentRegistry(
        signal_instruments=signals,
        execution_candidates=tuple(candidates),
    )
