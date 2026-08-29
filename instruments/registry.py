"""Loads instrument definitions from config/instruments.yaml.

Keeps the signal instrument and the execution candidates in separate
collections so that code cannot accidentally iterate "all instruments" and
treat the index as tradable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from core.types import InstrumentKind, InstrumentRole
from instruments.instrument import Instrument

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "instruments.yaml"


@dataclass(frozen=True)
class ExecutionCandidate:
    """A candidate execution instrument that has not been selected or verified.

    Wrapping the Instrument rather than subclassing it keeps "is this verified"
    separate from "what is this", and makes it impossible to pass an unverified
    candidate where a verified Instrument is required without unwrapping it
    explicitly.
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


def _build_instrument(raw: dict) -> Instrument | None:
    symbol = raw.get("symbol")
    if not symbol:
        return None
    lot_size = raw.get("lot_size")
    return Instrument(
        symbol=symbol,
        name=raw.get("name", symbol),
        kind=InstrumentKind(raw["kind"]),
        role=InstrumentRole(raw["role"]),
        lot_size=int(lot_size) if lot_size else 1,
        tick_size=raw.get("tick_size"),
        exchange=raw.get("exchange", "NSE"),
        notes=raw.get("notes", "").strip(),
    )


def load_registry(path: str | Path | None = None) -> InstrumentRegistry:
    """Read and validate the instrument configuration."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    signals: dict[str, Instrument] = {}
    for entry in raw.get("signal_instruments") or []:
        instrument = _build_instrument(entry)
        if instrument is not None:
            signals[instrument.symbol] = instrument

    candidates: list[ExecutionCandidate] = []
    for entry in raw.get("execution_candidates") or []:
        # A candidate with a null lot_size cannot be constructed as a valid
        # Instrument, which is correct: it is genuinely underspecified. Keep the
        # metadata so it can still be listed and reported on.
        instrument = None
        if entry.get("symbol") and entry.get("lot_size"):
            instrument = _build_instrument(entry)
        candidates.append(
            ExecutionCandidate(
                instrument=instrument,
                verified=bool(entry.get("verified", False)),
                name=entry.get("name", entry.get("symbol") or "unnamed"),
                notes=(entry.get("notes") or "").strip(),
            )
        )

    return InstrumentRegistry(
        signal_instruments=signals,
        execution_candidates=tuple(candidates),
    )
