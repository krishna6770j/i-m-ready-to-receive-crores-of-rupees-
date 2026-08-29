"""Instrument model enforcing the signal/execution separation.

This is the architectural boundary described in Phase 0 section C. The NIFTY 50
index generates signals but cannot be traded; whatever we eventually trade is a
different instrument with its own lot size, tick size and cost model.

The separation is enforced by behaviour, not convention: an instrument whose
role is SIGNAL raises if anything asks it to size a position or price a trade.
A conflation bug therefore fails immediately instead of silently producing a
backtest of an untradable series.

NOTE ON SELECTION: no execution instrument has been chosen. The registry
deliberately holds several candidates so they can be evaluated against each
other later. Nothing in this module expresses a preference between them.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import InstrumentKind, InstrumentRole


class NotTradableError(RuntimeError):
    """Raised when code attempts to trade or price a signal-only instrument."""


@dataclass(frozen=True)
class Instrument:
    """A market instrument.

    Attributes:
        symbol: Broker symbol, e.g. "NSE:NIFTY50-INDEX".
        name: Human-readable name.
        kind: What it is, for cost/margin modelling.
        role: Whether it may generate signals, be traded, or both.
        lot_size: Units per lot. 1 for cash-segment instruments.
        tick_size: Minimum price increment, in rupees. None if not applicable.
        exchange: Exchange code parsed from the symbol.
        notes: Free-text provenance or caveats.
    """

    symbol: str
    name: str
    kind: InstrumentKind
    role: InstrumentRole
    lot_size: int = 1
    tick_size: float | None = None
    exchange: str = "NSE"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.lot_size < 1:
            raise ValueError(f"{self.symbol}: lot_size must be >= 1, got {self.lot_size}")
        if self.tick_size is not None and self.tick_size <= 0:
            raise ValueError(
                f"{self.symbol}: tick_size must be positive, got {self.tick_size}"
            )
        if self.kind is InstrumentKind.INDEX and self.role is not InstrumentRole.SIGNAL:
            raise ValueError(
                f"{self.symbol}: an INDEX cannot have role {self.role.value}. "
                "An index is not tradable; trade a derivative or ETF that tracks it."
            )

    @property
    def is_tradable(self) -> bool:
        return self.role in (InstrumentRole.EXECUTION, InstrumentRole.BOTH)

    @property
    def can_generate_signals(self) -> bool:
        return self.role in (InstrumentRole.SIGNAL, InstrumentRole.BOTH)

    def require_tradable(self) -> None:
        """Guard called before any sizing, costing or P&L computation."""
        if not self.is_tradable:
            raise NotTradableError(
                f"{self.symbol} has role '{self.role.value}' and cannot be traded, "
                "sized, or used to compute P&L. It is a signal source only. "
                "Map the signal to an execution instrument first."
            )

    def round_to_tick(self, price: float) -> float:
        """Round a price to the instrument's tick size."""
        self.require_tradable()
        if self.tick_size is None:
            return float(price)
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def lots_for_quantity(self, quantity: int) -> int:
        """Whole lots represented by ``quantity`` units (floor)."""
        self.require_tradable()
        return quantity // self.lot_size
