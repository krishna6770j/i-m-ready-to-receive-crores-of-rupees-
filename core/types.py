"""Shared domain types.

Deliberately minimal for Phase 1. Order/Trade/Position types are NOT defined
here yet because Phase 1 has no execution path; defining unused types invites
them to drift out of sync with the engine that eventually uses them. They will
be added in the phase that first needs them.
"""

from __future__ import annotations

from enum import Enum


class TradingMode(str, Enum):
    """Operating mode of the system.

    The default everywhere is PAPER. LIVE is defined so that config can
    recognise and explicitly reject it; no live execution code exists.
    """

    BACKTEST = "backtest"
    LIVE_SIGNAL = "live_signal"
    PAPER = "paper"
    LIVE = "live"


class InstrumentRole(str, Enum):
    """Why an instrument exists in this system.

    The signal/execution split is a hard architectural boundary. SIGNAL
    instruments generate signals and can never be traded; EXECUTION
    instruments carry positions, costs and P&L. An instrument may be both
    (a liquid equity could signal and trade), which is why this is a role
    rather than a mutually exclusive type.
    """

    SIGNAL = "signal"
    EXECUTION = "execution"
    BOTH = "both"


class InstrumentKind(str, Enum):
    """What the instrument actually is, for cost and margin modelling."""

    INDEX = "index"
    ETF = "etf"
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"


class Resolution(str, Enum):
    """Candle resolutions supported by the FYERS history API.

    Source: ``FyersModel.history`` docstring, fyers-apiv3 3.1.16, which states:
    "'Day' or '1D', '1', '2', '3', '5', '10', '15', '20', '30', '60', '120', '240'".

    Note that '45' and '180' appear in some community posts but are NOT in the
    installed SDK's documented list, so they are excluded here rather than
    assumed. Add them only if verified against the API.
    """

    M1 = "1"
    M2 = "2"
    M3 = "3"
    M5 = "5"
    M10 = "10"
    M15 = "15"
    M20 = "20"
    M30 = "30"
    M60 = "60"
    M120 = "120"
    M240 = "240"
    DAY = "1D"

    @property
    def minutes(self) -> int | None:
        """Bar length in minutes, or None for daily bars."""
        if self is Resolution.DAY:
            return None
        return int(self.value)
