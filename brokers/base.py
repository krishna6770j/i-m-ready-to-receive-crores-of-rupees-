"""Abstract broker interfaces.

Phase 1 defines ONLY the read-only market data interface. There is deliberately
no order-placement interface, abstract or otherwise: an abstract method that
nothing implements is still a shape that invites implementation, and the
cheapest guarantee that this codebase cannot place an order is that no method
for doing so exists anywhere in it.

When execution is eventually approved, an ``OrderExecutionProvider`` will be
added as a SEPARATE protocol, and the risk manager will sit between the
strategy and any implementation of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class BrokerError(RuntimeError):
    """Base class for broker adapter failures."""


class BrokerAuthError(BrokerError):
    """Authentication failed or the token has expired."""


class BrokerRateLimitError(BrokerError):
    """The broker signalled that we exceeded a rate limit."""


class BrokerDataError(BrokerError):
    """The broker returned a malformed or unexpected payload."""


class HistoricalDataProvider(ABC):
    """Read-only access to historical candles.

    Implementations must return frames in the canonical schema defined in
    ``marketdata.schemas`` so that callers cannot tell one vendor from another.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier recorded in dataset provenance, e.g. 'fyers:history'."""

    @abstractmethod
    def fetch_candles(
        self,
        symbol: str,
        resolution: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch candles for an inclusive date range.

        Must return a canonical OHLCV frame, empty if the range holds no data.
        Must NOT silently truncate a range that exceeds the vendor's per-request
        limit; chunking is the implementation's responsibility.
        """
