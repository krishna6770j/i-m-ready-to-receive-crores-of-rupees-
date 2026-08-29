"""Shared test fixtures.

The synthetic OHLCV generator is deterministic (fixed seed) so that tests are
reproducible and so that reproducibility itself can be tested.

IMPORTANT: this generator produces SYNTHETIC data for exercising the pipeline.
It is not market data and must never be used for strategy research or presented
as a backtest input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.timeutils import IST_NAME  # noqa: E402
from marketdata.schemas import (  # noqa: E402
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    TS,
    VOLUME,
    normalise,
)


def make_ohlcv(
    n: int = 100,
    *,
    start: str = "2026-01-01 09:15",
    freq: str = "1min",
    seed: int = 42,
    start_price: float = 24000.0,
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame in canonical form."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n, freq=freq, tz=IST_NAME)

    steps = rng.normal(0.0, 5.0, size=n)
    close = start_price + np.cumsum(steps)
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    span = np.abs(rng.normal(0.0, 3.0, size=n)) + 0.5
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.integers(100, 10_000, size=n)

    return normalise(
        pd.DataFrame(
            {
                TS: ts,
                OPEN: open_,
                HIGH: high,
                LOW: low,
                CLOSE: close,
                VOLUME: volume,
            }
        )
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    d = tmp_path / "data_store"
    d.mkdir()
    return d


class FakeFyersClient:
    """Stands in for FyersModel, implementing only ``history(data) -> dict``.

    Records every request so tests can assert on chunking behaviour without a
    network connection or credentials.
    """

    def __init__(self, responses: list[dict] | dict):
        self._responses = responses if isinstance(responses, list) else [responses]
        self._index = 0
        self.requests: list[dict] = []

    def history(self, data=None):
        self.requests.append(dict(data or {}))
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response


def candles_payload(n: int = 5, *, start_epoch: int = 1767239100) -> dict:
    """A well-formed FYERS /history success payload.

    Row order is [epoch, open, high, low, close, volume]; start_epoch
    corresponds to 2026-01-01 09:15:00 IST.
    """
    rows = []
    for i in range(n):
        base = 24000.0 + i
        rows.append(
            [start_epoch + i * 60, base, base + 5, base - 5, base + 1, 1000 + i]
        )
    return {"s": "ok", "candles": rows}
