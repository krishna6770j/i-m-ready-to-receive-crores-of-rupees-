"""Signal/execution separation tests.

This boundary is the one most likely to be violated silently, so it is tested
behaviourally: the index must actively refuse to be traded.
"""

from __future__ import annotations

import pytest

from core.types import InstrumentKind, InstrumentRole
from instruments.instrument import Instrument, NotTradableError
from instruments.registry import load_registry


def nifty_index() -> Instrument:
    return Instrument(
        symbol="NSE:NIFTY50-INDEX",
        name="NIFTY 50",
        kind=InstrumentKind.INDEX,
        role=InstrumentRole.SIGNAL,
    )


def test_index_cannot_be_traded():
    with pytest.raises(NotTradableError, match="signal source only"):
        nifty_index().require_tradable()


def test_index_cannot_be_priced_to_tick():
    with pytest.raises(NotTradableError):
        nifty_index().round_to_tick(24000.07)


def test_index_cannot_compute_lots():
    with pytest.raises(NotTradableError):
        nifty_index().lots_for_quantity(100)


def test_index_can_generate_signals():
    assert nifty_index().can_generate_signals
    assert not nifty_index().is_tradable


def test_index_with_execution_role_is_rejected_at_construction():
    """An index is not tradable regardless of how it is configured."""
    with pytest.raises(ValueError, match="cannot have role"):
        Instrument(
            symbol="NSE:NIFTY50-INDEX",
            name="NIFTY 50",
            kind=InstrumentKind.INDEX,
            role=InstrumentRole.EXECUTION,
        )


def test_tradable_instrument_rounds_to_tick():
    etf = Instrument(
        symbol="NSE:NIFTYBEES-EQ",
        name="NiftyBeES",
        kind=InstrumentKind.ETF,
        role=InstrumentRole.EXECUTION,
        tick_size=0.01,
    )
    assert etf.round_to_tick(280.117) == pytest.approx(280.12)


def test_lot_size_must_be_positive():
    with pytest.raises(ValueError, match="lot_size"):
        Instrument(
            symbol="X",
            name="X",
            kind=InstrumentKind.FUTURES,
            role=InstrumentRole.EXECUTION,
            lot_size=0,
        )


def test_tick_size_must_be_positive():
    with pytest.raises(ValueError, match="tick_size"):
        Instrument(
            symbol="X",
            name="X",
            kind=InstrumentKind.EQUITY,
            role=InstrumentRole.EXECUTION,
            tick_size=0.0,
        )


# --- registry ------------------------------------------------------------


def test_registry_loads_signal_instrument():
    registry = load_registry()
    nifty = registry.signal("NSE:NIFTY50-INDEX")
    assert nifty.kind is InstrumentKind.INDEX
    assert not nifty.is_tradable


def test_registry_holds_multiple_execution_candidates():
    """Manager correction #1: multiple candidates, none selected."""
    registry = load_registry()
    assert len(registry.execution_candidates) >= 2


def test_no_execution_instrument_is_selected():
    registry = load_registry()
    with pytest.raises(NotImplementedError, match="No execution instrument"):
        _ = registry.selected_execution_instrument


def test_all_execution_candidates_are_unverified():
    registry = load_registry()
    assert all(not c.verified for c in registry.execution_candidates)


def test_unverified_candidate_refuses_use():
    registry = load_registry()
    candidate = registry.execution_candidates[0]
    with pytest.raises(ValueError, match="not verified"):
        candidate.require_verified()


def test_unknown_signal_symbol_raises():
    registry = load_registry()
    with pytest.raises(KeyError):
        registry.signal("NSE:DOESNOTEXIST")
