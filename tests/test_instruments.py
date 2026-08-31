"""Signal/execution separation tests.

This boundary is the one most likely to be violated silently, so it is tested
behaviourally: the index must actively refuse to be traded.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from core.types import InstrumentKind, InstrumentRole
from instruments.instrument import Instrument, NotTradableError
from instruments.registry import InstrumentConfigError, load_registry


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


# ---------------------------------------------------------------------------
# Unit 15: strict instrument configuration validation (architecture section
# 23). Every test below constructs a minimal, otherwise-valid config in
# isolation and mutates exactly one thing, so a failure always points at the
# one rule under test rather than at unrelated real-config content.
# ---------------------------------------------------------------------------


def _base_config() -> dict:
    return {
        "schema_version": 1,
        "signal_instruments": [
            {
                "symbol": "NSE:TEST-INDEX",
                "name": "Test Index",
                "kind": "index",
                "role": "signal",
                "lot_size": 1,
                "tick_size": None,
                "exchange": "NSE",
                "notes": "a signal instrument",
            }
        ],
        "execution_candidates": [
            {
                "symbol": "NSE:TEST-ETF",
                "name": "Test ETF",
                "kind": "etf",
                "role": "execution",
                "lot_size": 1,
                "tick_size": 0.01,
                "exchange": "NSE",
                "verified": False,
                "notes": "an execution candidate",
            }
        ],
    }


def _load(tmp_path, config: dict):
    path = tmp_path / "instruments.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return load_registry(path)


def _load_ok(tmp_path, config: dict):
    """Load and assert it succeeded, returning the registry."""
    return _load(tmp_path, config)


# --- schema_version --------------------------------------------------------


def test_schema_version_missing_rejected(tmp_path):
    config = _base_config()
    del config["schema_version"]
    with pytest.raises(InstrumentConfigError, match="schema_version"):
        _load(tmp_path, config)


def test_schema_version_bool_rejected(tmp_path):
    config = _base_config()
    config["schema_version"] = True
    with pytest.raises(InstrumentConfigError, match="schema_version"):
        _load(tmp_path, config)


def test_schema_version_wrong_number_rejected(tmp_path):
    config = _base_config()
    config["schema_version"] = 2
    with pytest.raises(InstrumentConfigError, match="schema_version"):
        _load(tmp_path, config)


def test_schema_version_float_rejected(tmp_path):
    config = _base_config()
    config["schema_version"] = 1.0
    with pytest.raises(InstrumentConfigError, match="schema_version"):
        _load(tmp_path, config)


def test_schema_version_string_rejected(tmp_path):
    config = _base_config()
    config["schema_version"] = "1"
    with pytest.raises(InstrumentConfigError, match="schema_version"):
        _load(tmp_path, config)


def test_unknown_top_level_key_rejected(tmp_path):
    config = _base_config()
    config["session_boundaries"] = []
    with pytest.raises(InstrumentConfigError, match="unknown top-level key"):
        _load(tmp_path, config)


# --- lot_size ----------------------------------------------------------


@pytest.mark.parametrize("bad_value", [0, -1, True, False, 1.0, "1", ""])
def test_signal_lot_size_rejected(tmp_path, bad_value):
    config = _base_config()
    config["signal_instruments"][0]["lot_size"] = bad_value
    with pytest.raises(InstrumentConfigError, match="lot_size"):
        _load(tmp_path, config)


def test_verified_candidate_lot_size_zero_rejected_and_never_becomes_one(tmp_path):
    """Critical regression: lot_size: 0 must raise, and must NEVER silently
    become 1 (the exact defect ``int(lot_size) if lot_size else 1``)."""
    config = _base_config()
    config["execution_candidates"][0]["verified"] = True
    config["execution_candidates"][0]["lot_size"] = 0
    with pytest.raises(InstrumentConfigError, match="lot_size"):
        registry = load_registry(_write(tmp_path, config))
        # If this line were ever reached, lot_size must not have become 1.
        assert registry.execution_candidates[0].instrument.lot_size != 1


def _write(tmp_path, config: dict):
    path = tmp_path / "instruments.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.mark.parametrize("bad_value", [0, -1, True, False, 1.0, "1", ""])
def test_verified_candidate_lot_size_rejected(tmp_path, bad_value):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = True
    config["execution_candidates"][0]["lot_size"] = bad_value
    with pytest.raises(InstrumentConfigError, match="lot_size"):
        _load(tmp_path, config)


def test_unverified_candidate_lot_size_none_accepted(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = False
    config["execution_candidates"][0]["lot_size"] = None
    registry = _load_ok(tmp_path, config)
    candidate = registry.execution_candidates[0]
    assert candidate.instrument is None


def test_unverified_candidate_positive_lot_size_accepted(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = False
    config["execution_candidates"][0]["lot_size"] = 5
    registry = _load_ok(tmp_path, config)
    assert registry.execution_candidates[0].instrument.lot_size == 5


# --- verified ------------------------------------------------------------


def test_verified_absent_defaults_to_false(tmp_path):
    config = _base_config()
    del config["execution_candidates"][0]["verified"]
    registry = _load_ok(tmp_path, config)
    assert registry.execution_candidates[0].verified is False


def test_verified_exact_false_accepted(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = False
    registry = _load_ok(tmp_path, config)
    assert registry.execution_candidates[0].verified is False


def test_verified_exact_true_accepted_when_all_requirements_met(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = True
    registry = _load_ok(tmp_path, config)
    candidate = registry.execution_candidates[0]
    assert candidate.verified is True
    assert candidate.instrument is not None


def test_verified_string_false_rejected_and_never_becomes_true(tmp_path):
    """Critical regression: verified: "false" must raise, and must NEVER
    silently become True (the exact defect
    ``bool(entry.get("verified", False))``, under which
    ``bool("false") == True``)."""
    config = _base_config()
    config["execution_candidates"][0]["verified"] = "false"
    with pytest.raises(InstrumentConfigError, match="verified"):
        registry = load_registry(_write(tmp_path, config))
        assert registry.execution_candidates[0].verified is not True


def test_verified_int_zero_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = 0
    with pytest.raises(InstrumentConfigError, match="verified"):
        _load(tmp_path, config)


def test_verified_int_one_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = 1
    with pytest.raises(InstrumentConfigError, match="verified"):
        _load(tmp_path, config)


def test_verified_explicit_none_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = None
    with pytest.raises(InstrumentConfigError, match="verified"):
        _load(tmp_path, config)


def test_signal_entry_must_not_contain_verified(tmp_path):
    config = _base_config()
    config["signal_instruments"][0]["verified"] = False
    with pytest.raises(InstrumentConfigError, match="unknown field"):
        _load(tmp_path, config)


# --- tick_size -----------------------------------------------------------


def test_unverified_candidate_tick_size_none_accepted(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = False
    config["execution_candidates"][0]["tick_size"] = None
    registry = _load_ok(tmp_path, config)
    assert registry.execution_candidates[0].instrument.tick_size is None


def test_unverified_candidate_positive_tick_size_accepted(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = False
    config["execution_candidates"][0]["tick_size"] = 0.05
    registry = _load_ok(tmp_path, config)
    assert registry.execution_candidates[0].instrument.tick_size == pytest.approx(0.05)


@pytest.mark.parametrize("bad_value", [0, -0.01, True, False, float("nan"), float("inf"), float("-inf"), "0.01"])
def test_tick_size_rejected(tmp_path, bad_value):
    config = _base_config()
    config["execution_candidates"][0]["tick_size"] = bad_value
    with pytest.raises(InstrumentConfigError, match="tick_size"):
        _load(tmp_path, config)


def test_verified_candidate_missing_tick_size_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = True
    config["execution_candidates"][0]["tick_size"] = None
    with pytest.raises(InstrumentConfigError, match="tick_size"):
        _load(tmp_path, config)


# --- structure -------------------------------------------------------------


def test_missing_required_symbol_on_signal_rejected(tmp_path):
    config = _base_config()
    del config["signal_instruments"][0]["symbol"]
    with pytest.raises(InstrumentConfigError, match="symbol"):
        _load(tmp_path, config)


def test_symbol_none_on_unverified_candidate_accepted(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = False
    config["execution_candidates"][0]["symbol"] = None
    registry = _load_ok(tmp_path, config)
    candidate = registry.execution_candidates[0]
    assert candidate.instrument is None
    assert candidate.verified is False


def test_symbol_none_on_verified_candidate_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = True
    config["execution_candidates"][0]["symbol"] = None
    with pytest.raises(InstrumentConfigError, match="symbol"):
        _load(tmp_path, config)


def test_wrong_section_role_on_signal_rejected(tmp_path):
    config = _base_config()
    config["signal_instruments"][0]["role"] = "execution"
    with pytest.raises(InstrumentConfigError, match="role"):
        _load(tmp_path, config)


def test_wrong_section_role_on_candidate_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["role"] = "signal"
    with pytest.raises(InstrumentConfigError, match="role"):
        _load(tmp_path, config)


def test_unknown_entry_key_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["margin_pct"] = 0.15
    with pytest.raises(InstrumentConfigError, match="unknown field"):
        _load(tmp_path, config)


def test_duplicate_signal_symbol_rejected(tmp_path):
    config = _base_config()
    config["signal_instruments"].append(copy.deepcopy(config["signal_instruments"][0]))
    with pytest.raises(InstrumentConfigError, match="duplicate symbol"):
        _load(tmp_path, config)


def test_duplicate_candidate_symbol_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"].append(
        copy.deepcopy(config["execution_candidates"][0])
    )
    with pytest.raises(InstrumentConfigError, match="duplicate symbol"):
        _load(tmp_path, config)


def test_duplicate_signal_and_candidate_symbol_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["symbol"] = config["signal_instruments"][0]["symbol"]
    with pytest.raises(InstrumentConfigError, match="duplicate symbol"):
        _load(tmp_path, config)


def test_multiple_none_symbol_candidates_are_not_duplicates(tmp_path):
    """architecture section 23 has no exact rule against multiple None
    symbols; None carries no instrument identity to collide on."""
    config = _base_config()
    config["execution_candidates"][0]["verified"] = False
    config["execution_candidates"][0]["symbol"] = None
    config["execution_candidates"].append(
        {
            "symbol": None,
            "name": "Another unverified candidate",
            "kind": "options",
            "role": "execution",
            "lot_size": None,
            "tick_size": None,
            "exchange": "NSE",
            "verified": False,
            "notes": "",
        }
    )
    registry = _load_ok(tmp_path, config)
    assert len(registry.execution_candidates) == 2
    assert all(c.instrument is None for c in registry.execution_candidates)


def test_malformed_section_type_rejected(tmp_path):
    config = _base_config()
    config["signal_instruments"] = "not a list"
    with pytest.raises(InstrumentConfigError, match="signal_instruments"):
        _load(tmp_path, config)


def test_malformed_entry_type_rejected(tmp_path):
    config = _base_config()
    config["execution_candidates"][0] = ["not", "a", "mapping"]
    with pytest.raises(InstrumentConfigError, match="mapping"):
        _load(tmp_path, config)


def test_missing_signal_instruments_key_rejected(tmp_path):
    config = _base_config()
    del config["signal_instruments"]
    with pytest.raises(InstrumentConfigError, match="signal_instruments"):
        _load(tmp_path, config)


def test_missing_execution_candidates_key_rejected(tmp_path):
    config = _base_config()
    del config["execution_candidates"]
    with pytest.raises(InstrumentConfigError, match="execution_candidates"):
        _load(tmp_path, config)


def test_identity_field_with_whitespace_rejected(tmp_path):
    config = _base_config()
    config["signal_instruments"][0]["symbol"] = "  NSE:TEST-INDEX  "
    with pytest.raises(InstrumentConfigError, match="symbol"):
        _load(tmp_path, config)


def test_identity_field_empty_string_rejected(tmp_path):
    config = _base_config()
    config["signal_instruments"][0]["exchange"] = ""
    with pytest.raises(InstrumentConfigError, match="exchange"):
        _load(tmp_path, config)


def test_identity_field_wrong_type_rejected(tmp_path):
    config = _base_config()
    config["signal_instruments"][0]["kind"] = 123
    with pytest.raises(InstrumentConfigError, match="kind"):
        _load(tmp_path, config)


def test_invalid_kind_enum_value_rejected(tmp_path):
    config = _base_config()
    config["signal_instruments"][0]["kind"] = "not_a_real_kind"
    with pytest.raises(InstrumentConfigError, match="kind"):
        _load(tmp_path, config)


def test_name_defaults_to_symbol_when_absent(tmp_path):
    config = _base_config()
    del config["execution_candidates"][0]["name"]
    registry = _load_ok(tmp_path, config)
    assert registry.execution_candidates[0].name == "NSE:TEST-ETF"


def test_name_defaults_to_unnamed_when_symbol_and_name_absent(tmp_path):
    config = _base_config()
    config["execution_candidates"][0]["verified"] = False
    config["execution_candidates"][0]["symbol"] = None
    config["execution_candidates"][0]["lot_size"] = None
    del config["execution_candidates"][0]["name"]
    registry = _load_ok(tmp_path, config)
    assert registry.execution_candidates[0].name == "unnamed"


def test_notes_defaults_to_empty_string_when_absent(tmp_path):
    config = _base_config()
    del config["signal_instruments"][0]["notes"]
    registry = _load_ok(tmp_path, config)
    assert registry.signal_instruments["NSE:TEST-INDEX"].notes == ""


def test_valid_config_still_loads_the_real_file():
    """The actual config/instruments.yaml must remain loadable after adding
    schema_version, with no execution instrument selected or verified."""
    registry = load_registry()
    assert registry.signal("NSE:NIFTY50-INDEX").kind is InstrumentKind.INDEX
    niftybees = next(
        c for c in registry.execution_candidates if c.name.startswith("Nippon")
    )
    assert niftybees.instrument is not None
    assert niftybees.instrument.lot_size == 1
    assert niftybees.instrument.tick_size is None
    nifty_fut = next(
        c for c in registry.execution_candidates if "Futures" in c.name
    )
    assert nifty_fut.instrument is None
    options = next(c for c in registry.execution_candidates if "Options" in c.name)
    assert options.instrument is None
    assert all(not c.verified for c in registry.execution_candidates)
