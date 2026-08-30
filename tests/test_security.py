"""Security and mode-safety tests (Phase 1 deliverable G).

These assert the safeguards actually hold, rather than trusting that they were
implemented. They are the tests most worth keeping green as the project grows.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from config.settings import (
    LiveTradingNotImplementedError,
    MissingCredentialError,
    FyersCredentials,
    load_settings,
)
from core.logging_setup import REDACTED, RedactSecretsFilter, setup_logging
from core.types import TradingMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIRS = ("config", "core", "instruments", "marketdata", "brokers", "scripts")


# --- repository hygiene --------------------------------------------------


def test_gitignore_excludes_env():
    content = (PROJECT_ROOT / ".gitignore").read_text()
    assert ".env" in content
    assert "data_store/" in content
    assert "logs/" in content


def test_env_example_contains_no_values():
    """The template must have empty right-hand sides for every secret."""
    lines = (PROJECT_ROOT / ".env.example").read_text().splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if any(
            marker in key.upper()
            for marker in ("TOKEN", "SECRET", "KEY", "PASSWORD")
        ):
            assert value == "", f"{key} must be empty in .env.example, got {value!r}"


def test_no_real_env_file_is_committed():
    assert not (PROJECT_ROOT / ".env").exists() or ".env" in (
        PROJECT_ROOT / ".gitignore"
    ).read_text()


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        files.extend((PROJECT_ROOT / directory).rglob("*.py"))
    return files


def test_no_hardcoded_credentials_in_source():
    """Scan for assignments that look like embedded secrets."""
    # Matches e.g. access_token = "abc123..." but not access_token = None,
    # os.getenv(...), or a keyword argument referencing a variable.
    pattern = re.compile(
        r"""(secret_key|access_token|client_id|password|totp)\s*=\s*["'][A-Za-z0-9_\-]{8,}["']""",
        re.IGNORECASE,
    )
    offenders = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}")
    assert not offenders, f"Possible hardcoded credentials: {offenders}"


def test_no_order_placement_calls_in_source():
    """No code path may CALL an order-mutating SDK method.

    Parses the AST rather than grepping, so that naming a forbidden method
    inside a docstring or a blocklist string is not mistaken for invoking it.
    Only genuine call expressions count.
    """
    import ast

    forbidden = {
        "place_order",
        "modify_order",
        "cancel_order",
        "exit_positions",
        "place_basket_orders",
        "place_multileg_order",
        "place_gtt_order",
        "convert_position",
    }
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name in forbidden:
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{name}"
                )
    assert not offenders, f"Order-placement calls found: {offenders}"


def test_no_source_file_prints_a_secret_value():
    """`print(f"...{token}")` style leaks must not exist.

    Catches the specific mistake of echoing a credential to stdout, which is
    not caught by log redaction because it never reaches the logging system.
    """
    import ast

    secret_names = {"token", "access_token", "secret_key", "secret", "password"}
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id in secret_names:
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                    )
    assert not offenders, f"Secret printed to stdout at: {offenders}"


def test_login_writes_token_without_printing_it(tmp_path):
    """The token reaches .env, with other keys preserved and mode 0600."""
    import stat
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from fyers_login import write_token_to_env

    env_path = tmp_path / ".env"
    env_path.write_text(
        "TRADING_MODE=paper\nFYERS_CLIENT_ID=ABC-100\nFYERS_ACCESS_TOKEN=old\n"
    )
    write_token_to_env("brandnewtoken123", env_path)

    content = env_path.read_text()
    assert "FYERS_ACCESS_TOKEN=brandnewtoken123" in content
    assert "TRADING_MODE=paper" in content
    assert "FYERS_CLIENT_ID=ABC-100" in content
    assert "old" not in content

    mode = stat.S_IMODE(env_path.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_login_appends_token_when_absent(tmp_path):
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from fyers_login import write_token_to_env

    env_path = tmp_path / ".env"
    env_path.write_text("TRADING_MODE=paper\n")
    write_token_to_env("tok", env_path)
    assert "FYERS_ACCESS_TOKEN=tok" in env_path.read_text()
    assert "TRADING_MODE=paper" in env_path.read_text()


# --- trading mode safety -------------------------------------------------


def test_default_mode_is_paper(monkeypatch, tmp_path):
    monkeypatch.delenv("TRADING_MODE", raising=False)
    settings = load_settings(env_file=tmp_path / "nonexistent.env")
    assert settings.trading_mode is TradingMode.PAPER


def test_empty_mode_falls_back_to_paper(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "")
    settings = load_settings(env_file=tmp_path / "nonexistent.env")
    assert settings.trading_mode is TradingMode.PAPER


def test_live_mode_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "live")
    with pytest.raises(LiveTradingNotImplementedError, match="refused"):
        load_settings(env_file=tmp_path / "nonexistent.env")


def test_live_mode_refused_case_insensitively(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    with pytest.raises(LiveTradingNotImplementedError):
        load_settings(env_file=tmp_path / "nonexistent.env")


def test_unknown_mode_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "yolo")
    with pytest.raises(ValueError, match="not recognised"):
        load_settings(env_file=tmp_path / "nonexistent.env")


def test_paper_and_backtest_modes_load(monkeypatch, tmp_path):
    for mode in ("paper", "backtest", "live_signal"):
        monkeypatch.setenv("TRADING_MODE", mode)
        settings = load_settings(env_file=tmp_path / "nonexistent.env")
        assert settings.trading_mode.value == mode


# --- credential handling -------------------------------------------------


def test_credentials_repr_never_exposes_values():
    creds = FyersCredentials(
        client_id="ABCD1234-100",
        secret_key="supersecretvalue",
        access_token="tok_abcdef123456",
    )
    text = repr(creds)
    assert "supersecretvalue" not in text
    assert "tok_abcdef123456" not in text
    assert "set" in text


def test_missing_credentials_raise_actionable_error():
    creds = FyersCredentials()
    with pytest.raises(MissingCredentialError, match="Never commit"):
        creds.require_for_data()


def test_incomplete_credentials_are_detected():
    assert not FyersCredentials(client_id="X").is_complete_for_data
    assert not FyersCredentials(access_token="Y").is_complete_for_data
    assert FyersCredentials(client_id="X", access_token="Y").is_complete_for_data


# --- log redaction -------------------------------------------------------


def test_redaction_filter_scrubs_configured_secret(monkeypatch):
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "tok_verysecret_1234567890")
    log_filter = RedactSecretsFilter()
    record = logging.LogRecord(
        "t", logging.INFO, __file__, 1,
        "calling api with tok_verysecret_1234567890", None, None,
    )
    log_filter.filter(record)
    assert "tok_verysecret_1234567890" not in record.msg
    assert REDACTED in record.msg


def test_redaction_filter_scrubs_url_query_parameters():
    log_filter = RedactSecretsFilter()
    record = logging.LogRecord(
        "t", logging.INFO, __file__, 1,
        "GET https://example.com/cb?auth_code=SECRETCODE123&state=x", None, None,
    )
    log_filter.filter(record)
    assert "SECRETCODE123" not in record.msg


def test_token_does_not_reach_the_log_file(tmp_path, monkeypatch):
    """End-to-end: a secret logged by accident must not land on disk."""
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "tok_ontheDISK_9876543210")
    run_id = setup_logging(tmp_path, console=False)
    logging.getLogger("test").info(
        "token is tok_ontheDISK_9876543210 and should not appear"
    )
    logging.shutdown()

    contents = (tmp_path / f"run_{run_id}.log").read_text()
    assert "tok_ontheDISK_9876543210" not in contents
    assert REDACTED in contents


# --- read-only client guard ----------------------------------------------


class _OrderCapableSDK:
    """Stands in for FyersModel: has read methods AND order methods."""

    def history(self, data=None):
        return {"s": "ok", "candles": []}

    def quotes(self, data=None):
        return {"s": "ok"}

    def get_profile(self):
        return {"s": "ok"}

    def market_status(self):
        return {"s": "ok"}

    def place_order(self, data):
        raise AssertionError("place_order must never be reachable in a test")

    def positions(self):
        raise AssertionError("positions must never be reachable in a test")


def test_read_only_client_blocks_order_methods():
    from brokers.fyers.client import OrderPlacementBlockedError, ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    for name in ("place_order", "modify_order", "cancel_order", "exit_positions"):
        with pytest.raises(OrderPlacementBlockedError, match="blocked"):
            getattr(client, name)


def test_read_only_client_blocks_position_and_funds_access():
    from brokers.fyers.client import OrderPlacementBlockedError, ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    for name in ("positions", "funds", "holdings"):
        with pytest.raises(OrderPlacementBlockedError):
            getattr(client, name)


def test_read_only_client_allows_history():
    from brokers.fyers.client import ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    assert client.history(data={})["s"] == "ok"


def test_read_only_client_rejects_unknown_attribute():
    from brokers.fyers.client import ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    with pytest.raises(AttributeError, match="allowlist"):
        getattr(client, "tradebook")


def test_read_only_client_is_immutable():
    from brokers.fyers.client import ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    with pytest.raises(AttributeError, match="immutable"):
        client.anything = 1


# --- the actual containment properties, not just the happy path ----------


def test_sdk_object_is_not_stored_as_an_attribute():
    """Regression: the SDK was reachable via client._inner in one step."""
    from brokers.fyers.client import ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    with pytest.raises((AttributeError, Exception)):
        getattr(client, "_inner")
    assert not hasattr(client, "__dict__"), "__slots__ must prevent an instance dict"


def test_vars_does_not_expose_the_sdk():
    """Regression: vars(client)['_inner'].place_order(...) used to work."""
    from brokers.fyers.client import ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    with pytest.raises(TypeError):
        vars(client)


def test_public_surface_is_exactly_the_allowlist():
    from brokers.fyers.client import ALLOWED_METHODS, ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    public = {n for n in dir(client) if not n.startswith("__")}
    assert public == set(ALLOWED_METHODS), f"unexpected public surface: {public}"


def test_forwarded_callables_do_not_expose_self():
    """A bound method would leak the SDK via __self__; closures do not."""
    from brokers.fyers.client import ReadOnlyFyersClient

    client = ReadOnlyFyersClient(_OrderCapableSDK())
    assert not hasattr(client.history, "__self__")
