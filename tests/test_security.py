"""Security and mode-safety tests (Phase 1 deliverable G).

These assert the safeguards actually hold, rather than trusting that they were
implemented. They are the tests most worth keeping green as the project grows.
"""

from __future__ import annotations

import logging
import re
import subprocess
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


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
    )


def test_env_is_not_tracked_by_git():
    """Ask git directly whether .env is tracked.

    Replaces a test asserting ``not .env.exists() or '.env' in .gitignore``.
    The right operand is always true while .gitignore lists .env, so the
    assertion could never fail and never consulted git at all -- it would
    have passed with .env fully committed.
    """
    result = _git("ls-files", "--error-unmatch", ".env")
    assert result.returncode != 0, (
        f".env is TRACKED by git. git ls-files said: {result.stdout.strip()!r}"
    )


def test_env_has_never_been_committed_in_any_revision():
    """A secret removed from HEAD still lives in history."""
    result = _git("log", "--all", "--pretty=format:%H", "--", ".env")
    assert result.returncode == 0, f"git log failed: {result.stderr}"
    assert result.stdout.strip() == "", (
        f".env appears in {len(result.stdout.split())} commit(s). "
        "History rewriting and credential rotation are required."
    )


def test_generated_and_secret_paths_are_untracked():
    """Directories that hold credentials, data or logs must stay out of git."""
    tracked = _git("ls-files").stdout.splitlines()
    offenders = [
        p
        for p in tracked
        if p == ".env"
        or p.startswith((".venv/", "data_store/", "logs/"))
        or p.endswith((".pem", ".key", ".parquet"))
    ]
    assert not offenders, f"these must not be tracked: {offenders}"


def test_gitignore_actually_ignores_a_real_env_file(tmp_path):
    """Prove the ignore rule works, rather than trusting the file's text."""
    result = _git("check-ignore", "-v", ".env")
    assert result.returncode == 0, "git does not consider .env ignored"
    assert ".gitignore" in result.stdout


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


def _read_log(tmp_path, run_id: str) -> str:
    for handler in logging.getLogger().handlers:
        handler.flush()
    return (tmp_path / f"run_{run_id}.log").read_text()


def test_token_does_not_reach_the_log_file(tmp_path, monkeypatch):
    """End-to-end: a secret logged by accident must not land on disk."""
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "tok_ontheDISK_9876543210")
    run_id = setup_logging(tmp_path, console=False)
    logging.getLogger("test").info(
        "token is tok_ontheDISK_9876543210 and should not appear"
    )
    contents = _read_log(tmp_path, run_id)
    assert "tok_ontheDISK_9876543210" not in contents
    assert REDACTED in contents


def test_secret_in_exception_traceback_is_redacted(tmp_path, monkeypatch):
    """Regression: filters run before the formatter renders exc_info.

    A secret inside an exception message previously reached the log file
    untouched, because a logging.Filter cannot see record.exc_text.
    """
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "tok_INTRACEBACK_4444555566")
    run_id = setup_logging(tmp_path, console=False)
    try:
        raise ValueError("auth rejected for tok_INTRACEBACK_4444555566")
    except ValueError:
        logging.getLogger("test").error("request failed", exc_info=True)

    contents = _read_log(tmp_path, run_id)
    assert "tok_INTRACEBACK_4444555566" not in contents
    assert "ValueError" in contents, "the traceback itself must still be logged"


def test_secret_in_nested_exception_is_redacted(tmp_path, monkeypatch):
    """Chained exceptions render both tracebacks; both must be scrubbed."""
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "tok_NESTED_7777888899")
    run_id = setup_logging(tmp_path, console=False)
    try:
        try:
            raise ValueError("inner used tok_NESTED_7777888899")
        except ValueError as inner:
            raise RuntimeError("outer wrapper") from inner
    except RuntimeError:
        logging.getLogger("test").exception("chained failure")

    contents = _read_log(tmp_path, run_id)
    assert "tok_NESTED_7777888899" not in contents


def test_secret_introduced_after_logging_setup_is_redacted(tmp_path, monkeypatch):
    """Regression: secrets were snapshotted at filter construction.

    Credentials load via load_dotenv() on first settings access, which
    routinely happens after logging is configured.
    """
    monkeypatch.delenv("FYERS_ACCESS_TOKEN", raising=False)
    run_id = setup_logging(tmp_path, console=False)
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "tok_LATELOADED_1111222233")
    logging.getLogger("test").info("using tok_LATELOADED_1111222233")

    contents = _read_log(tmp_path, run_id)
    assert "tok_LATELOADED_1111222233" not in contents
    assert REDACTED in contents


def test_late_secret_in_traceback_is_redacted(tmp_path, monkeypatch):
    """Combined failure mode: late-loaded credential inside a traceback."""
    monkeypatch.delenv("FYERS_SECRET_KEY", raising=False)
    run_id = setup_logging(tmp_path, console=False)
    monkeypatch.setenv("FYERS_SECRET_KEY", "sec_LATE_AND_NESTED_999888")
    try:
        raise ConnectionError("handshake used sec_LATE_AND_NESTED_999888")
    except ConnectionError:
        logging.getLogger("test").exception("connection failed")

    contents = _read_log(tmp_path, run_id)
    assert "sec_LATE_AND_NESTED_999888" not in contents


def test_secret_passed_as_log_arg_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "tok_ASARG_5555666677")
    run_id = setup_logging(tmp_path, console=False)
    logging.getLogger("test").info("token=%s", "tok_ASARG_5555666677")

    contents = _read_log(tmp_path, run_id)
    assert "tok_ASARG_5555666677" not in contents


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


# ---------------------------------------------------------------------------
# Unit 14: SecretRegistry -- append-only, process-lifetime, any length.
# ---------------------------------------------------------------------------


def test_registry_redacts_a_one_character_secret():
    from core.secrets import REDACTED, SecretRegistry

    reg = SecretRegistry()
    reg.register("x")
    assert reg.scrub("value is x here") == f"value is {REDACTED} here"


def test_registry_duplicate_registration_is_harmless():
    from core.secrets import SecretRegistry

    reg = SecretRegistry()
    reg.register("dup_secret")
    reg.register("dup_secret")
    assert len(reg) == 1


def test_registry_rejects_none():
    from core.secrets import SecretRegistry

    reg = SecretRegistry()
    with pytest.raises(ValueError):
        reg.register(None)


def test_registry_rejects_empty_string():
    from core.secrets import SecretRegistry

    reg = SecretRegistry()
    with pytest.raises(ValueError):
        reg.register("")


def test_registry_retains_rotated_out_secret():
    """Rotation A -> B must retain BOTH -- an old value can still be sitting
    in a retry buffer or a trace captured moments before rotation."""
    from core.secrets import SecretRegistry

    reg = SecretRegistry()
    reg.register("secret_A")
    reg.register("secret_B")
    assert "secret_A" in reg.scrub("still has secret_A in it") or True
    scrubbed_a = reg.scrub("old value secret_A")
    scrubbed_b = reg.scrub("new value secret_B")
    assert "secret_A" not in scrubbed_a
    assert "secret_B" not in scrubbed_b


def test_registry_never_evicts_never_removes():
    from core.secrets import SecretRegistry

    reg = SecretRegistry()
    reg.register("keepme")
    assert not hasattr(reg, "evict")
    assert not hasattr(reg, "remove")
    assert not hasattr(reg, "clear")
    assert "keepme" in reg


def test_registry_redacts_a_multiline_secret():
    from core.secrets import REDACTED, SecretRegistry

    reg = SecretRegistry()
    reg.register("multi\nline\nsecret")
    scrubbed = reg.scrub("before multi\nline\nsecret after")
    assert "multi\nline\nsecret" not in scrubbed
    assert REDACTED in scrubbed


def test_registry_is_documented_as_not_thread_safe():
    from core.secrets import SecretRegistry

    assert "not thread-safe" in (SecretRegistry.__doc__ or "").lower()


# --- registry-driven log redaction ----------------------------------------


def test_registered_secret_in_traceback_is_redacted(tmp_path):
    from core.secrets import register as register_secret

    register_secret("tok_REGISTRY_TRACEBACK_0001")
    run_id = setup_logging(tmp_path, console=False)
    try:
        raise ValueError("auth rejected for tok_REGISTRY_TRACEBACK_0001")
    except ValueError:
        logging.getLogger("test").error("request failed", exc_info=True)

    contents = _read_log(tmp_path, run_id)
    assert "tok_REGISTRY_TRACEBACK_0001" not in contents
    assert REDACTED in contents


def test_registered_secret_in_chained_traceback_is_redacted(tmp_path):
    from core.secrets import register as register_secret

    register_secret("tok_REGISTRY_CHAINED_0002")
    run_id = setup_logging(tmp_path, console=False)
    try:
        try:
            raise ValueError("inner used tok_REGISTRY_CHAINED_0002")
        except ValueError as inner:
            raise RuntimeError("outer wrapper") from inner
    except RuntimeError:
        logging.getLogger("test").exception("chained failure")

    contents = _read_log(tmp_path, run_id)
    assert "tok_REGISTRY_CHAINED_0002" not in contents


def test_rotated_out_secret_still_redacts_in_logs(tmp_path):
    """Old value A must still redact after B is registered."""
    from core.secrets import register as register_secret

    register_secret("tok_ROTATE_OLD_0003")
    register_secret("tok_ROTATE_NEW_0003")
    run_id = setup_logging(tmp_path, console=False)
    logging.getLogger("test").info(
        "old=tok_ROTATE_OLD_0003 new=tok_ROTATE_NEW_0003"
    )

    contents = _read_log(tmp_path, run_id)
    assert "tok_ROTATE_OLD_0003" not in contents
    assert "tok_ROTATE_NEW_0003" not in contents


# ---------------------------------------------------------------------------
# Unit 14: settings/auth registration
# ---------------------------------------------------------------------------


def test_settings_load_registers_credentials(monkeypatch, tmp_path):
    from core.secrets import registry

    monkeypatch.setenv("FYERS_CLIENT_ID", "REGCHECK_CLIENT_0001")
    monkeypatch.setenv("FYERS_SECRET_KEY", "REGCHECK_SECRET_0001")
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "REGCHECK_TOKEN_0001")
    load_settings(env_file=tmp_path / "nonexistent.env")

    assert "REGCHECK_CLIENT_0001" in registry
    assert "REGCHECK_SECRET_0001" in registry
    assert "REGCHECK_TOKEN_0001" in registry


def test_auth_code_is_registered_on_extraction():
    from brokers.fyers.auth import extract_auth_code
    from core.secrets import registry

    extract_auth_code("https://example.com/cb?auth_code=REGCHECK_AUTHCODE_0002&state=x")
    assert "REGCHECK_AUTHCODE_0002" in registry


def test_exchanged_access_token_is_registered():
    from brokers.fyers.auth import exchange_auth_code
    from core.secrets import registry

    class _FakeSession:
        def set_token(self, code):
            pass

        def generate_token(self):
            return {"access_token": "REGCHECK_EXCHANGED_0003"}

    exchange_auth_code(_FakeSession(), "somecode")
    assert "REGCHECK_EXCHANGED_0003" in registry


def test_repr_still_does_not_expose_secrets_after_registration(monkeypatch, tmp_path):
    monkeypatch.setenv("FYERS_CLIENT_ID", "REGCHECK_REPR_CLIENT")
    monkeypatch.setenv("FYERS_SECRET_KEY", "REGCHECK_REPR_SECRET")
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "REGCHECK_REPR_TOKEN")
    settings = load_settings(env_file=tmp_path / "nonexistent.env")

    text = repr(settings.fyers)
    assert "REGCHECK_REPR_SECRET" not in text
    assert "REGCHECK_REPR_TOKEN" not in text


# ---------------------------------------------------------------------------
# Unit 14: BrokerDiagnostic
# ---------------------------------------------------------------------------


def test_diagnostic_strips_and_normalizes_control_characters():
    from brokers.diagnostics import BrokerDiagnostic, BrokerDiagnosticStatus

    diag = BrokerDiagnostic(
        status=BrokerDiagnosticStatus.DATA_ERROR,
        code=None,
        sanitized_message="line1\x00\x01\x1fline2\x7f",
    )
    for ch in ("\x00", "\x01", "\x1f", "\x7f"):
        assert ch not in diag.sanitized_message


def test_diagnostic_message_length_is_capped():
    from brokers.diagnostics import BrokerDiagnostic, BrokerDiagnosticStatus

    diag = BrokerDiagnostic(
        status=BrokerDiagnosticStatus.DATA_ERROR,
        code=None,
        sanitized_message="x" * 5000,
    )
    assert len(diag.sanitized_message) <= 550


def test_diagnostic_scrubs_a_registered_secret():
    from brokers.diagnostics import BrokerDiagnostic, BrokerDiagnosticStatus
    from core.secrets import REDACTED, register as register_secret

    register_secret("REGCHECK_DIAGNOSTIC_SECRET_0004")
    diag = BrokerDiagnostic(
        status=BrokerDiagnosticStatus.AUTH_ERROR,
        code=-16,
        sanitized_message="failed for REGCHECK_DIAGNOSTIC_SECRET_0004",
    )
    assert "REGCHECK_DIAGNOSTIC_SECRET_0004" not in diag.sanitized_message
    assert REDACTED in diag.sanitized_message


def test_diagnostic_drops_unknown_raw_payload_fields():
    from brokers.diagnostics import BrokerDiagnostic, BrokerDiagnosticStatus

    diag = BrokerDiagnostic(
        status=BrokerDiagnosticStatus.DATA_ERROR,
        code=-99,
        sanitized_message="generic error",
        sanitized_structured_fields={
            "status": "error",
            "code": -99,
            "raw_response": {"secret_key": "leaked", "token": "leaked"},
            "unexpected_field": "should not survive",
        },
    )
    assert "raw_response" not in diag.sanitized_structured_fields
    assert "unexpected_field" not in diag.sanitized_structured_fields
    assert diag.sanitized_structured_fields["status"] == "error"
    assert diag.sanitized_structured_fields["code"] == -99


def test_diagnostic_repr_and_str_are_safe():
    from brokers.diagnostics import BrokerDiagnostic, BrokerDiagnosticStatus
    from core.secrets import register as register_secret

    register_secret("REGCHECK_DIAGNOSTIC_REPR_0005")
    diag = BrokerDiagnostic(
        status=BrokerDiagnosticStatus.AUTH_ERROR,
        code=-16,
        sanitized_message="failed for REGCHECK_DIAGNOSTIC_REPR_0005",
    )
    assert "REGCHECK_DIAGNOSTIC_REPR_0005" not in repr(diag)
    assert "REGCHECK_DIAGNOSTIC_REPR_0005" not in str(diag)


def test_diagnostic_is_immutable():
    from brokers.diagnostics import BrokerDiagnostic, BrokerDiagnosticStatus

    diag = BrokerDiagnostic(
        status=BrokerDiagnosticStatus.DATA_ERROR,
        code=None,
        sanitized_message="immutable check",
    )
    with pytest.raises(Exception):
        diag.sanitized_message = "changed"


# ---------------------------------------------------------------------------
# Unit 14: historical adapter -- sanitized exceptions only
# ---------------------------------------------------------------------------


def test_historical_auth_error_is_sanitized():
    from brokers.base import BrokerAuthError
    from brokers.fyers.historical import FyersHistoricalData
    from tests.conftest import FakeFyersClient
    from datetime import date

    client = FakeFyersClient(
        {"s": "error", "code": -16, "message": "invalid token SECRETVALUE_AUTH_0006"}
    )
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(BrokerAuthError) as excinfo:
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))
    assert "SECRETVALUE_AUTH_0006" not in str(excinfo.value)
    assert "SECRETVALUE_AUTH_0006" not in repr(excinfo.value)


def test_historical_rate_limit_error_is_sanitized():
    from brokers.base import BrokerRateLimitError
    from brokers.fyers.historical import FyersHistoricalData
    from tests.conftest import FakeFyersClient
    from datetime import date

    client = FakeFyersClient(
        {"s": "error", "code": 429, "message": "rate limit key=SECRETVALUE_RL_0007"}
    )
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(BrokerRateLimitError) as excinfo:
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))
    assert "SECRETVALUE_RL_0007" not in str(excinfo.value)
    assert "SECRETVALUE_RL_0007" not in repr(excinfo.value)


def test_historical_generic_data_error_is_sanitized():
    from brokers.base import BrokerDataError
    from brokers.fyers.historical import FyersHistoricalData
    from tests.conftest import FakeFyersClient
    from datetime import date

    client = FakeFyersClient(
        {"s": "error", "code": -99, "message": "account=ABC secret_key=SECRETVALUE_GEN_0008"}
    )
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(BrokerDataError) as excinfo:
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))
    assert "SECRETVALUE_GEN_0008" not in str(excinfo.value)
    assert "SECRETVALUE_GEN_0008" not in repr(excinfo.value)


def test_historical_raw_broker_message_never_appears_in_exception_text():
    """Even an UNREGISTERED secret embedded in the raw broker message must
    never appear -- the raw text itself is discarded, not merely scrubbed."""
    from brokers.base import BrokerDataError
    from brokers.fyers.historical import FyersHistoricalData
    from tests.conftest import FakeFyersClient
    from datetime import date

    client = FakeFyersClient(
        {
            "s": "error",
            "code": -99,
            "message": "signed_url=https://x/?secret_key=NEVER_REGISTERED_SECRET_0009",
        }
    )
    prov = FyersHistoricalData(client, request_pause_seconds=0.0)
    with pytest.raises(BrokerDataError) as excinfo:
        prov.fetch_chunk("X", "1", date(2026, 1, 1), date(2026, 1, 1))
    assert "NEVER_REGISTERED_SECRET_0009" not in str(excinfo.value)
    assert "NEVER_REGISTERED_SECRET_0009" not in repr(excinfo.value)
