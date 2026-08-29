"""Configuration loaded from environment variables.

Safety properties enforced here:

1. TRADING_MODE defaults to "paper" when unset.
2. TRADING_MODE="live" is REJECTED at load time. Live execution is not
   implemented, and this guard exists so that setting the variable by accident
   fails loudly instead of appearing to work.
3. Credentials are read from the environment only. They are never written to
   logs, never printed, and never defaulted to a placeholder that could be
   mistaken for a real value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from core.types import TradingMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LiveTradingNotImplementedError(RuntimeError):
    """Raised when TRADING_MODE=live is requested.

    Live execution does not exist in this codebase. This error is the first of
    several intended safeguards, not the only one.
    """


class MissingCredentialError(RuntimeError):
    """Raised when an operation needs a credential that is not configured."""


@dataclass(frozen=True)
class FyersCredentials:
    """FYERS API credentials sourced from the environment.

    ``__repr__`` is overridden so that these values cannot leak into a log line
    or traceback via ordinary object printing.
    """

    client_id: str | None = None
    secret_key: str | None = field(default=None, repr=False)
    redirect_uri: str | None = None
    access_token: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            "FyersCredentials("
            f"client_id={'set' if self.client_id else 'unset'}, "
            f"secret_key={'set' if self.secret_key else 'unset'}, "
            f"redirect_uri={'set' if self.redirect_uri else 'unset'}, "
            f"access_token={'set' if self.access_token else 'unset'})"
        )

    @property
    def is_complete_for_data(self) -> bool:
        """True if enough is present to make authenticated read-only calls."""
        return bool(self.client_id and self.access_token)

    def require_for_data(self) -> tuple[str, str]:
        """Return (client_id, access_token) or raise a actionable error."""
        if not self.is_complete_for_data:
            raise MissingCredentialError(
                "FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN must be set in your local "
                ".env file to make authenticated API calls. Copy .env.example to "
                ".env and fill them in. Never commit .env or paste these values "
                "into source code."
            )
        return self.client_id, self.access_token  # type: ignore[return-value]


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    trading_mode: TradingMode
    fyers: FyersCredentials
    data_store_dir: Path
    log_dir: Path


def _resolve_mode(raw: str | None) -> TradingMode:
    if raw is None or raw.strip() == "":
        return TradingMode.PAPER
    value = raw.strip().lower()
    try:
        mode = TradingMode(value)
    except ValueError as exc:
        allowed = ", ".join(m.value for m in TradingMode)
        raise ValueError(
            f"TRADING_MODE={raw!r} is not recognised. Allowed values: {allowed}."
        ) from exc
    if mode is TradingMode.LIVE:
        raise LiveTradingNotImplementedError(
            "TRADING_MODE=live is refused. Real-money order placement is not "
            "implemented in this codebase, and enabling it requires an explicit "
            "project decision plus SEBI-compliant static IP and App ID setup. "
            "Use 'paper' (default), 'backtest', or 'live_signal'."
        )
    return mode


def load_settings(env_file: str | Path | None = None, *, override: bool = False) -> Settings:
    """Load settings from a .env file (if present) and the environment."""
    dotenv_path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=override)

    data_dir = Path(os.getenv("DATA_STORE_DIR", "data_store"))
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir

    return Settings(
        trading_mode=_resolve_mode(os.getenv("TRADING_MODE")),
        fyers=FyersCredentials(
            client_id=os.getenv("FYERS_CLIENT_ID") or None,
            secret_key=os.getenv("FYERS_SECRET_KEY") or None,
            redirect_uri=os.getenv("FYERS_REDIRECT_URI") or None,
            access_token=os.getenv("FYERS_ACCESS_TOKEN") or None,
        ),
        data_store_dir=data_dir,
        log_dir=log_dir,
    )
