"""Construction of a read-only FYERS data client.

The SDK's ``FyersModel`` exposes order-placement methods. This module never
returns that object to application code directly: it wraps it in
``ReadOnlyFyersClient``, which forwards only the data methods we have verified
and refuses everything else.

That refusal is the point. It means an accidental ``client.place_order(...)``
raises an AttributeError naming the safeguard, instead of sending an order.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import FyersCredentials

logger = logging.getLogger(__name__)

# Methods this project is permitted to call. Read-only, verified to exist on
# FyersModel in fyers-apiv3 3.1.16 (see endpoints.py for provenance).
ALLOWED_METHODS = frozenset({"history", "quotes", "get_profile", "market_status"})


class OrderPlacementBlockedError(RuntimeError):
    """Raised when application code reaches for an order-placement method."""


class ReadOnlyFyersClient:
    """Whitelist wrapper over the FYERS SDK client.

    Only ``ALLOWED_METHODS`` are reachable. Anything else raises, with
    order-related names raising a distinct, loud error.
    """

    _ORDER_METHOD_MARKERS = (
        "order",
        "position",
        "exit",
        "convert",
        "smart",
        "alert",
        "gtt",
        "basket",
        "multileg",
    )

    def __init__(self, inner) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str):
        if name in ALLOWED_METHODS:
            return getattr(object.__getattribute__(self, "_inner"), name)
        lowered = name.lower()
        if any(marker in lowered for marker in self._ORDER_METHOD_MARKERS):
            raise OrderPlacementBlockedError(
                f"Access to {name!r} is blocked. This project is read-only: it has "
                "no approved execution path, and live order placement requires an "
                "explicit project decision plus SEBI-compliant static IP and App ID "
                "registration. Blocked deliberately, not by oversight."
            )
        raise AttributeError(
            f"{name!r} is not in the read-only allowlist {sorted(ALLOWED_METHODS)}. "
            "Add it only after verifying it against FYERS documentation and "
            "confirming it cannot mutate account state."
        )

    def __setattr__(self, name: str, value) -> None:  # pragma: no cover - guard
        raise AttributeError("ReadOnlyFyersClient is immutable.")

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ReadOnlyFyersClient(allowed={sorted(ALLOWED_METHODS)})"


def build_read_only_client(
    credentials: FyersCredentials, *, log_dir: Path | None = None
) -> ReadOnlyFyersClient:
    """Build a read-only FYERS client from credentials.

    Raises MissingCredentialError (from settings) when credentials are absent,
    with instructions that do not involve pasting secrets anywhere unsafe.
    """
    client_id, access_token = credentials.require_for_data()

    from fyers_apiv3 import fyersModel

    # The SDK writes fyersApi.log / fyersRequests.log relative to log_path.
    # Point them at our log directory so they are covered by .gitignore.
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    inner = fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        is_async=False,
        log_path=str(log_dir) if log_dir else None,
        log_level="ERROR",
    )
    logger.info("FYERS read-only client constructed (credentials not logged).")
    return ReadOnlyFyersClient(inner)
