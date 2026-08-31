"""FYERS OAuth login helpers.

Flow, verified against the installed SDK (fyers-apiv3 3.1.16):

  1. ``SessionModel(client_id, secret_key, redirect_uri, response_type="code",
     grant_type="authorization_code")``
  2. ``generate_authcode()`` returns
     ``{API_BASE}/generate-authcode?client_id=..&redirect_uri=..&response_type=..&state=..``
  3. The user opens that URL in a browser and authenticates with 2FA. FYERS
     redirects to ``redirect_uri`` with ``auth_code`` in the query string.
  4. ``set_token(auth_code)`` then ``generate_token()`` POSTs to
     ``{API_BASE}/validate-authcode`` with
     ``{"grant_type", "appIdHash": sha256("client_id:secret_key"), "code"}``
     and returns JSON containing the access token.

Security posture:
  * This module never writes a token to a log or to stdout.
  * It never embeds credentials in source.
  * The browser step is performed by the human. Nothing here automates a login
    or handles a password.

Operational note: per FYERS' notice on the SEBI framework effective
2026-04-01, 2FA is required once every trading day and continuous refresh-token
sessions are discontinued, so this flow is expected to run daily.
"""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlparse

from core.secrets import register as _register_secret

logger = logging.getLogger(__name__)

RESPONSE_TYPE = "code"
GRANT_TYPE = "authorization_code"

# Key observed in FYERS token responses. Not verified against the formal docs
# (see endpoints.py), so extraction falls back across plausible keys and raises
# a clear error rather than returning None.
_TOKEN_KEYS = ("access_token", "accessToken")

# Not observed in any response yet -- FYERS' 2FA-per-day flow described in the
# module docstring has no documented refresh token, but if one is ever added
# it must be registered the moment it is seen, same as the access token.
_REFRESH_TOKEN_KEYS = ("refresh_token", "refreshToken")


class FyersAuthError(RuntimeError):
    """Raised when the login flow cannot be completed."""


def build_session(client_id: str, secret_key: str, redirect_uri: str):
    """Construct a SessionModel. Imported lazily so tests need no SDK."""
    from fyers_apiv3 import fyersModel

    return fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type=RESPONSE_TYPE,
        grant_type=GRANT_TYPE,
    )


def login_url(session) -> str:
    """Return the URL the user must open to authenticate."""
    return session.generate_authcode()


def extract_auth_code(redirect_url: str) -> str:
    """Pull ``auth_code`` out of the URL the browser was redirected to.

    Accepts the full redirect URL so the user never has to hand-edit a query
    string. Raises with guidance if the parameter is absent.
    """
    parsed = urlparse(redirect_url.strip())
    params = parse_qs(parsed.query)
    for key in ("auth_code", "authcode", "code"):
        if key in params and params[key]:
            code = params[key][0]
            _register_secret(code)
            return code
    raise FyersAuthError(
        "No 'auth_code' parameter found in the redirect URL. Paste the FULL URL "
        "from the browser address bar after logging in, including everything "
        "after the '?'."
    )


def exchange_auth_code(session, auth_code: str) -> str:
    """Exchange an auth code for an access token.

    Returns the token. Never logs it.
    """
    _register_secret(auth_code)
    session.set_token(auth_code)
    response = session.generate_token()

    if not isinstance(response, dict):
        raise FyersAuthError(
            f"Unexpected token response type {type(response).__name__}."
        )
    for key in _REFRESH_TOKEN_KEYS:
        refresh_token = response.get(key)
        if refresh_token:
            _register_secret(str(refresh_token))

    for key in _TOKEN_KEYS:
        token = response.get(key)
        if token:
            token = str(token)
            _register_secret(token)
            logger.info("Access token obtained (value not logged).")
            return token

    # Surface diagnostic keys but never the payload values, which may contain
    # partial credentials.
    raise FyersAuthError(
        "Token response contained no access token. Keys present: "
        f"{sorted(response)}. Check that FYERS_SECRET_KEY and FYERS_REDIRECT_URI "
        "exactly match the app configuration in the FYERS API dashboard."
    )
