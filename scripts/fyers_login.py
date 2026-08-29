#!/usr/bin/env python3
"""Interactive FYERS login: obtain a daily access token.

Run this once per trading day. FYERS requires 2FA daily and has discontinued
continuous refresh-token sessions under the SEBI framework effective
2026-04-01, so a fresh token is expected each day.

    python scripts/fyers_login.py

The token is printed ONLY as an instruction to store it in .env; it is never
written to a log file. You paste it into your local .env yourself -- this
script does not write your secrets for you, and never sends them anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brokers.fyers import auth  # noqa: E402
from config.settings import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    creds = settings.fyers

    missing = [
        name
        for name, value in (
            ("FYERS_CLIENT_ID", creds.client_id),
            ("FYERS_SECRET_KEY", creds.secret_key),
            ("FYERS_REDIRECT_URI", creds.redirect_uri),
        )
        if not value
    ]
    if missing:
        print("Cannot start login. Missing in .env: " + ", ".join(missing))
        print("Copy .env.example to .env and fill these in from")
        print("https://myapi.fyers.in/dashboard/")
        return 1

    session = auth.build_session(creds.client_id, creds.secret_key, creds.redirect_uri)

    print("\n1. Open this URL in your browser and log in (2FA required):\n")
    print(auth.login_url(session))
    print("\n2. After login you will be redirected. Copy the FULL URL from the")
    print("   address bar (it contains auth_code=...) and paste it below.\n")

    redirect_url = input("Redirect URL: ").strip()
    if not redirect_url:
        print("No URL entered; aborting.")
        return 1

    try:
        code = auth.extract_auth_code(redirect_url)
        token = auth.exchange_auth_code(session, code)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        print(f"\nLogin failed: {exc}")
        return 1

    print("\nAccess token obtained. Add this line to your local .env file:\n")
    print(f"FYERS_ACCESS_TOKEN={token}")
    print("\nDo not commit .env. Do not paste this token into source code,")
    print("into chat, or into any shared location. It expires and must be")
    print("regenerated on the next trading day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
