#!/usr/bin/env python3
"""Interactive FYERS login: obtain a daily access token.

Run this once per trading day. FYERS requires 2FA daily and has discontinued
continuous refresh-token sessions under the SEBI framework effective
2026-04-01, so a fresh token is expected each day.

    python scripts/fyers_login.py

The token is NEVER printed and never logged. It is written directly into your
local .env file, which is gitignored and given owner-only permissions. The
value stays on this machine and is not transmitted anywhere by this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brokers.fyers import auth  # noqa: E402
from config.settings import PROJECT_ROOT, load_settings  # noqa: E402

TOKEN_KEY = "FYERS_ACCESS_TOKEN"


def write_token_to_env(token: str, env_path: Path) -> None:
    """Replace (or append) the token line in .env without printing the value.

    Preserves every other line. The file is written with mode 0600 so it is
    readable only by its owner.
    """
    lines = (
        env_path.read_text(encoding="utf-8").splitlines()
        if env_path.exists()
        else []
    )
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{TOKEN_KEY}="):
            lines[index] = f"{TOKEN_KEY}={token}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{TOKEN_KEY}={token}")

    # Create with restrictive permissions before any content is written.
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.chmod(env_path, 0o600)


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

    env_path = PROJECT_ROOT / ".env"
    try:
        write_token_to_env(token, env_path)
    except OSError as exc:
        print(f"\nToken obtained but could not be written to {env_path}: {exc}")
        print("Set FYERS_ACCESS_TOKEN in your .env manually. The value is not")
        print("printed here deliberately.")
        return 1

    print(f"\nAccess token obtained and written to {env_path} (mode 0600).")
    print("The value is not displayed, by design.")
    print("\n.env is gitignored. Do not commit it, do not paste the token into")
    print("source code or chat. It expires and must be regenerated on the next")
    print("trading day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
