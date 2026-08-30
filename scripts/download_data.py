#!/usr/bin/env python3
"""Download historical candles from FYERS, validate them, and store them.

    python scripts/download_data.py --symbol "NSE:NIFTY50-INDEX" \
        --resolution 1 --start 2026-01-01 --end 2026-06-30

Probe how far back data actually exists (manager correction #12) instead of
assuming a depth:

    python scripts/download_data.py --symbol "NSE:NIFTY50-INDEX" \
        --resolution 1 --probe-earliest --probe-from 2015-01-01

Requires FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in .env. This script only
reads market data; it cannot place orders.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brokers.base import BrokerAuthError  # noqa: E402
from brokers.fyers.client import build_read_only_client  # noqa: E402
from brokers.fyers.historical import FyersHistoricalData  # noqa: E402
from config.settings import MissingCredentialError, load_settings  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from core.types import Resolution  # noqa: E402
from marketdata.downloader import download  # noqa: E402


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid date; use YYYY-MM-DD"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--resolution",
        default=Resolution.M1.value,
        choices=[r.value for r in Resolution],
    )
    parser.add_argument("--start", type=_parse_date)
    parser.add_argument("--end", type=_parse_date)
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Persist even when validation fails or chunks failed. The dataset "
            "is permanently marked non-authoritative in its manifest."
        ),
    )
    parser.add_argument(
        "--probe-earliest",
        action="store_true",
        help="Determine the earliest date with data instead of downloading.",
    )
    parser.add_argument("--probe-from", type=_parse_date, default=date(2015, 1, 1))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    run_id = setup_logging(settings.log_dir)
    print(f"run_id={run_id}  trading_mode={settings.trading_mode.value}")

    try:
        client = build_read_only_client(settings.fyers, log_dir=settings.log_dir)
    except MissingCredentialError as exc:
        print(f"\nCannot start: {exc}\n")
        return 2
    provider = FyersHistoricalData(client)

    if args.probe_earliest:
        newest = args.end or date.today()
        print(
            f"Probing earliest available {args.resolution!r} data for "
            f"{args.symbol} between {args.probe_from} and {newest}..."
        )
        try:
            earliest = provider.probe_earliest_available(
                args.symbol,
                args.resolution,
                newest=newest,
                oldest_to_try=args.probe_from,
            )
        except BrokerAuthError as exc:
            print(f"\nAuthentication failed: {exc}\n"
                  "Run: python scripts/fyers_login.py\n")
            return 3
        if earliest is None:
            print("No data found anywhere in the probed range.")
            return 1
        print(f"Earliest window returning data begins: {earliest}")
        return 0

    if not args.start or not args.end:
        print("--start and --end are required unless --probe-earliest is used.")
        return 2

    try:
        outcome = download(
            provider,
            symbol=args.symbol,
            resolution=args.resolution,
            start=args.start,
            end=args.end,
            data_store_dir=settings.data_store_dir,
            persist=not args.no_persist,
            force=args.force,
        )
    except BrokerAuthError as exc:
        print(f"\nAuthentication failed: {exc}\n"
              "Run: python scripts/fyers_login.py\n")
        return 3
    print(outcome.summary())

    if args.no_persist:
        return 0 if outcome.validation.is_usable else 1
    # Exit non-zero unless a genuinely authoritative dataset was written, so
    # that a caller cannot mistake a refusal or a forced write for success.
    if outcome.manifest is not None and outcome.manifest.is_authoritative:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
