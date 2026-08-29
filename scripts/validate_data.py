#!/usr/bin/env python3
"""Re-validate a stored dataset and verify it still matches its manifest hash.

    python scripts/validate_data.py --symbol "NSE:NIFTY50-INDEX" --resolution 1

Exits non-zero if the data has ERROR-severity issues or if the stored content
no longer hashes to the value recorded when it was written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings  # noqa: E402
from core.types import Resolution  # noqa: E402
from marketdata import store  # noqa: E402
from marketdata.validator import validate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--resolution",
        default=Resolution.M1.value,
        choices=[r.value for r in Resolution],
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        frame, manifest = store.read(
            settings.data_store_dir, args.symbol, args.resolution
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    recomputed = store.content_hash(frame)
    hash_ok = recomputed == manifest.content_sha256

    print("PROVENANCE")
    print("-" * 72)
    print(f"  source      : {manifest.source}")
    print(f"  fetched_at  : {manifest.fetched_at_utc}")
    print(f"  requested   : {manifest.requested_range}")
    print(f"  manifest sha: {manifest.content_sha256}")
    print(f"  recomputed  : {recomputed}")
    print(f"  INTEGRITY   : {'OK' if hash_ok else 'MISMATCH'}")

    report = validate(
        frame,
        symbol=args.symbol,
        resolution=args.resolution,
        expected_interval_minutes=Resolution(args.resolution).minutes,
    )
    print(report.to_text())
    return 0 if (report.is_usable and hash_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
