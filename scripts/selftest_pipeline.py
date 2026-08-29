#!/usr/bin/env python3
"""Exercise the full data pipeline on SYNTHETIC data, without credentials.

    python scripts/selftest_pipeline.py

Purpose: prove that fetch -> clean -> validate -> store -> reload -> verify
works end to end, and that the validator detects injected defects, on a machine
that has no FYERS credentials. It also demonstrates reproducibility by running
the same input twice and comparing content hashes.

=======================================================================
THE DATA PRODUCED HERE IS SYNTHETIC. IT IS NOT MARKET DATA.
It must never be used for strategy research, backtesting, or any claim
about performance. It exists solely to exercise the plumbing.
=======================================================================
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brokers.fyers.historical import FyersHistoricalData  # noqa: E402
from marketdata import store  # noqa: E402
from marketdata.downloader import download  # noqa: E402
from marketdata.validator import validate  # noqa: E402
from tests.conftest import FakeFyersClient, candles_payload  # noqa: E402

SYMBOL = "SYNTHETIC:SELFTEST"
RESOLUTION = "1"


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def main() -> int:
    print(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # --- 1. clean synthetic dataset through the whole pipeline ---------
        banner("1. PIPELINE ON CLEAN SYNTHETIC DATA")
        provider = FyersHistoricalData(
            FakeFyersClient(candles_payload(120)), request_pause_seconds=0.0
        )
        outcome = download(
            provider,
            symbol=SYMBOL,
            resolution=RESOLUTION,
            start=date(2026, 1, 1),
            end=date(2026, 1, 5),
            data_store_dir=root,
            notes="SYNTHETIC self-test data. Not market data.",
        )
        print(outcome.summary())

        # --- 2. reload and verify integrity --------------------------------
        banner("2. RELOAD AND INTEGRITY CHECK")
        frame, manifest = store.read(root, SYMBOL, RESOLUTION)
        recomputed = store.content_hash(frame)
        print(f"  rows reloaded  : {len(frame)}")
        print(f"  manifest sha256: {manifest.content_sha256}")
        print(f"  recomputed     : {recomputed}")
        print(f"  MATCH          : {recomputed == manifest.content_sha256}")
        print(f"  source         : {manifest.source}")
        print(f"  timezone       : {manifest.timezone}")

        # --- 3. reproducibility --------------------------------------------
        banner("3. REPRODUCIBILITY: SAME INPUT TWICE")
        hashes = []
        for run in (1, 2):
            p = FyersHistoricalData(
                FakeFyersClient(candles_payload(120)), request_pause_seconds=0.0
            )
            o = download(
                p,
                symbol=SYMBOL,
                resolution=RESOLUTION,
                start=date(2026, 1, 1),
                end=date(2026, 1, 5),
                data_store_dir=root,
            )
            hashes.append(o.manifest.content_sha256)
            print(f"  run {run} sha256: {o.manifest.content_sha256}")
        print(f"  IDENTICAL     : {hashes[0] == hashes[1]}")

        # --- 4. defect detection --------------------------------------------
        banner("4. VALIDATOR DETECTS INJECTED DEFECTS")
        payload = candles_payload(60)
        payload["candles"][10] = [payload["candles"][10][0], 100.0, 50.0, 200.0, 120.0, 5]
        payload["candles"][20] = [payload["candles"][20][0], -1.0, 5.0, -9.0, 0.0, 5]
        payload["candles"].append(payload["candles"][30])  # duplicate timestamp

        bad_provider = FyersHistoricalData(
            FakeFyersClient(payload), request_pause_seconds=0.0
        )
        bad_frame = bad_provider.fetch_chunk(
            SYMBOL, RESOLUTION, date(2026, 1, 1), date(2026, 1, 2)
        )
        report = validate(
            bad_frame,
            symbol=SYMBOL,
            resolution=RESOLUTION,
            expected_interval_minutes=1,
        )
        print(report.to_text())

        detected = {i.code for i in report.issues}
        expected = {
            "OHLC_HIGH_BELOW_LOW",
            "NON_POSITIVE_PRICE",
            "DUPLICATE_TIMESTAMPS",
        }
        missing = expected - detected
        print(f"\n  expected defect codes : {sorted(expected)}")
        print(f"  detected              : {sorted(detected)}")
        print(f"  ALL EXPECTED DETECTED : {not missing}")
        if missing:
            print(f"  MISSING               : {sorted(missing)}")
            return 1

        banner("SELF-TEST COMPLETE (synthetic data only)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
