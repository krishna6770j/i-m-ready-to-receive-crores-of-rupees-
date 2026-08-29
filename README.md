# Algorithmic Trading Research System

Research and validation tooling for a rule-based intraday strategy on Indian
markets. **This is research software. It is not investment advice, and no
strategy in this repository has been shown to be profitable.**

## Current status

| | |
|---|---|
| **Phase** | 1 complete (data foundation) — awaiting manager review |
| **Trading mode** | `paper` by default; `live` is refused at config load |
| **Order placement** | Does not exist in this codebase |
| **Execution instrument** | **Not selected.** Multiple candidates held for evaluation |
| **Strategy** | Not implemented. No parameters tuned. No profitability claimed |
| **Real market data** | **Not yet downloaded** — requires FYERS credentials |

The project advances through validation gates. Each phase stops for review; a
failed gate stops progression rather than prompting a search for a more
permissive test.

```
Phase 0  research + architecture         [done]
Phase 1  data foundation                 [done, under review]
Phase 2  backtest engine                 [not started]
Phase 3  strategy implementation         [not started]
Phase 4  validation (OOS, walk-forward, robustness, Monte Carlo)
Phase 5  TradingView cross-check
Phase 6  live signal monitor (no orders)
Phase 7  paper trading
Phase 8  broker execution  [requires separate approval]
```

## Setup

Requires Python 3.12. This project uses a standalone interpreter managed by
`uv`, installed under `~/.local`, leaving the macOS system Python untouched.

```bash
uv venv --python 3.12 .venv
uv pip install --python ./.venv/bin/python -r requirements.txt
uv pip uninstall --python ./.venv/bin/python asyncio
```

That last line is required, not optional — see the note in `requirements.txt`.
`fyers-apiv3` depends on a PyPI package named `asyncio` which shadows the
standard library module and cannot run on Python 3.12.

Then configure credentials:

```bash
cp .env.example .env
# edit .env and fill in your FYERS values
```

`.env` is gitignored. Never commit it, never paste credentials into source
files, and never share your access token.

## Usage

Verify the pipeline without credentials (uses synthetic data):

```bash
./.venv/bin/python scripts/selftest_pipeline.py
```

Run the tests:

```bash
./.venv/bin/python -m pytest
```

Obtain a daily access token (FYERS requires 2FA once per trading day):

```bash
./.venv/bin/python scripts/fyers_login.py
```

Find out how far back data actually goes, rather than assuming:

```bash
./.venv/bin/python scripts/download_data.py --symbol "NSE:NIFTY50-INDEX" \
    --resolution 1 --probe-earliest --probe-from 2015-01-01
```

Download, validate and store a range:

```bash
./.venv/bin/python scripts/download_data.py --symbol "NSE:NIFTY50-INDEX" \
    --resolution 1 --start 2026-01-01 --end 2026-06-30
```

Re-validate stored data and check it against its provenance manifest:

```bash
./.venv/bin/python scripts/validate_data.py --symbol "NSE:NIFTY50-INDEX" --resolution 1
```

## Design rules

These are load-bearing. Changing one changes what the results mean.

**Signal instrument is not execution instrument.** The NIFTY 50 index generates
signals and cannot be traded. `Instrument.require_tradable()` raises if
anything tries to size, price or compute P&L on it. No execution instrument has
been selected; `instruments/registry.py` holds unverified candidates only.

**The validator flags, it never repairs.** Silent repair is how corrupt data
reaches a backtest and produces a plausible wrong answer. Gap filling,
interpolation and outlier smoothing are deliberately not implemented.
`marketdata/cleaner.py` applies only explicitly requested, logged operations,
and its default path changes nothing.

**Timestamps are tz-aware IST everywhere.** Naive timestamps are rejected, not
localised by assumption. A candle's timestamp is its OPEN time, matching the
FYERS convention.

**Every dataset carries provenance.** Source, fetch time, requested range and a
content hash live in a `.manifest.json` beside the Parquet file. Data without a
manifest is refused on read.

**Read-only by construction.** `ReadOnlyFyersClient` exposes an allowlist of
four data methods. Any order-related attribute raises
`OrderPlacementBlockedError`. A test parses the AST of every source file to
assert no order-placement call exists anywhere.

## Layout

```
config/       settings (mode + credentials) and instrument definitions
core/         timezone handling, shared enums, redacting logger
instruments/  signal/execution separation and the candidate registry
marketdata/   schema, validator, cleaner, Parquet store, download orchestration
brokers/      abstract read-only interface + verified FYERS adapter
scripts/      CLI entry points
tests/        123 tests covering the above
```

## Regulatory note

SEBI's framework for retail algorithmic trading has been fully applicable since
1 April 2026. Order placement requires a registered App ID bound to a
whitelisted static IP, and FYERS requires 2FA once per trading day with
continuous refresh-token sessions discontinued. Market data access does not
require a static IP, which is why phases 1–7 can proceed without one.

This summary reflects research current as of August 2026 and is not legal
advice. Re-verify against SEBI, exchange and broker documentation before any
live execution decision.
