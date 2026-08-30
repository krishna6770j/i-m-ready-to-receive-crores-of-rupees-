# Algorithmic Trading Research System

Research and validation tooling for a rule-based intraday strategy on Indian
markets. **This is research software. It is not investment advice, and no
strategy in this repository has been shown to be profitable.**

## Current status

| | |
|---|---|
| **Phase** | 1 data foundation — remediated after audit, awaiting manager review |
| **FYERS adapter** | Verified against SDK source and mocks only — **no live call ever made** |
| **Trading mode** | `paper` by default; `live` is refused at config load |
| **Order placement** | No call site, no endpoint defined; see Design rules |
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
interpolation and outlier smoothing are not implemented. `marketdata/cleaner.py`
applies only explicitly requested, logged operations, and its default path
changes no value.

**Normalisation preserves every market value.** `normalise()` may reorder
columns, convert dtypes losslessly, convert timezone preserving the instant,
and sort. It may not substitute, fabricate, or drop. Missing volume stays
missing (nullable `Int64`) rather than becoming `0`, which would assert that no
trading occurred. A value present in the source but unparseable raises
`SchemaError` naming it, instead of being coerced to `NaN` where it would be
indistinguishable from genuine missingness.

**Invalid data cannot become authoritative.** `store.write()` takes a required
validation report and refuses to persist a dataset with ERROR-severity defects,
or one whose acquisition had failed chunks. `force=True` permits an explicit
override and is recorded permanently; a forced dataset is never
`is_authoritative`. The manifest distinguishes complete+valid, complete+invalid,
partial+valid and partial+invalid.

**Gaps are measured, not assumed.** No trading calendar is configured, so the
validator cannot tell a weekend from missing data. Rather than calling every
cross-day gap expected, it reports the gap's span as a warning, escalates past
an optional `max_session_gap_days`, and treats anything beyond 30 days as an
error since no equity market closes that long.

**Timestamps are tz-aware IST everywhere.** Naive timestamps are rejected, not
localised by assumption. A candle's timestamp is its OPEN time, matching the
FYERS convention.

**Every dataset carries provenance.** Source, fetch time, requested range,
validation status, acquisition status and a content hash live in a
`.manifest.json` beside the Parquet file. Data without a manifest is refused on
read, and `is_authoritative` is recomputed on read so an edited manifest cannot
forge it.

**Read-only surface.** `ReadOnlyFyersClient` exposes exactly four data methods
and does **not** store the SDK object; each call is a closure, and `__slots__`
removes the instance dict, so `client._inner`, `vars(client)` and `dir(client)`
cannot reach an order-capable object. Order-shaped attribute names raise
`OrderPlacementBlockedError`. An AST test fails the suite if any order-placement
call is added to any source file.

Stated precisely, because the distinction matters: **the FYERS SDK object that
this project constructs does have order methods.** No order call site exists in
this codebase and no order endpoint is defined in `brokers/fyers/endpoints.py`,
but Python offers no true object containment — closure introspection can still
reach the SDK. The wrapper removes accidental reach and makes deliberate reach
obvious in review. It is not a sandbox, and this project should not be described
as incapable of placing an order.

## Layout

```
config/       settings (mode + credentials) and instrument definitions
core/         timezone handling, shared enums, redacting logger
instruments/  signal/execution separation and the candidate registry
marketdata/   schema, validator, cleaner, Parquet store, download orchestration
brokers/      read-only interface + FYERS adapter (SDK-source verified,
              mock-tested; never yet run against the live service)
scripts/      CLI entry points
tests/        176 tests covering the above
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
