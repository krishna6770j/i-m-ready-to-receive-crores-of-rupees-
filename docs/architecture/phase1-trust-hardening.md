# Phase 1 — Trusted Data Architecture

Design specification. **No production code, tests, dependencies, README or CI
changes accompany this document.** Implementation is not authorised.

---

## 1. Baseline

| | |
|---|---|
| Baseline commit | `e74d333832216d4c8e8ad77a099993e5323e657d` |
| Branch | `phase1-trust-hardening` |
| Manager verdict | Phase 1 REJECTED; Phase 2 NOT AUTHORISED; live FYERS NOT AUTHORISED |
| Tests at baseline | 176 passing |

Every claim was checked against code at this commit. Where an earlier revision
of this document was wrong, the correction is stated plainly rather than layered
over.

---

## 2. Glossary of certifications

The previous revision used *valid*, *usable*, *trusted*, *authoritative*,
*complete*, *reproducible* with overlapping meanings. Each term below now has
exactly one meaning. **There is no single `is_authoritative` boolean.**

| Dimension | Values | Meaning | Derivable from candles alone? |
|---|---|---|---|
| `ArtifactIntegrity` | INTACT / BROKEN | Stored bytes match the recorded data digest, and the provenance envelope binds to this generation | no — needs the envelope |
| `SchemaValidity` | VALID / INVALID | Frame conforms to the canonical schema and declared `schema_version` | yes |
| `MarketDataValidity` | VALID / INVALID | No ERROR-severity validation issue (OHLC rules, finiteness, duplicates, ordering) | yes |
| `AcquisitionEvidenceIntegrity` | INTACT / BROKEN | Acquisition snapshot is envelope-bound and internally consistent | no — historical claim |
| `ObservedCoverage` | a *record*, not a grade | Observed first/last dates, boundary-date presence, containment in the request window | yes, as fact |
| `ContinuityCertification` | CERTIFIED / NOT_CERTIFIED / FAILED | Whether interior gaps are explained | needs a calendar |
| `SessionCertification` | CERTIFIED / NOT_CERTIFIED / FAILED | Whether expected sessions and bar density are present | needs a calendar |
| `ReproducibilityCertification` | CERTIFIED / NOT_CERTIFIED | Whether the producing environment is pinned and clean | no |
| `Namespace` | TRUSTED / FORCED | Structural provenance state, from storage layout | no |

**`TrustedDataset` guarantees exactly:** `ArtifactIntegrity = INTACT`,
`SchemaValidity = VALID`, `MarketDataValidity = VALID`,
`AcquisitionEvidenceIntegrity = INTACT`, locator identity matches, and
`Namespace = TRUSTED`.

**`TrustedDataset` explicitly does NOT guarantee:** continuity, session
coverage, bar density, reproducibility, or that the broker returned the
instrument requested (§16). Those travel *on* the object as separate
certifications, and a consumer states which it requires.

This resolves the previous contradiction where an arbitrary gap threshold
decided whether the storage artifact itself was "trusted". Artifact
trustworthiness and market-history completeness are now different questions.

---

## 3. Problem statement

Sixteen confirmed defects share one root cause: the system trusts assertions
instead of deriving facts. But — correcting the previous revision — **not every
fact is derivable**, and the design must say which are which.

| # | Defect at baseline | Evidence |
|---|---|---|
| 1 | `store.write()` takes frame and report separately, never checking they correspond | Invalid frame + valid frame's report persists as `validation_status="valid"` |
| 2 | `is_authoritative` forgeable via `validation_status`/`fetch_status`/`forced` | Edited manifest yields `True` on invalid data |
| 3 | Completeness derived from `failed_chunks == []` | 3 candles stored as authoritative answer to a 1-year request |
| 4 | `ABSURD_GAP_DAYS = 30` magic number; 29-day hole is WARNING-only | 3/7/14/20/29-day gaps all `usable=True` |
| 5 | Source anomalies erased before validation | `normalise()` sorts before the validator can observe source ordering |
| 6 | Extra source columns silently dropped | `oi`, `vwap` vanish; contradicts the README |
| 7 | `content_hash` uses `%.10g` despite a `repr` docstring | 1e-9 price change hashes identically |
| 8 | Digest excludes source, symbol, resolution | Different instruments with equal candles collide |
| 9 | `read()` verifies nothing | Tampered Parquet returned silently |
| 10 | Two-file write is not transactional | Manifest failure leaves an orphan Parquet |
| 11 | `--force` overwrites the authoritative dataset in place | Good data silently replaced |
| 12 | Redaction reads only the current env value | A rotated-out token leaks |
| 13 | `probe_earliest_available()` binary-searches a non-monotonic predicate | Holiday/outage window empty while earlier data exists |
| 14 | Broker error text embedded in exceptions and printed to stdout | `historical.py` builds `detail` from raw broker `message`; four scripts `print(f"…{exc}")` |
| 15 | `write_token_to_env()` validates nothing | A newline in a token injects `.env` lines |
| 16 | Config coerces falsey values | `lot_size=int(lot_size) if lot_size else 1` turns `0` into `1` |

---

## 4. Threat model

**In scope.** Accidental manifest edits; stale or partially-written files;
process death mid-write; inconsistent output from future automation; manifests
copied between datasets; operator mistakes; malformed or hostile *broker
responses*; our own refactoring errors.

**Out of scope.** A malicious actor who controls the machine and source tree.
Such an actor can recompute every digest and rewrite the pointer, and the design
says so explicitly rather than implying protection it does not provide.

**No asymmetric signing.** The generation-integrity envelope (§7) detects every
in-scope failure. A key on the same machine would not raise the bar against the
out-of-scope actor.

---

## 5. Three classes of fact

The previous revision's blanket "derive, don't trust" was wrong. Correcting it:

| Class | Examples | Protection |
|---|---|---|
| **Data-derived** | OHLC validity, schema conformance, row count, observed first/last timestamp, data digest | **Recomputed** from the stored data at every use |
| **Provenance evidence** | requested range, chunks requested/failed/empty, broker attribution, raw column inventory, canonicalisation anomalies | **Cannot be recomputed from candles.** Integrity-bound to the generation via the envelope digest |
| **Operator declarations** | `forced`, `force_reason`, notes | Integrity-bound, plus **structural** demotion via namespace (§10) |

The manager's counterexample is decisive and is now handled: stored data
`Jan 1 – Jan 3`, actual request `Jan 1 – Dec 31`, manifest edited so
`requested_to = Jan 3`, `failed_chunks = []`, `acquisition = edge_complete`.
Every field is internally self-consistent and no amount of OHLCV revalidation
recovers the original request. **Only the envelope digest detects this**, because
editing the manifest changes the provenance digest, which no longer matches the
generation-integrity id recorded in the pointer.

---

## 6. Trust transitions and types

| Type | Constructed by | Guarantees | Does NOT guarantee | Phase 2 may consume |
|---|---|---|---|---|
| `RawObservations` | adapter parse | Shape matches **the adapter's currently configured parse contract** | That the contract matches live broker behaviour — that remains ASSUMED, never LIVE-VERIFIED | no |
| `CanonicalDataset` | `canonicalise()` | Deterministic canonical form **plus** transformation and source-anomaly evidence | Validity | no |
| `ValidatedDataset` | `ValidatedDataset.build()` | Canonical form, identity digest and validation evidence produced from that exact data | That the data stays unmutated afterwards | no |
| `UnverifiedDataset` | `store.read_unverified()` | Bytes were loaded from disk | Everything else | **no** |
| `TrustedDataset` | `store.read_trusted()` | The five properties in §2 | Continuity, session coverage, reproducibility, instrument correctness | **yes — the only permitted input** |

`ADAPTER-CONTRACT-VALID` and `LIVE-BROKER-VERIFIED` are distinct labels
throughout. Nothing in this repository is currently the latter.

---

## 7. Generation integrity envelope

```
data_digest        = SHA256( canonical encoding of identity + observations )   [§9]
provenance_digest  = SHA256( canonical encoding of the provenance envelope )
generation_id      = ULID (128-bit, content-independent)                       [§12]
integrity_id       = SHA256( data_digest || provenance_digest )
```

The **provenance envelope** contains, in canonical encoded form: identity
(`schema_version`, `source`, `symbol`, `resolution`); the acquisition snapshot
(§11); the canonicalisation snapshot (§14); operator declarations (`forced`,
`force_reason`); and the environment snapshot (§20).

The `CURRENT` pointer records `generation_id` **and** `integrity_id`. Therefore:

| Accidental change | Detected because |
|---|---|
| Parquet edited | `data_digest` changes → `integrity_id` mismatch |
| Manifest edited (any field) | `provenance_digest` changes → mismatch |
| Manifest and Parquet edited consistently | `integrity_id` still mismatches the pointer |
| Manifest copied from another dataset | Both digests mismatch |
| Pointer edited to another generation | That generation's `integrity_id` will not match the recorded one |

Defeating this requires recomputing both digests *and* rewriting the pointer —
which is precisely the out-of-scope actor. **Stated explicitly: this is
tamper-evidence against accidents and inconsistency, not against a malicious
user.**

---

## 8. Dataset identity and locator verification

Identity is an equivalence relation over:

> **(schema_version, source, symbol, resolution, canonical observation sequence
> including preserved named extra columns)**

In identity: schema version, source/vendor, symbol, resolution, canonical
observation columns, preserved named extra columns, timestamp instant, row
multiplicity, duplicate and conflicting timestamps, and adjustment state **when
introduced** (none exists today; it must enter identity the day it does).

Not in identity: timezone representation, row ordering, column ordering, dtype,
acquisition range, manifest metadata.

**Verified against what.** Correcting the previous revision, which said identity
is "reverified" without saying against which reference: `source`, `symbol`,
`resolution` and `schema_version` are **not** derivable from candles. They are
verified as a three-way agreement:

```
read_trusted(source=..., symbol=..., resolution=...)
        ↓
   caller's locator   ==   envelope identity   ==   digest identity inputs
```

A manifest may never redefine the identity of the object being requested.
Calling `read_trusted("fyers:history", "NSE:NIFTY50-INDEX", "1")` cannot return
a generation whose manifest was edited to say `SBIN`: the locator disagrees with
the envelope, and the envelope disagrees with the digest inputs.

---

## 9. Canonical encoding and digest

The previous `IDENTITY | field | field` scheme is withdrawn: pipe and newline
characters inside string values, column names containing separators, escape
sequences, Unicode normalisation differences, and an NA sentinel colliding with
a literal string all make it ambiguous once string columns exist.

**Replacement: typed, length-prefixed binary encoding.** No delimiters, so no
collision is possible.

```
field  := type_tag (1 byte) || length (8 bytes, big-endian) || payload
stream := field*
digest := SHA256(stream)
```

| Tag | Type | Payload |
|---|---|---|
| `0x01` | STR | UTF-8 bytes, **NFC-normalised** |
| `0x02` | F64 | 8 bytes, IEEE-754 big-endian |
| `0x03` | I64 | 8 bytes, two's complement big-endian |
| `0x04` | NA | zero length |
| `0x05` | NAN | zero length |
| `0x06` | POSINF | zero length |
| `0x07` | NEGINF | zero length |
| `0x08` | TS | 8 bytes, int64 **nanoseconds** since Unix epoch, UTC |

Rules: `-0.0` is normalised to `+0.0` before encoding (IEEE signed zero is not a
market distinction, and a zero price is invalid anyway); `NaN` and `±Inf` use
their own tags rather than any float payload, so they can never collide with a
finite value; missing volume uses `NA`, distinct from `I64(0)`; column names are
encoded as `STR` fields in the header; column order is the canonical order, not
the input order; row order is the canonical sort order.

Stream layout: header fields (`schema_version`, `source`, `symbol`,
`resolution`, column count, each column name) followed by each row's fields in
canonical column order.

Empty and one-row frames encode deterministically and differ from each other.

Verified with an in-memory prototype: `24000.12` ≠ `24000.13`; a 1e-9 change
differs (baseline `%.10g` says *same*); `0` ≠ `NA`; `NaN`, `+inf` all distinct;
duplicate row added and row removed both differ; symbol, resolution and source
each differ; while rows reversed, UTC-vs-IST representation, and reordered
columns all match. `np.float64` subclasses `float`, so extracting the IEEE bytes
is direct.

---

## 10. Forced generations — one semantic model

The previous revision contradicted itself, claiming both that forced generations
always fail trusted read and that forced data passing every check is
"authoritative on its own merits". **The second claim is withdrawn.**

**Force is an operational provenance state, not a statement about the candles.**

```
data_store/<source>/<symbol>/<resolution>/
    trusted_generations/<generation_id>/   data.parquet + manifest.json
    forced_generations/<generation_id>/    data.parquet + manifest.json
    CURRENT                                pointer file
```

- `force=True` writes into `forced_generations/` and requires a non-empty
  `force_reason`.
- `CURRENT` can only ever name a generation in `trusted_generations/`.
- `read_trusted()` **never traverses** `forced_generations/`.
- Even perfectly valid candles written through the forced path remain forensic
  until re-imported through the normal pipeline.
- Reachable only via the forensic API with an explicit generation id.

Selection therefore does not depend on an editable boolean at all. The manifest
still records `forced` and `force_reason` as evidence, but **the namespace
supplies the structural demotion**. Editing `forced=false` in a forced
generation's manifest changes nothing, because it is still in the forced
directory and additionally breaks the envelope digest.

---

## 11. Acquisition evidence and observed coverage

`failed_chunks == []` is not completeness. Six concepts stay separate:

| Concept | Answered by | Provable today? |
|---|---|---|
| 1. request success/failure | chunk results | yes |
| 2. observed coverage | §11.2 | yes, as **fact** |
| 3. internal continuity | §13 | only with a calendar |
| 4. expected session coverage | calendar | **no** |
| 5. bar density | calendar | **no** |
| 6. identity correctness | §8, with the §16 caveat | partially |

### 11.1 Acquisition classification

```
ACQUISITION_FAILED      every chunk errored
ACQUISITION_EMPTY       all chunks succeeded, all returned zero rows
ACQUISITION_PARTIAL     any chunk failed or returned an error state
ACQUISITION_SUCCEEDED   every chunk returned without error, ≥1 row overall
ACQUISITION_UNKNOWN     no acquisition evidence (fixtures, manual frames)
```

Renamed from `EDGE_COMPLETE`: this enum now describes **only whether the
requests succeeded**, not coverage. Coverage is a separate record, because
conflating them is exactly how "3 candles for a year" became authoritative.

### 11.2 `ObservedCoverage` — facts only

The previous revision compared requested edges to observed timestamps under a
generic tolerance. That is not meaningful: a request is expressed in **calendar
dates**, observations begin at an intraday market time, and a requested boundary
may fall on a weekend, holiday, half-day, future date or market closure.
Distance-to-midnight proves nothing without a calendar.

Recorded as fact, never graded:

- `earliest_observed_date`, `latest_observed_date`
- `observations_on_requested_start_date` (bool)
- `observations_on_requested_end_date` (bool)
- `all_observations_within_requested_window` (bool)
- `observed_span_days`, `requested_span_days`
- `distinct_observed_dates`

**Absence of observations on a requested boundary date is NOT called partial**
— that date may simply not have been a trading day. Grading requires
`CalendarCertifiedCoverage`, which is unavailable and is marked `NOT_CERTIFIED`.

The "3 candles for a one-year request" case is therefore recorded honestly:
`ACQUISITION_SUCCEEDED`, `distinct_observed_dates = 1`,
`requested_span_days = 365`, `SessionCertification = NOT_CERTIFIED`. It can
still be a `TrustedDataset` **as an artifact**, but no consumer requiring
session certification may use it.

---

## 12. History-depth probing — corrected

**Two rejected algorithms, with counterexamples.**

*Baseline binary search* on "this narrow window contains data" assumes
monotonicity. A window landing on a weekend, holiday cluster, broker outage or
sparsely-quoted period returns empty while both earlier and later periods
contain data; the search then discards a region that does contain data.

*The previous revision's proposed fix* — `any_data_in(candidate, known_good_newest)`
— is **mathematically broken**, as the manager identified. If
`known_good_newest` contains data, then every interval
`[candidate, known_good_newest]` contains that same observation, so the
predicate is constant-`TRUE` for all candidates and never produces the
false/true bracket a search requires. It degenerates immediately.

**What is being estimated.** These are not interchangeable, and the previous
revision blurred them:

- **(A)** earliest individual candle observed
- **(B)** earliest calendar date with any candle
- **(C)** earliest continuous-history boundary
- **(D)** broker retention boundary

We can establish **(A)** and **(B)** by observation. **(C)** requires a
calendar. **(D)** cannot be proven at all — absence of response is not proof of
absence of history.

**Algorithm: bounded backward window scan, then subdivide.**

1. Start from a recent known-good anchor.
2. Walk backwards in **non-overlapping coarse windows** (configured, e.g. 90 days).
3. Classify each window: `DATA`, `EMPTY_SUCCESS`, `ERROR`, `UNKNOWN`.
4. **`ERROR` is never evidence of absence** — it is recorded as unresolved and
   the scan continues.
5. Continue until a user-configured **search horizon** (no unbounded scanning).
6. Subdivide **only the oldest `DATA` window**, recursively, to the configured
   temporal resolution.
7. Report a **bracket plus evidence**, never a single retention date:
   - `earliest_observed_candle` (A)
   - `earliest_observed_date` (B)
   - `oldest_contiguous_empty_success_interval`
   - `unresolved_intervals` (errors/unknowns)
   - `search_horizon`

**Correctness argument.** The algorithm never concludes "no data before X" from
any single empty window; it only reports what it observed and what it could not
resolve. It makes no monotonicity assumption. Its output is an interval estimate
whose width is bounded by the configured resolution, and unresolved regions are
surfaced rather than silently treated as empty. Cost is O(horizon / window) plus
O(log) refinement of one window — more requests than binary search, which is
accepted: correctness outranks request count, subject to the §21 rate-limit note.

**No live probing until this is implemented and tested.**

---

## 13. Continuity, calendar and gap model

`ABSURD_GAP_DAYS = 30` is removed. Correcting the previous revision: moving the
number into configuration did **not** solve the knowledge problem, it only
relocated the arbitrariness. No `TradingCalendar` exists in the repository —
verified.

Three separate concepts:

- **`ObservedGap`** — elapsed time between consecutive observations. Always
  computable. Pure fact, carries no severity.
- **`CalendarExplanation`** — whether a configured calendar explains the gap.
  Without a calendar: `UNKNOWN` for every gap, uniformly.
- **`GapPolicy`** — a *consumer's* declared requirement, not a property of the
  artifact.

`TradingCalendar` is a protocol only — `is_session_day(date)`,
`expected_next_bar(ts, resolution)`, plus `calendar_id` and `calendar_version`
which enter provenance when a real calendar arrives. **No NSE holiday or session
data is invented in this phase.** Only `NullCalendar` ships, answering `UNKNOWN`
to everything.

**Certification, not gating:**

| Situation | `ContinuityCertification` |
|---|---|
| No calendar configured | `NOT_CERTIFIED` — uniformly, for every gap size |
| Calendar configured, all gaps explained | `CERTIFIED` |
| Calendar configured, unexplained gap found | `FAILED` |

`NOT_CERTIFIED` is **not** a failure — it is an honest statement that the
question cannot be answered. A `TrustedDataset` may carry
`ContinuityCertification = NOT_CERTIFIED`; whether that is acceptable is the
consumer's decision, declared through its own `GapPolicy`. No calendar-day
number decides whether the storage artifact is trusted.

This cleanly separates **trusted artifact** from **complete market history**
from **session-certified research input**.

---

## 14. Canonicalisation contract and anomaly severity

**Confirmed defect.** `normalise()` sorts before the validator runs, so
`TS_NOT_SORTED` can never fire on the production path — the source defect is
silently repaired.

**Invariant:** canonicalisation may produce a deterministic representation, but
it may never erase evidence that the source delivered anomalous structure.

```
canonicalise(raw) -> CanonicalisationResult(frame, transformations, source_anomalies)
```

Recorded: source row count, source column inventory, source ordering
(inversion count), duplicate timestamps at source, extra columns seen, timezone
conversion applied, dtype conversions, rows reordered, rows removed, values
changed, columns preserved or discarded.

**Severity — provenance with enforcement, not a diary:**

| Anomaly | Class | Reasoning |
|---|---|---|
| Timezone conversion preserving the instant | NORMAL TRANSFORMATION | No information changes |
| Column reordering to canonical order | NORMAL TRANSFORMATION | Representation only |
| Lossless dtype conversion | NORMAL TRANSFORMATION | Values preserved exactly |
| Stable sort of an unsorted source | **INFO, recorded** | Valid canonicalisation, but the source got it wrong and that is provenance |
| Exact duplicate rows at source | **WARNING** | Redundant but not contradictory |
| Source extra column, known and mapped | INFO | Preserved per §15 |
| **Conflicting duplicate timestamps** | **TRUST BLOCKER** | Two contradictory observations of one minute; choosing between them would discard evidence |
| **Unsupported / unmapped source field** | **TRUST BLOCKER** | §15 |
| **Row removal during canonicalisation** | **TRUST BLOCKER** | Never a normal transformation; only explicit cleaning may remove rows, and never on the trusted path |
| Values changed | **TRUST BLOCKER** | Canonicalisation must never change a value |

TRUST BLOCKER means `ValidatedDataset.build()` fails; nothing reaches storage.

---

## 15. Schema evolution and the positional payload

**Confirmed defect.** `frame.loc[:, OHLCV_COLUMNS]` silently discards extra
columns, contradicting the README.

The previous revision proposed preserving unknown extras as `x_*`. **Withdrawn**
— it contradicts the actual parser. `from_fyers_candles()` accepts exactly six
*positional* fields. A seventh positional value has no name and therefore no
meaning; inventing `x_unknown_7` would fabricate a research field out of an
unknown number.

**Policy by source shape:**

| Source shape | Policy |
|---|---|
| Positional payload (FYERS `/history`) | Exactly the six known fields. **Any other width → reject ingestion** until the source contract is verified. No named preservation is possible. |
| Named fields via an explicit source-schema mapping | Fields declared in the mapping (e.g. `oi`, `vwap`) are **preserved**, enter dataset identity, and are recorded in the column inventory |
| Named field absent from the mapping | **Canonicalisation failure** on the trusted path |
| Field with an unsupported dtype | **Canonicalisation failure** on the trusted path |

Accepting the manager's preference for §10 of the directive: on the trusted path
an unsupported field is a **failure**, not a recorded discard. Recorded discard
was audit-friendly but still destructive, and discarded source information can
change research meaning — a discarded `oi` column could be exactly what
distinguishes two otherwise identical datasets. DATA INTEGRITY > CONVENIENCE.

The forensic path (`read_unverified`, explicit import tooling) may preserve raw
values or drop them with a record; it makes no trust claim either way.

`schema_version` is recorded and is part of identity.

---

## 16. What identity does and does not prove

FYERS history rows carry no symbol; each row is `[epoch, o, h, l, c, v]`.
Therefore a successful response does **not** independently prove the broker
returned candles for the requested instrument.

Identity truthfully means:

> *candles returned by this provider in response to a request for symbol X*

and **not**:

> *proven observations of instrument X*

The requested symbol, provider, request parameters and response evidence all
enter acquisition provenance so the claim is at least auditable. Cross-instrument
verification would require an independent source and is out of scope for Phase 1.

---

## 17. `ValidatedDataset` strategy

**pandas DataFrames cannot be made immutable** — verified: frozen dataclasses
block only attribute reassignment; deep copy does not stop mutation through the
accessor; setting `arr.flags.writeable = False` on all six columns succeeded yet
pandas 3.0 copy-on-write **still permitted the write**.

Strategy is therefore **tamper-evidence plus non-exposure**:

1. `ValidatedDataset.build(...)` canonicalises, validates and computes digests
   **internally**. No parameter accepts a report, so a mismatched report cannot
   exist — defect 1 removed by construction.
2. A **private deep copy** is held; `.frame` returns a **fresh copy per call**.
3. Evidence is stored as **immutable snapshots**. Note: at baseline
   `ValidationReport`, `FetchReport` and `DatasetManifest` are all mutable
   `@dataclass` — that must change.
4. `__slots__`, frozen semantics, `__reduce__` raising so it cannot be pickled
   and revived carrying stale authority.
5. **Digest recomputed immediately before persistence**; divergence aborts.
6. **Digest recomputed again at trusted read.**

Mutation between build and write is *detected*, not *prevented*. That is the
strongest available guarantee in this language, stated rather than implied.

---

## 18. `TrustedDataset` construction rules

Constructible **iff every check passes**. Each failure has a distinct exception.

| # | Check | Failure |
|---|---|---|
| 1 | `CURRENT` names a complete generation in `trusted_generations/` | `DatasetNotFound` |
| 2 | Manifest present and parseable; `schema_version` known; no unknown or missing fields | `ProvenanceSchemaError` |
| 3 | Frame loads and is canonical | `SchemaError` |
| 4 | `data_digest` recomputed == envelope value | `IntegrityError` |
| 5 | `provenance_digest` recomputed == envelope value | `ProvenanceTampered` |
| 6 | `integrity_id` recomputed == value recorded in `CURRENT` | `GenerationIntegrityError` |
| 7 | Caller locator == envelope identity == digest identity inputs | `IdentityMismatch` |
| 8 | Validation re-run on loaded frame has no ERROR issues | `ValidationFailed` |
| 9 | Acquisition classification ∈ {`SUCCEEDED`} and evidence internally consistent | `AcquisitionEvidenceInvalid` |
| 10 | Generation resides in the trusted namespace | *(structural — a forced generation is unreachable, not rejected)* |

Continuity, session and reproducibility certifications are **attached, not
gated** — the consumer decides.

`read_unverified()` returns `UnverifiedDataset`, a distinct type with no
certification attributes, so it cannot be substituted even by accident.

---

## 19. Generation persistence, atomicity and durability

**Generation id** is a **ULID** (128-bit, content-independent). The previous
`g_<digest12>_<utc>` scheme is withdrawn: two writes can share a timestamp
resolution and digest prefixes can collide. The full identity digest lives in the
envelope, never as the sole filesystem uniqueness mechanism.

**`CURRENT` is an ordinary UTF-8 pointer file** (not a symlink) — atomically
replaceable and portable. It contains `generation_id` and `integrity_id`.

**Write sequence.** Create generation directory → write `data.parquet` → flush →
`fsync` file → write `manifest.json` → flush → `fsync` file → **`fsync` the
generation directory** → **`fsync` the `generations` parent** → write
`CURRENT.tmp` in the same parent → flush → `fsync CURRENT.tmp` →
`os.replace(CURRENT.tmp, CURRENT)` → **`fsync` the dataset parent directory**.

**Correct crash guarantee** — correcting the previous revision, which claimed
`CURRENT` still names the *previous* generation:

> After a crash at any point, `CURRENT` names **either the previous complete
> generation or the newly completed generation**. It never names an incomplete
> generation.

| Failure point | Outcome |
|---|---|
| directory creation | nothing changed |
| Parquet write / fsync | orphan generation; `CURRENT` unchanged |
| manifest write / fsync | orphan generation; `CURRENT` unchanged |
| generation/parent dir fsync | orphan generation; `CURRENT` unchanged |
| `CURRENT.tmp` write/fsync | orphan generation; `CURRENT` unchanged |
| `os.replace` | atomic — old or new, never partial |
| after replace, before parent fsync | either value may survive; **both are complete generations** |

**Platform assumptions, stated:** `os.replace` is atomic on POSIX and on Windows
for same-volume replacement; directory `fsync` is meaningful on Linux/macOS and a
no-op on some filesystems. Development and CI target macOS/Linux. Orphan
generations are inert, never selected, detectable by absence from `CURRENT`, and
safe to garbage-collect.

---

## 20. Reproducibility certification

`git_revision()` can return `unknown`, `not-a-git-repo` or `<sha>-dirty`, and
baseline still treats the dataset as authoritative. **Data validity and
reproducibility are different guarantees.**

The previous revision's "environment matching the lock" was underspecified.
Replacement: record an **`environment_spec_digest`** over a canonical
environment description containing Python major.minor.patch, OS, architecture,
direct dependency name/version pairs, full transitive dependency name/version
pairs, lock file digest, git commit SHA, and git dirty flag.

`ReproducibilityCertification = CERTIFIED` requires: git SHA known, tree clean,
and `environment_spec_digest` recorded. It is `NOT_CERTIFIED` otherwise.

**A different platform may produce logically identical data.** Reproducible
*environment* and identical *dataset* are separate claims; the digest certifies
the former, and the data digest independently establishes the latter.

---

## 21. Secret lifecycle

Baseline reads only the current environment value, so a rotated-out token becomes
unredactable — verified.

**Append-only, process-lifetime `SecretRegistry`.** Secrets are registered
explicitly when loaded (settings load, token exchange) and **never removed**, so
`TOKEN_A` stays redacted after rotation to `TOKEN_B`.

**No LRU eviction.** The previous revision proposed a bounded registry with
eviction; that reintroduces the exact defect being fixed. Memory is bounded in
practice by the number of credentials a process loads (single digits). If that
ever changes, it will be **measured** before any eviction policy is considered.

Registered: client secret, access token, auth code, any future refresh token, and
any other credential material loaded by settings or authentication.

Redacted across `record.msg`, `record.args`, exception messages, formatted
tracebacks, chained and nested tracebacks, URLs, and multiline values. Redaction
lives in the **formatter**, because filters run before `exc_info` is rendered.

**FYERS SDK-owned logging is OUTSIDE our formatter boundary.** The SDK builds its
own `FileHandler` and logs full URLs at debug level. Mitigation: keep it at
`ERROR`, point `log_path` into gitignored `logs/`, document the boundary. We do
**not** claim SDK logs are protected.

Secrets shorter than 8 characters remain unredacted by policy; real tokens are
long, and a lower threshold would redact ordinary words.

---

## 22. Broker error sanitisation

**Confirmed defect.** `historical.py` builds `detail` from the raw broker
`message` and embeds it in exception text; four scripts then `print(f"…{exc}")`.
Our redaction covers logging, **not** `print`.

Broker responses are **untrusted input** and could echo request URLs, client
identifiers, auth codes or tokens.

A `BrokerDiagnostic` value carries a numeric status/code, a **sanitised** message
(length-capped, control characters stripped, passed through the secret registry)
and redacted structured fields. Exceptions carry the diagnostic, not raw payload
text. CLI prints the diagnostic; raw payloads reach only the redacting logger at
debug level. Useful broker errors are not hidden — they are not trusted verbatim.

---

## 23. Configuration validation

**Confirmed defect.** `lot_size=int(lot_size) if lot_size else 1` silently turns
`0` and `""` into `1`.

The previous revision's blanket "missing optional → `None`, never a default" was
too broad; some fields legitimately have documented defaults. **What is forbidden
is truthiness-based repair.**

Per-field schema declares one of: `required`; `optional nullable`; or
`optional with an explicit documented default`.

| Field | Class |
|---|---|
| `symbol`, `kind`, `role`, `exchange` | required |
| `lot_size` | required positive integer **for a verified tradable instrument**; nullable while a candidate is unverified |
| `tick_size` | optional nullable while unverified |
| `verified` | optional, documented default `false` |

Present-but-invalid (`0`, negative, wrong type, malformed numeric) raises naming
the field, value and file. Unknown keys rejected. Duplicate symbols rejected.
Config carries a `schema_version`.

---

## 24. The no-authority-laundering test

One end-to-end integration test whose sole purpose is:

> Nothing invalid, incomplete, forged, stale, forced or tampered can be converted
> into a `TrustedDataset`.

Each must fail: invalid frame · validation from a different frame · frame mutated
after validation · source-order anomaly · **manifest `requested_range` shortened
to match sparse data** · **manifest chunk failures removed** · forged
`validation_status` · forged acquisition status · forged `forced` flag · forged
data digest · **provenance envelope modified without updating the pointer
digest** · **pointer `integrity_id` modified** · copied manifest from another
dataset · modified Parquet · missing manifest · partial acquisition · all chunks
empty · all chunks failed · huge interior gap where a policy requires continuity
· duplicate conflicting candles · unsupported schema field · **extra positional
broker field** · **ambiguous string containing encoding delimiters** · **valid
dataset written through the force path** · **forced manifest edited to
`force=false`** · interrupted generation write · **`CURRENT` crash recovery,
old-complete and new-complete cases** · **generation id collision attempt** ·
**wrong expected source / symbol / resolution locator**.

A dataset genuinely satisfying every invariant must still succeed — the test must
prove the gate is not simply refusing everything.

---

## 25. Test matrix

Correcting the previous revision: numeric expectations like "29-day hole blocks"
are **removed**. They encoded a magic number as semantics. Tests are
policy-based.

| Invariant | Unit | Integration | Adversarial | Mutation |
|---|---|---|---|---|
| Canonical encoding | type-tag matrix | write→read stable | delimiter-bearing strings, Unicode NFC, 1e-9, ±0.0, NaN, inf, NA/0 | revert to `%.10g` |
| Ingestion evidence | anomaly recorded | out-of-order response | source order erased | remove recording |
| Anomaly severity | each class | blocker halts build | conflicting duplicates | downgrade a blocker |
| Schema policy | mapped field preserved | round-trip with `oi` | 7-field positional payload | allow unknown field |
| Binding | build rejects a report arg | download→write | all mismatch cases | remove binding |
| Tamper-evidence | mutate after build | build→mutate→write | mutate via accessor | remove recompute |
| Envelope integrity | digest computation | manifest edit detected | edit each envelope field | skip envelope check |
| Locator identity | three-way agreement | wrong symbol locator | manifest says SBIN | drop locator check |
| Acquisition | 5 states | all/none/some fail | empty vs failed | `failed==0` shortcut |
| ObservedCoverage | facts recorded | 1 day for 1-year request | boundary on a weekend | grade absence as partial |
| Probe | window classification | bracket output | ERROR windows, holiday windows | restore binary search |
| Continuity | NOT_CERTIFIED without calendar | fake deterministic calendar identifies a known missing session | policy threshold N: gap > N gives expected policy result | remove certification |
| Trusted read | each of 9 checks | full pipeline | forge each field | delete a check |
| Generation atomicity | each failure point | crash between stages | old-complete and new-complete recovery | non-atomic rename |
| Generation id | uniqueness | concurrent writes | collision attempt | content-derived id |
| Force namespace | separate directory | force over trusted | edit `forced=false` | allow CURRENT to point at forced |
| Secrets | registry lifetime | rotation A→B | A in traceback after rotation | add eviction |
| Broker text | sanitiser | error → CLI | payload containing a token | echo raw |
| Config | per-field schema | load registry | `0`, `""`, unknown key | restore truthiness |
| Reproducibility | dirty tree | env digest | platform difference | merge with validity |

Every invariant must have at least one test that **fails when the production
check is removed**. Baseline mutation testing caught 9 of 9 attempted reversions;
that standard carries forward.

---

## 26. Failure-state matrix

Corrected: the previous revision claimed consistent manifest tampering is caught
by revalidation. **That is false** — a manifest can be altered in fields
unrelated to OHLC validity (requested range, chunk failures, force reason,
canonicalisation evidence) and stay self-consistent. Revalidation cannot see it;
the **envelope digest** can.

| State | `read_trusted()` | `read_unverified()` |
|---|---|---|
| No dataset / orphan only | `DatasetNotFound` | orphan reachable by explicit id |
| Manifest missing / unparseable / unknown fields | `ProvenanceSchemaError` | returns data |
| Parquet tampered | `IntegrityError` (data digest) | returns data |
| **Manifest tampered, self-consistent** | **`ProvenanceTampered` (envelope digest)** | returns data |
| Manifest and Parquet tampered consistently | `GenerationIntegrityError` (pointer) | returns data |
| Pointer edited | `GenerationIntegrityError` | n/a |
| Locator disagrees with envelope | `IdentityMismatch` | n/a |
| Validation errors | `ValidationFailed` | returns data |
| Acquisition partial/empty/failed/unknown | `AcquisitionEvidenceInvalid` | returns data |
| Forced generation | **unreachable** (namespace) | reachable by explicit id |
| Valid, continuity not certified | **succeeds**, `ContinuityCertification=NOT_CERTIFIED` | n/a |
| Valid, irreproducible | **succeeds**, `ReproducibilityCertification=NOT_CERTIFIED` | n/a |
| Fully valid | **succeeds** | n/a |

---

## 27. Backward compatibility

`data_store/` is **empty** at baseline — verified. The generation-directory
layout and the manifest/envelope schema are breaking on-disk changes that are
**free now and expensive later**. No migration path is required and none will be
written.

---

## 28. Implementation sequence

CI moves to the front, accepting the manager's reasoning: automation should
protect the review branch **during** this rewrite, not after it.

| # | Unit | Addresses | Note |
|---|---|---|---|
| 1 | **Minimal CI** on the review branch using the existing documented install (including the `asyncio` uninstall step) | — | Protects everything that follows; evolves once locking improves |
| 2 | Schema policy + canonicalisation contract with evidence and severity | 5, 6 | **Must precede the digest** — determines what it covers |
| 3 | Canonical encoding + dataset identity digest | 7, 8 | Depends on 2 |
| 4 | Immutable evidence snapshots | prerequisite | Baseline reports are mutable |
| 5 | `ValidatedDataset` + write accepting only it | 1 | Depends on 3, 4 |
| 6 | Provenance envelope + generation integrity id | 2 | Depends on 3 |
| 7 | Generation storage, namespaces, atomic `CURRENT` | 10, 11 | **Must precede the read boundary** |
| 8 | Acquisition classification + `ObservedCoverage` | 3 | |
| 9 | `TrustedDataset` read boundary | 9 | Depends on 5, 6, 7, 8 |
| 10 | Probe redesign | 13 | Independent; gates any live depth claim |
| 11 | Continuity/session certification + calendar protocol | 4 | |
| 12 | `SecretRegistry` + broker diagnostics | 12, 14 | Independent of the data path |
| 13 | Strict config validation | 16 | Independent |
| 14 | `write_token_to_env` validation | 15 | Small, independent |
| 15 | Dependency reproducibility (`pyproject`/lock/constraints) | — | CI upgraded alongside |
| 16 | Documentation reconciliation | — | Last, so it describes what exists |

Each unit: test first → implementation → full suite → adversarial test → commit →
push review branch → **STOP** for manager inspection on GitHub. **CI passing is
not a substitute for manager review.**

---

## 29. Open risks

1. **Tamper-evidence is weaker than immutability** — mutation between build and
   write is detected, not prevented. Unavoidable in pandas.
2. **The `/history` contract is ASSUMED**, never LIVE-VERIFIED. The positional
   rejection rule in §15 is designed against an unverified contract.
3. **Instrument correctness is unprovable** from candles alone (§16).
4. **Continuity and session coverage are uncertifiable** without a calendar, and
   no calendar exists.
5. **SDK logs remain outside the redaction boundary**, permanently.
6. **Probe cost rises** materially under §12 and interacts with rate limits;
   window size and horizon must be tuned against the documented daily cap.
7. **Digest cost is O(rows)** in Python; fine at ~90k rows/year, unmeasured beyond.
8. **Out-of-scope actor** can recompute all digests and the pointer. Stated, not
   defended against.

---

## 30. Non-goals

Not in Phase 1: strategy logic, indicators, signal generation, backtesting,
optimisation, walk-forward, Monte Carlo, TradingView or Pine integration,
webhooks, dashboards, paper trading, order execution, live-trading mode,
execution-instrument selection, NSE calendar data, and any performance or
profitability claim.

Also not goals: defending against an attacker with machine access; asymmetric
signing; making pandas immutable; certifying continuity, session coverage or bar
density without a calendar; proving instrument identity from candle data.
