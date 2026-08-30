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

Every claim was checked against code at this commit. Where an earlier revision of
this document was wrong, the correction is stated plainly rather than layered
over.

---

## 2. Glossary

Each term has exactly one meaning. **There is no `is_authoritative` boolean.**
The words *authoritative*, *complete* and *usable* are not used anywhere in this
design except where quoting a defect.

| Dimension | Values | Meaning | Derived from |
|---|---|---|---|
| `ArtifactIntegrity` | INTACT / BROKEN | The **loaded logical dataset** canonicalises to the recorded `data_digest`, the envelope canonicalises to `provenance_digest`, pointer/envelope integrity holds, and generation identity/location agree. See §2.3 — this is **not** byte-identity of `data.parquet` | data + envelope |
| `SchemaValidity` | VALID / INVALID | Frame conforms to the canonical schema and declared `schema_version` | data |
| `MarketDataValidity` | VALID / INVALID | No ERROR-severity validation issue (OHLC rules, finiteness, duplicates, ordering) | data |
| `AcquisitionRequestStatus` | see §11.1 | **Only** whether broker requests returned without error | provenance |
| `ObservedDataCoverage` | a record of facts | Earliest/latest observed date, distinct observed dates, observed span | **data alone** |
| `RequestedWindowComparison` | a record of facts | Whether observations exist on requested boundary dates; containment in the request window; requested span | **data + integrity-bound provenance** |
| `ContinuityCertification` | CERTIFIED / NOT_CERTIFIED / FAILED | Whether interior gaps are explained | needs a calendar |
| `SessionCertification` | CERTIFIED / NOT_CERTIFIED / FAILED | Whether expected sessions and bar density are present | needs a calendar |
| `ReproducibilityCertification` | CERTIFIED / NOT_CERTIFIED | Actual environment satisfies the declared expected environment policy | environment + policy |
| `Namespace` | TRUSTED / FORCED | Structural provenance state from storage layout | filesystem + envelope |
| `GenerationFreshness` | **not modelled** | See §5 — cannot be established | — |
| `ResearchReadiness` | READY / NOT_READY *(per policy)* | Whether a specific experiment's `ResearchDataPolicy` is satisfied | all of the above |

### 2.1 `ArtifactIntegrity` is logical, not byte-level

`data_digest` is computed over the **canonical logical dataset** (§8.2), not over
the raw `data.parquet` byte stream. Parquet embeds writer version, compression
settings, row-group layout and column metadata, all of which can differ between
runs without any observation differing.

`ArtifactIntegrity = INTACT` therefore means:

- the selected generation satisfies pointer and envelope integrity;
- the loaded logical dataset canonicalises to the recorded `data_digest`;
- the provenance envelope canonicalises to the recorded `provenance_digest`;
- generation identity and location agree (§7).

**It does not mean** every byte of `data.parquet` is identical to the file
originally written.

| Change to `data.parquet` | Outcome |
|---|---|
| Any change altering the decoded observations | **Detected** — `data_digest` differs |
| Rewrite preserving exactly the same logical canonical dataset (recompression, different row-group layout, writer version) | **Not detected — accepted** |

The second row is deliberate. Our scientific concern is the integrity of the
observations, not of a particular serialisation of them, and treating a
recompression as corruption would produce false alarms on any legitimate
re-encode. **No raw-Parquet file digest is added**: byte-for-byte artifact
identity has no demonstrated Phase-1 requirement, and adding a second digest
would create a second thing to keep consistent for no scientific gain.

### 2.2 `TrustedDataset` — exact guarantee

> `ArtifactIntegrity = INTACT` **and** `SchemaValidity = VALID` **and**
> `MarketDataValidity = VALID` **and** the acquisition provenance is
> integrity-bound and internally consistent **and** the caller's locator matches
> the envelope identity **and** `Namespace = TRUSTED`.

**It does NOT guarantee**: continuity, session coverage, bar density,
requested-window coverage, reproducibility, generation freshness, or that the
broker returned the instrument requested (§17). Those travel *on* the object as
separate records and certifications.

**A `TrustedDataset` is a sound artifact, not a research input.**

### 2.3 `ResearchReadyDataset` — exact guarantee

> Constructed **only** from a `TrustedDataset` plus an explicit
> `ResearchDataPolicy`, and only when every requirement the policy declares is
> satisfied.

A `ResearchDataPolicy` may require any of: `ContinuityCertification == CERTIFIED`;
`SessionCertification == CERTIFIED`; minimum distinct observed dates; minimum
fraction of the requested window observed; `ReproducibilityCertification ==
CERTIFIED`; expected source, symbol and resolution.

**Phase 2 accepts only `ResearchReadyDataset`, never `TrustedDataset`.** The
storage layer does not decide research suitability; the experiment declares what
it needs. This is what prevents a sparse three-candle artifact from reaching a
backtest, without relying on programmer discipline.

---

## 3. Problem statement

Sixteen confirmed defects share one root cause: the system trusts assertions
instead of deriving facts. But **not every fact is derivable**, and the design
says which are which (§4).

| # | Defect at baseline | Evidence |
|---|---|---|
| 1 | `store.write()` takes frame and report separately, never checking they correspond | Invalid frame + valid frame's report persists as `validation_status="valid"` |
| 2 | `is_authoritative` forgeable via `validation_status`/`fetch_status`/`forced` | Edited manifest yields `True` on invalid data |
| 3 | Completeness derived from `failed_chunks == []` | 3 candles stored as an authoritative answer to a 1-year request |
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
| 17 | Storage paths interpolate identifiers with only `:` and `/` replaced | `store.py:156`; a `..` component would traverse |

---

## 4. Threat model and the three classes of fact

**In scope.** Accidental manifest edits; stale or partially-written files;
process death mid-write; inconsistent output from future automation; manifests or
generation directories copied between datasets; operator mistakes; malformed or
hostile *broker responses*; unsafe identifiers reaching filesystem paths; our own
refactoring errors.

**Out of scope.** A malicious actor controlling the machine and source tree. Such
an actor can recompute every digest and rewrite the pointer. Stated explicitly
rather than implying protection that does not exist.

**No asymmetric signing.** The envelope (§6) detects every in-scope failure
except rollback (§5). A key on the same machine would not raise the bar against
the out-of-scope actor.

| Class | Examples | Protection |
|---|---|---|
| **Data-derived** | OHLC validity, schema conformance, observed dates, data digest | **Recomputed** from stored data at every use |
| **Provenance evidence** | requested range, chunks requested/failed/empty, broker attribution, raw column inventory, canonicalisation anomalies | **Cannot be recomputed from candles.** Integrity-bound via the envelope digest |
| **Operator declarations** | `forced`, `force_reason`, notes | Integrity-bound **plus** structural demotion via namespace (§10) |

The manager's counterexample is handled: stored data `Jan 1 – Jan 3`, actual
request `Jan 1 – Dec 31`, manifest edited so `requested_to = Jan 3`,
`failed_chunks = []`. Every field is internally self-consistent and no OHLCV
revalidation recovers the original request. **Only the envelope digest detects
it.**

---

## 5. What the design does NOT detect: generation rollback

The previous revision implied *pointer edited → `GenerationIntegrityError`*.
**That is false in general**, and the manager's counterexample is correct.

> `CURRENT` names generation B with integrity id B. An operator accidentally
> restores an older `CURRENT` file naming generation A with integrity id A.
> Generation A is complete, its data and provenance digests agree, and its
> integrity id is valid. **Every check in §12 passes.**

Rollback to an *earlier valid generation* is therefore **not detectable** from
generation-local evidence. Detecting it requires an external monotonic anchor —
an append-only journal with a separately verified head.

**Decision: no journal in Phase 1.** There is one operator, no concurrent
writers, and `data_store/` is empty. A journal would add a second consistency
problem (journal-vs-pointer divergence) to defend against a scenario with no
evidence of occurring. Complexity is not free, and unnecessary machinery is
itself a defect source.

**`GenerationFreshness` is deliberately NOT modelled.** An enum whose value is
always `UNKNOWN` would be decoration, not information.

**Stated limitation, carried into §29 and §30:**

> Artifact integrity detects corruption, forgery and inconsistency. It does
> **not** prove that the selected generation is the newest one ever written.

If evidence of rollback occurring ever appears, an append-only journal is the
designed remedy and should be added then, not speculatively now.

---

## 6. Generation integrity envelope

```
data_digest        = SHA256( canonical encoding of identity + observations )   [§8]
provenance_digest  = SHA256( canonical encoding of the provenance envelope )
generation_id      = uuid.uuid4()                                              [§13]
integrity_id       = SHA256( data_digest || provenance_digest )
```

**Generation id is `uuid.uuid4()` from the standard library.** ULID is withdrawn:
we need uniqueness, content-independence and timestamp-independence, but **not**
lexicographic time ordering, and adding a dependency or hand-rolling ULID for
sortability we do not use would be unjustified complexity.

The **provenance envelope** contains, canonically encoded:

- `provenance_schema_version` — **versioned independently** of the market-data
  `schema_version`, since acquisition fields will evolve while the candle schema
  stays stable. Trusted read rejects unknown provenance versions unless an
  explicit migration exists.
- Identity: `schema_version`, `source`, `symbol`, `resolution`
- **`generation_id` and `namespace`** (§7)
- Acquisition snapshot (§11)
- Canonicalisation snapshot (§14)
- Operator declarations: `forced`, `force_reason`
- Environment snapshot (§21)

| Accidental change | Detected because |
|---|---|
| Parquet edited | `data_digest` changes → `integrity_id` mismatch |
| Manifest edited (any field) | `provenance_digest` changes → mismatch |
| Both edited consistently | `integrity_id` still mismatches the pointer |
| Manifest copied from another dataset | Identity and both digests mismatch |
| **Generation directory copied elsewhere** | Envelope `generation_id`/`namespace` disagree with the location (§7) |
| Pointer names a generation whose digests disagree | mismatch |
| **Pointer restored to an older valid generation** | **NOT DETECTED — see §5** |

---

## 7. Generation identity binds to location

Filesystem location must never be an unverified implicit assertion. Four things
must agree:

| Source of truth | Must equal |
|---|---|
| Filesystem generation directory name | envelope `generation_id` |
| Pointer `generation_id` | envelope `generation_id` |
| Filesystem namespace directory (`trusted_generations` / `forced_generations`) | envelope `namespace` |
| Pointer `integrity_id` | recomputed `integrity_id` |

Because `generation_id` and `namespace` are inside the envelope and therefore
inside `provenance_digest`, copying a complete generation directory into another
location — or from `forced_generations/` into `trusted_generations/` — is
detected: the envelope still names its birth location, which no longer matches.

---

## 8. Canonical encoding and dataset identity

### 8.0 Schema version numbers — frozen

```
MARKET_DATA_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION  = 1
```

**`MARKET_DATA_SCHEMA_VERSION = 1`** identifies the currently frozen canonical
market-data representation defined in this document: the six-column canonical
frame (`ts`, `open`, `high`, `low`, `close`, `volume`) with the canonical
dtypes, timestamp semantics, missing-value semantics, and identity-defining
field interpretation described throughout §8 and §14–15. It is one of the
inputs to dataset identity (§8.1) and therefore to `data_digest`.

A **new** `MARKET_DATA_SCHEMA_VERSION` is required whenever a change would
alter what the canonical logical dataset *means* — for example: adding a
canonical observation field; changing timestamp meaning; changing canonical
numeric representation; changing missing-value semantics; changing the
interpretation of any identity-defining field.

A new version is **not** required for: implementation refactors; added tests;
performance changes; or storage-layout changes that do not alter logical
dataset identity.

**`PROVENANCE_SCHEMA_VERSION = 1`** identifies the first provenance-envelope
structure described in §6, independently of the market-data schema version —
acquisition/provenance fields are expected to evolve on their own timeline
while the candle schema stays stable. No provenance envelope implementation
is authorised by this section; this freezes the version number contract only,
ahead of that future unit.

Callers never choose or invent either version number. Both are fixed
constants defined once by this document (and mirrored as code constants where
each is implemented) and are never parameters.

### 8.1 Identity

> **(schema_version, source, symbol, resolution, canonical observation sequence
> including preserved named extra columns)**

`schema_version` here is `MARKET_DATA_SCHEMA_VERSION`, defined in §8.0.

In identity: schema version, source/vendor, symbol, resolution, canonical
observation columns, preserved named extra columns, timestamp instant, row
multiplicity, duplicate and conflicting timestamps, and adjustment state **when
introduced** (none today; it must enter identity the day it does).

Not in identity: timezone representation, row ordering, column ordering, dtype,
acquisition range, manifest metadata.

**Verified against what.** `source`, `symbol`, `resolution` and `schema_version`
are not derivable from candles. They are verified as a three-way agreement:

```
read_trusted(source=…, symbol=…, resolution=…)
        ↓
 caller locator  ==  envelope identity  ==  digest identity inputs
```

A manifest may never redefine the identity of the object requested. Asking for
`NSE:NIFTY50-INDEX` cannot return a generation whose manifest was edited to say
`SBIN`.

### 8.2 Encoding

Delimiter-based encoding is withdrawn — pipes, newlines, escapes and separator-
bearing column names make it ambiguous once string columns exist.

**Typed, length-prefixed binary. No delimiters, so no collision is possible.**

```
field  := type_tag (1 byte) || length (8 bytes, big-endian) || payload
digest := SHA256(field*)
```

| Tag | Type | Payload |
|---|---|---|
| `0x01` | STR | UTF-8 bytes, normalisation per §9 |
| `0x02` | F64 | 8 bytes, IEEE-754 big-endian |
| `0x03` | I64 | 8 bytes, two's complement big-endian |
| `0x04` | NA | zero length |
| `0x05` | NAN | zero length |
| `0x06` | POSINF | zero length |
| `0x07` | NEGINF | zero length |
| `0x08` | TS | 8 bytes, int64 **nanoseconds** since Unix epoch, UTC |

`-0.0` normalises to `+0.0` before encoding (IEEE signed zero is not a market
distinction and a zero price is invalid anyway). `NaN` and `±Inf` use their own
tags, so they can never collide with a finite value. Missing volume is `NA`,
distinct from `I64(0)`. Column names are `STR` header fields; column order is
canonical, not input order; row order is the canonical sort order.

Verified with an in-memory prototype: `24000.12` ≠ `24000.13`; a 1e-9 change
differs (baseline `%.10g` says *same*); `0` ≠ `NA`; `NaN` and `+inf` distinct;
duplicate row added and row removed differ; symbol, resolution and source each
differ; rows reversed, UTC-vs-IST representation and reordered columns all match.
Empty and one-row frames encode deterministically and differ.

### 8.3 Equal-timestamp identity ordering — frozen

**Confirmed defect, found in manager review of Unit 3 (commit `6239148`).**
§8.1 states row ordering is **not** identity. But canonicalisation performs
only a **stable** sort by timestamp, so when two or more canonical
observations share exactly one timestamp, their relative order in the
canonical frame is whatever the source delivered. §8.2's row-order encoding
("row order is the canonical sort order") then bakes that source-arrival
order into `data_digest` — making two datasets with the identical
observation multiset at a shared timestamp, differing only in which arrived
first, hash differently. That contradicts §8.1.

**Reproduced directly against `6239148`:** two canonical frames, each with
one timestamp `T` carrying two genuinely different observations `A` and `B`
(same multiset, same multiplicity — one of each), differing only in
arrival order (`T:A,B` vs `T:B,A`), produced two different `data_digest`
values.

**Root cause:** conflating two distinct concerns that must stay separate:

| Concept | What it is | Where it belongs |
|---|---|---|
| Source arrival order within a shared timestamp | Provenance evidence — a fact about what the broker sent and in what order | Canonicalisation snapshot / future `provenance_digest` (§6) — **may** differ between two acquisitions of the same logical dataset |
| The canonical observation multiset at a shared timestamp | Dataset identity | `data_digest` (§8.1) — **must not** differ based on arrival order alone |

`canonicalise()`'s stable sort is correct and **is not changed by this
section** — it must keep preserving source arrival order in
`CanonicalisationResult.frame` and in the future canonicalisation snapshot,
precisely so that provenance can later distinguish "FYERS sent B before A."
That fact is real and must not be erased. It is simply **not** a dataset
identity fact, and `data_digest` must stop depending on it.

**Frozen equivalence rule, schema v1:**

1. Timestamps establish primary observation ordering — unchanged.
2. Within one timestamp shared by two or more canonical observations,
   **source arrival order is not identity.**
3. The set of canonical observations at one timestamp is a **multiset**:
   multiplicity is identity (§7 already establishes this at the whole-dataset
   level; this makes it explicit per timestamp group).
4. Therefore identity encoding requires a **deterministic ordering of
   equal-timestamp observations that does not depend on source arrival
   order.**
5. That deterministic order is: **encode each observation's non-timestamp
   canonical fields (open, high, low, close, volume) using the exact §8.2
   typed field encoding, concatenate per observation, then sort the
   equal-timestamp group by those encoded byte strings, lexicographically.**
   Using the already-canonical encoded bytes — not `repr()`, not string
   formatting, not raw pandas/Python comparison — is required because it
   reuses the same normalisation §8.2 already defines (`-0.0`→`+0.0`, every
   `NaN` payload → one `NAN` token, `±Inf` distinct, volume `NA` distinct
   from `I64(0)`), so the tie-break ordering is exactly as deterministic and
   collision-free as the field encoding itself. Raw Python/pandas comparison
   is explicitly rejected here because `NaN` has no total order under `<`.
6. Every occurrence within the group is preserved — **sorting is not
   deduplication.** `T:A` and `T:A,A` remain different (the second has one
   more encoded-and-sorted element than the first).
7. Datasets whose timestamps are all strictly unique are **unaffected**:
   there is no equal-timestamp group to reorder, so this section changes
   nothing for the ordinary case where no timestamp repeats.

**Consequence for §8.1's identity statement.** "Row ordering is not
identity" now means precisely: not overall source row order (already true),
and not source arrival order *within* an equal-timestamp group either. What
**is** identity, per timestamp, is the timestamp itself plus the multiset of
observations sharing it (§7's row-multiplicity rule already covers
duplicate multiplicity generally; this section fixes how a multiset is
placed into a deterministic byte order for hashing).

**Required future tests** (implementation of this correction is a separate,
explicitly authorised step — not this document):

| # | Case | Expected `data_digest` |
|---|---|---|
| A | `T:A,B` vs `T:B,A` | SAME |
| B | `T:A,A,B` vs `T:B,A,A` | SAME |
| C | `T:A` vs `T:A,A` | DIFFERENT |
| D | `T:A,B` vs `T:A,C` | DIFFERENT |
| E | `NaN`-bearing equal-timestamp observations, reordered at source | SAME |
| F | volume-`NA`-bearing equal-timestamp observations, reordered at source | SAME |
| G | different (non-equal) timestamps reordered at source, then canonicalised | SAME (already covered by the existing reversed-rows test) |

---

## 9. Unicode normalisation is a per-field decision

Global NFC normalisation is withdrawn: it would silently rewrite byte-distinct
source text, which is a data transformation, not an encoding detail.

Field text policy is declared per field by the schema:

| Policy | Applies to | Behaviour |
|---|---|---|
| `TEXT_NFC` | Identity metadata: `source`, `symbol`, `resolution`, column names | NFC-normalised. These are our own identifiers, and semantic equality is what we want. |
| `TEXT_EXACT` | **Preserved source string columns** | Encoded byte-exact. Never normalised. |

Where NFC normalisation actually changes the bytes of an identity field, the
change is **recorded in the canonicalisation snapshot** (§14). No single global
rule may alter a research field.

---

## 10. Forced generations — one semantic model

Force is an **operational provenance state**, not a claim about candle validity.
The earlier "valid forced data is authoritative on its own merits" is withdrawn.

```
data_store/<source_slug>/<symbol_slug>/<resolution_slug>/
    trusted_generations/<uuid4>/   data.parquet + manifest.json
    forced_generations/<uuid4>/    data.parquet + manifest.json
    CURRENT
```

- `force=True` writes into `forced_generations/` and requires a non-empty
  `force_reason`.
- `CURRENT` may only ever name a generation in `trusted_generations/`.
- `read_trusted()` **never traverses** `forced_generations/`.
- Even perfectly valid candles written through the forced path remain forensic
  until re-imported through the normal pipeline.
- Reachable only via the forensic API with an explicit generation id.

Selection depends on structure, not on an editable boolean. Editing
`forced=false` in a forced generation's manifest changes nothing: it is still in
the forced directory, and the edit additionally breaks the envelope digest and
the §7 namespace agreement.

---

## 11. Acquisition evidence and coverage

### 11.1 `AcquisitionRequestStatus`

```
REQUESTS_FAILED      every chunk errored
REQUESTS_EMPTY       all chunks returned without error, all returned zero rows
REQUESTS_PARTIAL     at least one chunk errored
REQUESTS_SUCCEEDED   every chunk returned without error, ≥1 row overall
REQUESTS_UNKNOWN     no acquisition evidence (fixtures, manual frames)
```

Renamed from `ACQUISITION_SUCCEEDED` so the name cannot be misread as coverage.

> **`REQUESTS_SUCCEEDED` means only: every broker request returned a non-error
> result. It says nothing about requested-range coverage, interior completeness,
> trading days, bar density, or expected sessions.**

This sentence is load-bearing. No consumer may infer completeness from the enum.

### 11.2 `ObservedDataCoverage` — data alone

`earliest_observed_date`, `latest_observed_date`, `distinct_observed_dates`,
`observed_span_days`.

### 11.3 `RequestedWindowComparison` — data + integrity-bound provenance

`requested_from`, `requested_to`, `requested_span_days`,
`observations_on_requested_start_date`, `observations_on_requested_end_date`,
`all_observations_within_requested_window`, `observed_distinct_dates_ratio`.

Correcting the previous revision, which classified these as candle-only facts:
they depend on the request window, which lives in provenance and is trustworthy
only because it is envelope-bound.

**Absence of observations on a requested boundary date is NOT graded as
deficient** — that date may not have been a trading day. Grading requires
`SessionCertification`, which is `NOT_CERTIFIED` without a calendar.

The "3 candles for a one-year request" case is recorded honestly:
`REQUESTS_SUCCEEDED`, `distinct_observed_dates = 1`, `requested_span_days = 365`,
`SessionCertification = NOT_CERTIFIED`. It may be a sound `TrustedDataset`
artifact; it fails any `ResearchDataPolicy` requiring meaningful coverage.

---

## 12. `TrustedDataset` construction rules

| # | Check | Failure |
|---|---|---|
| 1 | `CURRENT` parses per §16, fields known, `generation_id` syntactically valid | `PointerFormatError` |
| 2 | `CURRENT` names a complete generation in `trusted_generations/` | `DatasetNotFound` |
| 3 | Manifest present, parseable, `schema_version` and `provenance_schema_version` known, no unknown/missing fields | `ProvenanceSchemaError` |
| 4 | Frame loads and is canonical | `SchemaError` |
| 5 | `data_digest` recomputed == envelope value | `IntegrityError` |
| 6 | `provenance_digest` recomputed == envelope value | `ProvenanceTampered` |
| 7 | `integrity_id` recomputed == pointer value | `GenerationIntegrityError` |
| 8 | Envelope `generation_id` and `namespace` == filesystem location and pointer (§7) | `GenerationLocationMismatch` |
| 9 | Caller locator == envelope identity == digest identity inputs | `IdentityMismatch` |
| 10 | Validation re-run on the loaded frame has no ERROR issues | `ValidationFailed` |
| 11 | `AcquisitionRequestStatus == REQUESTS_SUCCEEDED` and evidence internally consistent | `AcquisitionEvidenceInvalid` |

Forced generations are **unreachable** rather than rejected — a structural
property, not a check.

Continuity, session, reproducibility certifications and both coverage records are
**attached, not gated**. `ResearchDataPolicy` gates them (§2.2).

`read_unverified()` returns `UnverifiedDataset` — a distinct type with no
certification attributes, so it cannot be substituted even by accident.

---

## 13. Storage layout, path safety and persistence

### 13.1 Path safety (defect 17)

Baseline builds paths as `symbol.replace(":", "_").replace("/", "_")`. A `..`
component would traverse.

**Identifiers are never interpolated into paths.** Each of `source`, `symbol` and
`resolution` maps to a **safe slug**. The previous revision named a restricted
alphabet *and* percent-encoding, which contradicted itself: `%` is not in
`[A-Za-z0-9._-]`. Percent-encoding is withdrawn.

**Canonical mapping: readable prefix + digest suffix.**

```
slug = <sanitised_prefix> "-" <hex(sha256(utf8(identifier))[:8])>
```

- `sanitised_prefix` — the identifier's UTF-8 bytes with every character outside
  `[A-Za-z0-9_-]` replaced by `_`, truncated to 32 characters. `.` is **excluded
  from the alphabet**, so `.` and `..` components are impossible by construction
  rather than by a rejection rule that could be forgotten.
- `digest_suffix` — 16 hex characters, making the mapping collision-resistant
  even when two identifiers sanitise to the same prefix.

Properties, each satisfied by construction: the raw identifier is never a path
component; no `/` or `\` can appear; no `.`/`..` component is possible; no
absolute-path interpretation is possible; the mapping is deterministic and
collision-resistant; component length is bounded at 49 characters.

**The slug is one-way and decoding is never required.** The original identifiers
live in the envelope and in the identity digest, and lookup goes through the
envelope. **The slug is a locator only and is never treated as identity** (§8.1).

### 13.2 `CURRENT` pointer format

An ordinary UTF-8 file (**not a symlink**) — atomically replaceable and portable.
Versioned canonical JSON, sufficient because it contains no floats:

```json
{
  "pointer_version": 1,
  "generation_id": "…uuid4…",
  "integrity_id": "…sha256 hex…"
}
```

Deterministic serialisation on write (sorted keys, no whitespace variance);
strict rejection of unknown fields on read; **no absolute paths**;
`generation_id` syntax-validated as a UUID before any filesystem use.

### 13.3 Atomic write and durability

Create generation directory → write `data.parquet` → flush → `fsync` → write
`manifest.json` → flush → `fsync` → `fsync` generation directory → `fsync`
namespace directory → write `CURRENT.tmp` in the same parent → flush → `fsync` →
`os.replace(CURRENT.tmp, CURRENT)` → `fsync` dataset parent directory.

**Correct crash guarantee:**

> After a crash at any point, `CURRENT` names **either the previous complete
> generation or the newly completed generation**. It never names an incomplete
> generation.

| Failure point | Outcome |
|---|---|
| directory creation | nothing changed |
| Parquet or manifest write/fsync | orphan generation; `CURRENT` unchanged |
| generation/namespace fsync | orphan generation; `CURRENT` unchanged |
| `CURRENT.tmp` write/fsync | orphan generation; `CURRENT` unchanged |
| `os.replace` | atomic — old or new, never partial |
| after replace, before parent fsync | either value may survive; **both are complete generations** |

**Platform assumptions:** `os.replace` is atomic on POSIX and on Windows for
same-volume replacement; directory `fsync` is meaningful on Linux/macOS and a
no-op on some filesystems. Development and CI target macOS/Linux. Orphan
generations are inert, never selected, and safe to garbage-collect.

---

## 14. Canonicalisation contract and anomaly severity

**Confirmed defect.** `normalise()` sorts before the validator runs, so
`TS_NOT_SORTED` can never fire on the production path.

**Invariant:** canonicalisation may produce a deterministic representation, but
it may never erase evidence that the source delivered anomalous structure.

```
canonicalise(raw) -> CanonicalisationResult(frame, transformations, source_anomalies)
```

Recorded: source row count, source column inventory, source ordering (inversion
count), duplicate timestamps at source, extra columns seen, timezone conversion,
dtype conversions, **identity-field Unicode normalisation that changed bytes**
(§9), rows reordered, rows removed, values changed, columns preserved or
discarded.

| Anomaly | Class |
|---|---|
| Timezone conversion preserving the instant | NORMAL TRANSFORMATION |
| Column reordering to canonical order | NORMAL TRANSFORMATION |
| Lossless dtype conversion | NORMAL TRANSFORMATION |
| Stable sort of an unsorted source | **INFO, recorded** — valid canonicalisation, but the source got it wrong and that is provenance |
| Exact duplicate rows at source | **WARNING** |
| Known, mapped extra column | INFO — preserved per §15 |
| NFC change to an identity field | INFO, recorded |
| **Conflicting duplicate timestamps** | **TRUST BLOCKER** |
| **Unsupported / unmapped source field** | **TRUST BLOCKER** |
| **Row removal during canonicalisation** | **TRUST BLOCKER** |
| **Any value changed** | **TRUST BLOCKER** |

TRUST BLOCKER means `ValidatedDataset.build()` fails; nothing reaches storage.

---

## 15. Schema evolution and the positional payload

`from_fyers_candles()` accepts exactly six *positional* fields. A seventh
positional value has no name and therefore no meaning; `x_unknown_7` would
fabricate a research field out of an unknown number.

| Source shape | Policy |
|---|---|
| Positional payload (FYERS `/history`) | Exactly the six known fields. **Any other width → reject ingestion** until the source contract is verified |
| Named fields via an explicit source-schema mapping | Declared fields (e.g. `oi`, `vwap`) are **preserved**, enter identity, and are recorded in the column inventory |
| Named field absent from the mapping | **Canonicalisation failure** on the trusted path |
| Field with an unsupported dtype | **Canonicalisation failure** on the trusted path |

Recorded discard on the trusted path is withdrawn: it is audit-friendly but still
destructive, and discarded source information can change research meaning. DATA
INTEGRITY > CONVENIENCE.

The forensic path may preserve or drop raw values with a record; it makes no
trust claim.

---

## 16. `ValidatedDataset` strategy

**pandas DataFrames cannot be made immutable** — verified: frozen dataclasses
block only attribute reassignment; deep copy does not stop mutation through the
accessor; setting `arr.flags.writeable = False` on all six columns succeeded yet
pandas 3.0 copy-on-write **still permitted the write**.

Strategy: **tamper-evidence plus non-exposure.**

1. `ValidatedDataset.build(...)` canonicalises, validates and computes digests
   **internally**. No parameter accepts a report, so a mismatched report cannot
   exist — defect 1 removed by construction.
2. A **private deep copy** is held; `.frame` returns a **fresh copy per call**.
3. Evidence is stored as **immutable snapshots**. Baseline `ValidationReport`,
   `FetchReport` and `DatasetManifest` are mutable `@dataclass` — that must change.
4. `__slots__`, frozen semantics, `__reduce__` raising so it cannot be pickled and
   revived carrying stale evidence.
5. **Digest recomputed immediately before persistence**; divergence aborts.
6. **Digest recomputed again at trusted read.**

Mutation between build and write is *detected*, not *prevented* — the strongest
guarantee available in this language.

---

## 17. What identity does and does not prove

FYERS history rows carry no symbol; each row is `[epoch, o, h, l, c, v]`. A
successful response does **not** prove the broker returned the requested
instrument.

Identity means *candles returned by this provider in response to a request for
symbol X*, **not** *proven observations of instrument X*. The requested symbol,
provider, request parameters and response evidence enter acquisition provenance
so the claim is auditable. Cross-instrument verification needs an independent
source and is out of scope for Phase 1.

---

## 18. History-depth probing

**Two rejected algorithms.** Baseline binary search on "this narrow window
contains data" assumes monotonicity; a window on a weekend, holiday cluster,
outage or sparse period is empty while earlier and later data exist. The previous
revision's `any_data_in(candidate, known_good_newest)` is **mathematically
broken**: once the newest bound contains data, every interval containing it is
non-empty, so the predicate is constant-`TRUE` and never brackets.

**What is estimated** — not interchangeable: (A) earliest individual candle
observed; (B) earliest calendar date with any candle; (C) earliest
continuous-history boundary; (D) broker retention boundary. We establish (A) and
(B) by observation. (C) needs a calendar. **(D) cannot be proven** — absence of
response is not proof of absence of history.

**Algorithm: bounded backward window scan, then subdivide.**

1. Start from a recent known-good anchor.
2. Walk backwards in **non-overlapping coarse windows** (configured).
3. Classify each: `DATA`, `EMPTY_SUCCESS`, `ERROR`, `UNKNOWN`.
4. **`ERROR` is never evidence of absence** — recorded unresolved, scan continues.
5. Stop at a configured **search horizon**.
6. Subdivide **only the oldest `DATA` window**, recursively, to the configured
   resolution.
7. Report `earliest_observed_candle`, `earliest_observed_date`,
   `oldest_contiguous_empty_success_interval`, `unresolved_intervals`,
   `search_horizon` — a bracket with evidence, never a retention date.

**Correctness argument.** It never concludes "no data before X" from a single
empty window, makes no monotonicity assumption, bounds the estimate by the
configured resolution, and surfaces unresolved regions rather than treating them
as empty. **No live probing until implemented and tested.**

---

## 19. Continuity and session certification

`ABSURD_GAP_DAYS = 30` is removed. Moving the number into configuration did not
solve the knowledge problem, it only relocated the arbitrariness. No
`TradingCalendar` exists — verified.

- **`ObservedGap`** — elapsed time between consecutive observations. Pure fact,
  no severity.
- **`CalendarExplanation`** — whether a configured calendar explains the gap.
  Without a calendar: `UNKNOWN` for every gap, uniformly.
- **`ResearchDataPolicy`** — a *consumer's* requirement, not a property of the
  artifact.

`TradingCalendar` is a protocol only — `is_session_day(date)`,
`expected_next_bar(ts, resolution)`, plus `calendar_id`/`calendar_version` which
enter provenance when a real calendar arrives. **No NSE data is invented.** Only
`NullCalendar` ships.

| Situation | `ContinuityCertification` |
|---|---|
| No calendar configured | `NOT_CERTIFIED` — uniformly, every gap size |
| Calendar configured, all gaps explained | `CERTIFIED` |
| Calendar configured, unexplained gap | `FAILED` |

`NOT_CERTIFIED` is not a failure; it is an honest statement that the question
cannot be answered. A `TrustedDataset` may carry it. Whether that is acceptable
is decided by `ResearchDataPolicy`, not by a calendar-day threshold.

---

## 20. Secret lifecycle

Baseline reads only the current environment value, so a rotated-out token becomes
unredactable — verified.

**Append-only, process-lifetime `SecretRegistry`.** Registered explicitly when
loaded (settings load, token exchange) and **never removed**, so `TOKEN_A` stays
redacted after rotation to `TOKEN_B`. Rotation retains **both** entries.

**Explicitly registered secrets always redact, regardless of length.** Correcting
the previous revision: the `>= 8` rule is a heuristic for *automatically
discovered candidate* values only, and no such automatic scanning is planned. If
`register("x")` is called, `"x"` is protected.

Registry rules: deduplicate identical values; reject empty and `None`; **not
thread-safe by contract** — registration happens during single-threaded startup
and token exchange, and this constraint is documented rather than papered over
with a lock that would imply broader safety. **No eviction** — the previous
revision proposed bounded LRU, which reintroduces the exact defect being fixed.
Memory is bounded by the number of credentials a process loads (single digits);
if that changes it will be **measured** first.

Registered: client secret, access token, auth code, any future refresh token, and
any other credential material loaded by settings or authentication.

Redacted across `record.msg`, `record.args`, exception messages, formatted
tracebacks, chained and nested tracebacks, URLs, and multiline values. Redaction
lives in the **formatter**, because filters run before `exc_info` is rendered.

**FYERS SDK-owned logging is OUTSIDE our formatter boundary.** The SDK builds its
own `FileHandler` and logs full URLs at debug level. Mitigation: keep it at
`ERROR`, point `log_path` into gitignored `logs/`, document the boundary. We do
**not** claim SDK logs are protected.

---

## 21. Broker payloads and diagnostics

**Confirmed defect.** `historical.py` builds `detail` from the raw broker
`message` and embeds it in exception text; four scripts then `print(f"…{exc}")`.
Redaction covers logging, not `print`.

**Raw broker response payloads are never written to application logs — at any
level.** The previous revision permitted raw payloads at debug behind the
redacting logger; that is withdrawn. `SecretRegistry` protects only *known*
secrets, and a payload may contain unanticipated credential fields, signed URLs,
account data or server debug information that the registry cannot know about.

```
BrokerDiagnostic(status, code, sanitized_message, sanitized_structured_fields)
```

`sanitized_message` is length-capped, control characters stripped, and passed
through the registry. Exceptions carry the diagnostic, not raw payload text. CLI
prints the diagnostic. Raw responses exist transiently in memory for parsing and
are never logged. Any future forensic raw capture must be a separate, explicitly
security-reviewed feature — not debug logging.

---

## 22. Reproducibility certification

`git_revision()` can return `unknown`, `not-a-git-repo` or `<sha>-dirty`, and
baseline still treats the dataset as authoritative. Data validity and
reproducibility are different guarantees.

Correcting the previous revision, which recorded a digest without saying what it
is compared against:

- **`environment_actual_digest`** — canonical description of the running
  environment: Python major.minor.patch, OS, architecture, direct dependency
  name/version pairs, full transitive dependency name/version pairs, lock file
  digest, git commit SHA, git dirty flag.
- **`environment_expected_digest`** — derived from the declared environment
  policy (the lock file and pinned interpreter).

`ReproducibilityCertification = CERTIFIED` **only when the actual environment
satisfies the declared expected policy** and the git SHA is known and clean.
Recording a digest is not certification by itself.

**A different platform may produce logically identical data.** Reproducible
*environment* and identical *dataset* are separate claims: the environment
comparison certifies the former, the data digest independently establishes the
latter.

---

## 23. Configuration validation

**Confirmed defect.** `lot_size=int(lot_size) if lot_size else 1` silently turns
`0` and `""` into `1`.

The blanket "missing optional → `None`, never a default" was too broad. **What is
forbidden is truthiness-based repair.** Per-field schema declares one of:
`required`; `optional nullable`; `optional with an explicit documented default`.

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

## 24. Two-layer no-authority-laundering test

Correcting the previous revision, whose single test contradicted the glossary by
requiring interior gaps to block `TrustedDataset`.

### Layer A — trusted-artifact laundering

**Must NOT produce a `TrustedDataset`:** invalid frame · validation from a
different frame · frame mutated after validation · conflicting duplicate
timestamps · row removed during canonicalisation · value changed during
canonicalisation · unsupported schema field · extra positional broker field ·
manifest `requested_range` shortened to match sparse data · manifest chunk
failures removed · forged validation status · forged acquisition status · forged
`forced` flag · forged data digest · provenance envelope modified without
updating the pointer digest · **unknown `provenance_schema_version`** ·
**data/manifest copied into another generation directory** · manifest copied from
another dataset · Parquet modified so the decoded observations differ · missing
manifest · `REQUESTS_PARTIAL` · `REQUESTS_EMPTY` · `REQUESTS_FAILED` · **valid
dataset written through the force path** · **forced manifest edited to
`force=false`** · interrupted generation write · **`source`/`symbol` containing
path-traversal text** · ambiguous string containing encoding-delimiter characters
· wrong expected source / symbol / resolution locator.

**Pointer cases — must be rejected:**

| | Case |
|---|---|
| A | pointer `generation_id` does not match the target generation's envelope |
| B | pointer `integrity_id` does not match the target generation |
| C | pointer names a generation in the wrong namespace |
| D | pointer malformed (unknown field, bad JSON, wrong `pointer_version`) |
| E | pointer contains an invalid UUID or path-shaped material |
| F | pointer names a generation whose data/envelope does not satisfy its own `integrity_id` |

**Pointer case that must be ACCEPTED — known limitation:**

| | Case |
|---|---|
| G | `CURRENT` replaced wholesale by an older **valid** pointer carrying the correct `generation_id` and `integrity_id` for an older complete trusted generation |

Case G is generation rollback (§5) and is deliberately out of scope in Phase 1.
The test asserts G **succeeds**, so the limitation is encoded in the suite rather
than left as prose that could drift. **No other section of this document may
imply case G is detectable.**

**Must be ACCEPTED (see §24.1):** an otherwise sound dataset whose source arrived
out of order and was losslessly stable-sorted.

**A genuinely sound artifact must still succeed** — the test must prove the gate
discriminates rather than simply refusing everything.

### 24.1 Source-order anomaly — resolved

The previous revision listed "source-order anomaly" as a Layer A failure while
§14 classified a stable sort as `INFO, recorded`. That was a contradiction.

**Resolution — the §14 classification governs.** An otherwise valid dataset **may**
become a `TrustedDataset` after a lossless stable sort of an out-of-order source,
provided all of the following hold:

- the sort is deterministic;
- no observation is removed;
- no value changes;
- the source-ordering anomaly and `rows_reordered` are recorded in the
  canonicalisation snapshot;
- that snapshot is inside the provenance envelope and therefore integrity-bound.

If any of those fails, the relevant §14 TRUST BLOCKER applies instead (row
removal and value change are both blockers).

This is **not** a silent repair: the evidence that the source was disordered is
permanently bound to the generation and cannot be edited away without breaking
`provenance_digest`.

`ResearchDataPolicy` may declare `require_pristine_source_order=True`, in which
case Layer B rejects the dataset for that experiment while the artifact itself
remains sound.

### Layer B — research-readiness laundering

None of these may produce a `ResearchReadyDataset` under a policy requiring the
relevant certification: **sparse 1-year artifact (3 candles) that is a valid
`TrustedDataset`** · `ContinuityCertification = NOT_CERTIFIED` under a policy
requiring `CERTIFIED` · `ContinuityCertification = FAILED` · huge interior gap ·
`SessionCertification = NOT_CERTIFIED` under a policy requiring it ·
requested-window coverage below the policy minimum · `ReproducibilityCertification
= NOT_CERTIFIED` under a policy requiring it · **actual environment digest not
satisfying the expected environment policy** · locator source/symbol/resolution
not matching policy expectations · **source-order anomaly recorded in ingestion
provenance, under a policy declaring `require_pristine_source_order=True`**.

The same artifact must be accepted by a policy that does not require those
certifications — proving the boundary discriminates rather than blocks. In
particular the stable-sorted dataset from §24.1 is a valid `TrustedDataset`, is
rejected by a policy requiring pristine source order, and is accepted by one that
does not.

---

## 25. Test matrix

Numeric expectations like "29-day hole blocks" are removed; they encoded a magic
number as semantics. Tests are policy-based.

| Invariant | Unit | Integration | Adversarial | Mutation |
|---|---|---|---|---|
| Canonical encoding | type-tag matrix | write→read stable | delimiter-bearing strings, NFC-changing text, 1e-9, ±0.0, NaN, inf, NA/0 | revert to `%.10g` |
| Text policy | NFC vs EXACT per field | round-trip | source column with decomposable Unicode | normalise everything |
| Ingestion evidence | anomaly recorded | out-of-order response | source order erased | remove recording |
| Anomaly severity | each class | blocker halts build | conflicting duplicates | downgrade a blocker |
| Schema policy | mapped field preserved | round-trip with `oi` | 7-field positional payload | allow unknown field |
| Binding | build rejects a report arg | download→write | mismatch cases | remove binding |
| Tamper-evidence | mutate after build | build→mutate→write | mutate via accessor | remove recompute |
| Envelope integrity | digest computation | manifest edit detected | edit each envelope field | skip envelope check |
| Generation location | four-way agreement | copy dir between namespaces | forced→trusted copy | drop location check |
| Path safety | slug encoding | store/read round-trip | `..`, absolute, unicode, empty, colliding symbols | interpolate raw |
| Pointer format | strict parse | write→read | unknown field, malformed uuid, absolute path | accept unknown fields |
| Locator identity | three-way agreement | wrong symbol locator | manifest says SBIN | drop locator check |
| Requests status | 5 states | all/none/some fail | empty vs failed | infer coverage from status |
| Coverage records | data-only vs provenance-derived | 1 day for 1-year | boundary on a weekend | grade absence as deficient |
| Probe | window classification | bracket output | ERROR and holiday windows | restore binary search |
| Continuity | NOT_CERTIFIED without calendar | fake deterministic calendar finds a known missing session | policy threshold N: gap > N gives the expected policy result | remove certification |
| Trusted read | each of 11 checks | full pipeline | forge each field | delete a check |
| Research readiness | policy satisfied / not | sparse artifact rejected by policy | policy requiring each certification | accept TrustedDataset directly |
| Generation atomicity | each failure point | crash between stages | old-complete and new-complete recovery | non-atomic rename |
| Force namespace | separate directory | force over trusted | edit `forced=false` | let CURRENT point at forced |
| Secrets | registry lifetime; **1-char registered secret redacts** | rotation A→B | A in traceback after rotation | add eviction; restore length rule |
| Broker payload | diagnostic construction | error → CLI | payload with an unknown credential-shaped field | log raw payload |
| Config | per-field schema | load registry | `0`, `""`, unknown key, duplicate symbol | restore truthiness |
| Reproducibility | actual vs expected | dirty tree | platform difference; lock mismatch | certify on digest presence alone |

Every invariant must have at least one test that **fails when the production
check is removed**. Baseline mutation testing caught 9 of 9 attempted reversions;
that standard carries forward.

---

## 26. Failure-state matrix

| State | `read_trusted()` | `read_unverified()` |
|---|---|---|
| No dataset / orphan only | `DatasetNotFound` | orphan reachable by explicit id |
| Pointer malformed / unknown field / bad uuid | `PointerFormatError` | n/a |
| Manifest missing / unparseable / unknown fields | `ProvenanceSchemaError` | returns data |
| Unknown `provenance_schema_version` | `ProvenanceSchemaError` | returns data |
| Parquet modified so decoded observations differ | `IntegrityError` | returns data |
| Parquet re-encoded preserving the identical logical dataset | **succeeds — not detected, accepted (§2.1)** | returns data |
| Source arrived out of order, losslessly stable-sorted | **succeeds**; anomaly recorded; policy may reject (§24.1) | returns data |
| **Manifest tampered, self-consistent** | **`ProvenanceTampered`** (envelope digest, **not** revalidation) | returns data |
| Both tampered consistently | `GenerationIntegrityError` | returns data |
| Generation dir copied elsewhere | `GenerationLocationMismatch` | returns data |
| Locator disagrees with envelope | `IdentityMismatch` | n/a |
| Validation errors | `ValidationFailed` | returns data |
| Requests partial/empty/failed/unknown | `AcquisitionEvidenceInvalid` | returns data |
| Forced generation | **unreachable** (namespace) | reachable by explicit id |
| **Pointer rolled back to an older valid generation** | **succeeds — NOT DETECTED (§5)** | n/a |
| Valid, continuity not certified | **succeeds**; `ResearchDataPolicy` may still reject | n/a |
| Valid, irreproducible | **succeeds**; policy may still reject | n/a |
| Sound artifact | **succeeds** | n/a |

---

## 27. Backward compatibility

`data_store/` is **empty** at baseline — verified. The generation layout, slug
encoding, pointer format and envelope schema are breaking on-disk changes that
are **free now and expensive later**. No migration path is required and none will
be written.

---

## 28. Implementation sequence

CI is first: automation should protect the review branch **during** this rewrite.

| # | Unit | Addresses |
|---|---|---|
| 1 | **Minimal CI on the review branch** using the documented install (including the `asyncio` uninstall step) | — |
| 2 | Schema policy, text policy, canonicalisation contract with evidence and severity | 5, 6 |
| 3 | Canonical encoding + dataset identity digest | 7, 8 |
| 4 | Immutable evidence snapshots | prerequisite |
| 5 | `ValidatedDataset` + write accepting only it | 1 |
| 6 | Provenance envelope, versioning, generation integrity id | 2 |
| 7 | Path slug safety + pointer format | 17 |
| 8 | Generation storage, namespaces, atomic `CURRENT` | 10, 11 |
| 9 | Request status + coverage records | 3 |
| 10 | `TrustedDataset` read boundary | 9 |
| 11 | `ResearchDataPolicy` + `ResearchReadyDataset` | 3 |
| 12 | Probe redesign | 13 |
| 13 | Continuity/session certification + calendar protocol | 4 |
| 14 | `SecretRegistry` + broker diagnostics | 12, 14 |
| 15 | Strict config validation | 16 |
| 16 | `write_token_to_env` validation | 15 |
| 17 | Dependency reproducibility; CI upgraded alongside | — |
| 18 | Documentation reconciliation | — |

**Git protocol during implementation:** every logical change is committed and
pushed to `phase1-trust-hardening` only. Never to `main`. No merge without
manager approval. **STOP** after each commit for independent GitHub inspection.
**CI passing is not a substitute for manager review.**

---

## 29. Open risks

1. **Generation rollback is undetectable** (§5). Artifact integrity does not
   prove freshness.
2. **Tamper-evidence is weaker than immutability** — mutation between build and
   write is detected, not prevented. Unavoidable in pandas.
3. **The `/history` contract is ASSUMED**, never LIVE-VERIFIED. §15's rejection
   rule is designed against unverified behaviour.
4. **Instrument correctness is unprovable** from candles alone (§17).
5. **Continuity and session coverage are uncertifiable** without a calendar, and
   none exists.
6. **SDK logs remain outside the redaction boundary**, permanently.
7. **`SecretRegistry` is not thread-safe by contract**; registration is confined
   to single-threaded startup paths.
8. **Probe cost rises** materially under §18 and interacts with rate limits;
   window size and horizon must be tuned against the documented daily cap.
9. **Digest cost is O(rows)** in Python; fine at ~90k rows/year, unmeasured beyond.
10. **The out-of-scope actor** can recompute all digests and the pointer.

---

## 30. Architecture freeze for implementation

**This document is the Phase-1 implementation contract.**

- Implementation may not silently deviate from it. Code that disagrees with this
  document is a defect in one of the two, and which one must be decided
  deliberately.
- If implementation discovers a contradiction, an impossibility, or a better
  design, **STOP and amend this document first**, then implement. Do not resolve
  the disagreement in code and reconcile the design afterwards.
- Priority order for every ambiguity: **DATA INTEGRITY > REPRODUCIBILITY >
  AUDITABILITY > SAFETY > CONVENIENCE.**
- Every implementation unit requires: tests written first, implementation, the
  full suite, targeted adversarial tests, one logical commit, a push to the
  review branch, and **manager review before the next unit**.
- `phase1-trust-hardening` remains the only development branch. Nothing may
  remain only on a local machine.
- `main` remains the manager-approved baseline. No direct pushes, no
  self-merges.
- CI passing is **not** a substitute for manager review.

**Phase 1 is not complete.** This freeze covers the architecture only. Phase 1
remains REJECTED until the implementation exists, is tested, is reviewed, and
real market data has been acquired and validated through this pipeline.

---

## 31. Non-goals

Not in Phase 1: strategy logic, indicators, signal generation, backtesting,
optimisation, walk-forward, Monte Carlo, TradingView or Pine integration,
webhooks, dashboards, paper trading, order execution, live-trading mode,
execution-instrument selection, NSE calendar data, and any performance or
profitability claim.

Also not goals: defending against an attacker with machine access; asymmetric
signing; **detecting rollback to an earlier valid generation**; making pandas
immutable; certifying continuity, session coverage or bar density without a
calendar; proving instrument identity from candle data.
