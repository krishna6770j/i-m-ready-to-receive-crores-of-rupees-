# Phase 1 — Trusted Data Architecture

Design specification. **No production code, tests, dependencies or README changes
accompany this document.** Implementation is not authorised.

---

## 1. Baseline

| | |
|---|---|
| Baseline commit | `e74d333832216d4c8e8ad77a099993e5323e657d` |
| Branch | `phase1-trust-hardening`, branched from the baseline |
| Manager verdict at baseline | Phase 1 REJECTED; Phase 2 NOT AUTHORISED; live FYERS NOT AUTHORISED |
| Tests at baseline | 176 passing |

Every claim below was checked against the code at this commit. Where a previous
design assumption proved false, that is stated explicitly rather than quietly
corrected.

---

## 2. Problem statement

The central invariant is not established:

> An invalid, incomplete, tampered, or incorrectly-described dataset must never
> be accepted as authoritative.

Twelve independently confirmed defects share one root cause: **the system trusts
assertions instead of deriving facts.**

| # | Defect at baseline | Evidence |
|---|---|---|
| 1 | `store.write()` accepts a frame and a report as separate arguments and never checks the report describes that frame | Invalid frame + valid frame's report persists with `validation_status="valid"` |
| 2 | `is_authoritative` forgeable by editing `validation_status`, `fetch_status`, `forced` | Edited manifest yields `is_authoritative=True` on genuinely invalid data |
| 3 | Acquisition completeness derived from `failed_chunks == []` | 3 candles stored as authoritative answer to a 1-year request |
| 4 | `ABSURD_GAP_DAYS = 30` is a magic number; a 29-day hole is WARNING-only | 3/7/14/20/29-day gaps all `usable=True` |
| 5 | **Source anomalies erased before validation** | `normalise()` sorts before the validator can observe source ordering |
| 6 | Extra source columns silently dropped | `oi`, `vwap` vanish with no record; contradicts the README |
| 7 | `content_hash` uses `%.10g` despite a docstring claiming `repr` | 1e-9 price change hashes identically |
| 8 | Digest excludes source, symbol, resolution | Different instruments with equal candles collide |
| 9 | `read()` verifies nothing | Tampered Parquet returned silently |
| 10 | Two-file write is not transactional | Manifest failure leaves an orphan Parquet |
| 11 | `--force` overwrites the authoritative dataset in place | Good data silently replaced |
| 12 | Redaction reads only the *current* env value | A rotated-out token leaks |

Plus four found during this design review:

| # | Defect | Evidence |
|---|---|---|
| 13 | `probe_earliest_available()` binary-searches a **non-monotonic** predicate | A holiday/outage window is empty while earlier and later data exist |
| 14 | Broker-controlled error text is embedded in exceptions and printed to stdout | `historical.py` builds `detail` from the raw broker `message`; four scripts `print(f"...{exc}")` |
| 15 | `write_token_to_env()` performs no validation | A token containing a newline injects arbitrary `.env` lines |
| 16 | Config coerces falsey values | `lot_size=int(lot_size) if lot_size else 1` turns `0` and `""` into `1` |

---

## 3. Threat model

**In scope.** Accidental manifest edits ("I'll just set this to valid"); stale or
partially-written files; a process dying mid-write; inconsistent output from
future automation; manifests copied between datasets; operator mistakes;
malformed or hostile *broker responses*; our own refactoring errors.

**Out of scope.** An adversary who controls the machine and the source tree.
Anyone able to rewrite a manifest can equally rewrite `validator.py`, so a
signing key stored on the same machine would add ceremony without security.

**Consequence: no cryptographic signing.** Integrity is achieved by making the
reader *derive* every gating fact rather than read it. This is strictly more
robust against the in-scope threats than a signature would be, because it also
catches inconsistency that was never malicious.

---

## 4. Trust transitions and type boundaries

Five types, each corresponding to a genuine trust transition. No type exists
merely for naming.

| Type | Constructed by | Guarantees | Explicitly does NOT guarantee | Serialisable | Phase 2 may consume |
|---|---|---|---|---|---|
| `RawObservations` | adapter parse | Shape matches the source contract | Ordering, uniqueness, validity | No | No |
| `CanonicalDataset` | `canonicalise()` | Deterministic canonical form **plus** a record of every transformation applied | Validity | No | No |
| `ValidatedDataset` | `ValidatedDataset.build()` | Canonical form, identity digest, and validation evidence produced *from that exact data* | That the data is still unmutated *later* | No (`__reduce__` raises) | No |
| `UnverifiedDataset` | `store.read_unverified()` | Bytes were loaded from disk | Everything else | n/a | **No** |
| `TrustedDataset` | `store.read_trusted()` | Every §14 invariant re-derived at read time | Session coverage or bar density without a calendar | n/a | **Yes — the only permitted input** |

---

## 5. Raw ingestion integrity (manager finding §2)

**Confirmed defect.** `from_fyers_candles()` → `normalise()` sorts by timestamp
before the validator runs. An out-of-order FYERS response is therefore silently
repaired, and `TS_NOT_SORTED` can only ever fire on a hand-constructed frame —
never on the production path. The same applies to any transformation
canonicalisation performs.

**Invariant.**

> Canonicalisation may produce a deterministic representation, but it may never
> erase evidence that the source delivered anomalous structure.

**Chosen design: option B**, canonicalisation returns evidence alongside data.

```
canonicalise(raw) -> CanonicalisationResult(
    frame,                  # canonical
    transformations,        # what was done, deterministically
    source_anomalies,       # what the source got wrong
)
```

Option A (validate before normalising) was rejected because it requires a second
validator operating on non-canonical input, doubling the rules that must stay in
sync. Option B keeps one validator and makes the transformation itself
auditable.

`source_anomalies` records, at minimum: source row count, source column
inventory, whether rows arrived out of order (and how many inversions),
duplicate timestamps present at source, extra columns seen, timezone conversion
applied, dtype conversions applied, rows reordered, rows removed, values
changed, columns preserved or discarded.

Sorting remains valid canonicalisation. **Whether the source required sorting is
provenance**, and it will be recorded and carried into the manifest.

---

## 6. Canonicalisation contract

Permitted: column selection and ordering per the schema policy; dtype conversion
that preserves values exactly; timezone conversion preserving the instant;
deterministic stable sort by timestamp.

Forbidden: substituting missing values; fabricating volume; interpolating;
coercing unparseable source values into `NaN`; dropping rows; discarding columns
without a record.

Already fixed at baseline and to be preserved: volume is nullable `Int64` so
missing volume stays missing; unparseable values raise `SchemaError` naming the
offending value.

---

## 7. Schema evolution policy (manager finding §3)

**Confirmed defect.** `frame.loc[:, OHLCV_COLUMNS]` silently discards extra
columns, contradicting the README.

**Decision — (A) preserve, with (C) as the recorded fallback.** Applying
DATA INTEGRITY > CONVENIENCE, silent discard is removed entirely.

- The canonical six (`ts, open, high, low, close, volume`) remain the validated
  core.
- Recognised additional source fields are **preserved** in a namespaced
  `x_<name>` region (`x_oi`, `x_vwap`), recorded in the manifest column
  inventory, and **included in the identity digest**.
- A field that cannot be preserved (unsupported dtype) is **discarded only with
  an explicit provenance record naming it and the reason**.
- `schema_version` is recorded in the manifest and is part of dataset identity.
  A change in schema version changes the digest.

This ordering matters: **the digest contract cannot be finalised before this
policy**, because the policy determines what the digest covers. At baseline, an
extra `oi` column hashes identically to its absence — verified.

---

## 8. Dataset identity

The earlier unqualified "iff" is retracted. Identity is defined as an
equivalence relation over:

> **(schema_version, source, symbol, resolution, canonical observation sequence
> including preserved extra columns)**

| Attribute | In identity | Reasoning |
|---|---|---|
| schema version | yes | Changes what the data means |
| source / vendor | yes | Vendor differences are a research variable |
| symbol | yes | Equal candles for different instruments are different datasets |
| resolution | yes | Equal closes at 1-min and 5-min are different datasets |
| canonical observation columns | yes | The data itself |
| preserved extra columns | yes | Per §7; otherwise preservation is meaningless |
| timestamp instant | yes | Identity is the instant, not its representation |
| row multiplicity | yes | A duplicated row is a different sequence |
| duplicate/conflicting timestamps | yes | Part of the sequence; also a validation error |
| adjustment state | **yes, when introduced** | No adjustment logic exists today; it must enter identity the day it does |
| timezone representation | no | Instant-preserving; IST canonical form is identity |
| row ordering | no | Total deterministic sort; both orderings denote the same observations |
| column ordering | no | Representation |
| dtype | no | Values, not representation |
| acquisition range | no | A property of the *request*, recorded in provenance |
| manifest metadata | no | Evidence, not identity |

Consequences: identical candles under different symbols hash **differently**;
the same candles requested over different date ranges hash **the same**; subset
relationships are **not** derivable from digests, which is deliberate —
coverage answers that question, not identity.

---

## 9. Exact digest contract

The baseline digest is **not approved** as a binding mechanism: `%.10g`, and it
omits source/symbol/resolution.

Canonical serialisation, hashed with SHA-256:

```
IDENTITY | schema_version | source | symbol | resolution
COLUMNS  | <ordered canonical column inventory>
<row>    | ts.isoformat() | f(open) | f(high) | f(low) | f(close) | vol | x_*…
```

- **Floats** — `float.hex()`. Verified: `np.float64` is a genuine subclass of
  `float`, `.hex()` resolves directly and round-trips exactly.
- **`-0.0` collapses to `+0.0`.** `float.hex()` distinguishes them, but IEEE
  signed zero is not a market distinction and a zero price is invalid anyway.
  Recorded as a deliberate choice, not an accident.
- **`NaN` / `±inf`** — distinct stable tokens (`NaN`, `+Inf`, `-Inf`), never
  collapsed together.
- **Volume** — `NA` sentinel distinct from `0`.
- **Timestamps** — ISO-8601 with UTC offset.
- Serialised field-by-field, not via `to_csv`, removing dependence on pandas CSV
  formatting.

Verified distinctions with a prototype (in-memory only):

| Must differ | Must match |
|---|---|
| 24000.12 vs 24000.13 | identical copy |
| change at 1e-9 (baseline says *same*) | rows reversed |
| volume `0` vs `NA` | UTC vs IST representation of the same instants |
| `NaN`, `+inf` | columns reordered |
| duplicate row added; row removed | change below double epsilon (not representable) |
| timestamps shifted | |
| different symbol / resolution / source | |

Empty and one-row frames hash deterministically and differ from each other.

---

## 10. `ValidatedDataset` strategy

**Finding accepted: pandas DataFrames cannot be made immutable.** Verified:
`@dataclass(frozen=True)` blocks only attribute reassignment; deep copy does not
prevent mutation through the accessor; setting `arr.flags.writeable = False` on
all six columns succeeded yet pandas 3.0 copy-on-write **still permitted the
write**.

Therefore the strategy is **tamper-evidence plus non-exposure**, not immutability:

1. `ValidatedDataset.build(raw_or_frame, *, source, symbol, resolution,
   acquisition, policies)` canonicalises, validates and computes the digest
   **internally**. There is no parameter through which a report may be supplied,
   so a mismatched report cannot exist. This removes defect 1 by construction.
2. The instance holds a **private deep copy**; `.frame` is a property returning
   a **fresh copy per call**. The internal frame is never handed out.
3. Bound evidence is stored as **immutable snapshots** (frozen dataclasses,
   tuples). Note: at baseline `ValidationReport`, `FetchReport` and
   `DatasetManifest` are all plain mutable `@dataclass` — that must change.
4. `__slots__`, frozen semantics, and `__reduce__` raising so the object cannot
   be pickled and revived carrying stale authority.
5. **Digest recomputed immediately before persistence** and compared to the
   bound digest; divergence aborts the write.
6. **Digest recomputed again at trusted read.**

Mutation between `build()` and `write()` is *detected*, not *prevented*. That is
the strongest guarantee available in this language, and the design says so
rather than implying otherwise.

---

## 11. Acquisition evidence model

`failed_chunks == []` is not completeness. A successful call may return zero
rows, truncated rows, only part of the period, the wrong range, or a sparse
range.

Six independent concepts, never collapsed into one boolean:

| Concept | Answered by | Provable today? |
|---|---|---|
| 1. request success/failure | chunk results | yes |
| 2. observed edge coverage | requested vs observed first/last | yes |
| 3. internal continuity | gap model (§13) | partially — see §13 |
| 4. expected session coverage | trading calendar | **no** |
| 5. bar density | trading calendar | **no** |
| 6. identity correctness | digest identity (§8) | yes |

```
ACQUISITION_FAILED         every chunk errored
ACQUISITION_EMPTY          all chunks succeeded, all returned zero rows
ACQUISITION_PARTIAL        any chunk failed, OR edges outside tolerance
ACQUISITION_EDGE_COMPLETE  no chunk failed, ≥1 row, both edges within tolerance
ACQUISITION_UNKNOWN        no acquisition evidence (fixtures, manual frames)
```

`EDGE_COMPLETE` is named to be honest. It means **only**: every requested chunk
returned without error, and observed first/last timestamps reach both requested
edges within `coverage_tolerance`. It does **not** mean interior data is
continuous, that every session is present, that density is correct, or that the
instrument is the one requested. A six-month interior hole is `EDGE_COMPLETE`
*and* fails continuity — which is why authority requires both.

`coverage_tolerance` has **no default**; it must be configured deliberately.

`EMPTY` and `FAILED` are separate states, resolving the baseline behaviour where
both produced the identical message `'fetch returned zero rows; nothing to store'`.

---

## 12. History-depth probe redesign (manager finding §7)

**Confirmed defect.** `probe_earliest_available()` binary-searches on
"this narrow window contains data", which is **not monotonic**. Counterexamples:
a probe window landing entirely on a weekend, a long holiday cluster, a broker
outage, or a sparsely-quoted period returns empty while both earlier and later
periods contain data. Binary search then discards a region that does contain
data. A previous audit could not *demonstrate* a failure and reported it as
unconfirmed; that was too weak a conclusion — **correctness was never
established, and absence of a demonstrated failure is not evidence of
correctness.**

**Replacement:** exponential backward scan on a *monotonic* predicate, then
local refinement.

- Monotonic predicate: `any_data_in(candidate, known_good_newest)` — "does any
  data exist anywhere between the candidate and a date already known to have
  data". This is monotonic in the candidate because widening the window can only
  add data, never remove it.
- Phase 1: from a known-good recent anchor, probe backwards at exponentially
  increasing offsets until the predicate turns false.
- Phase 2: binary-search *within* the last bracketing interval, still using the
  monotonic predicate.
- Result reported as a **bracket** `[earliest_confirmed, earliest_possible]`,
  never a single date implying more precision than was established.

This costs more requests than the baseline; correctness outranks convenience.
**Must not be run against the live broker until implemented and tested.**

---

## 13. Continuity, calendar and gap model (manager finding §12)

`ABSURD_GAP_DAYS = 30` is **removed**. No principled basis for the value exists,
and an unjustified constant must not gate research data. No `TradingCalendar`
implementation exists anywhere in the repository — verified.

Three separate concepts, per the directive:

- **`ObservedGap`** — elapsed time between consecutive observations. Always
  computable. Pure fact.
- **`CalendarExplanation`** — whether a configured calendar explains the gap.
  Requires a calendar; otherwise `UNKNOWN`.
- **`GapPolicy`** — configuration deciding whether the available evidence
  suffices for a given consumer. No magic numbers in code.

`TradingCalendar` is a protocol only:

```
is_session_day(date) -> bool
expected_next_bar(ts, resolution) -> Timestamp
calendar_id / calendar_version        # provenanced when a real one arrives
```

**No NSE holiday or session data is invented in this phase.** Only
`NullCalendar` ships, and it answers `UNKNOWN` to everything.

Without a calendar the validator does not classify — it declines to certify,
emitting `CONTINUITY_NOT_CERTIFIED` plus the measured gap span. Any gap above
the configured `GapPolicy` threshold additionally emits `UNEXPLAINED_GAP`, which
blocks `TrustedDataset` construction. Uncertainty is represented as uncertainty,
never converted into an arbitrary number.

**A dataset may be a `TrustedDataset` without a calendar**, but only when no gap
exceeds the configured threshold — and the threshold has no default.

---

## 14. `TrustedDataset` construction rules

Constructible **iff every check passes**. Each failure has a distinct exception
type so callers cannot conflate them.

| # | Check | Failure |
|---|---|---|
| 1 | Generation selected by `CURRENT` exists and is complete | `DatasetNotFound` |
| 2 | Manifest present | `MissingProvenance` |
| 3 | Manifest parses; `schema_version` known; no unknown or missing fields | `ProvenanceSchemaError` |
| 4 | Frame loads and is canonical | `SchemaError` |
| 5 | **Digest recomputed** from loaded data == manifest digest | `IntegrityError` |
| 6 | Manifest identity (source, symbol, resolution, schema) == digest inputs | `IdentityMismatch` |
| 7 | **Validation re-run** on the loaded frame has no ERROR issues | `ValidationFailed` |
| 8 | Acquisition classification is `EDGE_COMPLETE` | `IncompleteAcquisition` |
| 9 | Continuity satisfies the active `GapPolicy` | `ContinuityNotCertified` |
| 10 | Generation is not forced | `ForcedGeneration` |
| 11 | Recorded acquisition evidence is internally consistent | `ProvenanceInconsistent` |

Scenario answers: valid Parquet + missing manifest → refuse (2). Valid manifest
+ tampered Parquet → refuse (5). Unknown or missing manifest fields → refuse
(3), never ignored. Contradictory fields → refuse (6/11). Validation passes but
acquisition incomplete → refuse (8). A faithfully copied dataset → **passes**,
correctly: a faithful copy *is* the same dataset.

`read_unverified()` returns `UnverifiedDataset`, a distinct type with no
authority attribute at all, so it cannot be substituted for a `TrustedDataset`
even by accident.

---

## 15. Manifest trust model

| Class | Fields | Reader behaviour |
|---|---|---|
| Derived, re-verified | `content_digest`, `schema_version`, `source`, `symbol`, `resolution` | Recomputed; mismatch → reject |
| Ignored for gating | `validation_status`, error/warning counts, error codes | Informational; validation is re-run |
| Source evidence | acquisition snapshot, ingestion/canonicalisation snapshot | Checked for internal consistency; inconsistency → reject |
| Operator declaration | `forced`, `force_reason`, `notes` | One-way only (below) |
| Environment | software versions, git revision, `fetched_at_utc` | Never gating for validity; see §22 |
| **Removed** | `is_authoritative` | Not stored, therefore not editable |

**`forced`, precisely.** `false → true` **demotes** and is honoured. `true →
false` **cannot promote**, because authority additionally requires re-run
validation, re-derived acquisition status and continuity to pass; a generation
was forced precisely because one of those fails. If a forced generation would
independently pass every check, it is authoritative on its own merits and the
flag was redundant. Recorded provenance may only reduce trust.

---

## 16. Generation-based atomic persistence

```
data_store/<source>/<symbol>/<resolution>/
    generations/
        g_<digest12>_<utc>/
            data.parquet
            manifest.json
    CURRENT
```

A dataset generation is written as a unit. Sequence: create a fresh generation
directory → write Parquet → `fsync` file → write manifest → `fsync` file →
`fsync` generation directory → atomically `os.replace` the `CURRENT` pointer.

| Failure point | Outcome |
|---|---|
| generation dir creation | nothing changed |
| Parquet write / fsync | orphan generation; `CURRENT` untouched |
| manifest write / fsync | orphan generation; `CURRENT` untouched |
| directory fsync | orphan generation; `CURRENT` untouched |
| `CURRENT` replace | atomic; either old or new, never partial |
| process death at any boundary | `CURRENT` still names the previous complete generation |

**Invariant: a failed write can never destroy or replace the last trusted
generation.** Orphan generations are inert — never selected by normal read,
detectable by absence of a `CURRENT` reference, and safe to garbage-collect.
This also resolves the baseline orphan-Parquet defect, since orphans are
structural rather than accidental.

`CURRENT` semantics: names exactly one generation directory; is updated only
after that generation is complete and fsynced; is never advanced by a forced
write when an authoritative generation already exists.

---

## 17. Forced-generation semantics

- `force=True` requires a non-empty `force_reason`, recorded in provenance. An
  operator cannot force accidentally or anonymously.
- A forced write creates a **separate generation** and does **not** advance
  `CURRENT` when an authoritative generation exists.
- Forced generations are **never** returned by `read_trusted()` — they fail
  check 10 and, independently, 7, 8 or 9.
- Reachable only through the explicit forensic API with an explicit generation
  id.
- **Force cannot be edited into authority**, because authority is re-derived and
  never read from the flag. "Permanently marked" is not offered as a security
  guarantee, since the mark is editable JSON — the guarantee comes from
  derivation.

Force may write invalid, incomplete or unknown-acquisition data. It is an escape
hatch for forensics, not a second definition of correctness.

---

## 18. Secret lifecycle

Baseline reads only the **current** environment value, so a rotated-out token
becomes unredactable — verified.

**Design: an append-only, process-lifetime `SecretRegistry`.** Secrets are
registered explicitly when loaded (settings load, token exchange) and **never
removed**, so `TOKEN_A` stays redacted after rotation to `TOKEN_B`.

**No LRU eviction.** The directive is explicit and correct: silently evicting a
secret reintroduces the defect. Memory is bounded in practice by the number of
credentials a process loads (single digits); if that ever changes, it will be
measured before any eviction policy is considered.

Registered: client secret, access token, auth code, any future refresh token,
and any other credential material loaded by settings or authentication.

Redacted across: `record.msg`, `record.args`, exception messages, formatted
tracebacks, chained and nested tracebacks, URLs, and multiline values. Redaction
lives in the **formatter**, because filters run before `exc_info` is rendered.

**Boundary stated honestly: FYERS SDK-owned logging is OUTSIDE our formatter.**
The SDK constructs its own `FileHandler` and logs full URLs at debug level.
Mitigation is to keep it at `ERROR`, point `log_path` into gitignored `logs/`,
and document the boundary. We do **not** claim SDK logs are protected.

Short secrets (< 8 characters) remain unredacted by policy; real tokens are long
and a lower threshold would redact ordinary words.

---

## 19. External broker-error sanitisation (manager finding §15)

**Confirmed defect.** `historical.py` builds `detail` from the raw broker
`message` and embeds it in exception text; four scripts then `print(f"…{exc}")`
to stdout. Our redaction covers the logging path, **not** `print`. Broker-
controlled text therefore reaches stdout unsanitised.

Broker responses are **untrusted input**. They could echo request URLs, client
identifiers, auth codes or tokens.

Design: a `BrokerDiagnostic` value carrying a numeric status/code, a **sanitised**
message (length-capped, control characters stripped, passed through the secret
registry), and redacted structured fields. Exceptions carry the diagnostic, not
raw payload text. CLI output prints the diagnostic; raw payloads go only to the
redacting logger at debug level. Useful broker errors are not hidden — they are
not trusted verbatim.

---

## 20. Configuration validation (manager finding §20)

**Confirmed defect.** `lot_size=int(lot_size) if lot_size else 1` silently turns
`0` and `""` into `1`.

Policy: **missing and invalid are different, and neither is repaired.**

- Missing optional field → explicit `None`, never a substituted default.
- Present but invalid (`0`, negative, wrong type, malformed numeric) → raise
  naming the field, the value and the file.
- Unknown keys → rejected, not ignored (they usually mean a typo).
- Duplicate symbols → rejected.
- Config carries a `schema_version`.

---

## 21. Reproducibility certification (manager finding §18)

`git_revision()` can return `unknown`, `not-a-git-repo` or `<sha>-dirty`, and the
dataset could still be treated as authoritative. **Data validity and
reproducibility are different guarantees and must not share one boolean.**

Two independent attributes on `TrustedDataset`:

- **`is_valid_market_data`** — §14 checks 1–11. Concerns the data.
- **`is_reproducible`** — git revision known and clean, package versions
  recorded, environment matching the lock. Concerns the *provenance of the code*
  that produced it.

A dataset may be trustworthy market data while not being a reproducible research
artifact (produced from a dirty tree, say). Phase 2 must be able to require
either or both. Collapsing them would either block legitimate work or silently
certify irreproducible results.

---

## 22. Dependency and environment strategy

Current clean install requires installing dependencies and then **manually
uninstalling `asyncio`** — fragile, and a forgotten step leaves a package that
shadows the standard library. `requirements.lock.txt` is a flat pinned snapshot
without hashes or platform markers.

No dependency changes in this design step. Target invariant:

> fresh environment → one documented install workflow → correct Python 3.12
> environment → no manual repair step → full tests pass

Candidates to evaluate later: `pyproject.toml` with a proper lock (`uv.lock`), a
constraints/override mechanism to neutralise the `asyncio` declaration, or a
verified wrapper install script that fails loudly if the environment is wrong.

---

## 23. CI strategy

The manager verified zero attached CI statuses at `e74d333`; `main` is
unprotected with no required checks. **No CI is added in this design step.**

Eventual pipeline: Python 3.12 setup → reproducible dependency install →
`pytest -W error` → `compileall` → source scans for prohibited operations and
order-placement calls → repository secret scan.

**CI success is not a substitute for manager review.** Until branch protection
exists, the gate is enforced by workflow: development on
`phase1-trust-hardening`, never pushed directly to `main`.

---

## 24. The no-authority-laundering test

One end-to-end integration test whose sole purpose is:

> Nothing invalid, incomplete, forged, stale, forced or tampered can be converted
> into a `TrustedDataset`.

Each of the following must fail to produce a `TrustedDataset`:

invalid frame · validation from a different frame · frame mutated after
validation · source-order anomaly · forged `validation_status` · forged
acquisition status · forged `forced` flag · forged content digest · copied
manifest from another dataset · edited manifest · modified Parquet · missing
manifest · partial acquisition · all chunks empty · all chunks failed ·
edge-only sparse acquisition · huge interior gap · duplicate conflicting candles
· unsupported schema fields · forced generation · interrupted generation write ·
stale `CURRENT` pointer · wrong symbol · wrong resolution · wrong source.

A dataset genuinely satisfying every invariant must still succeed — the test
must prove the gate is not simply refusing everything.

---

## 25. Test matrix

| Invariant | Unit | Integration | Adversarial | Mutation |
|---|---|---|---|---|
| Ingestion evidence | anomaly recorded | out-of-order FYERS response | source order erased | remove recording |
| Schema policy | extra column preserved | round-trip with `x_oi` | unsupported dtype | silent drop |
| Identity/digest | §9 matrix | write→read stable | 1e-9, ±0.0, NaN, inf, NA/0 | revert to `%.10g` |
| Binding | build rejects a report arg | download→write path | all 10 mismatch cases | remove binding |
| Tamper-evidence | mutate after build | build→mutate→write | mutate via accessor | remove recompute |
| Acquisition | 5 states | all/none/some chunks fail | empty vs failed | `failed==0` shortcut |
| Probe | monotonic predicate | bracket result | holiday/outage windows | restore binary search |
| Continuity | 1/3/7/14/29/30/31/90/180 d | with and without calendar | 29-day hole blocks | remove threshold |
| Trusted read | each of 11 checks | full pipeline | forge each field | delete a check |
| Generation atomicity | each failure point | interrupted write | kill between stages | non-atomic rename |
| Force | separate generation | force over authoritative | flip flag both ways | allow CURRENT advance |
| Secrets | registry lifetime | rotation A→B | A in traceback after rotation | remove registry |
| Broker text | sanitiser | error → CLI | payload containing a token | echo raw |
| Config | strict parse | load registry | `0`, `""`, unknown key | restore truthiness |
| Reproducibility | dirty tree | — | unknown revision | merge the two booleans |

Every invariant must have at least one test that **fails when the production
check is removed**. Mutation testing at baseline caught 9 of 9 attempted
reversions; that standard carries forward.

---

## 26. Failure-state matrix

| State | `read_trusted()` | `read_unverified()` |
|---|---|---|
| No dataset | `DatasetNotFound` | `DatasetNotFound` |
| Orphan generation only | `DatasetNotFound` | reachable by explicit id |
| Manifest missing | `MissingProvenance` | returns data, no authority |
| Manifest unparseable / unknown fields | `ProvenanceSchemaError` | returns data |
| Parquet tampered | `IntegrityError` | returns data |
| Manifest tampered consistently | `ValidationFailed` (re-run catches it) | returns data |
| Identity mismatch | `IdentityMismatch` | returns data |
| Validation errors | `ValidationFailed` | returns data |
| Acquisition partial/empty/failed/unknown | `IncompleteAcquisition` | returns data |
| Continuity uncertified | `ContinuityNotCertified` | returns data |
| Forced generation | `ForcedGeneration` | reachable by explicit id |
| Valid but irreproducible | **succeeds**, `is_reproducible=False` | n/a |
| Fully valid | **succeeds** | n/a |

---

## 27. Backward compatibility and migration

`data_store/` is **empty** at baseline — verified. The generation-directory
layout is a breaking on-disk format change that is **free now and expensive
later**. No migration path is required, and none will be written. This is the
one decision with irreversible-ish consequences; it was raised with the manager
before this document and the generation model was approved in principle.

The manifest schema also changes incompatibly (`is_authoritative` removed,
snapshots added). Same reasoning applies.

---

## 28. Implementation sequence

Reordered from the manager's suggested A–K where repository dependencies
require it. Reasons for each deviation are given.

| # | Unit | Addresses | Note |
|---|---|---|---|
| 1 | Schema policy + canonicalisation contract with evidence | 5, 6 | **Must precede the digest** — it determines what the digest covers |
| 2 | Exact dataset identity + digest | 7, 8 | Depends on 1; also fixes the false docstring |
| 3 | Immutable evidence snapshots | prerequisite | `ValidationReport`/`FetchReport` are mutable at baseline |
| 4 | `ValidatedDataset` + write accepting only it | 1 | Depends on 2, 3 |
| 5 | Generation storage + atomic `CURRENT` | 10, 11 | **Must precede the read boundary** — read selects a generation |
| 6 | Acquisition outcome + evidence snapshot | 3, 17 | |
| 7 | Probe redesign | 13 | Independent; must land before any live depth claim |
| 8 | `TrustedDataset` read boundary | 2, 9 | Depends on 4, 5, 6 |
| 9 | Gap/continuity policy + calendar protocol | 4 | |
| 10 | `SecretRegistry` + broker diagnostics | 12, 14 | Independent of the data path |
| 11 | Strict config validation | 16 | Independent |
| 12 | `write_token_to_env` validation | 15 | Small, independent |
| 13 | Dependency/CI hardening | — | |
| 14 | Documentation reconciliation | — | Last, so it describes what exists |

Deviations from A–K: schema policy moves ahead of the digest (§7 coupling);
generation storage moves ahead of the trusted read boundary; the probe redesign
is separated because it is independent of the trust chain and gates live use.

Each unit: test first → implementation → full suite → adversarial test → commit
→ push review branch → **STOP** for manager inspection on GitHub.

---

## 29. Open design risks

1. **Tamper-evidence is weaker than immutability.** A mutation between build and
   write is detected, not prevented. Unavoidable in pandas.
2. **`/history` response shape is still ASSUMED.** The schema policy is designed
   against an unverified contract; the extra-column namespace may need revision
   after the first live call.
3. **Without a calendar, density is never certified.** `TrustedDataset`
   guarantees less than "the data is complete". Phase 2 must not read more into
   it than §11 states.
4. **SDK logs remain outside the redaction boundary** — permanently.
5. **Digest cost is O(rows) in Python.** ~90k rows/year is fine; multi-year
   should be measured.
6. **Probe cost rises** under the corrected algorithm; correctness outranks
   request count, but the budget interacts with rate limits.
7. **No live verification of anything.** Every adapter behaviour remains
   MOCK-VERIFIED.

---

## 30. Non-goals

Explicitly **not** in Phase 1: strategy logic, indicators, signal generation,
backtesting, optimisation, walk-forward, Monte Carlo, TradingView or Pine
integration, webhooks, dashboards, paper trading, order execution of any kind,
live-trading mode, execution-instrument selection, NSE calendar data, and any
performance or profitability claim.

Also not goals: defending against an attacker with machine access;
cryptographic signing; making pandas immutable; certifying bar density without a
calendar.
