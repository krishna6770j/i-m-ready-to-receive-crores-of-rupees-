"""``ValidatedDataset``: binds a canonical frame to its own evidence.

Frozen architecture, section 16 ("ValidatedDataset strategy") and defect #1
(``docs/architecture/phase1-trust-hardening.md``): baseline ``store.write()``
takes a frame and a ``ValidationReport`` as SEPARATE parameters, with nothing
checking they correspond -- an invalid frame plus an unrelated valid frame's
report can be persisted as ``validation_status="valid"``.

Core invariant this module enforces structurally, not by convention: **a
validation report -- and, per manager review of an earlier revision,
canonicalisation evidence too -- can never be supplied separately from the
frame it claims to describe.** ``ValidatedDataset.build()`` is the only way
to construct one. It takes a RAW ``pd.DataFrame`` and calls ``canonicalise()``
on it INTERNALLY: an earlier revision of this module accepted a caller-
supplied ``CanonicalisationResult``, which let a caller bind one frame's
data to a completely different frame's transformations/anomalies/source
evidence (reproduced directly: a 3-row frame bound to evidence claiming 7
rows and a ``TIMEZONE_CONVERTED`` transformation that never happened to it).
Canonicalising internally makes that forgery structurally impossible, the
same way accepting no ``report``/``digest``/``schema_version`` parameter
makes THOSE substitutions impossible.

**Validated, possibly invalid.** Building a ``ValidatedDataset`` for OHLC-
invalid data (bad ordering, non-positive prices, ...) is ALLOWED: those are
``MarketDataValidity = INVALID`` facts recorded as evidence (frozen
architecture section 2 glossary), not a build-time gate. What DOES gate
``build()`` is a canonicalisation TRUST BLOCKER (section 14): "TRUST BLOCKER
means ``ValidatedDataset.build()`` fails; nothing reaches storage" is exact,
load-bearing architecture text, so a BLOCKER-severity
``CanonicalisationAnomaly`` (currently only
``CANONICAL_CONFLICTING_TIMESTAMPS``; the other three TRUST BLOCKER causes in
section 14's table already raise ``SchemaError`` inside ``canonicalise()``
itself, before any ``CanonicalisationResult`` exists) raises here instead of
producing an object. This is the intentional MarketDataValidity/TRUST-BLOCKER
split; do not conflate the two.

**Deciding whether to persist an invalid-but-buildable ``ValidatedDataset`` is
NOT this module's job.** That decision (today: ``store.write()``'s
``UnvalidatedDataError``/``force`` gate; per the frozen architecture,
eventually a ``TrustedDataset`` read-time check) belongs to a later,
explicitly authorised unit -- this module only guarantees the evidence is
truthful and bound to the exact data it describes.

**Pandas cannot be made immutable** (frozen architecture section 16, verified
there empirically: ``arr.flags.writeable = False`` on every column still
permitted a write under pandas 3.0 copy-on-write). This module does not
pretend otherwise: it holds a private deep copy internally, and ``.frame``
returns a FRESH deep copy on every access, so mutating what a caller
previously received back never reaches the bound internal frame -- tamper-
EVIDENCE, not tamper-prevention, "the strongest guarantee available in this
language" (section 16).

**Validation policy is bound evidence, not a build-time secret.** ``build()``
previously accepted ``expected_interval_minutes``/``sigma_threshold``/
``session_window``/``max_session_gap_days`` but discarded them once
``validate()`` returned -- a caller inspecting a built ``ValidatedDataset``
had no way to know which policy actually produced its validation evidence.
``ValidationPolicy`` is a frozen value object bundling exactly those four
fields; ``build()`` stores the exact policy instance it validated against
and exposes it as ``.validation_policy``, and validation always runs from
that stored policy, never from ad-hoc defaults recomputed elsewhere. The
policy is deliberately NOT part of ``data_digest``: the frozen architecture's
identity (section 8.1) is the canonical observation sequence plus
schema_version/source/symbol/resolution -- a validation policy is a
provenance/evidence fact about HOW the data was checked, not a fact about
WHAT the data is. The same frame + identity validated under two different
policies (e.g. differing ``max_session_gap_days``) can therefore legitimately
share one ``data_digest`` while producing different validation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from marketdata.evidence import ValidationReportSnapshot, snapshot_validation_report
from marketdata.identity import DatasetIdentity, dataset_digest
from marketdata.schemas import (
    AnomalySeverity,
    CanonicalisationAnomaly,
    CanonicalisationTransformation,
    SourceEvidence,
    assert_canonical,
    canonicalise,
)
from marketdata.validator import validate


class TrustBlockerError(ValueError):
    """Raised when canonicalisation evidence contains a TRUST BLOCKER anomaly.

    Frozen architecture section 14: "TRUST BLOCKER means
    ``ValidatedDataset.build()`` fails; nothing reaches storage." Distinct
    from ``MarketDataValidity = INVALID`` (an ordinary OHLC/finiteness/
    duplicate/ordering ERROR from ``validate()``), which does NOT raise here
    -- see the module docstring's "validated, possibly invalid" section.
    """


class MarketDataValidity(str, Enum):
    """Frozen architecture section 2 glossary: "No ERROR-severity validation
    issue (OHLC rules, finiteness, duplicates, ordering)". Derived from data
    (the bound frame's own ``validate()`` result), recomputed from
    ``ValidationReportSnapshot.is_usable`` rather than stored a second time.
    """

    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Exactly the configuration ``marketdata.validator.validate()`` accepts,
    bundled so it can be bound to a ``ValidatedDataset`` and inspected later
    instead of being silently discarded once ``validate()`` returns.

    NOT part of ``data_digest``: this describes HOW data was checked, not
    WHAT the data is (see the module docstring). Two builds of the same
    frame + identity under different policies may share one digest while
    differing in validation evidence.
    """

    expected_interval_minutes: int | None = None
    sigma_threshold: float = 10.0
    session_window: tuple | None = None
    max_session_gap_days: float | None = None


class ValidatedDataset:
    """A canonical frame permanently bound to its own validation evidence,
    canonicalisation evidence, identity and content digest.

    Construct ONLY via :meth:`build`. Direct instantiation
    (``ValidatedDataset(...)``) and attribute assignment both raise --
    mirroring the same "remove accidental reach, make deliberate reach
    obvious" boundary already used by
    ``brokers.fyers.client.ReadOnlyFyersClient`` elsewhere in this codebase.
    A sufficiently determined caller can still reach ``__slots__``-backed
    private attributes via ``object.__setattr__`` or by editing the
    ``_frame`` DataFrame in place through a name-mangled reference; nothing
    in a dynamic language can prevent that, and this class does not claim
    otherwise (frozen architecture section 16's own "tamper-evidence, not
    tamper-prevention" framing applies here too). What it removes is
    ACCIDENTAL mutation through the public API.
    """

    __slots__ = (
        "_identity",
        "_digest",
        "_validation",
        "_validation_policy",
        "_transformations",
        "_source_anomalies",
        "_source_evidence",
        "_frame",
    )

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "ValidatedDataset cannot be constructed directly; use "
            "ValidatedDataset.build(raw_frame, identity=...) instead, so "
            "canonicalisation and the validation report are always "
            "generated from the exact frame supplied."
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ValidatedDataset is immutable.")

    def __reduce__(self):
        # Frozen architecture section 16, item 4: pickling and reviving a
        # ValidatedDataset would let stale evidence outlive the process that
        # verified it, silently reintroducing the "unverified report treated
        # as current" defect this type exists to remove.
        raise TypeError(
            "ValidatedDataset cannot be pickled: a revived instance would "
            "carry evidence that was never re-verified in this process."
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ValidatedDataset(identity={self._identity!r}, "
            f"digest={self._digest!r}, "
            f"market_data_validity={self.market_data_validity.value})"
        )

    @classmethod
    def build(
        cls,
        raw_frame: pd.DataFrame,
        *,
        identity: DatasetIdentity,
        validation_policy: ValidationPolicy = ValidationPolicy(),
    ) -> "ValidatedDataset":
        """Build a ``ValidatedDataset`` from a RAW ``pd.DataFrame``.

        ``canonicalise(raw_frame)`` runs INTERNALLY, here -- there is
        deliberately no way to supply canonicalisation evidence (a
        ``CanonicalisationResult``) separately from the frame it describes.
        An earlier revision of this method accepted one as a parameter,
        which let a caller bind one frame's data to a different frame's
        transformations/anomalies/source evidence; canonicalising internally
        makes that forgery structurally impossible, exactly like accepting
        no ``report``/``digest``/``schema_version`` parameter does for those.

        Only a generic ``pd.DataFrame`` is accepted. A FYERS-positional-
        payload constructor (wrapping ``canonicalise_fyers_candles()``) is
        explicitly NOT implemented here -- out of scope for this unit.

        Raises ``TrustBlockerError`` if internal canonicalisation reports any
        BLOCKER-severity anomaly (section 14: "TRUST BLOCKER means
        ``ValidatedDataset.build()`` fails; nothing reaches storage").

        ``validation_policy`` is a ``ValidationPolicy`` (default: all of
        ``validate()``'s own defaults). It is stored EXACTLY as given and
        exposed via ``.validation_policy``; ``validate()`` runs from that
        stored policy's fields, never from ad-hoc values. ``symbol``/
        ``resolution`` passed to ``validate()`` are taken from ``identity``,
        never accepted separately, so the validation report's own symbol/
        resolution can never diverge from the identity this object is bound
        to.
        """
        canonicalisation = canonicalise(raw_frame)

        blockers = [
            a for a in canonicalisation.source_anomalies
            if a.severity is AnomalySeverity.BLOCKER
        ]
        if blockers:
            codes = ", ".join(a.code for a in blockers)
            raise TrustBlockerError(
                f"Refusing to build a ValidatedDataset: {len(blockers)} TRUST "
                f"BLOCKER anomaly(ies) present ({codes}). Frozen architecture "
                "section 14: a TRUST BLOCKER means nothing reaches storage."
            )

        assert_canonical(canonicalisation.frame)

        # Own, independent deep copy: decoupled not only from raw_frame
        # (canonicalise() already guarantees that) but also from
        # canonicalisation.frame itself, so mutating the raw frame the
        # caller still holds after this call can never reach this object's
        # internal state either.
        frame = canonicalisation.frame.copy(deep=True)

        report = validate(
            frame,
            symbol=identity.symbol,
            resolution=identity.resolution,
            expected_interval_minutes=validation_policy.expected_interval_minutes,
            sigma_threshold=validation_policy.sigma_threshold,
            session_window=validation_policy.session_window,
            max_session_gap_days=validation_policy.max_session_gap_days,
        )
        validation_snapshot = snapshot_validation_report(report)

        digest = dataset_digest(identity, frame)

        self = object.__new__(cls)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_digest", digest)
        object.__setattr__(self, "_validation", validation_snapshot)
        object.__setattr__(self, "_validation_policy", validation_policy)
        object.__setattr__(
            self, "_transformations", tuple(canonicalisation.transformations)
        )
        object.__setattr__(
            self, "_source_anomalies", tuple(canonicalisation.source_anomalies)
        )
        object.__setattr__(self, "_source_evidence", canonicalisation.source)
        object.__setattr__(self, "_frame", frame)
        return self

    # -- public, read-only surface -----------------------------------------

    @property
    def identity(self) -> DatasetIdentity:
        return self._identity

    @property
    def digest(self) -> str:
        """``dataset_digest(identity, frame)`` of the exact bound frame."""
        return self._digest

    @property
    def validation(self) -> ValidationReportSnapshot:
        return self._validation

    @property
    def validation_policy(self) -> ValidationPolicy:
        return self._validation_policy

    @property
    def market_data_validity(self) -> MarketDataValidity:
        return (
            MarketDataValidity.VALID
            if self._validation.is_usable
            else MarketDataValidity.INVALID
        )

    @property
    def transformations(self) -> tuple[CanonicalisationTransformation, ...]:
        return self._transformations

    @property
    def source_anomalies(self) -> tuple[CanonicalisationAnomaly, ...]:
        return self._source_anomalies

    @property
    def source_evidence(self) -> SourceEvidence:
        return self._source_evidence

    @property
    def frame(self) -> pd.DataFrame:
        """A fresh defensive copy of the bound canonical frame.

        Mutating the returned frame never affects this object: a NEW deep
        copy is made on every access, never the internal ``_frame`` itself.
        """
        return self._frame.copy(deep=True)
