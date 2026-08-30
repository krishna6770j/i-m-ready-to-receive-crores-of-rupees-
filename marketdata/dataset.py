"""``ValidatedDataset``: binds a canonical frame to its own evidence.

Frozen architecture, section 16 ("ValidatedDataset strategy") and defect #1
(``docs/architecture/phase1-trust-hardening.md``): baseline ``store.write()``
takes a frame and a ``ValidationReport`` as SEPARATE parameters, with nothing
checking they correspond -- an invalid frame plus an unrelated valid frame's
report can be persisted as ``validation_status="valid"``.

Core invariant this module enforces structurally, not by convention: **a
validation report can never be supplied separately from the frame it claims
to validate.** ``ValidatedDataset.build()`` is the only way to construct one,
it runs ``validate()`` internally on the exact frame it ends up holding, and
there is no parameter through which a caller could substitute a
pre-built ``ValidationReport``, an arbitrary digest, or a different
``MARKET_DATA_SCHEMA_VERSION``.

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
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

from marketdata.evidence import ValidationReportSnapshot, snapshot_validation_report
from marketdata.identity import DatasetIdentity, dataset_digest
from marketdata.schemas import (
    AnomalySeverity,
    CanonicalisationAnomaly,
    CanonicalisationResult,
    CanonicalisationTransformation,
    SourceEvidence,
    assert_canonical,
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
        "_transformations",
        "_source_anomalies",
        "_source_evidence",
        "_frame",
    )

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "ValidatedDataset cannot be constructed directly; use "
            "ValidatedDataset.build(canonicalisation, identity=...) instead, "
            "so the validation report is always generated from the exact "
            "frame it describes."
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
        canonicalisation: CanonicalisationResult,
        *,
        identity: DatasetIdentity,
        expected_interval_minutes: int | None = None,
        sigma_threshold: float = 10.0,
        session_window: tuple | None = None,
        max_session_gap_days: float | None = None,
    ) -> "ValidatedDataset":
        """Build a ``ValidatedDataset`` from canonicalisation evidence.

        ``canonicalisation`` must be the ``CanonicalisationResult`` produced
        by ``canonicalise()``/``canonicalise_fyers_candles()`` for the exact
        data this object is meant to describe -- there is deliberately no
        separate ``frame`` parameter, so a caller cannot pass a
        ``CanonicalisationResult`` for one frame alongside a different frame:
        the frame IS ``canonicalisation.frame``.

        Raises ``TrustBlockerError`` if ``canonicalisation.source_anomalies``
        contains any BLOCKER-severity anomaly (section 14). Raises
        ``SchemaError`` if ``canonicalisation.frame`` is not canonical
        (defence in depth against a hand-constructed, counterfeit
        ``CanonicalisationResult``; a genuine one from ``canonicalise()`` is
        always canonical).

        ``expected_interval_minutes``/``sigma_threshold``/``session_window``/
        ``max_session_gap_days`` are forwarded verbatim to ``validate()``;
        their defaults match ``validate()``'s own. ``symbol``/``resolution``
        are taken from ``identity``, never accepted separately, so the
        validation report's own symbol/resolution can never diverge from the
        identity this object is bound to.
        """
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

        # Own, independent deep copy: decoupled not only from whatever the
        # caller's original raw input was (canonicalise() already guarantees
        # that), but also from canonicalisation.frame itself, so mutating
        # the CanonicalisationResult the caller still holds after this call
        # can never reach this object's internal state either.
        frame = canonicalisation.frame.copy(deep=True)

        report = validate(
            frame,
            symbol=identity.symbol,
            resolution=identity.resolution,
            expected_interval_minutes=expected_interval_minutes,
            sigma_threshold=sigma_threshold,
            session_window=session_window,
            max_session_gap_days=max_session_gap_days,
        )
        validation_snapshot = snapshot_validation_report(report)

        digest = dataset_digest(identity, frame)

        self = object.__new__(cls)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_digest", digest)
        object.__setattr__(self, "_validation", validation_snapshot)
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
