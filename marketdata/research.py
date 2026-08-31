"""Research-readiness policy boundary, per the frozen architecture.

``TrustedDataset`` (``marketdata.trusted_reader``) certifies a SOUND STORED
ARTIFACT: every generation-write and read-time trust check has passed. It
does NOT certify that a given generation is fit for any particular
experiment -- three candles for a year is a perfectly sound, perfectly
trusted artifact that almost no research question should ever be allowed to
run against silently.

``ResearchReadyDataset`` is the second, separate gate: a ``TrustedDataset``
that additionally satisfies an explicit ``ResearchDataPolicy`` declared FOR
ONE EXPERIMENT. Phase 2 is expected to eventually accept
``ResearchReadyDataset`` only -- never a bare ``TrustedDataset``, and never a
raw frame.

**This module does not touch, weaken, or reinterpret ``TrustedDataset``
trust semantics.** It only reads already-certified facts already exposed by
``TrustedDataset`` (``.identity``, ``.observed_data_coverage``,
``.requested_window_comparison``, ``.source_evidence``) and evaluates them
against caller-declared thresholds.

**Continuity/session certification (Unit 13B)**: ``require_continuity_certified``/
``require_session_certified`` now check the actual
``trusted.continuity_certification``/``trusted.session_certification``
facts ``read_trusted()`` attaches (``marketdata.continuity``). Both
``NOT_CERTIFIED`` and ``FAILED`` are policy failures -- only an actual
``CERTIFIED`` status passes. This module still does not compute those
certifications itself, and still does not weaken or reinterpret
``TrustedDataset`` trust semantics: a ``TrustedDataset`` whose continuity/
session certification is ``FAILED`` is still a perfectly sound artifact:
whether that soundness is enough for a given experiment is exactly what
this policy decides.

**Reproducibility certification remains NOT AVAILABLE.** No such
certification exists anywhere in this codebase yet (a later, separately
authorised unit). Rather than inventing a placeholder verdict,
``require_reproducibility_certified=True`` ALWAYS evaluates as an unmet
requirement -- "not yet available" is reported honestly, never silently
treated as satisfied and never defaulted to "certified".

**Continuity/session certification does NOT imply**: requested-range
completeness, bar-density completeness, earliest available history, a
retention boundary, or data freshness. None of those claims are made here
or anywhere in ``marketdata.continuity``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from marketdata.continuity import CertificationStatus
from marketdata.trusted_reader import TrustedDataset


class ResearchPolicyError(ValueError):
    """Raised when a ``TrustedDataset`` does not satisfy every declared
    requirement of a ``ResearchDataPolicy``. ``unmet_requirements`` lists
    every failing requirement (not just the first) as human-readable
    strings, in the order the policy's own fields are checked.
    """

    def __init__(self, unmet_requirements: tuple[str, ...]) -> None:
        self.unmet_requirements = unmet_requirements
        message = "Research readiness requirements not met:\n" + "\n".join(
            f"  - {reason}" for reason in unmet_requirements
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ResearchDataPolicy:
    """One experiment's explicit research-readiness requirements.

    Every field is optional/``False`` by default -- a default-constructed
    policy imposes no requirement at all, so any ``TrustedDataset`` becomes
    research-ready under it. Fields are read-only value objects (frozen
    dataclass); ``__post_init__`` rejects malformed values outright rather
    than coercing them, matching ``ValidationPolicy``'s own self-validating
    shape elsewhere in this codebase.

    ``require_continuity_certified``/``require_session_certified``/
    ``require_reproducibility_certified`` correspond to certifications the
    frozen architecture anticipates but that do not exist in this codebase
    yet (Unit 13+). Setting any of them to ``True`` makes
    :meth:`ResearchReadyDataset.build` ALWAYS report that requirement as
    unmet -- never silently satisfied, never defaulted to "certified".
    """

    expected_source: str | None = None
    expected_symbol: str | None = None
    expected_resolution: str | None = None
    min_distinct_observed_dates: int | None = None
    min_requested_window_fraction: float | None = None
    require_pristine_source_order: bool = False
    require_continuity_certified: bool = False
    require_session_certified: bool = False
    require_reproducibility_certified: bool = False

    def __post_init__(self) -> None:
        for field_name in ("expected_source", "expected_symbol", "expected_resolution"):
            value = getattr(self, field_name)
            if value is not None:
                if type(value) is not str:
                    raise TypeError(
                        f"{field_name} must be None or a str, got {value!r} "
                        f"({type(value).__name__})"
                    )
                if value == "":
                    raise ValueError(f"{field_name} must be non-empty when supplied")

        if self.min_distinct_observed_dates is not None:
            value = self.min_distinct_observed_dates
            if type(value) is not int:
                raise TypeError(
                    "min_distinct_observed_dates must be None or an actual "
                    f"int, got {value!r} ({type(value).__name__})"
                )
            if value <= 0:
                raise ValueError(
                    f"min_distinct_observed_dates must be positive, got {value!r}"
                )

        if self.min_requested_window_fraction is not None:
            value = self.min_requested_window_fraction
            if type(value) not in (int, float):
                raise TypeError(
                    "min_requested_window_fraction must be None or a real "
                    f"number, got {value!r} ({type(value).__name__})"
                )
            value = float(value)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(
                    "min_requested_window_fraction must be finite and in "
                    f"[0, 1], got {self.min_requested_window_fraction!r}"
                )
            object.__setattr__(self, "min_requested_window_fraction", value)

        for field_name in (
            "require_pristine_source_order",
            "require_continuity_certified",
            "require_session_certified",
            "require_reproducibility_certified",
        ):
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise TypeError(
                    f"{field_name} must be an actual bool, got {value!r} "
                    f"({type(value).__name__})"
                )


def _evaluate(trusted: TrustedDataset, policy: ResearchDataPolicy) -> tuple[str, ...]:
    """Every unmet requirement, in field-declaration order. Uses only
    already-certified ``TrustedDataset`` facts -- derives no new coverage
    formula, infers nothing beyond what each named property states.
    """
    unmet: list[str] = []

    if policy.expected_source is not None and trusted.identity.source != policy.expected_source:
        unmet.append(
            f"expected_source={policy.expected_source!r} but "
            f"trusted.identity.source={trusted.identity.source!r}"
        )
    if policy.expected_symbol is not None and trusted.identity.symbol != policy.expected_symbol:
        unmet.append(
            f"expected_symbol={policy.expected_symbol!r} but "
            f"trusted.identity.symbol={trusted.identity.symbol!r}"
        )
    if (
        policy.expected_resolution is not None
        and trusted.identity.resolution != policy.expected_resolution
    ):
        unmet.append(
            f"expected_resolution={policy.expected_resolution!r} but "
            f"trusted.identity.resolution={trusted.identity.resolution!r}"
        )

    if policy.min_distinct_observed_dates is not None:
        actual = trusted.observed_data_coverage.distinct_observed_dates
        if actual < policy.min_distinct_observed_dates:
            unmet.append(
                "min_distinct_observed_dates="
                f"{policy.min_distinct_observed_dates} but "
                f"observed_data_coverage.distinct_observed_dates={actual}"
            )

    if policy.min_requested_window_fraction is not None:
        comparison = getattr(trusted, "requested_window_comparison", None)
        if comparison is None:
            unmet.append(
                "min_requested_window_fraction="
                f"{policy.min_requested_window_fraction} but "
                "RequestedWindowComparison is unavailable"
            )
        elif comparison.observed_distinct_dates_ratio < policy.min_requested_window_fraction:
            unmet.append(
                "min_requested_window_fraction="
                f"{policy.min_requested_window_fraction} but "
                "requested_window_comparison.observed_distinct_dates_ratio="
                f"{comparison.observed_distinct_dates_ratio}"
            )

    if policy.require_pristine_source_order and not trusted.source_evidence.timestamps_sorted:
        unmet.append(
            "require_pristine_source_order=True but "
            "source_evidence.timestamps_sorted is False"
        )

    if policy.require_continuity_certified:
        continuity_status = trusted.continuity_certification.status
        if continuity_status is not CertificationStatus.CERTIFIED:
            unmet.append(
                "require_continuity_certified=True but "
                f"continuity_certification.status={continuity_status.value} "
                "(NOT_CERTIFIED and FAILED are both policy failures; only "
                "CERTIFIED passes)"
            )
    if policy.require_session_certified:
        session_status = trusted.session_certification.status
        if session_status is not CertificationStatus.CERTIFIED:
            unmet.append(
                "require_session_certified=True but "
                f"session_certification.status={session_status.value} "
                "(NOT_CERTIFIED and FAILED are both policy failures; only "
                "CERTIFIED passes)"
            )

    # Not yet implemented anywhere in this codebase (a later, separately
    # authorised unit): True always reports unmet/not-available, never
    # silently satisfied.
    if policy.require_reproducibility_certified:
        unmet.append(
            "require_reproducibility_certified=True but reproducibility "
            "certification is not yet implemented -- NOT AVAILABLE"
        )

    return tuple(unmet)


class ResearchReadyDataset:
    """A ``TrustedDataset`` that has passed every declared requirement of a
    ``ResearchDataPolicy``. Construct ONLY via :meth:`build` -- mirrors
    ``TrustedDataset``/``ValidatedDataset``/``ProvenanceEnvelope``'s own
    hand-rolled ``__slots__`` shape (direct instantiation and attribute
    assignment both raise), for the same reason: a plain constructor would
    let a caller assemble an instance that never actually passed policy
    evaluation.
    """

    __slots__ = ("_trusted", "_policy")

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "ResearchReadyDataset cannot be constructed directly; use "
            "ResearchReadyDataset.build(trusted, policy) instead, so every "
            "declared policy requirement has actually been evaluated."
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ResearchReadyDataset is immutable.")

    def __reduce__(self):
        raise TypeError(
            "ResearchReadyDataset cannot be pickled: a revived instance "
            "would carry a readiness verdict that was never re-evaluated "
            "in this process."
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ResearchReadyDataset(identity={self._trusted.identity!r}, "
            f"generation_id={self._trusted.generation_id!s})"
        )

    @classmethod
    def build(cls, trusted: TrustedDataset, policy: ResearchDataPolicy) -> "ResearchReadyDataset":
        """Build a ``ResearchReadyDataset`` from an actual ``TrustedDataset``
        plus an actual ``ResearchDataPolicy``.

        ``trusted`` must be an actual ``TrustedDataset`` instance -- a raw
        ``pandas.DataFrame``, a ``ValidatedDataset``, an
        ``UnverifiedDataset``, or any duck-typed lookalike is rejected
        outright with ``TypeError``, never silently accepted because it
        happens to expose similarly-named attributes.

        Raises ``ResearchPolicyError`` (listing every unmet requirement, not
        just the first) if any declared policy requirement fails. Never
        returns a partially-ready object: either every requirement passed,
        or nothing is constructed at all.
        """
        if not isinstance(trusted, TrustedDataset):
            raise TypeError(
                "trusted must be an actual TrustedDataset instance (from "
                "marketdata.trusted_reader.read_trusted()), got "
                f"{type(trusted).__name__}."
            )
        if not isinstance(policy, ResearchDataPolicy):
            raise TypeError(
                "policy must be an actual ResearchDataPolicy instance, got "
                f"{type(policy).__name__}."
            )

        unmet = _evaluate(trusted, policy)
        if unmet:
            raise ResearchPolicyError(unmet)

        self = object.__new__(cls)
        object.__setattr__(self, "_trusted", trusted)
        object.__setattr__(self, "_policy", policy)
        return self

    # -- public, read-only surface -----------------------------------------

    @property
    def policy(self) -> ResearchDataPolicy:
        return self._policy

    @property
    def identity(self):
        return self._trusted.identity

    @property
    def generation_id(self):
        return self._trusted.generation_id

    @property
    def data_digest(self) -> str:
        return self._trusted.data_digest

    @property
    def integrity_id(self) -> str:
        return self._trusted.integrity_id

    @property
    def frame(self) -> pd.DataFrame:
        """A fresh defensive copy of the underlying trusted, verified
        canonical frame."""
        return self._trusted.frame
