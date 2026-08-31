"""Verified trusted dataset reader, per the frozen architecture
(docs/architecture/phase1-trust-hardening.md, section 12).

**``TrustedDataset`` exists only after every trust check below has already
passed.** There is no stored boolean like ``is_authoritative`` anywhere in
this module -- the TYPE itself is the only certification. Constructing one
outside :func:`read_trusted` is impossible (mirrors ``ValidatedDataset`` /
``ProvenanceEnvelope``'s own hand-rolled ``__slots__`` shape).

**``read_trusted()`` never repairs data while reading.** ``assert_canonical``
is checked directly against the loaded Parquet frame -- ``canonicalise()``
(a REPAIR step) is never invoked as a way to make a non-canonical file pass.
A byte-different Parquet file that decodes to the exact same logical
canonical frame is still accepted (the digest is a function of the LOGICAL
observation sequence, not of Parquet's own byte layout); a file that decodes
to a logically different frame is rejected outright, at whatever the very
first canonical/digest check catches it.

**The trusted-read pipeline** (frozen architecture section 12), each step
gating the next -- nothing after a failed step ever runs:

    1.  derive dataset location from caller identity via Unit-7 safe slugs
    2.  read ``CURRENT``
    3.  strict ``CurrentPointer`` parse
    4.  locate ONLY ``trusted_generations/<generation_id>``
    5.  require ``data.parquet`` + ``manifest.json`` to exist
    6.  strictly parse the manifest (``ReconstructedManifest``)
    7.  load the Parquet frame
    8.  ``assert_canonical`` the loaded frame (no repair)
    9.  recompute ``data_digest`` and compare to the manifest's stored value
    10. recompute ``provenance_digest`` (from the manifest's OWN stored
        software values, never the live environment) and compare
    11. recompute ``integrity_id`` and compare
    12. verify ``CURRENT.integrity_id`` against the (recomputed) integrity_id
    13. verify generation_id: filesystem directory name == pointer == manifest
    14. verify ``namespace == TRUSTED``
    15. verify caller identity == manifest identity == the identity the
        digest was recomputed against (all three are, structurally, the
        same ``DatasetIdentity`` value)
    16. re-run validation using the manifest's STORED ``ValidationPolicy``
        (never a manifest-supplied validation RESULT -- that would be
        trusting an assertion instead of deriving a fact)
    17. require ``MarketDataValidity == VALID``
    18. validate + cross-check the manifest's acquisition evidence
    19. require ``AcquisitionRequestStatus == REQUESTS_SUCCEEDED``
    20. derive ``ObservedDataCoverage``
    21. derive ``RequestedWindowComparison``
    22. only then construct ``TrustedDataset``

**Known, accepted rollback limitation** (frozen architecture, explicitly
accepted, not a defect): if ``CURRENT`` is wholesale-replaced with an OLDER
but completely valid trusted pointer, ``read_trusted()`` accepts that older
generation -- there is no generation-freshness detection anywhere in this
codebase, and this module does not invent one. A test proves this limitation
exists and is accepted, rather than silently "fixing" it out of scope.

**``read_unverified()``** is a deliberately different, narrower doorway
(section 12/9): it requires an EXPLICIT ``generation_id`` (and namespace --
``TRUSTED`` or ``FORCED``) and never consults ``CURRENT`` or performs any of
the trust-gating steps above. It exists for forensic inspection of a
specific generation, including one that would fail every check above (a
tampered manifest, non-canonical data, an unreachable-by-``CURRENT``
generation). It returns a distinct ``UnverifiedDataset`` type that does not
expose (and never silently reuses) any ``TrustedDataset`` certification
property -- there is no fallback path from ``read_trusted`` to
``read_unverified``; a caller must choose the unverified doorway explicitly.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from marketdata.acquisition import (
    AcquisitionError,
    AcquisitionRequestStatus,
    ObservedDataCoverage,
    RequestedWindowComparison,
    classify_acquisition_status,
    compute_observed_data_coverage,
    compute_requested_window_comparison,
    cross_check_fetch_against_dataset,
    validate_fetch_evidence,
)
from marketdata.dataset import MarketDataValidity, ValidationPolicy
from marketdata.evidence import (
    FetchReportSnapshot,
    ValidationReportSnapshot,
    snapshot_validation_report,
)
from marketdata.identity import DatasetIdentity, dataset_digest
from marketdata.locator import CurrentPointer, LocatorError, safe_slug
from marketdata.provenance import ManifestError, Namespace, ReconstructedManifest
from marketdata.schemas import (
    CanonicalisationAnomaly,
    CanonicalisationTransformation,
    SchemaError,
    SourceEvidence,
    assert_canonical,
)
from marketdata.validator import validate

_TRUSTED_DIRNAME = "trusted_generations"
_FORCED_DIRNAME = "forced_generations"
_DATA_FILENAME = "data.parquet"
_MANIFEST_FILENAME = "manifest.json"
_CURRENT_FILENAME = "CURRENT"


class TrustedReaderError(RuntimeError):
    """Base class for every error this module raises."""


class TrustedReadError(TrustedReaderError):
    """Raised when ``read_trusted()`` cannot certify a generation -- any
    failed step in the pipeline documented in the module docstring.
    """


class UnverifiedReadError(TrustedReaderError):
    """Raised when ``read_unverified()`` cannot even locate/parse the exact
    generation requested (missing files, malformed manifest JSON) -- never
    raised for a digest mismatch or any other trust judgement, which
    ``UnverifiedDataset`` deliberately does not make.
    """


class _FrameView:
    """A minimal ``.frame``-only object, so ``marketdata.acquisition``'s
    dataset-shaped helpers (``cross_check_fetch_against_dataset``,
    ``compute_observed_data_coverage``, ``compute_requested_window_comparison``)
    can run against an already-loaded, already-verified-canonical frame
    without going through ``ValidatedDataset.build()`` -- which would run
    ``canonicalise()`` (a REPAIR step) before this module's own
    ``assert_canonical`` gate ever gets a chance to reject non-canonical
    data outright. None of those helper functions touch anything on their
    ``dataset`` argument except ``.frame``.
    """

    __slots__ = ("_frame",)

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)


def _dataset_dir(root: Path, identity: DatasetIdentity) -> Path:
    return (
        Path(root)
        / safe_slug(identity.source)
        / safe_slug(identity.symbol)
        / safe_slug(identity.resolution)
    )


def _namespace_dirname(namespace: Namespace) -> str:
    return _TRUSTED_DIRNAME if namespace is Namespace.TRUSTED else _FORCED_DIRNAME


def _require_generation_id(value: object, field_name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError, TypeError) as exc:
            raise TrustedReaderError(
                f"{field_name} must be a valid UUID4, got {value!r}"
            ) from exc
    else:
        raise TrustedReaderError(
            f"{field_name} must be a uuid.UUID or a UUID string, got "
            f"{type(value).__name__}"
        )
    if parsed.version != 4:
        raise TrustedReaderError(
            f"{field_name} must be a version-4 UUID, got version "
            f"{parsed.version} ({parsed})"
        )
    return parsed


# ---------------------------------------------------------------------------
# TrustedDataset
# ---------------------------------------------------------------------------


class TrustedDataset:
    """A generation that has passed every trust check in
    :func:`read_trusted`'s pipeline (module docstring). Construct ONLY via
    :func:`read_trusted`.

    Deliberately does NOT expose ``is_authoritative`` or any other stored
    trust boolean -- this TYPE is the certification. Deliberately does NOT
    claim continuity certification, session-completeness certification,
    density completeness, reproducibility certification, or "freshest
    generation": none of those are checked by this pipeline, and claiming
    them here would be exactly the kind of invented guarantee the frozen
    architecture's problem statement warns against.
    """

    __slots__ = (
        "_identity",
        "_data_digest",
        "_provenance_digest",
        "_integrity_id",
        "_generation_id",
        "_validation_policy",
        "_validation",
        "_transformations",
        "_source_anomalies",
        "_source_evidence",
        "_acquisition_status",
        "_observed_data_coverage",
        "_requested_window_comparison",
        "_frame",
    )

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "TrustedDataset cannot be constructed directly; use "
            "marketdata.trusted_reader.read_trusted(...) instead, so every "
            "trust check has actually run before this object exists."
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("TrustedDataset is immutable.")

    def __reduce__(self):
        raise TypeError(
            "TrustedDataset cannot be pickled: a revived instance would "
            "carry a certification that was never re-verified in this "
            "process."
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"TrustedDataset(identity={self._identity!r}, "
            f"generation_id={self._generation_id!s}, "
            f"integrity_id={self._integrity_id!r})"
        )

    @classmethod
    def _construct(
        cls,
        *,
        identity: DatasetIdentity,
        data_digest: str,
        provenance_digest: str,
        integrity_id: str,
        generation_id: uuid.UUID,
        validation_policy: ValidationPolicy,
        validation: ValidationReportSnapshot,
        transformations: tuple[CanonicalisationTransformation, ...],
        source_anomalies: tuple[CanonicalisationAnomaly, ...],
        source_evidence: SourceEvidence,
        acquisition_status: AcquisitionRequestStatus,
        observed_data_coverage: ObservedDataCoverage,
        requested_window_comparison: RequestedWindowComparison,
        frame: pd.DataFrame,
    ) -> "TrustedDataset":
        self = object.__new__(cls)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_data_digest", data_digest)
        object.__setattr__(self, "_provenance_digest", provenance_digest)
        object.__setattr__(self, "_integrity_id", integrity_id)
        object.__setattr__(self, "_generation_id", generation_id)
        object.__setattr__(self, "_validation_policy", validation_policy)
        object.__setattr__(self, "_validation", validation)
        object.__setattr__(self, "_transformations", transformations)
        object.__setattr__(self, "_source_anomalies", source_anomalies)
        object.__setattr__(self, "_source_evidence", source_evidence)
        object.__setattr__(self, "_acquisition_status", acquisition_status)
        object.__setattr__(self, "_observed_data_coverage", observed_data_coverage)
        object.__setattr__(
            self, "_requested_window_comparison", requested_window_comparison
        )
        object.__setattr__(self, "_frame", frame)
        return self

    # -- public, read-only surface -----------------------------------------

    @property
    def identity(self) -> DatasetIdentity:
        return self._identity

    @property
    def data_digest(self) -> str:
        return self._data_digest

    @property
    def provenance_digest(self) -> str:
        return self._provenance_digest

    @property
    def integrity_id(self) -> str:
        return self._integrity_id

    @property
    def generation_id(self) -> uuid.UUID:
        return self._generation_id

    @property
    def validation_policy(self) -> ValidationPolicy:
        return self._validation_policy

    @property
    def validation(self) -> ValidationReportSnapshot:
        """The re-run validation snapshot (never the manifest's own,
        unverified claim)."""
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
    def acquisition_status(self) -> AcquisitionRequestStatus:
        """Always ``REQUESTS_SUCCEEDED`` for a constructed ``TrustedDataset``
        -- exposed for inspection, not because any other value could reach
        here. Certifies only that every broker request returned without
        error; certifies nothing about coverage, continuity, sessions, or
        density.
        """
        return self._acquisition_status

    @property
    def observed_data_coverage(self) -> ObservedDataCoverage:
        return self._observed_data_coverage

    @property
    def requested_window_comparison(self) -> RequestedWindowComparison:
        return self._requested_window_comparison

    @property
    def frame(self) -> pd.DataFrame:
        """A fresh defensive copy of the verified canonical frame."""
        return self._frame.copy(deep=True)


# ---------------------------------------------------------------------------
# UnverifiedDataset
# ---------------------------------------------------------------------------


class UnverifiedDataset:
    """One EXACT, EXPLICITLY-named generation (``trusted_generations`` or
    ``forced_generations``), read for forensic inspection ONLY. Construct
    ONLY via :func:`read_unverified`.

    Deliberately exposes the manifest's OWN, UNVERIFIED claims (its stored
    ``data_digest``/``provenance_digest``/``integrity_id``, exactly as
    persisted) rather than recomputed/certified values -- this type makes NO
    trust judgement and performs NO digest recomputation, gate, or
    validation re-run. It never exposes ``TrustedDataset``'s certification
    properties (``market_data_validity``, ``acquisition_status``,
    ``observed_data_coverage``, ``requested_window_comparison``) because
    none of the checks that would make those meaningful have run.
    """

    __slots__ = (
        "_identity",
        "_generation_id",
        "_namespace",
        "_stored_data_digest",
        "_stored_provenance_digest",
        "_stored_integrity_id",
        "_validation_policy",
        "_transformations",
        "_source_anomalies",
        "_source_evidence",
        "_fetch",
        "_forced",
        "_force_reason",
        "_software",
        "_frame",
    )

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "UnverifiedDataset cannot be constructed directly; use "
            "marketdata.trusted_reader.read_unverified(...) instead."
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("UnverifiedDataset is immutable.")

    def __reduce__(self):
        raise TypeError(
            "UnverifiedDataset cannot be pickled: a revived instance would "
            "carry evidence that was never re-read in this process."
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"UnverifiedDataset(identity={self._identity!r}, "
            f"generation_id={self._generation_id!s}, "
            f"namespace={self._namespace.value})"
        )

    @classmethod
    def _construct(
        cls,
        *,
        identity: DatasetIdentity,
        generation_id: uuid.UUID,
        namespace: Namespace,
        stored_data_digest: str,
        stored_provenance_digest: str,
        stored_integrity_id: str,
        validation_policy: ValidationPolicy,
        transformations: tuple[CanonicalisationTransformation, ...],
        source_anomalies: tuple[CanonicalisationAnomaly, ...],
        source_evidence: SourceEvidence,
        fetch: FetchReportSnapshot | None,
        forced: bool,
        force_reason: str | None,
        software,
        frame: pd.DataFrame,
    ) -> "UnverifiedDataset":
        self = object.__new__(cls)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_generation_id", generation_id)
        object.__setattr__(self, "_namespace", namespace)
        object.__setattr__(self, "_stored_data_digest", stored_data_digest)
        object.__setattr__(self, "_stored_provenance_digest", stored_provenance_digest)
        object.__setattr__(self, "_stored_integrity_id", stored_integrity_id)
        object.__setattr__(self, "_validation_policy", validation_policy)
        object.__setattr__(self, "_transformations", transformations)
        object.__setattr__(self, "_source_anomalies", source_anomalies)
        object.__setattr__(self, "_source_evidence", source_evidence)
        object.__setattr__(self, "_fetch", fetch)
        object.__setattr__(self, "_forced", forced)
        object.__setattr__(self, "_force_reason", force_reason)
        object.__setattr__(self, "_software", software)
        object.__setattr__(self, "_frame", frame)
        return self

    # -- public, read-only surface -----------------------------------------

    @property
    def identity(self) -> DatasetIdentity:
        return self._identity

    @property
    def generation_id(self) -> uuid.UUID:
        return self._generation_id

    @property
    def namespace(self) -> Namespace:
        return self._namespace

    @property
    def stored_data_digest(self) -> str:
        """The manifest's OWN claim -- never recomputed/verified here."""
        return self._stored_data_digest

    @property
    def stored_provenance_digest(self) -> str:
        return self._stored_provenance_digest

    @property
    def stored_integrity_id(self) -> str:
        return self._stored_integrity_id

    @property
    def validation_policy(self) -> ValidationPolicy:
        return self._validation_policy

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
    def fetch(self) -> FetchReportSnapshot | None:
        return self._fetch

    @property
    def forced(self) -> bool:
        return self._forced

    @property
    def force_reason(self) -> str | None:
        return self._force_reason

    @property
    def software(self):
        return self._software

    @property
    def frame(self) -> pd.DataFrame:
        """A fresh defensive copy of the frame exactly as read from disk --
        never checked for canonicality, never re-validated."""
        return self._frame.copy(deep=True)


# ---------------------------------------------------------------------------
# read_trusted
# ---------------------------------------------------------------------------


def read_trusted(root: Path, *, source: str, symbol: str, resolution: str) -> TrustedDataset:
    """Read and fully re-verify the generation ``CURRENT`` names for
    ``(source, symbol, resolution)``. See the module docstring for the
    exact pipeline; any failed step raises ``TrustedReadError`` and nothing
    later in the pipeline ever runs.
    """
    identity = DatasetIdentity(source=source, symbol=symbol, resolution=resolution)
    dataset_dir = _dataset_dir(root, identity)

    current_path = dataset_dir / _CURRENT_FILENAME
    if not current_path.is_file():
        raise TrustedReadError(f"No CURRENT pointer at {current_path}.")
    try:
        pointer_text = current_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrustedReadError(f"Could not read CURRENT at {current_path}: {exc}") from exc

    try:
        pointer = CurrentPointer.from_json(pointer_text)
    except LocatorError as exc:
        raise TrustedReadError(f"CURRENT pointer at {current_path} is invalid: {exc}") from exc

    generation_dir = dataset_dir / _TRUSTED_DIRNAME / str(pointer.generation_id)
    if not generation_dir.is_dir():
        raise TrustedReadError(
            f"Generation {pointer.generation_id} named by CURRENT does not "
            f"exist under {dataset_dir / _TRUSTED_DIRNAME}."
        )

    data_path = generation_dir / _DATA_FILENAME
    manifest_path = generation_dir / _MANIFEST_FILENAME
    if not data_path.is_file():
        raise TrustedReadError(f"Missing {_DATA_FILENAME} at {data_path}.")
    if not manifest_path.is_file():
        raise TrustedReadError(f"Missing {_MANIFEST_FILENAME} at {manifest_path}.")

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrustedReadError(f"Could not read manifest at {manifest_path}: {exc}") from exc

    try:
        manifest = ReconstructedManifest.from_manifest_json(manifest_text)
    except ManifestError as exc:
        raise TrustedReadError(f"Manifest at {manifest_path} is invalid: {exc}") from exc

    # Filesystem generation_id == pointer == manifest, all three.
    fs_generation_id = _require_generation_id(generation_dir.name, "generation directory name")
    if fs_generation_id != pointer.generation_id:
        raise TrustedReadError(
            f"Generation directory name {fs_generation_id} does not match "
            f"CURRENT pointer generation_id {pointer.generation_id}."
        )
    if manifest.generation_id != pointer.generation_id:
        raise TrustedReadError(
            f"Manifest generation_id {manifest.generation_id} does not "
            f"match CURRENT pointer generation_id {pointer.generation_id}."
        )

    if manifest.namespace is not Namespace.TRUSTED:
        raise TrustedReadError(
            f"Manifest namespace is {manifest.namespace.value}, expected TRUSTED."
        )
    # Defense-in-depth: ReconstructedManifest.from_manifest_json() already
    # rejects namespace/forced/force_reason inconsistency as a structurally
    # impossible manifest, but this trust boundary does not rely on that
    # ALONE -- forced/force_reason are checked explicitly here too, so a
    # future change to the parser's own invariant check can never silently
    # widen what read_trusted() accepts.
    if manifest.forced is not False:
        raise TrustedReadError(
            f"Manifest forced is {manifest.forced!r}, expected False for a "
            "TRUSTED read."
        )
    if manifest.force_reason is not None:
        raise TrustedReadError(
            f"Manifest force_reason is {manifest.force_reason!r}, expected "
            "None for a TRUSTED read."
        )

    manifest_identity = DatasetIdentity(
        source=manifest.source, symbol=manifest.symbol, resolution=manifest.resolution
    )
    if manifest_identity != identity:
        raise TrustedReadError(
            f"Manifest identity ({manifest_identity}) does not match caller "
            f"identity ({identity})."
        )

    try:
        frame = pd.read_parquet(data_path)
    except Exception as exc:  # noqa: BLE001 - any parquet-engine failure is a read failure
        raise TrustedReadError(f"Could not read Parquet at {data_path}: {exc}") from exc

    # Step: assert_canonical FIRST -- never canonicalise() as a repair step.
    try:
        assert_canonical(frame)
    except SchemaError as exc:
        raise TrustedReadError(
            f"Data at {data_path} is not canonical: {exc}"
        ) from exc

    recomputed_data_digest = dataset_digest(identity, frame)
    if recomputed_data_digest != manifest.data_digest:
        raise TrustedReadError(
            f"Recomputed data_digest ({recomputed_data_digest}) does not "
            f"match manifest.data_digest ({manifest.data_digest}). The "
            "logical observation sequence has changed since this "
            "generation was written."
        )

    recomputed_provenance_digest = manifest.recompute_provenance_digest()
    if recomputed_provenance_digest != manifest.provenance_digest:
        raise TrustedReadError(
            f"Recomputed provenance_digest ({recomputed_provenance_digest}) "
            f"does not match manifest.provenance_digest "
            f"({manifest.provenance_digest}). A provenance fact has been "
            "edited since this manifest was written."
        )

    recomputed_integrity_id = manifest.recompute_integrity_id()
    if recomputed_integrity_id != manifest.integrity_id:
        raise TrustedReadError(
            f"Recomputed integrity_id ({recomputed_integrity_id}) does not "
            f"match manifest.integrity_id ({manifest.integrity_id})."
        )

    if recomputed_integrity_id != pointer.integrity_id:
        raise TrustedReadError(
            f"Recomputed integrity_id ({recomputed_integrity_id}) does not "
            f"match CURRENT pointer integrity_id ({pointer.integrity_id})."
        )

    # Canonicalisation in schema v1 never drops rows, so the row count
    # source_evidence recorded at build time must still equal the loaded
    # canonical frame's row count. This is the ONLY source_evidence fact
    # genuinely re-derivable from the final frame -- source ordering,
    # representation-level duplicates, transformations, and anomalies are
    # all facts about the ORIGINAL input that cannot be reconstructed from
    # already-canonicalised data, and this check deliberately does not
    # pretend otherwise.
    if manifest.source_evidence.row_count != len(frame):
        raise TrustedReadError(
            f"manifest.source_evidence.row_count "
            f"({manifest.source_evidence.row_count}) does not match the "
            f"loaded canonical frame's row count ({len(frame)})."
        )

    # Re-run validation using the STORED ValidationPolicy -- never trust a
    # validation RESULT from the manifest (there is none stored anyway).
    report = validate(
        frame,
        symbol=identity.symbol,
        resolution=identity.resolution,
        expected_interval_minutes=manifest.validation_policy.expected_interval_minutes,
        sigma_threshold=manifest.validation_policy.sigma_threshold,
        session_window=manifest.validation_policy.session_window,
        max_session_gap_days=manifest.validation_policy.max_session_gap_days,
    )
    validation_snapshot = snapshot_validation_report(report)
    if not validation_snapshot.is_usable:
        raise TrustedReadError(
            "Re-run validation reports MarketDataValidity == INVALID "
            f"({len(validation_snapshot.errors)} error(s)); refusing "
            "trusted read."
        )

    frame_view = _FrameView(frame)

    if manifest.fetch is None:
        raise TrustedReadError(
            "No acquisition evidence (AcquisitionRequestStatus == "
            "REQUESTS_UNKNOWN); trusted read requires REQUESTS_SUCCEEDED."
        )
    try:
        validate_fetch_evidence(manifest.fetch)
        cross_check_fetch_against_dataset(manifest.fetch, frame_view)
    except AcquisitionError as exc:
        raise TrustedReadError(f"Acquisition evidence is invalid: {exc}") from exc

    acquisition_status = classify_acquisition_status(manifest.fetch)
    if acquisition_status is not AcquisitionRequestStatus.REQUESTS_SUCCEEDED:
        raise TrustedReadError(
            f"Acquisition status is {acquisition_status.value}; trusted "
            "read requires REQUESTS_SUCCEEDED. REQUESTS_SUCCEEDED itself "
            "certifies only that every broker request returned without "
            "error -- nothing about coverage, continuity, sessions, or "
            "density."
        )

    observed_data_coverage = compute_observed_data_coverage(frame_view)
    requested_window_comparison = compute_requested_window_comparison(frame_view, manifest.fetch)

    return TrustedDataset._construct(
        identity=identity,
        data_digest=recomputed_data_digest,
        provenance_digest=recomputed_provenance_digest,
        integrity_id=recomputed_integrity_id,
        generation_id=manifest.generation_id,
        validation_policy=manifest.validation_policy,
        validation=validation_snapshot,
        transformations=manifest.transformations,
        source_anomalies=manifest.source_anomalies,
        source_evidence=manifest.source_evidence,
        acquisition_status=acquisition_status,
        observed_data_coverage=observed_data_coverage,
        requested_window_comparison=requested_window_comparison,
        frame=frame,
    )


# ---------------------------------------------------------------------------
# read_unverified
# ---------------------------------------------------------------------------


def read_unverified(
    root: Path,
    *,
    source: str,
    symbol: str,
    resolution: str,
    namespace: Namespace,
    generation_id: uuid.UUID | str,
) -> UnverifiedDataset:
    """Read one EXACT, explicitly-named generation for forensic inspection.

    Never consults ``CURRENT``. Never falls back from a failed
    :func:`read_trusted` call -- there is no code path connecting the two
    functions at all. Performs no digest recomputation, no validation
    re-run, no acquisition gate: the returned ``UnverifiedDataset`` exposes
    exactly what is stored, however inconsistent that turns out to be.

    Raises ``UnverifiedReadError`` if the generation cannot even be located
    or its manifest does not structurally parse -- never for a digest
    mismatch or any other trust judgement.
    """
    if not isinstance(namespace, Namespace):
        raise UnverifiedReadError(
            f"namespace must be an actual Namespace member, got {namespace!r}"
        )
    resolved_generation_id = _require_generation_id(generation_id, "generation_id")

    identity = DatasetIdentity(source=source, symbol=symbol, resolution=resolution)
    dataset_dir = _dataset_dir(root, identity)
    generation_dir = dataset_dir / _namespace_dirname(namespace) / str(resolved_generation_id)

    if not generation_dir.is_dir():
        raise UnverifiedReadError(
            f"Generation {resolved_generation_id} does not exist under "
            f"{dataset_dir / _namespace_dirname(namespace)}."
        )

    data_path = generation_dir / _DATA_FILENAME
    manifest_path = generation_dir / _MANIFEST_FILENAME
    if not data_path.is_file():
        raise UnverifiedReadError(f"Missing {_DATA_FILENAME} at {data_path}.")
    if not manifest_path.is_file():
        raise UnverifiedReadError(f"Missing {_MANIFEST_FILENAME} at {manifest_path}.")

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UnverifiedReadError(f"Could not read manifest at {manifest_path}: {exc}") from exc

    try:
        manifest = ReconstructedManifest.from_manifest_json(manifest_text)
    except ManifestError as exc:
        raise UnverifiedReadError(f"Manifest at {manifest_path} is invalid: {exc}") from exc

    try:
        frame = pd.read_parquet(data_path)
    except Exception as exc:  # noqa: BLE001 - any parquet-engine failure is a read failure
        raise UnverifiedReadError(f"Could not read Parquet at {data_path}: {exc}") from exc

    manifest_identity = DatasetIdentity(
        source=manifest.source, symbol=manifest.symbol, resolution=manifest.resolution
    )

    return UnverifiedDataset._construct(
        identity=manifest_identity,
        generation_id=manifest.generation_id,
        namespace=manifest.namespace,
        stored_data_digest=manifest.data_digest,
        stored_provenance_digest=manifest.provenance_digest,
        stored_integrity_id=manifest.integrity_id,
        validation_policy=manifest.validation_policy,
        transformations=manifest.transformations,
        source_anomalies=manifest.source_anomalies,
        source_evidence=manifest.source_evidence,
        fetch=manifest.fetch,
        forced=manifest.forced,
        force_reason=manifest.force_reason,
        software=manifest.software,
        frame=frame,
    )
