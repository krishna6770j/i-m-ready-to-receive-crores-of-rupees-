"""Atomic generation storage, per the frozen architecture
(docs/architecture/phase1-trust-hardening.md, sections 10 and 13.3).

**On-disk layout:**

    data_store/<source_slug>/<symbol_slug>/<resolution_slug>/
        trusted_generations/<uuid4>/
            data.parquet
            manifest.json
        forced_generations/<uuid4>/
            data.parquet
            manifest.json
        CURRENT

Slugs are Unit 7's ``marketdata.locator.safe_slug`` -- raw
source/symbol/resolution text is never interpolated into a path. The
generation directory component is always the generation's canonical
version-4 UUID string.

**Input is always a (``ValidatedDataset``, ``ProvenanceEnvelope``) pair.**
Before any filesystem mutation, :func:`write_generation` verifies the two
actually describe each other -- ``dataset.digest == envelope.data_digest``,
matching identity, and matching canonicalisation/validation-policy
evidence -- closing the same class of "evidence supplied separately from
the data it describes" defect every earlier unit in this branch has had to
close for its own layer (``ValidatedDataset.build()`` for the frame/report,
``ProvenanceEnvelope.build()`` for the canonicalisation/fetch evidence).
A mismatched pair is rejected with no directory created and no file
written.

**Namespace is read from the envelope only** (``ProvenanceEnvelope`` binds
it, itself derived from ``forced`` at build time, per Unit 6). This module
never infers namespace from which directory happens to contain something --
"do not derive namespace from filesystem after the fact" is the frozen
architecture's own words for exactly this rule (section 10).

**Atomic write sequence** (section 13.3), executed exactly, and only for a
TRUSTED generation does step 8 (the ``CURRENT`` pointer replacement) run at
all -- a FORCED generation is written (steps 1-7) and ``CURRENT`` is never
touched:

    1. create generation directory (fails if it already exists -- an
       existing generation is never overwritten)
    2. write data.parquet
    3. flush + fsync the data file
    4. write manifest.json
    5. flush + fsync the manifest file
    6. fsync the generation directory
    7. fsync the namespace directory
    8. (TRUSTED only) write CURRENT.tmp in the dataset directory, flush +
       fsync it, ``os.replace(CURRENT.tmp, CURRENT)``, then fsync the
       dataset directory

Crash guarantee (unchanged from the architecture, restated here): after a
failure at any point before step 8's ``os.replace``, ``CURRENT`` still
names whatever complete generation it named before this write started (or
does not exist, if none ever completed) -- the new, incomplete generation
directory is left behind as an inert orphan, never selected by anything,
and never cleaned up by this module (garbage collection of orphans is
explicitly a different, later concern). After ``os.replace`` runs,
``CURRENT`` names the newly completed generation; a crash between the
replace and the final parent-directory fsync may still leave either the
old or the new value durable depending on timing, but in both cases the
named generation is complete. This module never deletes a previous
generation.

Every filesystem-touching step is a small, separately-named function
(``_write_data_parquet``, ``_write_text``, ``_fsync_file``, ``_fsync_dir``,
``_atomic_replace``) specifically so tests can monkeypatch one step to
fail, or record call order, without mocking ``write_generation`` itself or
faking real power loss.

**Explicitly out of scope for this unit**: reading/selecting a generation
via ``CURRENT`` ("trusted read"), classifying acquisition status, history
probing, and any migration of the existing ``marketdata.store`` module.
Only writing is implemented here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marketdata.dataset import MarketDataValidity, ValidatedDataset
from marketdata.locator import CurrentPointer, safe_slug
from marketdata.provenance import Namespace, ProvenanceEnvelope

_TRUSTED_DIRNAME = "trusted_generations"
_FORCED_DIRNAME = "forced_generations"
_DATA_FILENAME = "data.parquet"
_MANIFEST_FILENAME = "manifest.json"
_CURRENT_FILENAME = "CURRENT"
_CURRENT_TMP_FILENAME = "CURRENT.tmp"


class GenerationStoreError(RuntimeError):
    """Base class for generation-store domain errors."""


class GenerationConsistencyError(GenerationStoreError):
    """Raised when ``dataset`` and ``envelope`` do not describe each other."""


class GenerationAlreadyExistsError(GenerationStoreError):
    """Raised when the target generation directory already exists."""


@dataclass(frozen=True)
class GenerationWriteResult:
    """What :func:`write_generation` actually did."""

    generation_dir: Path
    namespace: Namespace
    current_updated: bool


# --- small, separately-named filesystem steps (deliberately injectable) ----


def _write_data_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace(tmp_path: Path, target_path: Path) -> None:
    os.replace(str(tmp_path), str(target_path))


# --- consistency check -------------------------------------------------------


def _verify_consistency(dataset: ValidatedDataset, envelope: ProvenanceEnvelope) -> None:
    if not isinstance(dataset, ValidatedDataset):
        raise TypeError(
            f"dataset must be an actual ValidatedDataset instance, got "
            f"{type(dataset).__name__}."
        )
    if not isinstance(envelope, ProvenanceEnvelope):
        raise TypeError(
            f"envelope must be an actual ProvenanceEnvelope instance, got "
            f"{type(envelope).__name__}."
        )
    if dataset.digest != envelope.data_digest:
        raise GenerationConsistencyError(
            f"dataset.digest ({dataset.digest!r}) does not match "
            f"envelope.data_digest ({envelope.data_digest!r}); refusing to "
            "write -- the envelope does not describe this dataset."
        )
    identity_match = (
        dataset.identity.source == envelope.source
        and dataset.identity.symbol == envelope.symbol
        and dataset.identity.resolution == envelope.resolution
    )
    if not identity_match:
        raise GenerationConsistencyError(
            "dataset.identity "
            f"(source={dataset.identity.source!r}, symbol={dataset.identity.symbol!r}, "
            f"resolution={dataset.identity.resolution!r}) does not match envelope "
            f"identity (source={envelope.source!r}, symbol={envelope.symbol!r}, "
            f"resolution={envelope.resolution!r})."
        )
    if dataset.transformations != envelope.transformations:
        raise GenerationConsistencyError(
            "dataset.transformations does not match envelope.transformations."
        )
    if dataset.source_anomalies != envelope.source_anomalies:
        raise GenerationConsistencyError(
            "dataset.source_anomalies does not match envelope.source_anomalies."
        )
    if dataset.source_evidence != envelope.source_evidence:
        raise GenerationConsistencyError(
            "dataset.source_evidence does not match envelope.source_evidence."
        )
    if dataset.validation_policy != envelope.validation_policy:
        raise GenerationConsistencyError(
            "dataset.validation_policy does not match envelope.validation_policy."
        )
    if (
        envelope.namespace is Namespace.TRUSTED
        and dataset.market_data_validity is not MarketDataValidity.VALID
    ):
        raise GenerationConsistencyError(
            f"Refusing to write: dataset.market_data_validity is "
            f"{dataset.market_data_validity.value}, but envelope.namespace is "
            "TRUSTED. A TRUSTED write requires MarketDataValidity == VALID "
            "(frozen architecture section 10/13). An invalid-but-buildable "
            "dataset may only be persisted via an explicit FORCED envelope "
            "(ProvenanceEnvelope.build(dataset, forced=True, "
            "force_reason=<non-empty>)) -- this is never silently converted "
            "here; the operator must choose force explicitly."
        )


# --- location helpers (reuses Unit 7's safe slugs) --------------------------


def _namespace_dirname(namespace: Namespace) -> str:
    return _TRUSTED_DIRNAME if namespace is Namespace.TRUSTED else _FORCED_DIRNAME


def _require_existing_dir(path: Path, label: str) -> Path:
    """``root`` (and only ``root``) must already exist as a directory --
    this module never creates it implicitly. Below ``root``, every missing
    hierarchy component IS created, deliberately, by
    :func:`_ensure_dir_component`.
    """
    if not path.exists():
        raise GenerationStoreError(
            f"{label} {path} does not exist. It must be created explicitly "
            "before writing generations -- this module never creates "
            "arbitrary root ancestors implicitly."
        )
    if not path.is_dir():
        raise GenerationStoreError(f"{label} {path} exists but is not a directory.")
    return path


def _ensure_dir_component(parent: Path, name: str) -> Path:
    """Create ``parent/name`` if it does not already exist, then fsync
    ``parent`` -- a NEWLY-CREATED directory ENTRY is made crash-durable by
    fsyncing its PARENT, never by fsyncing the new directory itself (there
    is nothing inside it yet to make durable that way; the fact that the
    entry now exists in ``parent``'s listing is what fsyncing ``parent``
    commits). If ``parent/name`` already exists, nothing is created and no
    fsync happens here -- its entry's durability was already established
    whenever it was first created; this module does not re-fsync an
    unchanged, pre-existing directory on every write.
    """
    child = parent / name
    if child.exists():
        if not child.is_dir():
            raise GenerationStoreError(f"{child} exists but is not a directory.")
        return child
    child.mkdir()
    _fsync_dir(parent)
    return child


def _ensure_dataset_dir(root: Path, envelope: ProvenanceEnvelope) -> Path:
    """``root/<source_slug>/<symbol_slug>/<resolution_slug>``, creating any
    missing component deliberately -- never one opaque
    ``mkdir(parents=True)`` -- so each newly-created component's parent is
    fsynced immediately, making the new entry crash-durable before the next
    component (or generation persistence) proceeds. ``root`` itself must
    already exist (see :func:`_require_existing_dir`).
    """
    root = _require_existing_dir(Path(root), "root")
    source_dir = _ensure_dir_component(root, safe_slug(envelope.source))
    symbol_dir = _ensure_dir_component(source_dir, safe_slug(envelope.symbol))
    resolution_dir = _ensure_dir_component(symbol_dir, safe_slug(envelope.resolution))
    return resolution_dir


# --- public write API --------------------------------------------------------


def write_generation(
    dataset: ValidatedDataset, envelope: ProvenanceEnvelope, root: Path
) -> GenerationWriteResult:
    """Persist one generation, atomically advancing ``CURRENT`` if (and
    only if) ``envelope.namespace is Namespace.TRUSTED``.

    Raises ``GenerationConsistencyError`` if ``dataset``/``envelope`` do not
    describe each other (checked first, before anything touches the
    filesystem). Raises ``GenerationAlreadyExistsError`` if the target
    generation directory already exists -- an existing generation is never
    overwritten.
    """
    _verify_consistency(dataset, envelope)

    # Component-by-component, parent-fsynced hierarchy creation (section
    # 13.3 correction): root/source/symbol/resolution/namespace. Each
    # newly-created component's parent is fsynced immediately -- see
    # _ensure_dir_component -- rather than one opaque
    # mkdir(parents=True, exist_ok=True), which would leave every newly
    # created intermediate directory entry un-fsynced and therefore not
    # crash-durable.
    dataset_dir = _ensure_dataset_dir(root, envelope)
    namespace_dir = _ensure_dir_component(dataset_dir, _namespace_dirname(envelope.namespace))
    generation_dir = namespace_dir / str(envelope.generation_id)

    if generation_dir.exists():
        raise GenerationAlreadyExistsError(
            f"Generation {envelope.generation_id} already exists at "
            f"{generation_dir}; refusing to overwrite an existing generation."
        )

    try:
        generation_dir.mkdir()
    except FileExistsError as exc:
        raise GenerationAlreadyExistsError(
            f"Generation {envelope.generation_id} already exists at "
            f"{generation_dir}; refusing to overwrite an existing generation."
        ) from exc

    data_path = generation_dir / _DATA_FILENAME
    manifest_path = generation_dir / _MANIFEST_FILENAME

    # Steps 2-3.
    _write_data_parquet(data_path, dataset.frame)
    _fsync_file(data_path)

    # Steps 4-5.
    _write_text(manifest_path, envelope.to_manifest_json())
    _fsync_file(manifest_path)

    # Steps 6-7.
    _fsync_dir(generation_dir)
    _fsync_dir(namespace_dir)

    current_updated = False
    if envelope.namespace is Namespace.TRUSTED:
        # Step 8: FORCED generations never reach this -- CURRENT may only
        # ever name a generation in trusted_generations/ (section 10).
        pointer = CurrentPointer(
            generation_id=envelope.generation_id, integrity_id=envelope.integrity_id
        )
        current_path = dataset_dir / _CURRENT_FILENAME
        current_tmp_path = dataset_dir / _CURRENT_TMP_FILENAME

        _write_text(current_tmp_path, pointer.to_json())
        _fsync_file(current_tmp_path)
        _atomic_replace(current_tmp_path, current_path)
        _fsync_dir(dataset_dir)
        current_updated = True

    return GenerationWriteResult(
        generation_dir=generation_dir,
        namespace=envelope.namespace,
        current_updated=current_updated,
    )
