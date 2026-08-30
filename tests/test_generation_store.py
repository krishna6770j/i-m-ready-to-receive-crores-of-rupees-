"""Atomic generation storage tests.

Frozen architecture sections 10 and 13.3. All tests use temporary
directories only.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest

import marketdata.generation_store as generation_store
from core.timeutils import IST_NAME
from marketdata.dataset import MarketDataValidity, ValidatedDataset, ValidationPolicy
from marketdata.evidence import ChunkResultSnapshot, FetchReportSnapshot
from marketdata.generation_store import (
    GenerationAlreadyExistsError,
    GenerationConsistencyError,
    GenerationWriteResult,
    write_generation,
)
from marketdata.identity import DatasetIdentity
from marketdata.locator import CurrentPointer, safe_slug
from marketdata.provenance import Namespace, ProvenanceEnvelope
from marketdata.schemas import CLOSE, HIGH, LOW, OPEN, TS, VOLUME, empty_ohlcv


def _identity(**overrides) -> DatasetIdentity:
    fields = {"source": "fyers:history", "symbol": "NIFTY", "resolution": "1"}
    fields.update(overrides)
    return DatasetIdentity(**fields)


def _frame(n: int = 3, *, base: float = 100.0) -> pd.DataFrame:
    ts0 = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
    rows = []
    for i in range(n):
        rows.append(
            {
                TS: ts0 + pd.Timedelta(minutes=i),
                OPEN: base + i,
                HIGH: base + i + 5,
                LOW: base + i - 5,
                CLOSE: base + i + 1,
                VOLUME: 1000 + i,
            }
        )
    return pd.DataFrame(rows)


def _dataset(*, base: float = 100.0, **identity_overrides) -> ValidatedDataset:
    return ValidatedDataset.build(_frame(base=base), identity=_identity(**identity_overrides))


def _invalid_frame() -> pd.DataFrame:
    # high < low: an impossible bar -- MarketDataValidity.INVALID.
    return pd.DataFrame(
        {
            TS: [pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)],
            OPEN: [100.0],
            HIGH: [50.0],
            LOW: [200.0],
            CLOSE: [100.0],
            VOLUME: [1000],
        }
    )


def _invalid_dataset(**identity_overrides) -> ValidatedDataset:
    ds = ValidatedDataset.build(_invalid_frame(), identity=_identity(**identity_overrides))
    assert ds.market_data_validity is MarketDataValidity.INVALID
    return ds


def _current_path(result: GenerationWriteResult) -> Path:
    return result.generation_dir.parent.parent / "CURRENT"


def _dataset_dir_for(root: Path, envelope: ProvenanceEnvelope) -> Path:
    return (
        root
        / safe_slug(envelope.source)
        / safe_slug(envelope.symbol)
        / safe_slug(envelope.resolution)
    )


def _fetch_for(ds: ValidatedDataset, *, requested_date: str = "2026-01-01") -> FetchReportSnapshot:
    """A coherent, cross-check-passing FetchReportSnapshot for one dataset
    whose candles all fall on a single calendar date (every dataset built
    from ``_dataset()``/``_invalid_dataset()`` in this file does) --
    REQUESTS_SUCCEEDED, so a TRUSTED write is permitted by Unit 9's
    acquisition gate.
    """
    frame = ds.frame
    first_ts = frame[TS].iloc[0].isoformat() if len(frame) else None
    last_ts = frame[TS].iloc[-1].isoformat() if len(frame) else None
    return FetchReportSnapshot(
        symbol=ds.identity.symbol,
        resolution=ds.identity.resolution,
        requested_from=requested_date,
        requested_to=requested_date,
        chunks=(ChunkResultSnapshot(requested_date, requested_date, len(frame), True, None),),
        total_rows=len(frame),
        first_ts=first_ts,
        last_ts=last_ts,
        duplicate_rows_removed=0,
        conflicting_timestamps=0,
    )


# ---------------------------------------------------------------------------
# basic trusted/forced behaviour
# ---------------------------------------------------------------------------


def test_first_trusted_write_creates_complete_generation_and_current(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    result = write_generation(ds, env, tmp_path)

    assert result.namespace is Namespace.TRUSTED
    assert result.current_updated is True
    assert (result.generation_dir / "data.parquet").exists()
    assert (result.generation_dir / "manifest.json").exists()

    current_path = _current_path(result)
    assert current_path.exists()
    pointer = CurrentPointer.from_json(current_path.read_text())
    assert pointer.generation_id == env.generation_id
    assert pointer.integrity_id == env.integrity_id


def test_second_trusted_write_advances_current_only_after_full_generation_exists(tmp_path):
    identity_kwargs = {}
    ds1 = _dataset(base=100.0, **identity_kwargs)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    result1 = write_generation(ds1, env1, tmp_path)

    ds2 = _dataset(base=200.0, **identity_kwargs)
    env2 = ProvenanceEnvelope.build(ds2, fetch=_fetch_for(ds2))
    result2 = write_generation(ds2, env2, tmp_path)

    current_path = _current_path(result1)
    pointer = CurrentPointer.from_json(current_path.read_text())
    assert pointer.generation_id == env2.generation_id

    # gen1 remains fully intact.
    assert (result1.generation_dir / "data.parquet").exists()
    assert (result1.generation_dir / "manifest.json").exists()
    assert result2.generation_dir != result1.generation_dir


def test_forced_write_creates_forced_generation_and_leaves_current_unchanged(tmp_path):
    ds1 = _dataset(base=100.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    result1 = write_generation(ds1, env1, tmp_path)
    current_path = _current_path(result1)
    before = current_path.read_text()

    ds2 = _dataset(base=300.0)
    env2 = ProvenanceEnvelope.build(ds2, forced=True, force_reason="backfill")
    result2 = write_generation(ds2, env2, tmp_path)

    assert result2.namespace is Namespace.FORCED
    assert result2.current_updated is False
    assert "forced_generations" in str(result2.generation_dir)
    assert current_path.read_text() == before


def test_forced_generation_can_never_become_current_even_as_first_write(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, forced=True, force_reason="manual import")
    result = write_generation(ds, env, tmp_path)
    current_path = _current_path(result)
    assert not current_path.exists()


def test_current_points_to_trusted_namespace_only(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    result = write_generation(ds, env, tmp_path)
    assert "trusted_generations" in str(result.generation_dir)
    assert "forced_generations" not in str(result.generation_dir)


# ---------------------------------------------------------------------------
# consistency checks
# ---------------------------------------------------------------------------


def test_mismatched_dataset_and_envelope_rejected_before_write(tmp_path):
    ds_a = _dataset(base=100.0)
    ds_b = _dataset(base=999.0)
    env_b = ProvenanceEnvelope.build(ds_b)  # built from ds_b, not ds_a

    with pytest.raises(GenerationConsistencyError):
        write_generation(ds_a, env_b, tmp_path)

    # No filesystem mutation happened at all.
    assert list(tmp_path.iterdir()) == []


def test_data_digest_mismatch_rejected_before_any_write(tmp_path):
    ds_a = _dataset(base=100.0)
    ds_b = _dataset(base=200.0)
    env_b = ProvenanceEnvelope.build(ds_b)
    assert ds_a.digest != env_b.data_digest

    with pytest.raises(GenerationConsistencyError):
        write_generation(ds_a, env_b, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_generation_identity_mismatch_rejected(tmp_path):
    ds = _dataset(symbol="NIFTY")
    env_other_symbol = ProvenanceEnvelope.build(_dataset(symbol="SBIN"))
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds, env_other_symbol, tmp_path)


def test_fake_dataset_rejected():
    class FakeDataset:
        pass

    with pytest.raises(TypeError):
        write_generation(FakeDataset(), ProvenanceEnvelope.build(_dataset()), Path("/tmp/x"))


# ---------------------------------------------------------------------------
# path safety
# ---------------------------------------------------------------------------


def test_raw_dangerous_identifiers_stay_inside_safe_slug_paths(tmp_path):
    ds = _dataset(source="../../etc", symbol="../passwd", resolution="../1")
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    result = write_generation(ds, env, tmp_path)

    resolved = result.generation_dir.resolve()
    assert str(resolved).startswith(str(tmp_path.resolve()))
    assert ".." not in result.generation_dir.parts


# ---------------------------------------------------------------------------
# no overwrite
# ---------------------------------------------------------------------------


def test_existing_generation_never_overwritten(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    result = write_generation(ds, env, tmp_path)

    original_manifest = (result.generation_dir / "manifest.json").read_text()

    with pytest.raises(GenerationAlreadyExistsError):
        write_generation(ds, env, tmp_path)

    assert (result.generation_dir / "manifest.json").read_text() == original_manifest


# ---------------------------------------------------------------------------
# manifest/data both present before CURRENT update
# ---------------------------------------------------------------------------


def test_manifest_and_data_both_present_before_current_written(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    result = write_generation(ds, env, tmp_path)
    assert (result.generation_dir / "data.parquet").exists()
    assert (result.generation_dir / "manifest.json").exists()
    assert _current_path(result).exists()


# ---------------------------------------------------------------------------
# exact write-sequence / ordering
# ---------------------------------------------------------------------------


def test_write_sequence_order_matches_architecture(tmp_path, monkeypatch):
    calls: list[tuple] = []

    orig_write_parquet = generation_store._write_data_parquet
    orig_write_text = generation_store._write_text
    orig_fsync_file = generation_store._fsync_file
    orig_fsync_dir = generation_store._fsync_dir
    orig_replace = generation_store._atomic_replace

    def write_parquet(path, frame):
        calls.append(("write_data_parquet", Path(path).name))
        return orig_write_parquet(path, frame)

    def write_text(path, text):
        calls.append(("write_text", Path(path).name))
        return orig_write_text(path, text)

    def fsync_file(path):
        calls.append(("fsync_file", Path(path).name))
        return orig_fsync_file(path)

    def fsync_dir(path):
        calls.append(("fsync_dir", Path(path).name))
        return orig_fsync_dir(path)

    def replace(tmp, target):
        calls.append(("replace", None))
        return orig_replace(tmp, target)

    monkeypatch.setattr(generation_store, "_write_data_parquet", write_parquet)
    monkeypatch.setattr(generation_store, "_write_text", write_text)
    monkeypatch.setattr(generation_store, "_fsync_file", fsync_file)
    monkeypatch.setattr(generation_store, "_fsync_dir", fsync_dir)
    monkeypatch.setattr(generation_store, "_atomic_replace", replace)

    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    result = write_generation(ds, env, tmp_path)

    gid = str(env.generation_id)
    source_slug = safe_slug(env.source)
    symbol_slug = safe_slug(env.symbol)
    resolution_slug = safe_slug(env.resolution)

    assert calls == [
        # Hierarchy creation: every component here is newly created (empty
        # tmp_path), so each is followed immediately by an fsync of its
        # parent (section 13.3 correction).
        ("fsync_dir", tmp_path.name),  # created source_dir -> fsync(root)
        ("fsync_dir", source_slug),  # created symbol_dir -> fsync(source_dir)
        ("fsync_dir", symbol_slug),  # created resolution_dir -> fsync(symbol_dir)
        ("fsync_dir", resolution_slug),  # created namespace_dir -> fsync(resolution_dir)
        ("write_data_parquet", "data.parquet"),
        ("fsync_file", "data.parquet"),
        ("write_text", "manifest.json"),
        ("fsync_file", "manifest.json"),
        ("fsync_dir", gid),
        ("fsync_dir", "trusted_generations"),
        ("write_text", "CURRENT.tmp"),
        ("fsync_file", "CURRENT.tmp"),
        ("replace", None),
        ("fsync_dir", resolution_slug),
    ]


def test_forced_write_sequence_has_no_current_steps(tmp_path, monkeypatch):
    calls: list[str] = []
    orig_replace = generation_store._atomic_replace

    def replace(tmp, target):
        calls.append("replace")
        return orig_replace(tmp, target)

    monkeypatch.setattr(generation_store, "_atomic_replace", replace)

    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, forced=True, force_reason="x")
    write_generation(ds, env, tmp_path)

    assert calls == []


# ---------------------------------------------------------------------------
# failure injection
# ---------------------------------------------------------------------------


def test_failed_parquet_write_leaves_current_unchanged(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    def boom(path, frame):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(generation_store, "_write_data_parquet", boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before
    # Orphan generation directory exists but has no data file.
    orphan_dir = _dataset_dir_for(tmp_path, env1) / "trusted_generations" / str(env1.generation_id)
    assert orphan_dir.exists()
    assert not (orphan_dir / "data.parquet").exists()


def test_failed_manifest_write_leaves_current_unchanged(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    orig_write_text = generation_store._write_text

    def selective_boom(path, text):
        if Path(path).name == "manifest.json":
            raise OSError("simulated disk failure")
        return orig_write_text(path, text)

    monkeypatch.setattr(generation_store, "_write_text", selective_boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before
    orphan_dir = _dataset_dir_for(tmp_path, env1) / "trusted_generations" / str(env1.generation_id)
    assert (orphan_dir / "data.parquet").exists()
    assert not (orphan_dir / "manifest.json").exists()


def test_failed_current_tmp_write_leaves_current_unchanged(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    orig_write_text = generation_store._write_text

    def selective_boom(path, text):
        if Path(path).name == "CURRENT.tmp":
            raise OSError("simulated disk failure")
        return orig_write_text(path, text)

    monkeypatch.setattr(generation_store, "_write_text", selective_boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before
    # The new generation itself is fully written (data + manifest); only
    # the CURRENT advance failed.
    orphan_dir = _dataset_dir_for(tmp_path, env1) / "trusted_generations" / str(env1.generation_id)
    assert (orphan_dir / "data.parquet").exists()
    assert (orphan_dir / "manifest.json").exists()


def test_failed_os_replace_leaves_current_unchanged(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    def boom(tmp, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(generation_store, "_atomic_replace", boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before


def test_orphan_generation_is_inert(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)

    def boom(tmp, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(generation_store, "_atomic_replace", boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    # CURRENT still resolves to gen0, not the orphan gen1.
    pointer = CurrentPointer.from_json(_current_path(result0).read_text())
    assert pointer.generation_id == env0.generation_id
    assert pointer.generation_id != env1.generation_id

    # The orphan is still there on disk (not cleaned up), but nothing
    # references it.
    orphan_dir = _dataset_dir_for(tmp_path, env1) / "trusted_generations" / str(env1.generation_id)
    assert orphan_dir.exists()


def test_previous_trusted_generation_remains_intact_after_failed_new_write(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    original_data = (result0.generation_dir / "data.parquet").read_bytes()
    original_manifest = (result0.generation_dir / "manifest.json").read_text()

    def boom(path, frame):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(generation_store, "_write_data_parquet", boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert (result0.generation_dir / "data.parquet").read_bytes() == original_data
    assert (result0.generation_dir / "manifest.json").read_text() == original_manifest
    pointer = CurrentPointer.from_json(_current_path(result0).read_text())
    assert pointer.generation_id == env0.generation_id


# ---------------------------------------------------------------------------
# hierarchy durability (manager correction)
# ---------------------------------------------------------------------------


def test_missing_hierarchy_created_component_by_component(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    write_generation(ds, env, tmp_path)

    source_dir = tmp_path / safe_slug(env.source)
    symbol_dir = source_dir / safe_slug(env.symbol)
    resolution_dir = symbol_dir / safe_slug(env.resolution)
    assert source_dir.is_dir()
    assert symbol_dir.is_dir()
    assert resolution_dir.is_dir()
    assert (resolution_dir / "trusted_generations").is_dir()


def test_parent_fsync_follows_each_newly_created_hierarchy_component(tmp_path, monkeypatch):
    fsynced_dirs: list[str] = []
    orig_fsync_dir = generation_store._fsync_dir

    def fsync_dir(path):
        fsynced_dirs.append(Path(path).name)
        return orig_fsync_dir(path)

    monkeypatch.setattr(generation_store, "_fsync_dir", fsync_dir)

    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    write_generation(ds, env, tmp_path)

    source_slug = safe_slug(env.source)
    symbol_slug = safe_slug(env.symbol)
    resolution_slug = safe_slug(env.resolution)

    # The first four fsync_dir calls are the hierarchy-creation parent
    # fsyncs, one per newly-created component, in creation order.
    assert fsynced_dirs[:4] == [tmp_path.name, source_slug, symbol_slug, resolution_slug]


def test_existing_hierarchy_does_not_require_recreation(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    write_generation(ds0, env0, tmp_path)

    fsynced_dirs: list[str] = []
    orig_fsync_dir = generation_store._fsync_dir

    def fsync_dir(path):
        fsynced_dirs.append(Path(path).name)
        return orig_fsync_dir(path)

    monkeypatch.setattr(generation_store, "_fsync_dir", fsync_dir)

    ds1 = _dataset(base=200.0)  # same identity -> hierarchy already exists
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    write_generation(ds1, env1, tmp_path)

    source_slug = safe_slug(env1.source)
    symbol_slug = safe_slug(env1.symbol)
    resolution_slug = safe_slug(env1.resolution)
    gid = str(env1.generation_id)

    # No hierarchy-creation fsyncs at all: every component already existed,
    # so _ensure_dir_component short-circuits before ever calling
    # _fsync_dir for source/symbol/resolution/namespace.
    assert tmp_path.name not in fsynced_dirs
    assert source_slug not in fsynced_dirs
    assert symbol_slug not in fsynced_dirs
    # Only the ordinary per-write fsyncs remain: generation dir (step 6),
    # namespace dir (step 7, unconditional -- not part of hierarchy
    # creation), and the dataset-dir fsync after CURRENT replacement (step 8).
    assert fsynced_dirs == [gid, "trusted_generations", resolution_slug]


def test_forced_first_write_gets_same_durable_hierarchy_treatment(tmp_path, monkeypatch):
    fsynced_dirs: list[str] = []
    orig_fsync_dir = generation_store._fsync_dir

    def fsync_dir(path):
        fsynced_dirs.append(Path(path).name)
        return orig_fsync_dir(path)

    monkeypatch.setattr(generation_store, "_fsync_dir", fsync_dir)

    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, forced=True, force_reason="backfill")
    write_generation(ds, env, tmp_path)

    source_slug = safe_slug(env.source)
    symbol_slug = safe_slug(env.symbol)
    resolution_slug = safe_slug(env.resolution)
    assert fsynced_dirs[:4] == [tmp_path.name, source_slug, symbol_slug, resolution_slug]


def test_missing_root_raises_clear_error_without_implicit_creation(tmp_path):
    missing_root = tmp_path / "does_not_exist_yet"
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))

    with pytest.raises(generation_store.GenerationStoreError):
        write_generation(ds, env, missing_root)

    assert not missing_root.exists()


def test_root_is_a_file_raises_clear_error(tmp_path):
    file_root = tmp_path / "not_a_directory"
    file_root.write_text("surprise")
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))

    with pytest.raises(generation_store.GenerationStoreError):
        write_generation(ds, env, file_root)


# ---------------------------------------------------------------------------
# hierarchy / component fsync failure injection
# ---------------------------------------------------------------------------


def test_hierarchy_parent_fsync_failure_leaves_no_current(tmp_path, monkeypatch):
    def boom(path):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(generation_store, "_fsync_dir", boom)

    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    with pytest.raises(OSError):
        write_generation(ds, env, tmp_path)

    current_path = tmp_path / safe_slug(env.source) / safe_slug(env.symbol) / safe_slug(env.resolution) / "CURRENT"
    assert not current_path.exists()


def test_data_file_fsync_failure_leaves_previous_current_unchanged(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    orig_fsync_file = generation_store._fsync_file

    def selective_boom(path):
        if Path(path).name == "data.parquet":
            raise OSError("simulated fsync failure")
        return orig_fsync_file(path)

    monkeypatch.setattr(generation_store, "_fsync_file", selective_boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before


def test_manifest_fsync_failure_leaves_previous_current_unchanged(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    orig_fsync_file = generation_store._fsync_file

    def selective_boom(path):
        if Path(path).name == "manifest.json":
            raise OSError("simulated fsync failure")
        return orig_fsync_file(path)

    monkeypatch.setattr(generation_store, "_fsync_file", selective_boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before


def test_generation_dir_fsync_failure_leaves_previous_current_unchanged(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    orig_fsync_dir = generation_store._fsync_dir

    def selective_boom(path):
        if Path(path).name == str(env1.generation_id):
            raise OSError("simulated fsync failure")
        return orig_fsync_dir(path)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    monkeypatch.setattr(generation_store, "_fsync_dir", selective_boom)

    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before


def test_namespace_dir_fsync_failure_leaves_previous_current_unchanged(tmp_path, monkeypatch):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    orig_fsync_dir = generation_store._fsync_dir
    calls_to_namespace = {"count": 0}

    def selective_boom(path):
        if Path(path).name == "trusted_generations":
            raise OSError("simulated fsync failure")
        return orig_fsync_dir(path)

    monkeypatch.setattr(generation_store, "_fsync_dir", selective_boom)

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_for(ds1))
    with pytest.raises(OSError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before


# ---------------------------------------------------------------------------
# trust gate: MarketDataValidity.INVALID must never reach TRUSTED storage
# ---------------------------------------------------------------------------


def test_invalid_dataset_with_trusted_envelope_rejected(tmp_path):
    ds = _invalid_dataset()
    env = ProvenanceEnvelope.build(ds, forced=False)
    assert env.namespace is Namespace.TRUSTED

    with pytest.raises(GenerationConsistencyError):
        write_generation(ds, env, tmp_path)


def test_invalid_trusted_rejection_happens_before_any_filesystem_mutation(tmp_path):
    ds = _invalid_dataset()
    env = ProvenanceEnvelope.build(ds, forced=False)

    with pytest.raises(GenerationConsistencyError):
        write_generation(ds, env, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_existing_current_byte_identical_after_rejected_invalid_trusted_write(tmp_path):
    ds_valid = _dataset()
    env_valid = ProvenanceEnvelope.build(ds_valid, fetch=_fetch_for(ds_valid))
    result_valid = write_generation(ds_valid, env_valid, tmp_path)
    current_before = _current_path(result_valid).read_text()

    ds_invalid = _invalid_dataset()
    env_invalid = ProvenanceEnvelope.build(ds_invalid, forced=False)
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds_invalid, env_invalid, tmp_path)

    assert _current_path(result_valid).read_text() == current_before


def test_invalid_dataset_with_explicit_forced_envelope_persisted_under_forced(tmp_path):
    ds = _invalid_dataset()
    env = ProvenanceEnvelope.build(ds, forced=True, force_reason="forensic inspection")
    assert env.namespace is Namespace.FORCED

    result = write_generation(ds, env, tmp_path)
    assert result.namespace is Namespace.FORCED
    assert (result.generation_dir / "data.parquet").exists()
    assert (result.generation_dir / "manifest.json").exists()


def test_forced_invalid_write_never_changes_current(tmp_path):
    ds_valid = _dataset()
    env_valid = ProvenanceEnvelope.build(ds_valid, fetch=_fetch_for(ds_valid))
    result_valid = write_generation(ds_valid, env_valid, tmp_path)
    current_before = _current_path(result_valid).read_text()

    ds_invalid = _invalid_dataset()
    env_invalid = ProvenanceEnvelope.build(ds_invalid, forced=True, force_reason="forensic")
    result_invalid = write_generation(ds_invalid, env_invalid, tmp_path)

    assert result_invalid.current_updated is False
    assert _current_path(result_valid).read_text() == current_before


def test_valid_trusted_dataset_still_writes_normally(tmp_path):
    ds = _dataset()
    assert ds.market_data_validity is MarketDataValidity.VALID
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    result = write_generation(ds, env, tmp_path)
    assert result.namespace is Namespace.TRUSTED
    assert result.current_updated is True


def test_validation_policy_causing_invalid_also_triggers_the_gate(tmp_path):
    # A frame that is OHLC-valid under default validation but INVALID under
    # a stricter policy (a large gap flagged as EXCESSIVE_DATA_GAP).
    ts0 = pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)
    raw = pd.DataFrame(
        {
            TS: [ts0, ts0 + pd.Timedelta(minutes=1), ts0 + pd.Timedelta(days=10)],
            OPEN: [100.0, 101.0, 102.0],
            HIGH: [105.0, 106.0, 107.0],
            LOW: [95.0, 96.0, 97.0],
            CLOSE: [101.0, 102.0, 103.0],
            VOLUME: [1000, 1001, 1002],
        }
    )
    lenient_policy = ValidationPolicy(expected_interval_minutes=1)
    strict_policy = ValidationPolicy(expected_interval_minutes=1, max_session_gap_days=1.0)

    ds_lenient = ValidatedDataset.build(raw, identity=_identity(), validation_policy=lenient_policy)
    ds_strict = ValidatedDataset.build(raw, identity=_identity(), validation_policy=strict_policy)
    assert ds_lenient.market_data_validity is MarketDataValidity.VALID
    assert ds_strict.market_data_validity is MarketDataValidity.INVALID

    # This dataset spans two calendar dates (Jan 1 and Jan 11), so the
    # generic single-day _fetch_for() helper does not apply -- build a
    # matching two-chunk fetch covering the full requested span.
    fetch_lenient = FetchReportSnapshot(
        symbol="NIFTY", resolution="1",
        requested_from="2026-01-01", requested_to="2026-01-11",
        chunks=(
            ChunkResultSnapshot("2026-01-01", "2026-01-10", 2, True, None),
            ChunkResultSnapshot("2026-01-11", "2026-01-11", 1, True, None),
        ),
        total_rows=3,
        first_ts=ds_lenient.frame[TS].iloc[0].isoformat(),
        last_ts=ds_lenient.frame[TS].iloc[-1].isoformat(),
        duplicate_rows_removed=0, conflicting_timestamps=0,
    )
    env_lenient = ProvenanceEnvelope.build(
        ds_lenient, fetch=fetch_lenient, generation_id=uuid.uuid4()
    )
    # lenient writes fine
    write_generation(ds_lenient, env_lenient, tmp_path)

    env_strict = ProvenanceEnvelope.build(ds_strict, forced=False, generation_id=uuid.uuid4())
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds_strict, env_strict, tmp_path)


def test_adversarial_valid_then_invalid_trusted_attempt_leaves_current_and_disk_untouched(tmp_path):
    # 1. write valid trusted generation A
    ds_a = _dataset(base=100.0)
    env_a = ProvenanceEnvelope.build(ds_a, fetch=_fetch_for(ds_a))
    result_a = write_generation(ds_a, env_a, tmp_path)
    current_before = _current_path(result_a).read_text()

    trusted_dir = _dataset_dir_for(tmp_path, env_a) / "trusted_generations"
    generations_before = set(trusted_dir.iterdir())

    # 2. build invalid dataset B
    ds_b = _invalid_dataset()
    env_b = ProvenanceEnvelope.build(ds_b, forced=False)

    # 3. attempt trusted write B
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds_b, env_b, tmp_path)

    # 4. assertions
    assert _current_path(result_a).read_text() == current_before
    b_generation_dir = trusted_dir / str(env_b.generation_id)
    assert not b_generation_dir.exists()
    assert set(trusted_dir.iterdir()) == generations_before


# ---------------------------------------------------------------------------
# Unit 9: acquisition-status trust gate
# ---------------------------------------------------------------------------


def _fetch_with_status(ds: ValidatedDataset, status: str, *, requested_date: str = "2026-01-01") -> FetchReportSnapshot:
    frame = ds.frame
    first_ts = frame[TS].iloc[0].isoformat() if len(frame) else None
    last_ts = frame[TS].iloc[-1].isoformat() if len(frame) else None
    if status == "FAILED":
        return FetchReportSnapshot(
            symbol=ds.identity.symbol, resolution=ds.identity.resolution,
            requested_from=requested_date, requested_to=requested_date,
            chunks=(ChunkResultSnapshot(requested_date, requested_date, 0, False, "boom"),),
            total_rows=0, first_ts=None, last_ts=None,
            duplicate_rows_removed=0, conflicting_timestamps=0,
        )
    if status == "EMPTY":
        return FetchReportSnapshot(
            symbol=ds.identity.symbol, resolution=ds.identity.resolution,
            requested_from=requested_date, requested_to=requested_date,
            chunks=(ChunkResultSnapshot(requested_date, requested_date, 0, True, None),),
            total_rows=0, first_ts=None, last_ts=None,
            duplicate_rows_removed=0, conflicting_timestamps=0,
        )
    if status == "PARTIAL":
        # Two-day request: one ok chunk (matching this dataset's actual
        # rows), one failed chunk -- overall PARTIAL.
        return FetchReportSnapshot(
            symbol=ds.identity.symbol, resolution=ds.identity.resolution,
            requested_from=requested_date, requested_to="2026-01-02",
            chunks=(
                ChunkResultSnapshot(requested_date, requested_date, len(frame), True, None),
                ChunkResultSnapshot("2026-01-02", "2026-01-02", 0, False, "rate limited"),
            ),
            total_rows=len(frame), first_ts=first_ts, last_ts=last_ts,
            duplicate_rows_removed=0, conflicting_timestamps=0,
        )
    raise ValueError(status)


def test_only_succeeded_status_may_update_current(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_for(ds))
    result = write_generation(ds, env, tmp_path)
    assert result.current_updated is True


def test_unknown_status_trusted_write_rejected_before_filesystem_mutation(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, forced=False)  # fetch=None -> UNKNOWN
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds, env, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_failed_status_trusted_write_rejected_before_filesystem_mutation(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_with_status(ds, "FAILED"))
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds, env, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_empty_status_trusted_write_rejected_before_filesystem_mutation(tmp_path):
    ds = ValidatedDataset.build(empty_ohlcv(), identity=_identity())
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_with_status(ds, "EMPTY"))
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds, env, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_partial_status_trusted_write_rejected_before_filesystem_mutation(tmp_path):
    ds = _dataset()
    env = ProvenanceEnvelope.build(ds, fetch=_fetch_with_status(ds, "PARTIAL"))
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds, env, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_all_four_non_succeeded_statuses_persistable_through_forced_storage(tmp_path):
    for status in ("FAILED", "EMPTY", "PARTIAL"):
        # FAILED/EMPTY both correspond to zero actual rows (no chunk
        # contributed data); PARTIAL's one successful chunk must match a
        # real, non-empty dataset.
        ds = (
            _dataset(base=100.0 + hash(status) % 50)
            if status == "PARTIAL"
            else ValidatedDataset.build(empty_ohlcv(), identity=_identity())
        )
        fetch = _fetch_with_status(ds, status)
        env = ProvenanceEnvelope.build(ds, fetch=fetch, forced=True, force_reason=f"forensic {status}")
        result = write_generation(ds, env, tmp_path)
        assert result.namespace is Namespace.FORCED
        assert result.current_updated is False

    # UNKNOWN (fetch=None) forced write.
    ds_unknown = _dataset(base=999.0)
    env_unknown = ProvenanceEnvelope.build(ds_unknown, forced=True, force_reason="forensic UNKNOWN")
    result_unknown = write_generation(ds_unknown, env_unknown, tmp_path)
    assert result_unknown.namespace is Namespace.FORCED
    assert result_unknown.current_updated is False


def test_previous_current_unchanged_after_rejected_acquisition_status(tmp_path):
    ds0 = _dataset(base=100.0)
    env0 = ProvenanceEnvelope.build(ds0, fetch=_fetch_for(ds0))
    result0 = write_generation(ds0, env0, tmp_path)
    current_before = _current_path(result0).read_text()

    ds1 = _dataset(base=200.0)
    env1 = ProvenanceEnvelope.build(ds1, fetch=_fetch_with_status(ds1, "FAILED"))
    with pytest.raises(GenerationConsistencyError):
        write_generation(ds1, env1, tmp_path)

    assert _current_path(result0).read_text() == current_before
