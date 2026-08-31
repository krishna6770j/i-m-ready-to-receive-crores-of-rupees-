"""Research-readiness policy boundary tests.

``ResearchDataPolicy``/``ResearchReadyDataset`` (``marketdata.research``):
a ``TrustedDataset`` (sound stored artifact) becomes research-ready only
after satisfying an explicit, per-experiment policy.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.dataset import ValidatedDataset
from marketdata.evidence import ChunkResultSnapshot, FetchReportSnapshot
from marketdata.generation_store import write_generation
from marketdata.identity import DatasetIdentity
from marketdata.provenance import Namespace, ProvenanceEnvelope
from marketdata.research import ResearchDataPolicy, ResearchPolicyError, ResearchReadyDataset
from marketdata.schemas import CLOSE, HIGH, LOW, OPEN, TS, VOLUME
from marketdata.trusted_reader import TrustedDataset, read_trusted, read_unverified


def _identity(**overrides) -> DatasetIdentity:
    fields = {"source": "fyers:history", "symbol": "NIFTY", "resolution": "1"}
    fields.update(overrides)
    return DatasetIdentity(**fields)


def _rows_for_date(date_str: str, n: int, base: float) -> list[dict]:
    ts0 = pd.Timestamp(f"{date_str} 09:15", tz=IST_NAME)
    return [
        {
            TS: ts0 + pd.Timedelta(minutes=i),
            OPEN: base + i,
            HIGH: base + i + 5,
            LOW: base + i - 5,
            CLOSE: base + i + 1,
            VOLUME: 1000 + i,
        }
        for i in range(n)
    ]


def _frame_for_dates(dates: list[str], n_per_day: int = 3, *, base: float = 100.0) -> pd.DataFrame:
    rows: list[dict] = []
    for date_str in dates:
        rows.extend(_rows_for_date(date_str, n_per_day, base))
    return pd.DataFrame(rows)


def _dataset(dates: list[str], n_per_day: int = 3, *, base: float = 100.0, **id_overrides) -> ValidatedDataset:
    return ValidatedDataset.build(
        _frame_for_dates(dates, n_per_day, base=base), identity=_identity(**id_overrides)
    )


def _fetch_covering(ds: ValidatedDataset, requested_from: str, requested_to: str) -> FetchReportSnapshot:
    frame = ds.frame
    return FetchReportSnapshot(
        symbol=ds.identity.symbol,
        resolution=ds.identity.resolution,
        requested_from=requested_from,
        requested_to=requested_to,
        chunks=(ChunkResultSnapshot(requested_from, requested_to, len(frame), True, None),),
        total_rows=len(frame),
        first_ts=frame[TS].iloc[0].isoformat(),
        last_ts=frame[TS].iloc[-1].isoformat(),
        duplicate_rows_removed=0,
        conflicting_timestamps=0,
    )


def _write_and_read_trusted(root, ds: ValidatedDataset, fetch: FetchReportSnapshot) -> TrustedDataset:
    env = ProvenanceEnvelope.build(ds, fetch=fetch)
    write_generation(ds, env, root)
    return read_trusted(
        root,
        source=ds.identity.source,
        symbol=ds.identity.symbol,
        resolution=ds.identity.resolution,
    )


def _one_day_trusted(tmp_path, **id_overrides) -> TrustedDataset:
    ds = _dataset(["2026-01-01"], **id_overrides)
    fetch = _fetch_covering(ds, "2026-01-01", "2026-01-01")
    return _write_and_read_trusted(tmp_path, ds, fetch)


# ---------------------------------------------------------------------------
# Happy path / no requirements
# ---------------------------------------------------------------------------


def test_no_requirements_any_trusted_dataset_becomes_research_ready(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    ready = ResearchReadyDataset.build(trusted, ResearchDataPolicy())
    assert ready.identity == trusted.identity
    assert ready.generation_id == trusted.generation_id
    assert ready.policy == ResearchDataPolicy()
    assert len(ready.frame) == 3


# ---------------------------------------------------------------------------
# Identity requirements
# ---------------------------------------------------------------------------


def test_expected_source_match_succeeds(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    ResearchReadyDataset.build(trusted, ResearchDataPolicy(expected_source="fyers:history"))


def test_expected_source_mismatch_fails(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    with pytest.raises(ResearchPolicyError):
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(expected_source="other:source"))


def test_expected_symbol_match_succeeds(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    ResearchReadyDataset.build(trusted, ResearchDataPolicy(expected_symbol="NIFTY"))


def test_expected_symbol_mismatch_fails(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    with pytest.raises(ResearchPolicyError):
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(expected_symbol="BANKNIFTY"))


def test_expected_resolution_match_succeeds(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    ResearchReadyDataset.build(trusted, ResearchDataPolicy(expected_resolution="1"))


def test_expected_resolution_mismatch_fails(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    with pytest.raises(ResearchPolicyError):
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(expected_resolution="5"))


# ---------------------------------------------------------------------------
# min_distinct_observed_dates
# ---------------------------------------------------------------------------


def test_minimum_distinct_dates_pass(tmp_path):
    ds = _dataset(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
    fetch = _fetch_covering(ds, "2026-01-01", "2026-01-10")
    trusted = _write_and_read_trusted(tmp_path, ds, fetch)
    ResearchReadyDataset.build(trusted, ResearchDataPolicy(min_distinct_observed_dates=5))


def test_minimum_distinct_dates_fail(tmp_path):
    ds = _dataset(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
    fetch = _fetch_covering(ds, "2026-01-01", "2026-01-10")
    trusted = _write_and_read_trusted(tmp_path, ds, fetch)
    with pytest.raises(ResearchPolicyError):
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(min_distinct_observed_dates=6))


def test_critical_sparse_data_rejected_when_minimum_larger(tmp_path):
    """The exact failure class this boundary exists to prevent: a
    technically sound, fully TRUSTED artifact covering an entire year's
    request but observing only a single date (three candles) must be
    rejected by a policy requiring meaningfully more coverage.
    """
    ds = _dataset(["2026-01-01"])  # 3 candles, one single date
    fetch = _fetch_covering(ds, "2026-01-01", "2026-12-31")  # requested a full year
    trusted = _write_and_read_trusted(tmp_path, ds, fetch)
    assert trusted.observed_data_coverage.distinct_observed_dates == 1
    with pytest.raises(ResearchPolicyError) as excinfo:
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(min_distinct_observed_dates=50))
    assert any("min_distinct_observed_dates" in reason for reason in excinfo.value.unmet_requirements)


# ---------------------------------------------------------------------------
# min_requested_window_fraction
# ---------------------------------------------------------------------------


def test_minimum_requested_window_fraction_pass(tmp_path):
    ds = _dataset(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
    fetch = _fetch_covering(ds, "2026-01-01", "2026-01-10")  # ratio = 5/10 = 0.5
    trusted = _write_and_read_trusted(tmp_path, ds, fetch)
    assert trusted.requested_window_comparison.observed_distinct_dates_ratio == 0.5
    ResearchReadyDataset.build(trusted, ResearchDataPolicy(min_requested_window_fraction=0.4))


def test_minimum_requested_window_fraction_fail(tmp_path):
    ds = _dataset(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
    fetch = _fetch_covering(ds, "2026-01-01", "2026-01-10")  # ratio = 0.5
    trusted = _write_and_read_trusted(tmp_path, ds, fetch)
    with pytest.raises(ResearchPolicyError):
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(min_requested_window_fraction=0.6))


def test_missing_requested_window_comparison_with_required_fraction_fails(tmp_path, monkeypatch):
    trusted = _one_day_trusted(tmp_path)
    monkeypatch.setattr(
        type(trusted), "requested_window_comparison", property(lambda self: None)
    )
    with pytest.raises(ResearchPolicyError) as excinfo:
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(min_requested_window_fraction=0.1))
    assert any("unavailable" in reason for reason in excinfo.value.unmet_requirements)


# ---------------------------------------------------------------------------
# require_pristine_source_order
# ---------------------------------------------------------------------------


def test_pristine_source_order_required_and_pristine_passes(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    assert trusted.source_evidence.timestamps_sorted is True
    ResearchReadyDataset.build(trusted, ResearchDataPolicy(require_pristine_source_order=True))


def test_pristine_source_order_required_but_source_unsorted_fails(tmp_path):
    # Build from a RAW frame whose rows arrive out of timestamp order --
    # canonicalise() still produces a valid, sorted canonical frame (a
    # legitimate TRUSTED artifact), but source_evidence.timestamps_sorted
    # records the ORIGINAL, unsorted arrival order as a fact.
    day1 = _rows_for_date("2026-01-01", 3, 100.0)
    day2 = _rows_for_date("2026-01-02", 3, 100.0)
    raw = pd.DataFrame(day2 + day1)  # day2 arrives before day1 -> unsorted source
    ds = ValidatedDataset.build(raw, identity=_identity())
    assert ds.source_evidence.timestamps_sorted is False
    fetch = _fetch_covering(ds, "2026-01-01", "2026-01-02")
    trusted = _write_and_read_trusted(tmp_path, ds, fetch)
    assert trusted.source_evidence.timestamps_sorted is False

    # Still a perfectly sound TrustedDataset -- the research policy is what
    # rejects it, not trust certification itself.
    with pytest.raises(ResearchPolicyError) as excinfo:
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(require_pristine_source_order=True))
    assert any("pristine_source_order" in reason for reason in excinfo.value.unmet_requirements)

    # Without that requirement, the same TrustedDataset is research-ready.
    ResearchReadyDataset.build(trusted, ResearchDataPolicy())


# ---------------------------------------------------------------------------
# Multiple failures reported together
# ---------------------------------------------------------------------------


def test_multiple_failed_requirements_reported_together(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    policy = ResearchDataPolicy(
        expected_symbol="BANKNIFTY",
        expected_resolution="5",
        min_distinct_observed_dates=99,
    )
    with pytest.raises(ResearchPolicyError) as excinfo:
        ResearchReadyDataset.build(trusted, policy)
    assert len(excinfo.value.unmet_requirements) == 3


# ---------------------------------------------------------------------------
# Non-bypassable type boundary
# ---------------------------------------------------------------------------


def test_raw_dataframe_rejected(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    with pytest.raises(TypeError):
        ResearchReadyDataset.build(trusted.frame, ResearchDataPolicy())


def test_validated_dataset_rejected(tmp_path):
    ds = _dataset(["2026-01-01"])
    with pytest.raises(TypeError):
        ResearchReadyDataset.build(ds, ResearchDataPolicy())


def test_unverified_dataset_rejected(tmp_path):
    ds = _dataset(["2026-01-01"])
    fetch = _fetch_covering(ds, "2026-01-01", "2026-01-01")
    env = ProvenanceEnvelope.build(ds, fetch=fetch)
    write_generation(ds, env, tmp_path)
    unverified = read_unverified(
        tmp_path,
        source="fyers:history",
        symbol="NIFTY",
        resolution="1",
        namespace=Namespace.TRUSTED,
        generation_id=env.generation_id,
    )
    with pytest.raises(TypeError):
        ResearchReadyDataset.build(unverified, ResearchDataPolicy())


def test_fake_duck_typed_trusted_object_rejected(tmp_path):
    real_trusted = _one_day_trusted(tmp_path)

    class FakeTrusted:
        identity = real_trusted.identity
        observed_data_coverage = real_trusted.observed_data_coverage
        requested_window_comparison = real_trusted.requested_window_comparison
        source_evidence = real_trusted.source_evidence
        frame = real_trusted.frame

    with pytest.raises(TypeError):
        ResearchReadyDataset.build(FakeTrusted(), ResearchDataPolicy())


def test_policy_must_be_actual_research_data_policy(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    with pytest.raises(TypeError):
        ResearchReadyDataset.build(trusted, object())


# ---------------------------------------------------------------------------
# Tamper resistance
# ---------------------------------------------------------------------------


def test_policy_mutation_cannot_change_already_built_dataset(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    ready = ResearchReadyDataset.build(trusted, ResearchDataPolicy(expected_symbol="NIFTY"))
    with pytest.raises(Exception):
        ready.policy.expected_symbol = "BANKNIFTY"
    assert ready.policy.expected_symbol == "NIFTY"


def test_returned_frame_mutation_cannot_affect_internal_state(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    ready = ResearchReadyDataset.build(trusted, ResearchDataPolicy())
    first = ready.frame
    first.loc[0, CLOSE] = 999999.0
    second = ready.frame
    assert second.loc[0, CLOSE] != 999999.0


def test_research_ready_dataset_direct_construction_blocked(tmp_path):
    with pytest.raises(TypeError):
        ResearchReadyDataset()


def test_research_ready_dataset_cannot_be_pickled(tmp_path):
    import pickle

    trusted = _one_day_trusted(tmp_path)
    ready = ResearchReadyDataset.build(trusted, ResearchDataPolicy())
    with pytest.raises(TypeError):
        pickle.dumps(ready)


def test_research_ready_dataset_attribute_assignment_blocked(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    ready = ResearchReadyDataset.build(trusted, ResearchDataPolicy())
    with pytest.raises(AttributeError):
        ready.policy = ResearchDataPolicy(expected_symbol="BANKNIFTY")


# ---------------------------------------------------------------------------
# Future certification flags: never silently pass
# ---------------------------------------------------------------------------


def test_require_continuity_certified_always_fails_today(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    with pytest.raises(ResearchPolicyError) as excinfo:
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(require_continuity_certified=True))
    assert any("continuity" in reason and "NOT AVAILABLE" in reason for reason in excinfo.value.unmet_requirements)


def test_require_session_certified_always_fails_today(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    with pytest.raises(ResearchPolicyError) as excinfo:
        ResearchReadyDataset.build(trusted, ResearchDataPolicy(require_session_certified=True))
    assert any("session" in reason and "NOT AVAILABLE" in reason for reason in excinfo.value.unmet_requirements)


def test_require_reproducibility_certified_always_fails_today(tmp_path):
    trusted = _one_day_trusted(tmp_path)
    with pytest.raises(ResearchPolicyError) as excinfo:
        ResearchReadyDataset.build(
            trusted, ResearchDataPolicy(require_reproducibility_certified=True)
        )
    assert any(
        "reproducibility" in reason and "NOT AVAILABLE" in reason
        for reason in excinfo.value.unmet_requirements
    )


# ---------------------------------------------------------------------------
# ResearchDataPolicy strict validation
# ---------------------------------------------------------------------------


def test_policy_rejects_non_policy_type_at_build():
    # Covered above (test_policy_must_be_actual_research_data_policy); this
    # section focuses on ResearchDataPolicy's OWN __post_init__ validation.
    pass


@pytest.mark.parametrize("field_name", ["expected_source", "expected_symbol", "expected_resolution"])
def test_empty_expected_identity_string_rejected(field_name):
    with pytest.raises(ValueError):
        ResearchDataPolicy(**{field_name: ""})


@pytest.mark.parametrize("field_name", ["expected_source", "expected_symbol", "expected_resolution"])
def test_non_str_expected_identity_rejected(field_name):
    with pytest.raises(TypeError):
        ResearchDataPolicy(**{field_name: 123})


def test_min_distinct_observed_dates_must_be_positive():
    with pytest.raises(ValueError):
        ResearchDataPolicy(min_distinct_observed_dates=0)
    with pytest.raises(ValueError):
        ResearchDataPolicy(min_distinct_observed_dates=-1)


def test_min_distinct_observed_dates_bool_rejected():
    with pytest.raises(TypeError):
        ResearchDataPolicy(min_distinct_observed_dates=True)


def test_min_requested_window_fraction_out_of_range_rejected():
    with pytest.raises(ValueError):
        ResearchDataPolicy(min_requested_window_fraction=-0.1)
    with pytest.raises(ValueError):
        ResearchDataPolicy(min_requested_window_fraction=1.1)


def test_min_requested_window_fraction_bool_rejected():
    with pytest.raises(TypeError):
        ResearchDataPolicy(min_requested_window_fraction=True)


def test_min_requested_window_fraction_nan_rejected():
    with pytest.raises(ValueError):
        ResearchDataPolicy(min_requested_window_fraction=float("nan"))


def test_require_pristine_source_order_must_be_bool():
    with pytest.raises(TypeError):
        ResearchDataPolicy(require_pristine_source_order=1)


def test_boundary_fraction_values_accepted():
    ResearchDataPolicy(min_requested_window_fraction=0.0)
    ResearchDataPolicy(min_requested_window_fraction=1.0)
