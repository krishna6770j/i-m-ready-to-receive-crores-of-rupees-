"""Exact canonical dataset identity + digest tests.

Written before ``marketdata/identity.py`` exists (TDD): every test in this
file failed on collection with ``ModuleNotFoundError`` until the module was
created, then failed on assertions until behaviour matched, per the frozen
architecture (docs/architecture/phase1-trust-hardening.md, sections 8.0-8.2).
"""

from __future__ import annotations

import math
import re
import struct

import numpy as np
import pandas as pd
import pytest

from core.timeutils import IST_NAME
from marketdata.identity import (
    DatasetIdentity,
    DatasetIdentityError,
    _encode_dataset,
    dataset_digest,
)
from marketdata.schemas import (
    CLOSE,
    HIGH,
    LOW,
    OHLCV_COLUMNS,
    OPEN,
    TS,
    VOLUME,
    SchemaError,
    canonicalise,
    empty_ohlcv,
)

_HEX64 = re.compile(r"^[a-f0-9]{64}$")


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """Build a canonical OHLCV frame from (ts, open, high, low, close, volume)
    tuples. ``ts`` may be any tz-aware-constructible value or already a
    ``pd.Timestamp``; ``volume`` may be ``None`` for missing.
    """
    raw = pd.DataFrame(
        {
            TS: [pd.Timestamp(r[0]).tz_convert(IST_NAME) if pd.Timestamp(r[0]).tzinfo else pd.Timestamp(r[0], tz=IST_NAME) for r in rows],
            OPEN: [r[1] for r in rows],
            HIGH: [r[2] for r in rows],
            LOW: [r[3] for r in rows],
            CLOSE: [r[4] for r in rows],
            VOLUME: [r[5] for r in rows],
        }
    )
    return canonicalise(raw).frame


def _identity(**overrides) -> DatasetIdentity:
    fields = {"source": "fyers:history", "symbol": "NIFTY", "resolution": "1"}
    fields.update(overrides)
    return DatasetIdentity(**fields)


_BASE_ROWS = [
    ("2026-01-01 09:15", 100.0, 101.0, 99.0, 100.5, 1000),
    ("2026-01-01 09:16", 100.5, 102.0, 100.0, 101.5, 2000),
]


# --- 1-2: determinism -------------------------------------------------------


def test_digest_is_deterministic_on_repeated_calls():
    identity = _identity()
    frame = _frame(_BASE_ROWS)
    assert dataset_digest(identity, frame) == dataset_digest(identity, frame)


def test_identical_identity_and_frame_same_digest():
    identity_a = _identity()
    identity_b = _identity()
    frame_a = _frame(_BASE_ROWS)
    frame_b = _frame(_BASE_ROWS)
    assert dataset_digest(identity_a, frame_a) == dataset_digest(identity_b, frame_b)


# --- 3-5: identity fields participate ---------------------------------------


def test_different_symbol_changes_digest():
    frame = _frame(_BASE_ROWS)
    d1 = dataset_digest(_identity(symbol="NIFTY"), frame)
    d2 = dataset_digest(_identity(symbol="SBIN"), frame)
    assert d1 != d2


def test_different_source_changes_digest():
    frame = _frame(_BASE_ROWS)
    d1 = dataset_digest(_identity(source="fyers:history"), frame)
    d2 = dataset_digest(_identity(source="other"), frame)
    assert d1 != d2


def test_different_resolution_changes_digest():
    frame = _frame(_BASE_ROWS)
    d1 = dataset_digest(_identity(resolution="1"), frame)
    d2 = dataset_digest(_identity(resolution="5"), frame)
    assert d1 != d2


# --- 6: schema version is represented, never caller-supplied ---------------


def test_schema_version_cannot_be_passed_by_caller():
    with pytest.raises(TypeError):
        DatasetIdentity(
            source="fyers:history", symbol="NIFTY", resolution="1", schema_version=2
        )


def test_schema_version_is_actually_encoded(monkeypatch):
    frame = _frame(_BASE_ROWS)
    identity = _identity()
    original = dataset_digest(identity, frame)

    import marketdata.identity as identity_module

    monkeypatch.setattr(identity_module, "MARKET_DATA_SCHEMA_VERSION", 999)
    changed = dataset_digest(identity, frame)

    assert original != changed


# --- 7-9: float64 exactness -------------------------------------------------


def test_adjacent_decimal_prices_differ():
    f1 = _frame([("2026-01-01 09:15", 24000.12, 24000.12, 24000.12, 24000.12, 1)])
    f2 = _frame([("2026-01-01 09:15", 24000.13, 24000.13, 24000.13, 24000.13, 1)])
    identity = _identity()
    assert dataset_digest(identity, f1) != dataset_digest(identity, f2)


def test_nextafter_adjacent_float64_values_differ():
    a = 24000.0
    b = float(np.nextafter(a, np.inf))
    assert a != b
    f1 = _frame([("2026-01-01 09:15", a, a, a, a, 1)])
    f2 = _frame([("2026-01-01 09:15", b, b, b, b, 1)])
    identity = _identity()
    assert dataset_digest(identity, f1) != dataset_digest(identity, f2)


def test_old_percent_10g_collision_pair_differs_under_new_digest():
    # Direct regression for the baseline defect (architecture section 2,
    # defect #7): store.content_hash() used `%.10g`, which collapses these
    # two distinct, adjacent float64 values to the identical text "24000".
    a = 24000.0
    b = float(np.nextafter(a, np.inf))
    assert a != b
    assert format(a, ".10g") == format(b, ".10g") == "24000"

    f1 = _frame([("2026-01-01 09:15", a, a, a, a, 1)])
    f2 = _frame([("2026-01-01 09:15", b, b, b, b, 1)])
    identity = _identity()
    assert dataset_digest(identity, f1) != dataset_digest(identity, f2)


# --- 10: signed zero --------------------------------------------------------


def test_positive_and_negative_zero_price_hash_the_same():
    f_pos = _frame([("2026-01-01 09:15", 0.0, 0.0, 0.0, 0.0, 1)])
    f_neg = _frame([("2026-01-01 09:15", -0.0, -0.0, -0.0, -0.0, 1)])
    identity = _identity()
    assert dataset_digest(identity, f_pos) == dataset_digest(identity, f_neg)


# --- 11: NaN determinism -----------------------------------------------------


def _canonical_shaped_frame(open_value: float) -> pd.DataFrame:
    """A structurally canonical single-row frame built WITHOUT going through
    ``canonicalise()``.

    ``canonicalise()`` treats any NaN as ordinary missingness and, via
    ``_convert_numeric_exact``, reconstructs a fresh, single-payload
    ``float("nan")`` placeholder for every missing price -- so a NaN-payload
    test that first calls ``canonicalise()`` would pass even if
    ``identity.py`` did nothing at all to canonicalise NaN itself, because
    the payload was already collapsed one layer up. Building the frame
    directly, with the target dtypes already in place, is what makes this a
    real test of this module's own ``_encode_f64`` behaviour.
    """
    return pd.DataFrame(
        {
            TS: pd.Series([pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)]),
            OPEN: pd.Series([open_value], dtype="float64"),
            HIGH: pd.Series([1.0], dtype="float64"),
            LOW: pd.Series([1.0], dtype="float64"),
            CLOSE: pd.Series([1.0], dtype="float64"),
            VOLUME: pd.Series([1], dtype="Int64"),
        }
    )


def test_nan_payload_bits_do_not_affect_digest():
    nan_a = float("nan")
    # A different NaN bit pattern (alternate payload) from the default one.
    nan_b = np.frombuffer(
        struct.pack(">Q", 0x7FF8000000000001), dtype=">f8"
    )[0].item()
    assert math.isnan(nan_a) and math.isnan(nan_b)
    assert struct.pack(">d", nan_a) != struct.pack(">d", nan_b)

    f_a = _canonical_shaped_frame(nan_a)
    f_b = _canonical_shaped_frame(nan_b)
    identity = _identity()
    assert dataset_digest(identity, f_a) == dataset_digest(identity, f_b)


# --- 12: infinity ------------------------------------------------------------


def test_positive_and_negative_infinity_differ():
    f_pos = _frame([("2026-01-01 09:15", float("inf"), float("inf"), float("inf"), float("inf"), 1)])
    f_neg = _frame([("2026-01-01 09:15", float("-inf"), float("-inf"), float("-inf"), float("-inf"), 1)])
    identity = _identity()
    assert dataset_digest(identity, f_pos) != dataset_digest(identity, f_neg)


# --- 13-14: volume -----------------------------------------------------------


def test_volume_zero_differs_from_volume_na():
    f_zero = _frame([("2026-01-01 09:15", 1.0, 1.0, 1.0, 1.0, 0)])
    f_na = _frame([("2026-01-01 09:15", 1.0, 1.0, 1.0, 1.0, None)])
    identity = _identity()
    assert dataset_digest(identity, f_zero) != dataset_digest(identity, f_na)


def test_volume_one_differs_from_volume_two():
    f1 = _frame([("2026-01-01 09:15", 1.0, 1.0, 1.0, 1.0, 1)])
    f2 = _frame([("2026-01-01 09:15", 1.0, 1.0, 1.0, 1.0, 2)])
    identity = _identity()
    assert dataset_digest(identity, f1) != dataset_digest(identity, f2)


# --- 15: timestamp nanosecond precision --------------------------------------


def test_timestamp_plus_one_nanosecond_differs():
    base = pd.Timestamp("2026-01-01 09:15:00", tz=IST_NAME)
    plus_ns = base + pd.Timedelta(nanoseconds=1)
    f1 = _frame([(base, 1.0, 1.0, 1.0, 1.0, 1)])
    f2 = _frame([(plus_ns, 1.0, 1.0, 1.0, 1.0, 1)])
    identity = _identity()
    assert dataset_digest(identity, f1) != dataset_digest(identity, f2)


def test_same_instant_utc_vs_ist_after_canonicalisation_same_digest():
    instant_ist = pd.Timestamp("2026-01-01 09:15:00", tz=IST_NAME)
    instant_utc = instant_ist.tz_convert("UTC")

    raw_ist = pd.DataFrame(
        {TS: [instant_ist], OPEN: [1.0], HIGH: [1.0], LOW: [1.0], CLOSE: [1.0], VOLUME: [1]}
    )
    raw_utc = pd.DataFrame(
        {TS: [instant_utc], OPEN: [1.0], HIGH: [1.0], LOW: [1.0], CLOSE: [1.0], VOLUME: [1]}
    )
    frame_ist = canonicalise(raw_ist).frame
    frame_utc = canonicalise(raw_utc).frame

    identity = _identity()
    assert dataset_digest(identity, frame_ist) == dataset_digest(identity, frame_utc)


# --- 17-19: row multiplicity / empty / one-row -------------------------------


def test_duplicate_row_added_changes_digest():
    identity = _identity()
    base = dataset_digest(identity, _frame(_BASE_ROWS))
    with_dup = dataset_digest(identity, _frame(_BASE_ROWS + [_BASE_ROWS[-1]]))
    assert base != with_dup


def test_row_removed_changes_digest():
    identity = _identity()
    base = dataset_digest(identity, _frame(_BASE_ROWS))
    fewer = dataset_digest(identity, _frame(_BASE_ROWS[:-1]))
    assert base != fewer


def test_empty_differs_from_one_row():
    identity = _identity()
    empty_digest = dataset_digest(identity, empty_ohlcv())
    one_row = dataset_digest(identity, _frame(_BASE_ROWS[:1]))
    assert empty_digest != one_row


def test_empty_frame_digest_is_deterministic():
    identity = _identity()
    assert dataset_digest(identity, empty_ohlcv()) == dataset_digest(identity, empty_ohlcv())


# --- 20-21: reordering is absorbed by canonicalisation, not by the digest --


def test_reversed_source_rows_then_canonicalised_same_digest():
    raw_forward = pd.DataFrame(
        {
            TS: [pd.Timestamp(r[0], tz=IST_NAME) for r in _BASE_ROWS],
            OPEN: [r[1] for r in _BASE_ROWS],
            HIGH: [r[2] for r in _BASE_ROWS],
            LOW: [r[3] for r in _BASE_ROWS],
            CLOSE: [r[4] for r in _BASE_ROWS],
            VOLUME: [r[5] for r in _BASE_ROWS],
        }
    )
    raw_reversed = raw_forward.iloc[::-1].reset_index(drop=True)

    identity = _identity()
    forward_digest = dataset_digest(identity, canonicalise(raw_forward).frame)
    reversed_digest = dataset_digest(identity, canonicalise(raw_reversed).frame)
    assert forward_digest == reversed_digest


def test_shuffled_input_columns_then_canonicalised_same_digest():
    raw = pd.DataFrame(
        {
            TS: [pd.Timestamp(r[0], tz=IST_NAME) for r in _BASE_ROWS],
            OPEN: [r[1] for r in _BASE_ROWS],
            HIGH: [r[2] for r in _BASE_ROWS],
            LOW: [r[3] for r in _BASE_ROWS],
            CLOSE: [r[4] for r in _BASE_ROWS],
            VOLUME: [r[5] for r in _BASE_ROWS],
        }
    )
    shuffled = raw.loc[:, [VOLUME, CLOSE, LOW, HIGH, OPEN, TS]]

    identity = _identity()
    canonical_digest = dataset_digest(identity, canonicalise(raw).frame)
    shuffled_digest = dataset_digest(identity, canonicalise(shuffled).frame)
    assert canonical_digest == shuffled_digest


# --- 22: digest requires an already-canonical frame -------------------------


def test_noncanonical_unsorted_frame_rejected_directly():
    raw = pd.DataFrame(
        {
            TS: [
                pd.Timestamp("2026-01-01 09:16", tz=IST_NAME),
                pd.Timestamp("2026-01-01 09:15", tz=IST_NAME),
            ],
            OPEN: [1.0, 1.0],
            HIGH: [1.0, 1.0],
            LOW: [1.0, 1.0],
            CLOSE: [1.0, 1.0],
            VOLUME: [1, 1],
        }
    )
    identity = _identity()
    with pytest.raises(SchemaError):
        dataset_digest(identity, raw)


def test_wrong_column_order_frame_rejected_directly():
    raw = pd.DataFrame(
        {
            VOLUME: [1],
            CLOSE: [1.0],
            LOW: [1.0],
            HIGH: [1.0],
            OPEN: [1.0],
            TS: [pd.Timestamp("2026-01-01 09:15", tz=IST_NAME)],
        }
    )
    identity = _identity()
    with pytest.raises(SchemaError):
        dataset_digest(identity, raw)


# --- 23: malformed identity metadata -----------------------------------------


@pytest.mark.parametrize("field", ["source", "symbol", "resolution"])
def test_empty_identity_field_rejected(field):
    with pytest.raises(DatasetIdentityError):
        _identity(**{field: ""})


def test_non_string_identity_field_rejected():
    with pytest.raises(DatasetIdentityError):
        _identity(symbol=123)


# --- 24: no delimiter/concatenation ambiguity --------------------------------


def test_metadata_concatenation_ambiguity_is_impossible():
    frame = _frame(_BASE_ROWS)
    d1 = dataset_digest(_identity(source="ab", symbol="c"), frame)
    d2 = dataset_digest(_identity(source="a", symbol="bc"), frame)
    assert d1 != d2


def test_metadata_pipe_character_does_not_collide():
    frame = _frame(_BASE_ROWS)
    d1 = dataset_digest(_identity(source="a|b", symbol="c"), frame)
    d2 = dataset_digest(_identity(source="a", symbol="b|c"), frame)
    assert d1 != d2


# --- 25: Unicode NFC equivalence for identity metadata -----------------------


def test_nfc_equivalent_symbol_hashes_the_same():
    precomposed = "NIFTY-é"  # é as one code point
    decomposed = "NIFTY-é"  # e + combining acute accent
    assert precomposed != decomposed  # distinct as raw Python strings

    frame = _frame(_BASE_ROWS)
    d1 = dataset_digest(_identity(symbol=precomposed), frame)
    d2 = dataset_digest(_identity(symbol=decomposed), frame)
    assert d1 == d2


# --- 26-27: adversarial metadata content -------------------------------------


def test_embedded_newline_in_metadata_is_safely_framed():
    frame = _frame(_BASE_ROWS)
    d_plain = dataset_digest(_identity(symbol="NIFTY"), frame)
    d_newline = dataset_digest(_identity(symbol="NIFTY\n50"), frame)
    assert d_plain != d_newline


def test_nul_character_in_metadata_is_rejected():
    with pytest.raises(DatasetIdentityError):
        _identity(symbol="NIFTY\x00X")


# --- 28: digest format --------------------------------------------------------


def test_digest_is_lowercase_64_char_hex():
    digest = dataset_digest(_identity(), _frame(_BASE_ROWS))
    assert _HEX64.match(digest), digest


# --- security: no secrets in identity ----------------------------------------


def test_dataset_identity_has_no_secret_fields():
    identity = _identity()
    field_names = {f.name for f in identity.__dataclass_fields__.values()}
    assert field_names == {"source", "symbol", "resolution"}


# ============================================================================
# Section 8.3: equal-timestamp identity ordering (docs/architecture/
# phase1-trust-hardening.md). Source arrival order within a shared timestamp
# is provenance, not dataset identity: canonicalise()'s stable sort correctly
# preserves that arrival order in CanonicalisationResult.frame, but the digest
# must order equal-timestamp observations by their own canonical encoded
# non-timestamp bytes, not by arrival position.
# ============================================================================

_T1 = "2026-01-01 09:15"
_T2 = "2026-01-01 09:16"
_T3 = "2026-01-01 09:17"

_OBS_A = (100.0, 101.0, 99.0, 100.5, 1000)
_OBS_B = (200.0, 201.0, 199.0, 200.5, 2000)
_OBS_C = (300.0, 301.0, 299.0, 300.5, 3000)
_OBS_D = (400.0, 401.0, 399.0, 400.5, 4000)
_OBS_E = (410.0, 411.0, 409.0, 410.5, 4100)
_OBS_F = (420.0, 421.0, 419.0, 420.5, 4200)
_OBS_NAN = (float("nan"), 1.0, 1.0, 1.0, 1)
_OBS_VOL_NA = (500.0, 501.0, 499.0, 500.5, None)
_OBS_ZERO_POS = (0.0, 1.0, 1.0, 1.0, 1)
_OBS_ZERO_NEG = (-0.0, 1.0, 1.0, 1.0, 1)

# Captured directly against commit 6239148 (before the section 8.3
# correction), using the unique-timestamp _BASE_ROWS fixture already defined
# above. Architecture section 8.3, item 7: datasets with no repeated
# timestamp are unaffected by this correction -- this is the compatibility
# proof that the fix does not gratuitously invalidate ordinary Unit-3 hashes.
_KNOWN_PRE_FIX_UNIQUE_TIMESTAMP_DIGEST = (
    "bcbd610d113aa6e21febdf6f118e25783644dc99357d16780aca6543af1afda9"
)


def _grouped_frame(ts: str, observations: list[tuple]) -> pd.DataFrame:
    return _frame([(ts, *obs) for obs in observations])


def _multi_group_frame(groups: list[tuple[str, list[tuple]]]) -> pd.DataFrame:
    rows = []
    for ts, observations in groups:
        for obs in observations:
            rows.append((ts, *obs))
    return _frame(rows)


def test_unique_timestamp_digest_unchanged_by_ordering_correction():
    assert (
        dataset_digest(_identity(), _frame(_BASE_ROWS))
        == _KNOWN_PRE_FIX_UNIQUE_TIMESTAMP_DIGEST
    )


# --- A: T:A,B vs T:B,A -> SAME ------------------------------------------------


def test_equal_timestamp_ab_vs_ba_same_digest():
    identity = _identity()
    d1 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_B]))
    d2 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_B, _OBS_A]))
    assert d1 == d2


# --- B: T:A,A,B vs T:B,A,A -> SAME --------------------------------------------


def test_equal_timestamp_aab_vs_baa_same_digest():
    identity = _identity()
    d1 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_A, _OBS_B]))
    d2 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_B, _OBS_A, _OBS_A]))
    assert d1 == d2


def test_equal_timestamp_multiplicity_distinguishes_same_distinct_set():
    # T:A,A,B and T:A,B,B share the same DISTINCT observation set {A, B} and
    # the same total row count (3), differing only in how many times each
    # occurs. A tie-break that deduplicates within a timestamp group (rather
    # than only sorting) would collapse both to the same two-element result
    # and collide -- the frame-level row_count field alone does not catch
    # this, since it is identical for both frames.
    identity = _identity()
    d_aab = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_A, _OBS_B]))
    d_abb = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_B, _OBS_B]))
    assert d_aab != d_abb


# --- C: T:A vs T:A,A -> DIFFERENT ---------------------------------------------


def test_equal_timestamp_a_vs_aa_different_digest():
    identity = _identity()
    d1 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A]))
    d2 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_A]))
    assert d1 != d2


# --- D: T:A,B vs T:A,C -> DIFFERENT -------------------------------------------


def test_equal_timestamp_ab_vs_ac_different_digest():
    identity = _identity()
    d1 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_B]))
    d2 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_C]))
    assert d1 != d2


# --- E: NaN-bearing equal-timestamp group, reordered source -> SAME ----------


def test_equal_timestamp_group_with_nan_reordered_same_digest():
    identity = _identity()
    d1 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_NAN, _OBS_A]))
    d2 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_NAN]))
    assert d1 == d2


# --- F: volume-NA-bearing equal-timestamp group, reordered source -> SAME ----


def test_equal_timestamp_group_with_volume_na_reordered_same_digest():
    identity = _identity()
    d1 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_VOL_NA, _OBS_A]))
    d2 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_A, _OBS_VOL_NA]))
    assert d1 == d2


# --- G: +0.0 / -0.0 within an equal-timestamp group -> deterministic ---------


def test_equal_timestamp_group_with_signed_zero_deterministic():
    identity = _identity()
    d1 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_ZERO_POS, _OBS_A]))
    d2 = dataset_digest(identity, _grouped_frame(_T1, [_OBS_ZERO_NEG, _OBS_A]))
    # +0.0 and -0.0 collapse to the identical encoded bytes (section 8.2), so
    # they are indistinguishable for both the tie-break sort and the digest.
    assert d1 == d2


# --- multi-group: only within-group reordering must be identity-preserving --


def test_multi_group_within_group_reordering_same_digest():
    identity = _identity()
    groups1 = [(_T1, [_OBS_A, _OBS_B]), (_T2, [_OBS_C]), (_T3, [_OBS_D, _OBS_E, _OBS_F])]
    groups2 = [(_T1, [_OBS_B, _OBS_A]), (_T2, [_OBS_C]), (_T3, [_OBS_F, _OBS_D, _OBS_E])]
    d1 = dataset_digest(identity, _multi_group_frame(groups1))
    d2 = dataset_digest(identity, _multi_group_frame(groups2))
    assert d1 == d2


def test_multi_group_changed_observation_in_later_group_different_digest():
    identity = _identity()
    groups1 = [(_T1, [_OBS_A, _OBS_B]), (_T2, [_OBS_C]), (_T3, [_OBS_D, _OBS_E, _OBS_F])]
    groups2 = [(_T1, [_OBS_A, _OBS_B]), (_T2, [_OBS_C]), (_T3, [_OBS_D, _OBS_E, _OBS_C])]
    d1 = dataset_digest(identity, _multi_group_frame(groups1))
    d2 = dataset_digest(identity, _multi_group_frame(groups2))
    assert d1 != d2


def test_timestamp_groups_are_not_globally_sorted_by_observation_bytes():
    # T1 carries an observation whose encoded bytes are lexicographically
    # LARGER than T2's observation. If equal-timestamp ordering were
    # mistakenly applied dataset-wide instead of within each timestamp group
    # (mutation M3), a byte-value sort would place T2's group ahead of T1's
    # -- violating "timestamp remains the primary key". This inspects the
    # actual byte stream, not just the resulting hash.
    from marketdata.identity import _TAG_TS, _encode_field

    high_value_obs = (900000.0, 900001.0, 899999.0, 900000.5, 999999)
    low_value_obs = (1.0, 2.0, 0.5, 1.5, 1)
    frame = _multi_group_frame([(_T1, [high_value_obs]), (_T2, [low_value_obs])])
    encoded = _encode_dataset(_identity(), frame)

    t1_ns = int(pd.Timestamp(_T1, tz=IST_NAME).value)
    t2_ns = int(pd.Timestamp(_T2, tz=IST_NAME).value)
    t1_field = _encode_field(_TAG_TS, struct.pack(">q", t1_ns))
    t2_field = _encode_field(_TAG_TS, struct.pack(">q", t2_ns))

    assert encoded.index(t1_field) < encoded.index(t2_field)


def test_equal_timestamp_tie_break_uses_encoded_bytes_not_repr():
    # Adversarial pair where string/repr ordering and canonical-byte ordering
    # DISAGREE: repr("10.0") < repr("9.0") lexically (string comparison sees
    # '1' < '9'), but struct.pack(">d", 9.0) < struct.pack(">d", 10.0) as
    # bytes (IEEE-754 big-endian preserves numeric order for positive
    # finite floats). A tie-break using repr()/string formatting would
    # therefore place the 10.0 observation first; the required canonical-
    # bytes tie-break must place the 9.0 observation first. This is checked
    # by locating each observation's OPEN field bytes directly in the
    # encoded stream, not merely by comparing hashes.
    obs_nine = (9.0, 1.0, 1.0, 1.0, 1)
    obs_ten = (10.0, 1.0, 1.0, 1.0, 1)
    frame = _grouped_frame(_T1, [obs_ten, obs_nine])  # arrival order: 10, then 9
    encoded = _encode_dataset(_identity(), frame)

    open_nine_field = struct.pack(">d", 9.0)
    open_ten_field = struct.pack(">d", 10.0)
    assert encoded.index(open_nine_field) < encoded.index(open_ten_field)


def test_equal_timestamp_reordering_yields_identical_byte_stream():
    # Proves byte-stream equality directly, not merely hash equality.
    identity = _identity()
    encoded1 = _encode_dataset(identity, _grouped_frame(_T1, [_OBS_A, _OBS_B]))
    encoded2 = _encode_dataset(identity, _grouped_frame(_T1, [_OBS_B, _OBS_A]))
    assert encoded1 == encoded2
