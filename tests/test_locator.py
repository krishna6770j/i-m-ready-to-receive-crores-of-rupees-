"""Safe dataset locator (slug) and CURRENT pointer format tests.

Frozen architecture sections 13.1 (identifier slug) and 13.2 (CURRENT
pointer). No filesystem I/O is exercised or expected here -- this module
produces pure values only.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest

from marketdata.locator import (
    POINTER_VERSION,
    CurrentPointer,
    LocatorError,
    dataset_relative_path,
    safe_slug,
)

_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}-[0-9a-f]{16}$")
_HEX64 = re.compile(r"^[a-f0-9]{64}$")


# ---------------------------------------------------------------------------
# safe_slug
# ---------------------------------------------------------------------------


def test_normal_symbol_slug_is_deterministic():
    d1 = safe_slug("NIFTY")
    d2 = safe_slug("NIFTY")
    assert d1 == d2
    assert _SLUG_RE.match(d1)
    assert d1.startswith("NIFTY-")


def test_dotdot_sanitised():
    slug = safe_slug("..")
    assert ".." not in slug
    assert "." not in slug.split("-")[0]  # sanitised prefix has no dot
    assert _SLUG_RE.match(slug)


def test_single_dot_sanitised():
    slug = safe_slug(".")
    assert "." not in slug.rsplit("-", 1)[0]
    assert _SLUG_RE.match(slug)


def test_dotdot_slash_x_sanitised():
    slug = safe_slug("../x")
    assert "/" not in slug
    assert ".." not in slug
    assert _SLUG_RE.match(slug)


def test_absolute_unix_path_sanitised():
    slug = safe_slug("/tmp/x")
    assert "/" not in slug
    assert _SLUG_RE.match(slug)


def test_backslash_sanitised():
    slug = safe_slug("a\\b")
    assert "\\" not in slug
    assert _SLUG_RE.match(slug)


def test_unicode_identifier_sanitised_and_deterministic():
    slug1 = safe_slug("NIFTY:नमस्ते")
    slug2 = safe_slug("NIFTY:नमस्ते")
    assert slug1 == slug2
    assert _SLUG_RE.match(slug1)


def test_empty_string_rejected():
    with pytest.raises(LocatorError):
        safe_slug("")


def test_non_string_rejected():
    with pytest.raises(LocatorError):
        safe_slug(12345)


def test_sanitised_prefix_collision_gets_different_suffix():
    # "AB@" and "AB#" both sanitise to the identical prefix "AB_", but are
    # genuinely different raw identifiers -- their slugs must still differ.
    slug_a = safe_slug("AB@")
    slug_b = safe_slug("AB#")
    prefix_a, suffix_a = slug_a.rsplit("-", 1)
    prefix_b, suffix_b = slug_b.rsplit("-", 1)
    assert prefix_a == prefix_b == "AB_"
    assert suffix_a != suffix_b


def test_output_matches_allowed_alphabet_and_length():
    for identifier in ["NIFTY", "..", "../x", "/tmp/x", "a\\b", "NIFTY:नमस्ते", "x" * 100]:
        slug = safe_slug(identifier)
        assert _SLUG_RE.match(slug), slug
        assert len(slug) <= 49


def test_raw_dangerous_identifier_never_appears_as_path_component():
    dangerous = ["..", "../../etc/passwd", "/etc/passwd", "a/b", "a\\b", "."]
    for identifier in dangerous:
        slug = safe_slug(identifier)
        # No / or \ can survive sanitisation, and no "." character can
        # appear anywhere in the sanitised prefix -- so neither a "/"-
        # separated traversal nor a bare "."/".." component is possible.
        assert "/" not in slug
        assert "\\" not in slug
        assert "." not in slug.rsplit("-", 1)[0]
        assert slug not in (".", "..")


def test_prefix_truncated_to_32_characters():
    long_identifier = "A" * 100
    slug = safe_slug(long_identifier)
    prefix = slug.rsplit("-", 1)[0]
    assert len(prefix) == 32


def test_different_identifiers_produce_different_slugs():
    assert safe_slug("NIFTY") != safe_slug("SBIN")


# ---------------------------------------------------------------------------
# dataset_relative_path
# ---------------------------------------------------------------------------


def test_dataset_relative_path_has_three_safe_components():
    path = dataset_relative_path(source="fyers:history", symbol="NIFTY", resolution="1")
    parts = path.parts
    assert len(parts) == 3
    for part in parts:
        assert _SLUG_RE.match(part)


def test_dataset_relative_path_rejects_dangerous_identifiers_safely():
    path = dataset_relative_path(source="../etc", symbol="NIFTY", resolution="1")
    assert ".." not in str(path)
    assert not str(path).startswith("/")


# ---------------------------------------------------------------------------
# CurrentPointer
# ---------------------------------------------------------------------------


def _pointer(**overrides) -> CurrentPointer:
    fields = dict(generation_id=uuid.uuid4(), integrity_id="a" * 64)
    fields.update(overrides)
    return CurrentPointer(**fields)


def test_pointer_roundtrip():
    original = _pointer()
    text = original.to_json()
    restored = CurrentPointer.from_json(text)
    assert restored.generation_id == original.generation_id
    assert restored.integrity_id == original.integrity_id
    assert restored.pointer_version == POINTER_VERSION


def test_pointer_json_is_deterministic():
    p = _pointer()
    assert p.to_json() == p.to_json()


def test_pointer_json_is_canonical_sorted_compact():
    p = _pointer()
    text = p.to_json()
    assert " " not in text  # compact, no incidental whitespace
    payload = json.loads(text)
    assert list(payload.keys()) == sorted(payload.keys())


def test_pointer_version_cannot_be_overridden():
    with pytest.raises(TypeError):
        CurrentPointer(pointer_version=2, generation_id=uuid.uuid4(), integrity_id="a" * 64)


def test_malformed_json_rejected():
    with pytest.raises(LocatorError):
        CurrentPointer.from_json("{not valid json")


def test_non_object_json_rejected():
    with pytest.raises(LocatorError):
        CurrentPointer.from_json("[1, 2, 3]")
    with pytest.raises(LocatorError):
        CurrentPointer.from_json('"just a string"')


def test_unknown_field_rejected():
    payload = {
        "pointer_version": 1,
        "generation_id": str(uuid.uuid4()),
        "integrity_id": "a" * 64,
        "extra_field": "surprise",
    }
    with pytest.raises(LocatorError):
        CurrentPointer.from_json(json.dumps(payload))


def test_missing_field_rejected():
    payload = {"pointer_version": 1, "generation_id": str(uuid.uuid4())}
    with pytest.raises(LocatorError):
        CurrentPointer.from_json(json.dumps(payload))


def test_wrong_pointer_version_rejected():
    payload = {
        "pointer_version": 2,
        "generation_id": str(uuid.uuid4()),
        "integrity_id": "a" * 64,
    }
    with pytest.raises(LocatorError):
        CurrentPointer.from_json(json.dumps(payload))


def test_malformed_uuid_rejected():
    with pytest.raises(LocatorError):
        _pointer(generation_id="not-a-uuid")


def test_uuid1_rejected():
    with pytest.raises(LocatorError):
        _pointer(generation_id=uuid.uuid1())


def test_malformed_integrity_digest_rejected():
    with pytest.raises(LocatorError):
        _pointer(integrity_id="not-a-valid-digest")


def test_uppercase_integrity_digest_rejected():
    with pytest.raises(LocatorError):
        _pointer(integrity_id="A" * 64)


def test_short_integrity_digest_rejected():
    with pytest.raises(LocatorError):
        _pointer(integrity_id="a" * 63)


def test_extra_path_like_field_rejected_because_unknown():
    payload = {
        "pointer_version": 1,
        "generation_id": str(uuid.uuid4()),
        "integrity_id": "a" * 64,
        "path": "../../etc/passwd",
    }
    with pytest.raises(LocatorError):
        CurrentPointer.from_json(json.dumps(payload))


def test_no_path_string_exists_inside_pointer_structure():
    p = _pointer()
    text = p.to_json()
    assert "/" not in text
    assert "\\" not in text
    assert ".." not in text
