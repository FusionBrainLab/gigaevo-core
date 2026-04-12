"""Tests for RunSpec parser -- canonical run specification parsing."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from gigaevo.monitoring.run_spec import RunSpec

# ---------------------------------------------------------------------------
# a) Basic parsing tests (parametrize)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_prefix, expected_db, expected_label",
    [
        (
            "chains/hotpotqa/static@4:O",
            "chains/hotpotqa/static",
            4,
            "O",
        ),
        (
            "chains/hotpotqa/static@4",
            "chains/hotpotqa/static",
            4,
            "chains/hotpotqa/static@4",
        ),
        ("prefix@0:label", "prefix", 0, "label"),
        ("a@1:b", "a", 1, "b"),
    ],
)
def test_basic_parsing(
    raw: str,
    expected_prefix: str,
    expected_db: int,
    expected_label: str,
) -> None:
    result = RunSpec.parse(raw)
    assert result.prefix == expected_prefix
    assert result.db == expected_db
    assert result.label == expected_label


# ---------------------------------------------------------------------------
# b) Edge case tests
# ---------------------------------------------------------------------------


def test_double_quoted_string() -> None:
    result = RunSpec.parse('"chains/hotpotqa/static@4:O"')
    assert result.prefix == "chains/hotpotqa/static"
    assert result.db == 4
    assert result.label == "O"


def test_single_quoted_string() -> None:
    result = RunSpec.parse("'chains/hotpotqa/static@4:O'")
    assert result.prefix == "chains/hotpotqa/static"
    assert result.db == 4
    assert result.label == "O"


def test_whitespace_stripped() -> None:
    result = RunSpec.parse("  chains/hotpotqa/static@4:O  ")
    assert result.prefix == "chains/hotpotqa/static"
    assert result.db == 4
    assert result.label == "O"


def test_label_with_special_chars() -> None:
    result = RunSpec.parse("prefix@3:my-label_v2")
    assert result.label == "my-label_v2"


def test_label_with_colons() -> None:
    result = RunSpec.parse("prefix@3:label:with:colons")
    assert result.label == "label:with:colons"


def test_prefix_with_slashes() -> None:
    result = RunSpec.parse("chains/hover/static_soft@6:T1_A")
    assert result.prefix == "chains/hover/static_soft"
    assert result.db == 6
    assert result.label == "T1_A"


# ---------------------------------------------------------------------------
# c) Error tests (ValueError)
# ---------------------------------------------------------------------------


def test_error_empty_string() -> None:
    with pytest.raises(ValueError):
        RunSpec.parse("")


def test_error_no_at_sign() -> None:
    with pytest.raises(ValueError):
        RunSpec.parse("prefix_only")


def test_error_non_numeric_db() -> None:
    with pytest.raises(ValueError):
        RunSpec.parse("prefix@abc:label")


def test_error_negative_db() -> None:
    with pytest.raises(ValueError):
        RunSpec.parse("prefix@-1:label")


def test_error_missing_prefix() -> None:
    with pytest.raises(ValueError):
        RunSpec.parse("@4:label")


def test_error_just_at() -> None:
    with pytest.raises(ValueError):
        RunSpec.parse("@")


# ---------------------------------------------------------------------------
# d) Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

_prefix_strategy = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz/_-0123456789")),
    min_size=1,
    max_size=50,
)
_db_strategy = st.integers(min_value=0, max_value=15)
_label_strategy = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-0123456789")
    ),
    min_size=1,
    max_size=20,
)


@given(prefix=_prefix_strategy, db=_db_strategy, label=_label_strategy)
@settings(max_examples=100)
def test_roundtrip_with_label(prefix: str, db: int, label: str) -> None:
    raw = f"{prefix}@{db}:{label}"
    parsed = RunSpec.parse(raw)
    assert parsed.prefix == prefix
    assert parsed.db == db
    assert parsed.label == label


@given(prefix=_prefix_strategy, db=_db_strategy)
@settings(max_examples=50)
def test_roundtrip_without_label(prefix: str, db: int) -> None:
    raw = f"{prefix}@{db}"
    parsed = RunSpec.parse(raw)
    assert parsed.prefix == prefix
    assert parsed.db == db
    assert parsed.label == f"{prefix}@{db}"


# ---------------------------------------------------------------------------
# e) Equality and hashing tests
# ---------------------------------------------------------------------------


def test_equality() -> None:
    a = RunSpec(prefix="p", db=1, label="L")
    b = RunSpec(prefix="p", db=1, label="L")
    assert a == b


def test_hashable() -> None:
    a = RunSpec(prefix="p", db=1, label="L")
    b = RunSpec(prefix="p", db=1, label="L")
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_frozen() -> None:
    spec = RunSpec(prefix="p", db=1, label="L")
    with pytest.raises(AttributeError):
        spec.prefix = "q"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# f) display_name property tests
# ---------------------------------------------------------------------------


def test_display_name_with_explicit_label() -> None:
    spec = RunSpec(prefix="chains/hotpotqa/static", db=4, label="O")
    assert spec.display_name == "O"


def test_display_name_with_auto_label() -> None:
    spec = RunSpec(
        prefix="chains/hotpotqa/static",
        db=4,
        label="chains/hotpotqa/static@4",
    )
    assert spec.display_name == "chains/hotpotqa/static@4"
