"""Tests for derive_canonical_key and related verb-extraction helpers.

These are RED-phase TDD tests written before implementation.
They lock in the v2 canonical-key derivation contract from
CARD_STRUCTURE_v2.md §2.1.
"""

from __future__ import annotations

import pytest

from gigaevo.memory.ideas_tracker.idea_bank import (
    derive_canonical_key,
    normalize_canonical_value,
    parse_packed_description,
)


class TestNormalizeCanonicalValue:
    def test_none_normalizes_to_underscore(self) -> None:
        assert normalize_canonical_value(None) == "_"

    def test_integer_normalizes_to_3_sig_digit_string(self) -> None:
        assert normalize_canonical_value(7) == "7"
        assert normalize_canonical_value(1000) == "1e+03"

    def test_float_normalizes_to_3_sig_digit_string(self) -> None:
        # 3sig collapses 1.000000001 → 1, 1.999999999 → 2
        assert normalize_canonical_value(1.000000001) == "1"
        assert normalize_canonical_value(1.999999999) == "2"

    def test_float_zero_normalizes_to_zero(self) -> None:
        assert normalize_canonical_value(0.0) == "0"

    def test_negative_float_preserved(self) -> None:
        assert normalize_canonical_value(-0.05) == "-0.05"

    def test_string_lowercases_and_strips(self) -> None:
        assert normalize_canonical_value("  RandomForest  ") == "randomforest"

    def test_string_alias_collapse(self) -> None:
        # Estimator-class aliases per v2 §2.1
        assert normalize_canonical_value("rf") == "randomforest"
        assert normalize_canonical_value("xgb") == "xgboost"
        assert normalize_canonical_value("lr") == "linear"
        assert normalize_canonical_value("lgbm") == "lightgbm"
        assert normalize_canonical_value("cb") == "catboost"

    def test_dict_fallback_to_hash(self) -> None:
        result = normalize_canonical_value({"a": 1})
        # Hash form: 8-char hex
        assert len(result) == 8
        assert all(c in "0123456789abcdef" for c in result)


class TestDeriveCanonicalKey:
    def test_update_simple(self) -> None:
        key = derive_canonical_key("UPDATE", "depth", 6, 7)
        assert key == "UPDATE:depth:6:7"

    def test_update_uppercases_verb(self) -> None:
        key = derive_canonical_key("update", "depth", 6, 7)
        assert key == "UPDATE:depth:6:7"

    def test_target_lowercased_and_stripped(self) -> None:
        key = derive_canonical_key("UPDATE", "  L2_LEAF_REG  ", 1.0, 2.0)
        assert key == "UPDATE:l2_leaf_reg:1:2"

    def test_numeric_drift_collapses(self) -> None:
        # The canonical case that v1's keys missed
        key_a = derive_canonical_key("UPDATE", "l2_leaf_reg", 1.0, 2.0)
        key_b = derive_canonical_key("UPDATE", "l2_leaf_reg", 1.000000001, 1.999999999)
        assert key_a == key_b

    def test_estimator_alias_collapse(self) -> None:
        key_a = derive_canonical_key("SWAP", "estimator", "RandomForest", "CatBoost")
        key_b = derive_canonical_key("SWAP", "estimator", "rf", "cb")
        assert key_a == key_b

    def test_add_no_old_value(self) -> None:
        key = derive_canonical_key("ADD", "log1p_population", None, None)
        assert key == "ADD:log1p_population:_:_"

    def test_remove_no_new_value(self) -> None:
        key = derive_canonical_key("REMOVE", "target_log_transform", None, None)
        assert key == "REMOVE:target_log_transform:_:_"

    def test_different_verb_different_key(self) -> None:
        add_key = derive_canonical_key("ADD", "log1p_pop", None, None)
        remove_key = derive_canonical_key("REMOVE", "log1p_pop", None, None)
        assert add_key != remove_key


class TestParsePackedDescription:
    """v2 §1.1 grammar: `<VERB> <target>[ <old>→<new>]: <mechanism>; support=N; …`"""

    def test_parse_add(self) -> None:
        desc = "ADD log1p_population: log-transform stabilises tails; support=3; Δbest=-0.041; co=[a*,b*]"
        parsed = parse_packed_description(desc)
        assert parsed["verb"] == "ADD"
        assert parsed["target"] == "log1p_population"
        assert parsed["old"] is None
        assert parsed["new"] is None
        assert "log-transform stabilises" in parsed["mechanism"]
        assert parsed["support"] == 3
        assert parsed["delta_best"] == pytest.approx(-0.041)
        assert parsed["co"] == ["a*", "b*"]

    def test_parse_update(self) -> None:
        desc = "UPDATE depth 6→7: shallower trees underfitting; support=2; Δbest=-0.012; co=[]"
        parsed = parse_packed_description(desc)
        assert parsed["verb"] == "UPDATE"
        assert parsed["target"] == "depth"
        assert parsed["old"] == "6"
        assert parsed["new"] == "7"
        assert parsed["co"] == []

    def test_parse_use_with_equals(self) -> None:
        desc = "USE early_stopping_rounds = 100: convergence aid; support=1; Δbest=-0.015; co=[]"
        parsed = parse_packed_description(desc)
        assert parsed["verb"] == "USE"
        assert parsed["target"] == "early_stopping_rounds"
        assert parsed["new"] == "100"

    def test_parse_remove(self) -> None:
        desc = "REMOVE target_log_transform: floor at 0.15 penalty; support=5; Δbest=-0.018; co=[depth,early*]"
        parsed = parse_packed_description(desc)
        assert parsed["verb"] == "REMOVE"
        assert parsed["target"] == "target_log_transform"

    def test_parse_unverified_marker(self) -> None:
        # Per v2 §1.1 + v2_claude_voice amendment: (UNVERIFIED) parenthetical, single-`:` preserved
        desc = "USE early_stopping_rounds = 100 (UNVERIFIED): convergence not derivable; support=1; Δbest=-0.015; co=[]"
        parsed = parse_packed_description(desc)
        assert parsed["verified"] is False
        assert parsed["verb"] == "USE"

    def test_parse_verified_default(self) -> None:
        desc = "ADD foo: bar; support=1; Δbest=0; co=[]"
        parsed = parse_packed_description(desc)
        assert parsed["verified"] is True

    def test_parse_rejects_double_colon(self) -> None:
        # v2 §1.1 single-`:` invariant
        desc = "ADD foo: bar: baz; support=1; Δbest=0; co=[]"
        with pytest.raises(ValueError, match="single-`:`|invariant"):
            parse_packed_description(desc)

    def test_parse_rejects_unknown_verb(self) -> None:
        desc = "WIGGLE foo: bar; support=1; Δbest=0; co=[]"
        with pytest.raises(ValueError, match="verb"):
            parse_packed_description(desc)

    def test_parse_truncated_targets_in_co_list(self) -> None:
        desc = "ADD x: y; support=1; Δbest=0; co=[room_occup*,target_log*]"
        parsed = parse_packed_description(desc)
        assert parsed["co"] == ["room_occup*", "target_log*"]
