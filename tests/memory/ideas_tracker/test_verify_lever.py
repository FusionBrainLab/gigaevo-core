"""Tests for verify_lever predicate (CARD_STRUCTURE_v2 §2 Stage A-PRE).

RED-phase TDD tests written before implementation.
"""

from __future__ import annotations

from gigaevo.memory.ideas_tracker.idea_bank import (
    verify_lever,
)

# ---------------------------------------------------------------------------
# UPDATE verb
# ---------------------------------------------------------------------------


class TestVerifyLeverUpdate:
    def test_update_value_present_in_parent_and_child(self) -> None:
        parent = "model = CatBoostRegressor(depth=6, l2_leaf_reg=1.0)"
        child = "model = CatBoostRegressor(depth=6, l2_leaf_reg=2.0)"
        result = verify_lever(parent, child, "UPDATE", "l2_leaf_reg", 1.0, 2.0)
        assert result.verified is True
        assert result.method == "ast_diff"

    def test_update_value_missing_in_parent(self) -> None:
        parent = "model = CatBoostRegressor(depth=6)"
        child = "model = CatBoostRegressor(depth=6, l2_leaf_reg=2.0)"
        result = verify_lever(parent, child, "UPDATE", "l2_leaf_reg", 1.0, 2.0)
        assert result.verified is False

    def test_update_value_missing_in_child(self) -> None:
        parent = "model = CatBoostRegressor(depth=6, l2_leaf_reg=1.0)"
        child = "model = CatBoostRegressor(depth=6, l2_leaf_reg=1.0)"
        result = verify_lever(parent, child, "UPDATE", "l2_leaf_reg", 1.0, 2.0)
        assert result.verified is False

    def test_update_tolerates_whitespace(self) -> None:
        parent = "model = CatBoostRegressor(depth = 6, l2_leaf_reg = 1.0)"
        child = "model = CatBoostRegressor(depth = 6,  l2_leaf_reg = 2.0)"
        result = verify_lever(parent, child, "UPDATE", "l2_leaf_reg", 1.0, 2.0)
        assert result.verified is True

    def test_update_tolerates_quotes(self) -> None:
        parent = 'config = {"depth": "6"}'
        child = 'config = {"depth": "7"}'
        result = verify_lever(parent, child, "UPDATE", "depth", 6, 7)
        assert result.verified is True


# ---------------------------------------------------------------------------
# SWAP verb (alias for UPDATE on string-valued targets)
# ---------------------------------------------------------------------------


class TestVerifyLeverSwap:
    def test_swap_estimator_class(self) -> None:
        parent = "model = RandomForestRegressor()"
        child = "model = CatBoostRegressor()"
        result = verify_lever(
            parent, child, "SWAP", "estimator", "RandomForest", "CatBoost"
        )
        assert result.verified is True


# ---------------------------------------------------------------------------
# ADD verb
# ---------------------------------------------------------------------------


class TestVerifyLeverAdd:
    def test_add_identifier_absent_in_parent_present_in_child(self) -> None:
        parent = "X['mean'] = X.mean(axis=1)"
        child = "X['log1p_population'] = np.log1p(X['population'])"
        result = verify_lever(parent, child, "ADD", "log1p_population", None, None)
        assert result.verified is True

    def test_add_identifier_already_in_parent(self) -> None:
        parent = "X['log1p_population'] = np.log1p(X['population'])"
        child = "X['log1p_population'] = np.log1p(X['population']) * 2"
        result = verify_lever(parent, child, "ADD", "log1p_population", None, None)
        assert result.verified is False

    def test_add_identifier_missing_in_child(self) -> None:
        parent = "X['mean'] = X.mean(axis=1)"
        child = "X['median'] = X.median(axis=1)"
        result = verify_lever(parent, child, "ADD", "log1p_population", None, None)
        assert result.verified is False


# ---------------------------------------------------------------------------
# REMOVE verb
# ---------------------------------------------------------------------------


class TestVerifyLeverRemove:
    def test_remove_identifier_present_in_parent_absent_in_child(self) -> None:
        parent = "y_train = np.log1p(y_train)"
        child = "# no log transform"
        result = verify_lever(parent, child, "REMOVE", "log1p", None, None)
        assert result.verified is True

    def test_remove_identifier_still_in_child(self) -> None:
        parent = "y_train = np.log1p(y_train)"
        child = "y_train = np.log1p(y_train) * 2"
        result = verify_lever(parent, child, "REMOVE", "log1p", None, None)
        assert result.verified is False


# ---------------------------------------------------------------------------
# USE verb (no diff verification; always returns unverified)
# ---------------------------------------------------------------------------


class TestVerifyLeverUse:
    def test_use_always_unverified(self) -> None:
        parent = "model = CatBoostRegressor()"
        child = "model = CatBoostRegressor(early_stopping_rounds=100)"
        result = verify_lever(parent, child, "USE", "early_stopping_rounds", None, 100)
        assert result.verified is False
        assert result.method == "absent"


# ---------------------------------------------------------------------------
# Mechanism truthfulness gate
# ---------------------------------------------------------------------------


class TestMechanismMentionsTarget:
    def test_mechanism_mentions_target_literally(self) -> None:
        from gigaevo.memory.ideas_tracker.idea_bank import mechanism_mentions_target

        assert mechanism_mentions_target(
            "shallower l2_leaf_reg permits broader generalisation",
            "l2_leaf_reg",
        )

    def test_mechanism_mentions_target_via_alias(self) -> None:
        from gigaevo.memory.ideas_tracker.idea_bank import mechanism_mentions_target

        # 1-step alias: l2_leaf_reg ~ L2 regularization
        assert mechanism_mentions_target(
            "stronger L2 regularization smooths leaf values",
            "l2_leaf_reg",
        )

    def test_mechanism_does_not_mention_target(self) -> None:
        from gigaevo.memory.ideas_tracker.idea_bank import mechanism_mentions_target

        # "captures slower-converging geographic patterns" doesn't mention
        # early_stopping_rounds — this is the 2026-05-23 defect to catch
        assert not mechanism_mentions_target(
            "captures slower-converging geographic patterns",
            "early_stopping_rounds",
        )


# ---------------------------------------------------------------------------
# VerificationResult shape
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_has_verified_and_method(self) -> None:
        parent = "depth=6"
        child = "depth=7"
        result = verify_lever(parent, child, "UPDATE", "depth", 6, 7)
        assert hasattr(result, "verified")
        assert hasattr(result, "method")
        assert result.method in {"ast_diff", "regex_diff", "absent"}
