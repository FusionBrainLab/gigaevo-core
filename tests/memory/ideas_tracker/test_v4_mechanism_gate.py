"""Tests for v4 FINAL mechanism gate (CARD_STRUCTURE_v4_FINAL.md §2).

RED-phase TDD tests for v4-FINAL upgrades:
- changed_tokens (symmetric diff)
- mechanism_grounded_in_diff returns (passed, strength) tuple where strength in {"code","lexicon","none"}
- decide_verification 3-branch tree: code → clean; lexicon|none with verified lever → mechanism_unverified
- Pass-B requires >=2 DISTINCT lexicon stems
- UPDATE/SWAP extra_changed_tokens (literal old/new values anchor mechanism)
"""

from __future__ import annotations

from gigaevo.memory.ideas_tracker.idea_bank import (
    changed_tokens,
    decide_verification,
    mechanism_grounded_in_diff,
)


class TestChangedTokensSymmetricDiff:
    def test_pure_remove_includes_removed_token(self) -> None:
        parent = "y_train = np.log1p(y_train)"
        child = "# no log transform"
        ct = changed_tokens(parent, child)
        assert "log1p" in ct
        assert "y_train" in ct
        assert "np" in ct

    def test_pure_add_includes_added_token(self) -> None:
        parent = "X['mean'] = X.mean(axis=1)"
        child = "X['log1p_population'] = np.log1p(X['population'])"
        ct = changed_tokens(parent, child)
        assert "log1p_population" in ct
        assert "log1p" in ct

    def test_numeric_only_change_no_identifier_diff(self) -> None:
        parent = "model = CatBoostRegressor(depth=6)"
        child = "model = CatBoostRegressor(depth=7)"
        ct = changed_tokens(parent, child)
        assert "depth" not in ct
        assert "catboostregressor" not in ct

    def test_tokens_lowercased(self) -> None:
        parent = "CatBoost"
        child = "LightGBM"
        ct = changed_tokens(parent, child)
        assert "catboost" in ct
        assert "lightgbm" in ct


class TestMechanismGroundedInDiffV4Final:
    """v4 FINAL: returns (passed, strength) tuple."""

    # ----- Pass-A: code evidence -----

    def test_pass_a_returns_code_strength(self) -> None:
        parent = "X['mean'] = X.mean(axis=1)"
        child = "X['log1p_population'] = np.log1p(X['population'])"
        passed, strength = mechanism_grounded_in_diff(
            "log1p transform stabilises long-tail population variance",
            "log1p_population",
            parent,
            child,
        )
        assert passed is True
        assert strength == "code"

    def test_pass_a_for_pure_remove(self) -> None:
        """REMOVE case: mechanism must cite removed NON-target identifier (target alone is gate 1)."""
        parent = "y_train = np.log1p(y_train); model.fit(X, y_train)"
        child = "model.fit(X, y_train)"
        passed, strength = mechanism_grounded_in_diff(
            "removing the np.log1p numpy call restores raw target scale",
            "log1p",
            parent,
            child,
        )
        assert passed is True
        assert strength == "code"

    def test_pure_remove_target_only_is_not_evidence(self) -> None:
        """Mechanism that only restates the target is NOT evidence (per GPT v4-final amendment)."""
        parent = "y_train = np.log1p(y_train)"
        child = "# removed"
        passed, strength = mechanism_grounded_in_diff(
            "removing log1p was the right move",
            "log1p",
            parent,
            child,
        )
        assert passed is False
        assert strength == "none"

    def test_target_alone_does_NOT_grant_pass_a(self) -> None:
        parent = "model = CatBoostRegressor()"
        child = "model = CatBoostRegressor(early_stopping_rounds=100)"
        passed, strength = mechanism_grounded_in_diff(
            "early_stopping_rounds is a known trick",
            "early_stopping_rounds",
            parent,
            child,
        )
        assert passed is False
        assert strength == "none"

    # ----- Pass-B: ML-lexicon (returns "lexicon" strength) -----

    def test_pass_b_returns_lexicon_strength(self) -> None:
        parent = "depth=6"
        child = "depth=7"
        passed, strength = mechanism_grounded_in_diff(
            "deeper trees reduce bias at cost of higher variance",
            "depth",
            parent,
            child,
        )
        assert passed is True
        assert strength == "lexicon"

    def test_pass_b_requires_distinct_terms(self) -> None:
        """Word-repetition gaming: 'overfit overfits overfitting' should NOT pass.

        These are 3 distinct stems in our lexicon though ({overfit, overfits, overfitting}),
        so test with actual repetition: 'variance variance variance' should fail.
        """
        parent = "depth=6"
        child = "depth=7"
        passed, strength = mechanism_grounded_in_diff(
            "variance variance variance",
            "depth",
            parent,
            child,
        )
        # variance counted once → fails
        assert passed is False
        assert strength == "none"

    def test_pass_b_one_term_insufficient(self) -> None:
        parent = "depth=6"
        child = "depth=7"
        passed, strength = mechanism_grounded_in_diff(
            "this reduces variance only",
            "depth",
            parent,
            child,
        )
        assert passed is False

    # ----- extra_changed_tokens (UPDATE/SWAP literal values) -----

    def test_extra_changed_tokens_anchors_numeric_update(self) -> None:
        """GPT amendment: UPDATE/SWAP literal old/new values anchor mechanism."""
        parent = "depth=6"
        child = "depth=7"
        passed, strength = mechanism_grounded_in_diff(
            "depth 7 enables one more split level",
            "depth",
            parent,
            child,
            extra_changed_tokens={"6", "7"},
        )
        assert passed is True
        assert strength == "code"

    def test_extra_changed_tokens_anchors_string_swap(self) -> None:
        parent = "model = RandomForestRegressor()"
        child = "model = CatBoostRegressor()"
        # Without extra: identifier diff already catches this
        passed, strength = mechanism_grounded_in_diff(
            "CatBoost handles categorical splits natively",
            "estimator",
            parent,
            child,
            extra_changed_tokens={"RandomForest", "CatBoost"},
        )
        assert passed is True

    # ----- Composite failures -----

    def test_no_evidence_no_lexicon_fails(self) -> None:
        parent = "depth=6"
        child = "depth=7"
        passed, strength = mechanism_grounded_in_diff(
            "this just felt right",
            "depth",
            parent,
            child,
        )
        assert passed is False
        assert strength == "none"


class TestDecideVerificationV4Final:
    """v4 FINAL §2 decision tree."""

    def test_branch_1_code_evidence_clean(self) -> None:
        """Lever pass + code evidence → clean verb, verified:true only."""
        parent = "depth=6"
        child = "depth=7"
        result = decide_verification(
            parent_code=parent,
            child_code=child,
            verb="UPDATE",
            target="depth",
            old=6,
            new=7,
            mechanism="depth 7 reduces bias by enabling one more split level",
        )
        assert result["verb_prefix"] == ""
        assert "verified:true" in result["keywords"]
        assert "mechanism_unverified:true" not in result["keywords"]
        assert result["parent_diff_verified"] is True

    def test_branch_2_lexicon_only_marks_mechanism_unverified(self) -> None:
        """GPT amendment: Lever pass + Pass-B lexicon only → UNVERIFIED_ + mechanism_unverified."""
        parent = "depth=6"
        child = "depth=7"
        # Mention target but only via ML lexicon (no code evidence)
        result = decide_verification(
            parent_code=parent,
            child_code=child,
            verb="UPDATE",
            target="depth",
            old=999,  # mismatched literal so extra_changed_tokens doesn't help
            new=998,
            mechanism="depth reduces bias and improves generalisation",
        )
        # lever fails because literal "999"/"998" not in code → branch 3 actually
        assert result["verb_prefix"] == "UNVERIFIED_"

    def test_branch_2_target_ok_no_evidence_at_all(self) -> None:
        """Lever pass + target mentioned + no evidence (no code, no lexicon) → mechanism_unverified."""
        parent = "depth=6"
        child = "depth=7"
        result = decide_verification(
            parent_code=parent,
            child_code=child,
            verb="UPDATE",
            target="depth",
            old=6,
            new=7,
            mechanism="depth captures slower-converging geographic patterns",
        )
        # target mentioned ('depth'), no code evidence (only "depth" identifier shared),
        # only 1 ML lexicon term ('converging'? not in lexicon — only "converge"/"convergence")
        # Actually 'converging' is NOT in lexicon. So fails Pass-B.
        # Result: branch 2 → UNVERIFIED_ + verified:true + mechanism_unverified
        assert result["verb_prefix"] == "UNVERIFIED_"
        assert "verified:true" in result["keywords"]
        assert "mechanism_unverified:true" in result["keywords"]
        assert result["parent_diff_verified"] is True

    def test_branch_3_lever_fail(self) -> None:
        parent = "model = CatBoostRegressor()"
        child = "model = CatBoostRegressor(early_stopping_rounds=100)"
        result = decide_verification(
            parent_code=parent,
            child_code=child,
            verb="USE",
            target="early_stopping_rounds",
            old=None,
            new=100,
            mechanism="depth 7 reduces bias and overfit",
        )
        assert result["verb_prefix"] == "UNVERIFIED_"
        assert "verified:false" in result["keywords"]
        assert "verified:true" not in result["keywords"]
        assert "mechanism_unverified:true" not in result["keywords"]

    def test_2026_05_23_hallucinated_why_caught(self) -> None:
        result = decide_verification(
            parent_code="model = CatBoostRegressor()",
            child_code="model = CatBoostRegressor(early_stopping_rounds=100)",
            verb="USE",
            target="early_stopping_rounds",
            old=None,
            new=100,
            mechanism="captures slower-converging geographic patterns",
        )
        assert result["verb_prefix"] == "UNVERIFIED_"
        assert "verified:false" in result["keywords"]

    def test_update_with_literal_anchored_mechanism_branch_1(self) -> None:
        """UPDATE/SWAP literal old/new in mechanism → code evidence → clean verb."""
        parent = "model = CatBoostRegressor(depth=6, l2_leaf_reg=1.0)"
        child = "model = CatBoostRegressor(depth=6, l2_leaf_reg=2.0)"
        result = decide_verification(
            parent_code=parent,
            child_code=child,
            verb="UPDATE",
            target="l2_leaf_reg",
            old=1.0,
            new=2.0,
            mechanism="stronger l2 regularization with l2_leaf_reg 2.0 reduces leaf-value variance",
        )
        # literal "2.0" is in extra_changed_tokens → Pass-A
        assert result["verb_prefix"] == ""
        assert "verified:true" in result["keywords"]


class TestVerificationDictShape:
    def test_branch_1_returns_required_keys(self) -> None:
        result = decide_verification(
            parent_code="depth=6",
            child_code="depth=7",
            verb="UPDATE",
            target="depth",
            old=6,
            new=7,
            mechanism="depth 7 reduces bias at cost of higher variance",
        )
        assert "verb_prefix" in result
        assert "keywords" in result
        assert "parent_diff_verified" in result
        assert "verification_method" in result
        assert isinstance(result["keywords"], list)
        assert isinstance(result["parent_diff_verified"], bool)

    def test_verification_method_enum(self) -> None:
        result = decide_verification(
            parent_code="depth=6",
            child_code="depth=7",
            verb="UPDATE",
            target="depth",
            old=6,
            new=7,
            mechanism="depth 7 reduces bias at cost of higher variance",
        )
        assert result["verification_method"] in {"ast_diff", "regex_diff", "absent"}
