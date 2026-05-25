"""TDD test for enrich_with_verification helper."""

from __future__ import annotations

from gigaevo.memory.ideas_tracker.idea_bank import enrich_with_verification


class TestEnrichWithVerification:
    def test_unparsable_description_returns_original(self) -> None:
        result = enrich_with_verification(
            description="Removed target_log_transform log->raw: matches scale",
            parent_code="y = np.log1p(y)",
            child_code="y = y",
        )
        assert (
            result["description"]
            == "Removed target_log_transform log->raw: matches scale"
        )
        assert result["keywords"] == []
        assert result["parent_diff_verified"] is False

    def test_parsable_update_lever_passes(self) -> None:
        result = enrich_with_verification(
            description="UPDATE l2_leaf_reg 1.0→2.0: stronger l2_leaf_reg 2.0 reduces leaf variance; support=1; Δbest=+0.012; co=[]",
            parent_code="model = CatBoostRegressor(l2_leaf_reg=1.0)",
            child_code="model = CatBoostRegressor(l2_leaf_reg=2.0)",
        )
        assert not result["description"].startswith("UNVERIFIED_")
        assert "verified:true" in result["keywords"]
        assert result["parent_diff_verified"] is True

    def test_parsable_use_lever_fails(self) -> None:
        result = enrich_with_verification(
            description="USE early_stopping_rounds = 100: improves convergence; support=1; Δbest=+0.015; co=[]",
            parent_code="model = CatBoostRegressor()",
            child_code="model = CatBoostRegressor(early_stopping_rounds=100)",
        )
        assert result["description"].startswith("UNVERIFIED_USE")
        assert "verified:false" in result["keywords"]

    def test_lever_pass_no_code_evidence_marks_mechanism_unverified(self) -> None:
        result = enrich_with_verification(
            description="UPDATE depth 6→7: captures geographic patterns; support=1; Δbest=+0.005; co=[]",
            parent_code="model = CatBoostRegressor(depth=6)",
            child_code="model = CatBoostRegressor(depth=7)",
        )
        assert result["description"].startswith("UNVERIFIED_UPDATE")
        assert "verified:true" in result["keywords"]
        assert "mechanism_unverified:true" in result["keywords"]

    def test_no_parent_code_skips_verification(self) -> None:
        result = enrich_with_verification(
            description="UPDATE depth 6→7: bias-variance; support=1; Δbest=+0.005; co=[]",
            parent_code="",
            child_code="model = ...",
        )
        # No parent code → can't verify lever; pass through but don't mark verified
        assert result["parent_diff_verified"] is False
        assert (
            result["description"]
            == "UPDATE depth 6→7: bias-variance; support=1; Δbest=+0.005; co=[]"
        )

    def test_idempotent_on_already_unverified(self) -> None:
        result = enrich_with_verification(
            description="UNVERIFIED_USE early_stopping_rounds = 100: x; support=1; Δbest=+0.0; co=[]",
            parent_code="model = CatBoostRegressor()",
            child_code="model = CatBoostRegressor(early_stopping_rounds=100)",
        )
        # Don't double-prefix
        assert result["description"].count("UNVERIFIED_") == 1
