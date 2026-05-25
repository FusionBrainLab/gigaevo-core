from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.memory_quality_audit import (
    AuditReport,
    audit_run,
    is_stub_description,
    is_tautology,
    normalize_target_stem,
)

PRE_V4_RUN = Path(
    "~/gigaevo/output/tabular_regression_intra_extra_20260523_161718"
)


def _make_card(category: str, description: str, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": extra.pop("id", f"card-{abs(hash(description)) % 100000}"),
        "category": category,
        "description": description,
        "keywords": extra.pop("keywords", []),
    }
    base.update(extra)
    return base


def _make_store(cards: list[dict[str, object]], tmp_path: Path) -> Path:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    api_index = {
        "entity_by_card_id": {},
        "entity_version_by_entity": {},
        "memory_cards": {c["id"]: c for c in cards},
    }
    (memory_dir / "api_index.json").write_text(json.dumps(api_index))
    return tmp_path


class TestIsStubDescription:
    def test_canonical_pending_analysis_stub(self) -> None:
        desc = "Top-3 program (fitness=-0.428681); no recorded idea lineage - inspect `code` field for mechanism."
        assert is_stub_description(desc) is True

    def test_canonical_packed_grammar_not_stub(self) -> None:
        desc = "Removed target_log_transform log->raw: gradient updates match evaluation metric scale"
        assert is_stub_description(desc) is False

    def test_empty_description_treated_as_stub(self) -> None:
        assert is_stub_description("") is True


class TestIsTautology:
    @pytest.mark.parametrize(
        "mechanism",
        [
            "models complex non-linear interactions",
            "captures slower-converging geographic patterns",
            "allows more complex trees without overfitting",
            "fundamental demographic unit for demand",
            "captures spatial efficiency per household",
            "captures occupancy density patterns",
            "reflects housing density in urban vs rural",
        ],
    )
    def test_seed_tautology_templates_from_pre_v4(self, mechanism: str) -> None:
        assert is_tautology(mechanism) is True

    @pytest.mark.parametrize(
        "mechanism",
        [
            "gradient updates match evaluation metric scale",
            "aligns training objective with raw-scale RMSE metric",
            "avoids RMSE penalty from sub-0.15 predictions",
            "tree splits model non-linearities; early stopping halts at val min",
            "models heavy-tailed population count",
            "out-of-fold meta-learner captures non-linear model interactions",
            "reduces skew in heavy-tailed population distribution",
            "balances rare capped class",
        ],
    )
    def test_specific_mechanisms_are_not_tautology(self, mechanism: str) -> None:
        assert is_tautology(mechanism) is False


class TestNormalizeTargetStem:
    def test_strips_log_transform_suffix(self) -> None:
        assert normalize_target_stem("target_log_transform") == "target_transform"

    def test_strips_train_suffix(self) -> None:
        assert normalize_target_stem("household_count_train") == "household_count"

    def test_does_not_merge_distinct_levers(self) -> None:
        assert normalize_target_stem("n_neighbors") != normalize_target_stem(
            "n_clusters"
        )

    def test_idempotent(self) -> None:
        assert normalize_target_stem("learning_rate") == "learning_rate"


class TestAuditRunOnPreV4:
    """End-to-end gate per plans/memory-system-quality-boost.md §3 A.1.

    Harness must reproduce the hand-grading on
    output/tabular_regression_intra_extra_20260523_161718 within ±2 cards on each metric.
    """

    @pytest.fixture(scope="class")
    def report(self) -> AuditReport:
        if not (PRE_V4_RUN / "memory" / "api_index.json").exists():
            pytest.skip(f"PRE-v4 run dir missing: {PRE_V4_RUN}")
        return audit_run(PRE_V4_RUN)

    def test_total_cards_exact(self, report: AuditReport) -> None:
        assert report.total_cards == 73

    def test_program_card_count_exact(self, report: AuditReport) -> None:
        assert report.program_count == 55

    def test_general_card_count_exact(self, report: AuditReport) -> None:
        assert report.general_count == 18

    def test_stub_count_within_tolerance(self, report: AuditReport) -> None:
        # Hand-graded: 45/55 stubs. Tolerance ±2 cards.
        assert 43 <= report.stub_count <= 47

    def test_specific_idea_count_within_tolerance(self, report: AuditReport) -> None:
        # Hand-graded: 11/18 specific. Tolerance ±2 cards.
        assert 9 <= report.specific_idea_count <= 13

    def test_known_dedup_pair_target_transform_flagged(
        self, report: AuditReport
    ) -> None:
        # #00 "Removed target_log_transform" + #01 "Replaced target_transform" → same lever
        flagged_targets = {t for group in report.dedup_collisions for t in group}
        assert "target_transform" in flagged_targets

    def test_known_dedup_pair_n_clusters_flagged(self, report: AuditReport) -> None:
        # #11 "Raised n_clusters 10->15" + #13 "Lowered n_clusters 50->15" — same target
        flagged_targets = {t for group in report.dedup_collisions for t in group}
        assert "n_clusters" in flagged_targets


class TestAuditRunSyntheticStores:
    """Unit-level audit_run against constructed stores."""

    def test_empty_store_yields_zero_counts(self, tmp_path: Path) -> None:
        run = _make_store([], tmp_path)
        report = audit_run(run)
        assert report.total_cards == 0
        assert report.program_count == 0
        assert report.general_count == 0
        assert report.stub_count == 0
        assert report.dedup_collisions == []

    def test_distinct_levers_not_flagged_as_dup(self, tmp_path: Path) -> None:
        cards = [
            _make_card(
                "general", "Raised n_neighbors 5->7: smooths urban density noise"
            ),
            _make_card(
                "general",
                "Raised n_clusters 10->15: improves coastal-inland resolution",
            ),
            _make_card(
                "general",
                "Raised early_stopping_rounds 50->100: captures slow convergence",
            ),
        ]
        run = _make_store(cards, tmp_path)
        report = audit_run(run)
        assert report.dedup_collisions == []

    def test_target_stem_collision_flagged(self, tmp_path: Path) -> None:
        cards = [
            _make_card(
                "general",
                "Removed target_log_transform log->raw: gradient matches metric",
            ),
            _make_card(
                "general",
                "Replaced target_transform log->raw: aligns objective with metric",
            ),
        ]
        run = _make_store(cards, tmp_path)
        report = audit_run(run)
        flagged_targets = {t for group in report.dedup_collisions for t in group}
        assert "target_transform" in flagged_targets

    def test_specificity_rate_computed_on_general_only(self, tmp_path: Path) -> None:
        cards = [
            _make_card(
                "general", "Raised depth 6->7: models complex non-linear interactions"
            ),  # tautology
            _make_card(
                "general",
                "Removed target_log_transform log->raw: gradient matches metric",
            ),  # specific
            _make_card(
                "program", "Top-1 program; no recorded idea lineage"
            ),  # stub, doesn't affect rate
        ]
        run = _make_store(cards, tmp_path)
        report = audit_run(run)
        assert report.general_count == 2
        assert report.specific_idea_count == 1

    def test_stub_count_uses_program_only(self, tmp_path: Path) -> None:
        cards = [
            _make_card(
                "general", "Raised depth 6->7: models complex non-linear interactions"
            ),  # general, not stub
            _make_card("program", "Top-1 program; no recorded idea lineage"),  # stub
            _make_card(
                "program", "Raised depth 6->7: shallow trees reduce variance"
            ),  # real packed
        ]
        run = _make_store(cards, tmp_path)
        report = audit_run(run)
        assert report.program_count == 2
        assert report.stub_count == 1
