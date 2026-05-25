"""Tests for apply_render_filters (CARD_STRUCTURE_v3 §3).

RED-phase TDD tests written before implementation.

Contract: apply_render_filters takes a list[AnyCard] and applies the v3
render-time filtering pipeline:
  1. MemoryCards: drop unverified-with-low-support, drop near-duplicates,
     rank by support * max(delta_best, 0), dedup by canonical key.
  2. ProgramCards: drop pending-analysis, drop empty-with-no-ideas,
     rank by fitness, append after MemoryCards.
  3. Missing evolution_statistics defaults to zero-evidence shape.
"""

from __future__ import annotations

from gigaevo.memory.shared_memory.card_search import apply_render_filters
from gigaevo.memory.shared_memory.models import (
    MemoryCard,
    MemoryCardExplanation,
    ProgramCard,
)


def _mcard(
    id: str,
    description: str = "ADD x: y; support=1; Δbest=+0.01; co=[]",
    keywords: list[str] | None = None,
    evolution_statistics: dict | None = None,
) -> MemoryCard:
    return MemoryCard(
        id=id,
        description=description,
        keywords=keywords or [],
        evolution_statistics=evolution_statistics or {},
        explanation=MemoryCardExplanation(),
    )


def _pcard(
    id: str,
    description: str = "PROGRAM rank=1: ...; fitness=0.05",
    keywords: list[str] | None = None,
    fitness: float | None = 0.05,
    connected_ideas: list | None = None,
) -> ProgramCard:
    return ProgramCard(
        id=id,
        program_id=id.replace("program-", ""),
        description=description,
        keywords=keywords or [],
        fitness=fitness,
        connected_ideas=connected_ideas or [],
    )


# ---------------------------------------------------------------------------
# MemoryCard filters
# ---------------------------------------------------------------------------


class TestMemoryCardFilters:
    def test_drop_unverified_with_low_support(self) -> None:
        cards = [
            _mcard(
                "a",
                keywords=["verified:false", "canonical:USE:x:_:1"],
                evolution_statistics={"support": 2, "delta_best": 0.01},
            ),
        ]
        result = apply_render_filters(cards)
        assert result == []

    def test_keep_unverified_with_high_support(self) -> None:
        cards = [
            _mcard(
                "a",
                keywords=["verified:false", "canonical:USE:x:_:1"],
                evolution_statistics={"support": 3, "delta_best": 0.01},
            ),
        ]
        result = apply_render_filters(cards)
        assert [c.id for c in result] == ["a"]

    def test_dedup_by_canonical_key(self) -> None:
        cards = [
            _mcard(
                "a",
                keywords=["canonical:UPDATE:depth:6:7"],
                evolution_statistics={"support": 1, "delta_best": 0.01},
            ),
            _mcard(
                "b",
                keywords=["canonical:UPDATE:depth:6:7"],
                evolution_statistics={"support": 1, "delta_best": 0.02},
            ),
        ]
        result = apply_render_filters(cards)
        assert len(result) == 1
        # Higher delta_best wins
        assert result[0].id == "b"

    def test_rank_by_support_times_positive_delta_best(self) -> None:
        cards = [
            _mcard(
                "low_score",
                keywords=["canonical:A:1:_:_"],
                evolution_statistics={"support": 1, "delta_best": 0.05},
            ),
            _mcard(
                "high_score",
                keywords=["canonical:B:1:_:_"],
                evolution_statistics={"support": 5, "delta_best": 0.10},
            ),
            _mcard(
                "regressed",
                keywords=["canonical:C:1:_:_"],
                evolution_statistics={"support": 10, "delta_best": -0.05},
            ),
        ]
        result = apply_render_filters(cards)
        assert [c.id for c in result] == ["high_score", "low_score", "regressed"]

    def test_missing_evolution_statistics_defaults_to_zero_evidence(self) -> None:
        cards = [
            _mcard(
                "no_stats",
                keywords=["canonical:A:1:_:_"],
                evolution_statistics={},
            ),
        ]
        result = apply_render_filters(cards)
        # Survives but ranked last (zero evidence)
        assert [c.id for c in result] == ["no_stats"]


# ---------------------------------------------------------------------------
# ProgramCard filters
# ---------------------------------------------------------------------------


class TestProgramCardFilters:
    def test_drop_pending_analysis(self) -> None:
        cards = [_pcard("program-1", keywords=["pending_analysis:true"])]
        result = apply_render_filters(cards)
        assert result == []

    def test_drop_empty_description_and_no_connected_ideas(self) -> None:
        cards = [_pcard("program-1", description="", connected_ideas=[])]
        result = apply_render_filters(cards)
        assert result == []

    def test_keep_program_with_connected_ideas(self) -> None:
        from gigaevo.memory.shared_memory.models import ConnectedIdea

        cards = [
            _pcard(
                "program-1",
                description="PROGRAM rank=1",
                connected_ideas=[ConnectedIdea(idea_id="i1", description="d")],
            )
        ]
        result = apply_render_filters(cards)
        assert [c.id for c in result] == ["program-1"]

    def test_keep_program_with_nonempty_description_even_no_ideas(self) -> None:
        cards = [_pcard("program-1", description="PROGRAM rank=1", connected_ideas=[])]
        result = apply_render_filters(cards)
        assert [c.id for c in result] == ["program-1"]


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


class TestRenderOrdering:
    def test_memory_cards_render_before_program_cards(self) -> None:
        mcard = _mcard(
            "m1",
            keywords=["canonical:A:1:_:_"],
            evolution_statistics={"support": 1, "delta_best": 0.01},
        )
        pcard = _pcard("program-1", description="PROGRAM rank=1")
        result = apply_render_filters([pcard, mcard])
        assert [c.id for c in result] == ["m1", "program-1"]

    def test_empty_input_returns_empty(self) -> None:
        assert apply_render_filters([]) == []
