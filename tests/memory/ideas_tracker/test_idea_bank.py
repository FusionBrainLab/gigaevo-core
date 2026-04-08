# tests/memory/ideas_tracker/test_idea_bank.py
"""Tests for gigaevo.memory.ideas_tracker.idea_bank.IdeaBank."""
from __future__ import annotations

import pytest

from gigaevo.memory.ideas_tracker.idea_bank import IdeaBank, merge_usage_payloads
from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    Idea,
    IdeaExplanation,
    IdeaUpdate,
)


def _idea(description: str = "Test idea", **kwargs) -> Idea:
    return Idea(description=description, **kwargs)


class TestIdeaBankAdd:
    def test_add_single_idea(self) -> None:
        bank = IdeaBank()
        idea = _idea("Use BFS")
        bank.add(idea)
        assert len(bank.all_ideas()) == 1

    def test_add_reassigns_duplicate_id(self) -> None:
        bank = IdeaBank()
        idea = _idea("First")
        bank.add(idea)
        duplicate = idea.model_copy()
        bank.add(duplicate)
        ids = [i.id for i in bank.all_ideas()]
        assert len(set(ids)) == 2  # both ideas have unique ids

    def test_get_returns_idea_by_id(self) -> None:
        bank = IdeaBank()
        idea = _idea("Cache results")
        bank.add(idea)
        assert bank.get(idea.id) is not None
        assert bank.get(idea.id).description == "Cache results"

    def test_get_returns_none_for_unknown_id(self) -> None:
        bank = IdeaBank()
        assert bank.get("nonexistent") is None


class TestIdeaBankApply:
    def test_apply_adds_new_ideas(self) -> None:
        bank = IdeaBank()
        result = AnalysisResult(new_ideas=[_idea("A"), _idea("B")])
        bank.apply(result)
        assert len(bank.all_ideas()) == 2

    def test_apply_runs_updates(self) -> None:
        bank = IdeaBank()
        idea = _idea("Original description", programs=["p0"])
        bank.add(idea)
        upd = IdeaUpdate(idea_id=idea.id, programs=["p1"], generation=5)
        bank.apply(AnalysisResult(updates=[upd]))
        updated = bank.get(idea.id)
        assert "p1" in updated.programs
        assert updated.last_generation == 5


class TestIdeaBankUpdate:
    def test_update_appends_programs(self) -> None:
        bank = IdeaBank()
        idea = _idea("Idea", programs=["p0"])
        bank.add(idea)
        bank.update(IdeaUpdate(idea_id=idea.id, programs=["p1", "p2"]))
        assert set(bank.get(idea.id).programs) == {"p0", "p1", "p2"}

    def test_update_deduplicates_programs(self) -> None:
        bank = IdeaBank()
        idea = _idea("Idea", programs=["p0"])
        bank.add(idea)
        bank.update(IdeaUpdate(idea_id=idea.id, programs=["p0"]))
        assert bank.get(idea.id).programs.count("p0") == 1

    def test_update_bumps_generation_if_greater(self) -> None:
        bank = IdeaBank()
        idea = _idea("Idea", last_generation=3)
        bank.add(idea)
        bank.update(IdeaUpdate(idea_id=idea.id, generation=7))
        assert bank.get(idea.id).last_generation == 7

    def test_update_does_not_lower_generation(self) -> None:
        bank = IdeaBank()
        idea = _idea("Idea", last_generation=10)
        bank.add(idea)
        bank.update(IdeaUpdate(idea_id=idea.id, generation=2))
        assert bank.get(idea.id).last_generation == 10

    def test_update_new_description_archives_old(self) -> None:
        bank = IdeaBank()
        idea = _idea("Old description")
        bank.add(idea)
        bank.update(IdeaUpdate(idea_id=idea.id, new_description="New description"))
        updated = bank.get(idea.id)
        assert updated.description == "New description"
        assert len(updated.aliases) == 1

    def test_update_appends_motivation_to_explanation(self) -> None:
        bank = IdeaBank()
        idea = _idea("Idea", explanation=IdeaExplanation(entries=["first"]))
        bank.add(idea)
        bank.update(IdeaUpdate(idea_id=idea.id, motivation="second reason"))
        updated = bank.get(idea.id)
        assert "second reason" in updated.explanation.entries

    def test_update_returns_false_for_unknown_id(self) -> None:
        bank = IdeaBank()
        assert bank.update(IdeaUpdate(idea_id="ghost")) is False


class TestIdeaBankEnrich:
    def test_enrich_sets_keywords_and_summary(self) -> None:
        bank = IdeaBank()
        idea = _idea("Some idea")
        bank.add(idea)
        bank.enrich(idea.id, keywords=["bfs", "graph"], summary="Uses BFS", task_summary="TSP")
        enriched = bank.get(idea.id)
        assert enriched.keywords == ["bfs", "graph"]
        assert enriched.explanation.summary == "Uses BFS"
        assert enriched.task_description_summary == "TSP"

    def test_enrich_returns_false_for_unknown_id(self) -> None:
        bank = IdeaBank()
        assert bank.enrich("ghost", keywords=[], summary="", task_summary="") is False


class TestIdeaBankChunks:
    def test_empty_bank_returns_empty_chunks(self) -> None:
        bank = IdeaBank(chunk_size=5)
        assert bank.classification_chunks() == []

    def test_chunk_size_splits_correctly(self) -> None:
        bank = IdeaBank(chunk_size=3)
        for i in range(7):
            bank.add(_idea(f"Idea {i}"))
        chunks = bank.classification_chunks()
        assert len(chunks) == 3  # 3 + 3 + 1

    def test_chunk_text_contains_short_ids(self) -> None:
        bank = IdeaBank(chunk_size=5)
        bank.add(_idea("Cache calls"))
        chunk = bank.classification_chunks()[0]
        assert "Cache calls" in chunk.text
        assert len(chunk.short_ids) == 1
        assert "short_id" in chunk.short_ids[0]


class TestMergeUsagePayloads:
    def test_merge_combines_deltas(self) -> None:
        existing = {"used": {"entries": [{"task_description_summary": "task", "fitness_delta_per_use": [2.0], "used_count": 1, "median_delta_fitness": 2.0}], "total": {"total_used": 1, "median_delta_fitness": 2.0}}}
        incoming = {"used": {"entries": [{"task_description_summary": "task", "fitness_delta_per_use": [4.0], "used_count": 1, "median_delta_fitness": 4.0}], "total": {"total_used": 1, "median_delta_fitness": 4.0}}}
        merged = merge_usage_payloads(existing, incoming)
        deltas = merged["used"]["entries"][0]["fitness_delta_per_use"]
        assert sorted(deltas) == [2.0, 4.0]
