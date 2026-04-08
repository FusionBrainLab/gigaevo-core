# tests/memory/ideas_tracker/test_models.py
"""Tests for gigaevo.memory.ideas_tracker.models."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    ClassificationChunk,
    EmbeddedIdea,
    Idea,
    IdeaExplanation,
    IdeaUpdate,
    ProgramRecord,
    normalize_improvement_item,
    normalize_improvements,
    program_to_record,
    programs_to_records,
)


class TestNormalizeImprovementItem:
    def test_string_becomes_description(self) -> None:
        result = normalize_improvement_item("Use BFS traversal")
        assert result == {"description": "Use BFS traversal", "explanation": ""}

    def test_dict_with_description_and_explanation(self) -> None:
        result = normalize_improvement_item({"description": "Add cache", "explanation": "reduces calls"})
        assert result["description"] == "Add cache"
        assert result["explanation"] == "reduces calls"

    def test_dict_with_alternative_description_key(self) -> None:
        result = normalize_improvement_item({"summary": "Switched algo", "reason": "faster"})
        assert result["description"] == "Switched algo"
        assert result["explanation"] == "faster"

    def test_non_dict_non_string_uses_stringify(self) -> None:
        result = normalize_improvement_item(42)
        assert result["description"] == "42"
        assert result["explanation"] == ""

    def test_empty_dict_returns_unspecified(self) -> None:
        result = normalize_improvement_item({})
        assert result["description"] == "Unspecified change"

    def test_none_returns_unspecified(self) -> None:
        result = normalize_improvement_item(None)
        assert result["description"] == "Unspecified change"

    def test_whitespace_only_string_returns_unspecified(self) -> None:
        result = normalize_improvement_item("   ")
        assert result["description"] == "Unspecified change"


class TestNormalizeImprovements:
    def test_none_returns_empty_list(self) -> None:
        assert normalize_improvements(None) == []

    def test_list_of_dicts(self) -> None:
        result = normalize_improvements([{"description": "A"}, {"description": "B"}])
        assert len(result) == 2
        assert result[0]["description"] == "A"

    def test_single_non_list_is_wrapped(self) -> None:
        result = normalize_improvements("Single change")
        assert len(result) == 1
        assert result[0]["description"] == "Single change"


class TestIdeaModel:
    def test_id_auto_generated(self) -> None:
        idea = Idea(description="Use BFS")
        assert len(idea.id) == 36  # UUID4 length

    def test_two_ideas_have_different_ids(self) -> None:
        a = Idea(description="A")
        b = Idea(description="B")
        assert a.id != b.id

    def test_explanation_defaults_to_empty(self) -> None:
        idea = Idea(description="test")
        assert idea.explanation.entries == []
        assert idea.explanation.summary == ""

    def test_model_dump_is_serialisable(self) -> None:
        idea = Idea(description="test", programs=["p1"])
        d = idea.model_dump()
        assert d["description"] == "test"
        assert d["programs"] == ["p1"]


class TestAnalysisResult:
    def test_defaults_to_empty_lists(self) -> None:
        result = AnalysisResult()
        assert result.new_ideas == []
        assert result.updates == []

    def test_holds_ideas_and_updates(self) -> None:
        idea = Idea(description="Cache retrieval")
        update = IdeaUpdate(idea_id="abc-123", programs=["p1"])
        result = AnalysisResult(new_ideas=[idea], updates=[update])
        assert len(result.new_ideas) == 1
        assert len(result.updates) == 1


class TestProgramToRecord:
    def _make_program(
        self,
        *,
        fitness: float = 0.75,
        fitness_key: str = "fitness",
        generation: int = 3,
        parents: list[str] | None = None,
        mutation_output: dict | None = None,
    ) -> MagicMock:
        prog = MagicMock()
        prog.id = "prog-uuid-001"
        prog.code = "def solve(): return 42"
        prog.metrics = {fitness_key: fitness}
        prog.lineage.generation = generation
        prog.lineage.parents = parents or ["parent-uuid-001"]
        prog.metadata = {}
        if mutation_output is not None:
            prog.metadata["mutation_output"] = mutation_output
        return prog

    def test_basic_field_mapping(self) -> None:
        prog = self._make_program(
            fitness=7.5,
            generation=4,
            mutation_output={"insights_used": ["Use BFS"], "archetype": "exploration"},
        )
        record = program_to_record(prog, "Solve TSP", "TSP optimisation")
        assert record.id == "prog-uuid-001"
        assert record.fitness == 7.5
        assert record.generation == 4
        assert record.insights == ["Use BFS"]
        assert record.strategy == "exploration"
        assert record.task_description == "Solve TSP"
        assert record.task_description_summary == "TSP optimisation"

    def test_missing_mutation_output_defaults_to_empty(self) -> None:
        prog = self._make_program()
        record = program_to_record(prog, "task", "summary")
        assert record.insights == []
        assert record.strategy == ""
        assert record.improvements == []

    def test_invalid_mutation_output_type_defaults_to_empty(self) -> None:
        prog = self._make_program()
        prog.metadata["mutation_output"] = "not a dict"
        record = program_to_record(prog, "task", "summary")
        assert record.insights == []

    def test_custom_fitness_key(self) -> None:
        prog = self._make_program(fitness_key="accuracy", fitness=0.95)
        record = program_to_record(prog, "task", "summary", fitness_key="accuracy")
        assert record.fitness == 0.95

    def test_programs_to_records_returns_ids(self) -> None:
        progs = [self._make_program() for _ in range(3)]
        for i, p in enumerate(progs):
            p.id = f"id-{i}"
        records, ids = programs_to_records(progs, "task", "summary")
        assert len(records) == 3
        assert ids == {"id-0", "id-1", "id-2"}
