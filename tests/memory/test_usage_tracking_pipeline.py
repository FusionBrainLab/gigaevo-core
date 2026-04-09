"""Tests for the usage-tracking sub-pipeline of IdeaTracker.

Layer 1 (unit) — two test classes:
  TestBuildUsageUpdatesExtended: _build_usage_updates (ideas_tracker.py)
  TestMergeUsagePayloadsWithObjects: merge_usage_payloads (idea_bank.py)

Regression tests in TestMergeUsagePayloadsWithObjects expose a bug in
_extract_task_deltas: model_dump() output (a dict with keys "entries",
"total_used", "median_delta_fitness" but no "used" top-level key) is
silently treated as empty, causing incoming deltas to be lost.
"""

from __future__ import annotations

import uuid
from typing import Any

from gigaevo.memory.ideas_tracker.idea_bank import (
    build_usage_payload,
    merge_usage_payloads,
)
from gigaevo.memory.ideas_tracker.ideas_tracker import _build_usage_updates
from gigaevo.memory.ideas_tracker.models import Idea, UsagePayload
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState

_TEST_NAMESPACE = uuid.NAMESPACE_DNS


def _uuid(test_id: str) -> str:
    return str(uuid.uuid5(_TEST_NAMESPACE, test_id))


def _make_program(
    *,
    code: str = "def solve(): return 42",
    fitness: float = 0.75,
    is_valid: float = 1.0,
    fitness_key: str = "fitness",
    generation: int = 3,
    parents: list[str] | None = None,
    memory_ids: list[str] | None = None,
    state: ProgramState = ProgramState.DONE,
    program_id: str | None = None,
) -> Program:
    """Create a test Program with sensible defaults."""
    metadata: dict[str, Any] = {}
    if memory_ids is not None:
        metadata["memory_selected_idea_ids"] = memory_ids
    parent_list = parents or (["parent-1"] if generation > 1 else [])
    parent_uuids = [_uuid(p) if isinstance(p, str) else p for p in parent_list]
    lineage = Lineage(parents=parent_uuids, generation=max(generation, 1))
    prog = Program(
        code=code,
        state=state,
        metrics={fitness_key: fitness, "is_valid": is_valid},
        metadata=metadata,
        lineage=lineage,
    )
    if program_id is not None:
        object.__setattr__(prog, "id", _uuid(program_id))
    return prog


def _make_memory_program(
    *,
    fitness: float = 8.0,
    is_valid: float = 1.0,
    parent_id: str = "parent-a",
    card_ids: list[str] | None = None,
    generation: int = 5,
) -> Program:
    """Create a program that used memory cards during mutation."""
    return _make_program(
        fitness=fitness,
        is_valid=is_valid,
        generation=generation,
        parents=[parent_id],
        memory_ids=card_ids or ["idea-001", "idea-002"],
    )


def _make_single_task_payload(
    task: str = "task-A",
    deltas: list[float] | None = None,
) -> UsagePayload:
    """Create a single-task UsagePayload via build_usage_payload."""
    if deltas is None:
        deltas = [1.0, 3.0]
    return build_usage_payload({task: deltas})


def _make_multi_task_payload() -> UsagePayload:
    """Create a multi-task UsagePayload with 3 tasks and varied deltas."""
    return build_usage_payload(
        {
            "task-alpha": [0.1, 0.2, 0.3],
            "task-beta": [-0.5, 0.4],
            "task-gamma": [1.0],
        }
    )


def _make_idea_with_usage(
    *,
    idea_id: str = "idea-test",
    usage: UsagePayload | None = None,
) -> Idea:
    """Create an Idea with optional pre-populated usage."""
    idea = Idea(description=f"Test idea {idea_id}")
    if usage is not None:
        idea = idea.model_copy(update={"usage": usage})
    return idea


class TestBuildUsageUpdatesExtended:
    def test_build_usage_updates_from_single_mutation(self) -> None:
        parent = _make_program(program_id="p1", fitness=5.0, parents=[], generation=1)
        child = _make_memory_program(fitness=8.0, parent_id="p1", card_ids=["card-1"])
        result = _build_usage_updates([parent, child], "task-A", "fitness")

        assert "card-1" in result
        assert isinstance(result["card-1"], UsagePayload)
        assert result["card-1"].total_used == 1
        assert result["card-1"].entries[0].task_description_summary == "task-A"
        assert result["card-1"].entries[0].fitness_delta_per_use == [3.0]
        assert result["card-1"].median_delta_fitness == 3.0

    def test_build_usage_updates_deduplicates_repeated_card_ids(self) -> None:
        parent = _make_program(program_id="p1", fitness=1.0, parents=[], generation=1)
        child = _make_memory_program(
            fitness=2.0, parent_id="p1", card_ids=["dup", "dup", "dup"]
        )
        result = _build_usage_updates([parent, child], "task-A", "fitness")

        assert "dup" in result
        assert result["dup"].total_used == 1
        assert result["dup"].entries[0].used_count == 1

    def test_build_usage_updates_accumulates_across_multiple_programs(self) -> None:
        parent = _make_program(program_id="p0", fitness=4.0, parents=[], generation=1)
        child1 = _make_memory_program(
            fitness=6.0, parent_id="p0", generation=2, card_ids=["card-1"]
        )
        child2 = _make_memory_program(
            fitness=7.0, parent_id="p0", generation=2, card_ids=["card-1", "card-2"]
        )
        result = _build_usage_updates([parent, child1, child2], "task", "fitness")

        assert result["card-1"].total_used == 2
        assert result["card-1"].entries[0].fitness_delta_per_use == [2.0, 3.0]
        assert result["card-2"].total_used == 1
        assert result["card-2"].entries[0].fitness_delta_per_use == [3.0]

    def test_build_usage_updates_result_is_usage_payload_not_dict(self) -> None:
        parent = _make_program(program_id="p1", fitness=5.0, parents=[], generation=1)
        child = _make_memory_program(fitness=6.0, parent_id="p1", card_ids=["c1"])
        result = _build_usage_updates([parent, child], "task", "fitness")

        for value in result.values():
            assert isinstance(value, UsagePayload)


class TestMergeUsagePayloadsWithObjects:
    def test_merge_usage_payloads_combines_per_task_deltas(self) -> None:
        existing = _make_single_task_payload("task-A", [1.0])
        incoming = _make_single_task_payload("task-A", [3.0])

        merged = merge_usage_payloads(existing, incoming)

        assert isinstance(merged, UsagePayload)
        assert merged.total_used == 2
        assert len(merged.entries) == 1
        assert merged.entries[0].fitness_delta_per_use == [1.0, 3.0]
        assert merged.median_delta_fitness == 2.0

    def test_merge_usage_payloads_object_plus_model_dump_dict(self) -> None:
        existing = _make_single_task_payload("task-A", [1.0])
        incoming_dict = _make_single_task_payload("task-A", [3.0]).model_dump()

        merged = merge_usage_payloads(existing, incoming_dict)

        assert merged.total_used == 2
        assert merged.entries[0].fitness_delta_per_use == [1.0, 3.0]
        assert merged.median_delta_fitness == 2.0

    def test_merge_usage_payloads_model_dump_plus_object(self) -> None:
        existing_dict = _make_single_task_payload("task-A", [1.0]).model_dump()
        incoming = _make_single_task_payload("task-A", [3.0])

        merged = merge_usage_payloads(existing_dict, incoming)

        assert merged.total_used == 2
        assert merged.entries[0].fitness_delta_per_use == [1.0, 3.0]

    def test_merge_usage_payloads_multi_task_objects(self) -> None:
        existing = _make_multi_task_payload()
        incoming = build_usage_payload(
            {
                "task-alpha": [0.9],
                "task-delta": [0.5],
            }
        )

        merged = merge_usage_payloads(existing, incoming)

        alpha_entry = next(
            e for e in merged.entries if e.task_description_summary == "task-alpha"
        )
        assert len(alpha_entry.fitness_delta_per_use) == 4
        delta_entry = next(
            e for e in merged.entries if e.task_description_summary == "task-delta"
        )
        assert len(delta_entry.fitness_delta_per_use) == 1

    def test_merge_usage_payloads_empty_object_is_identity(self) -> None:
        existing = _make_single_task_payload("task-A", [1.0, 3.0])
        incoming = UsagePayload()

        merged = merge_usage_payloads(existing, incoming)

        assert merged.total_used == existing.total_used
        assert merged.entries == existing.entries
        assert merged.median_delta_fitness == existing.median_delta_fitness
