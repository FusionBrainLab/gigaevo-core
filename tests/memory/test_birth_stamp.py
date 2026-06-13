"""Birth-time memory attribution stamp on child programs.

``generate_one_mutation`` must freeze the union of the parents' prompt-time
selected card ids onto the child. Posterior attribution reads the child's
stamp, so later re-selection on the parent (NO_CACHE MemoryContextStage
re-runs the stochastic auction on every requeue) cannot drift the credit.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from gigaevo.evolution.engine.mutation import generate_one_mutation
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_USED_METADATA_KEY,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _make_parent(selected: list[str] | None = None) -> Program:
    parent = Program(code="def solve(): return 42", state=ProgramState.DONE)
    if selected is not None:
        parent.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, selected)
    return parent


def _make_deps(parents: list[Program]) -> tuple[Any, Any, Any]:
    storage = AsyncMock()
    state_manager = AsyncMock()
    mutator = AsyncMock()
    storage.get.return_value = parents[0]
    mutator.mutate_single.return_value = MutationSpec(
        code="def solve(): return 0",
        parents=parents,
        name="mut",
        metadata={},
    )
    return mutator, storage, state_manager


async def _run_and_capture_child(parents: list[Program]) -> Program:
    mutator, storage, state_manager = _make_deps(parents)
    child_id = await generate_one_mutation(
        parents,
        mutator=mutator,
        storage=storage,
        state_manager=state_manager,
        iteration=1,
    )
    assert child_id is not None
    (child,) = storage.add.call_args.args
    return child


class TestBirthStamp:
    @pytest.mark.asyncio
    async def test_child_stamped_with_sorted_union_of_parent_slates(self) -> None:
        parents = [_make_parent(["c2", "c1"]), _make_parent(["c1", "c3"])]
        child = await _run_and_capture_child(parents)
        assert child.get_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY) == [
            "c1",
            "c2",
            "c3",
        ]
        assert child.get_metadata(MUTATION_MEMORY_USED_METADATA_KEY) is True

    @pytest.mark.asyncio
    async def test_child_stamped_empty_when_parents_have_empty_slates(self) -> None:
        parents = [_make_parent([]), _make_parent(None)]
        child = await _run_and_capture_child(parents)
        assert child.get_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY) == []
        assert child.get_metadata(MUTATION_MEMORY_USED_METADATA_KEY) is False

    @pytest.mark.asyncio
    async def test_stamp_survives_parent_slate_overwrite_after_birth(self) -> None:
        parent = _make_parent(["c1"])
        child = await _run_and_capture_child([parent])
        parent.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, ["c9"])
        assert child.get_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY) == ["c1"]

    @pytest.mark.asyncio
    async def test_blank_ids_excluded_from_stamp(self) -> None:
        parents = [_make_parent(["", "c1"])]
        child = await _run_and_capture_child(parents)
        assert child.get_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY) == ["c1"]
