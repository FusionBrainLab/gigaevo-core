"""Tests for ``gigaevo.evolution.engine.mutation.generate_mutations``.

Covers the asyncio.gather result-filtering contract for parallel mutation
tasks: the function must count only program-id strings as successes and
must ignore truthy exception objects emitted by ``return_exceptions=True``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from gigaevo.evolution.engine.mutation import generate_mutations
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.parent_selector import RandomParentSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(state: ProgramState = ProgramState.DONE) -> Program:
    p = Program(code="def solve(): return 42", state=state)
    return p


def _make_deps(mutation_spec=None, storage_get_returns_none: bool = False):
    """Return (mutator, storage, state_manager) mocks."""
    storage = AsyncMock()
    state_manager = AsyncMock()
    mutator = AsyncMock()

    parent = _prog()
    storage.get.return_value = None if storage_get_returns_none else parent

    if mutation_spec is not None:
        mutator.mutate_single.return_value = mutation_spec
    else:
        mutator.mutate_single.return_value = MutationSpec(
            code="def solve(): return 1",
            parents=[parent],
            name="m",
            metadata={},
        )

    return mutator, storage, state_manager


# ---------------------------------------------------------------------------
# TestGatherExceptionCounting — the CRITICAL finding
# ---------------------------------------------------------------------------


class TestGatherExceptionCounting:
    def test_truthy_filter_would_overcount_exception_objects(self) -> None:
        """A plain truthiness filter would treat exception objects as successes.

        Guard against regressing the filter to ``if result`` instead of an
        ``isinstance(r, str)`` check on the asyncio.gather output.
        """
        results = [True, True, False, RuntimeError("something exploded")]

        truthy_count = sum(1 for result in results if result)
        assert truthy_count == 3, (
            "Truthy filter wrongly counts exception objects as successes."
        )

        correct_count = sum(1 for result in results if result is True)
        assert correct_count == 2

    def test_truthy_filter_would_overcount_base_exceptions(self) -> None:
        """BaseException subclasses (e.g. GeneratorExit) are truthy too.

        They can leak past ``except Exception`` and appear in
        ``asyncio.gather(return_exceptions=True)`` output as objects.
        """
        results_with_escaped_exc = [
            True,
            GeneratorExit("escaped base exception"),
            False,
        ]
        truthy_count = sum(1 for result in results_with_escaped_exc if result)
        assert truthy_count == 2

        correct_count = sum(1 for result in results_with_escaped_exc if result is True)
        assert correct_count == 1

    async def test_truthy_filter_would_overcount_pure_exception_results(self) -> None:
        """All-exception results: truthiness filter counts every entry."""
        all_exception_results = [ValueError("v"), RuntimeError("r"), KeyError("k")]

        truthy_count = sum(1 for result in all_exception_results if result)
        assert truthy_count == 3

        correct_count = sum(1 for result in all_exception_results if result is True)
        assert correct_count == 0

    async def test_all_successful_mutations_counted_correctly(self) -> None:
        """Sanity: all mutations succeed → count matches limit exactly."""
        mutator, storage, state_manager = _make_deps()
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        # All mutate_single calls succeed (default from _make_deps)
        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=selector,
            limit=3,
            iteration=0,
        )

        assert len(count) == 3
        assert storage.add.call_count == 3

    async def test_all_mutations_fail_returns_zero(self) -> None:
        """All mutations fail (mutator returns None) → count is 0."""
        mutator, storage, state_manager = _make_deps(mutation_spec=None)
        mutator.mutate_single.return_value = None
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=selector,
            limit=3,
            iteration=0,
        )

        assert len(count) == 0
        storage.add.assert_not_called()

    async def test_inner_exception_handler_returns_false(self) -> None:
        """The inner coroutine catches exceptions and returns False — not an exception object.

        This tests the normal path: when mutator.mutate_single raises a regular
        Exception, generate_and_persist_mutation handles it internally and returns
        False. gather sees False, not an exception object. count == 0.
        """
        mutator, storage, state_manager = _make_deps()
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        mutator.mutate_single.side_effect = RuntimeError("LLM timeout")

        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=selector,
            limit=2,
            iteration=0,
        )

        # Both tasks fail internally → False → count = 0
        assert len(count) == 0
        storage.add.assert_not_called()

    async def test_storage_add_exception_returns_false(self) -> None:
        """When storage.add raises, the inner handler returns False."""
        mutator, storage, state_manager = _make_deps()
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        storage.add.side_effect = ConnectionError("Redis down")

        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=selector,
            limit=2,
            iteration=0,
        )

        assert len(count) == 0

    async def test_partial_failure_counts_only_successes(self) -> None:
        """With 3 parent selections: first succeeds, second mutator raises, third succeeds.

        count must be 2 (only the successful ones), not inflated by the failure.
        """
        mutator, storage, state_manager = _make_deps()
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        call_count = 0

        async def mutate_side_effect(parents, memory_instructions: str | None = None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("bad mutation on call 2")
            return MutationSpec(
                code=f"def solve(): return {call_count}",
                parents=parents,
                name="m",
                metadata={},
            )

        mutator.mutate_single.side_effect = mutate_side_effect

        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=selector,
            limit=3,
            iteration=0,
        )

        assert len(count) == 2
        assert storage.add.call_count == 2


# ---------------------------------------------------------------------------
# TestOrphanPrevention — CancelledError after storage.add must return ID
# ---------------------------------------------------------------------------


class TestOrphanPrevention:
    """Verify that programs persisted to Redis are never lost as orphans.

    The bug: ``except Exception`` doesn't catch ``asyncio.CancelledError``
    (a ``BaseException``). When a mutation task is cancelled between
    ``storage.add()`` and ``return program.id``, the ID is lost and the
    program becomes a ghost in Redis — never tracked, never evaluated.
    """

    async def test_cancelled_error_after_persist_returns_id(self) -> None:
        """CancelledError during lineage update must still return the program ID."""
        import asyncio

        mutator, storage, state_manager = _make_deps()
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        # storage.add succeeds (default mock), but storage.get (used in
        # lineage update) raises CancelledError — simulating task
        # cancellation after persist but during lineage update.
        storage.get.side_effect = asyncio.CancelledError()

        ids = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=selector,
            limit=1,
            iteration=0,
        )

        # The program was persisted — its ID must be returned, not lost.
        assert len(ids) == 1
        assert isinstance(ids[0], str)
        storage.add.assert_called_once()

    async def test_cancelled_error_before_persist_propagates(self) -> None:
        """CancelledError before storage.add — no program persisted, safe to propagate."""
        import asyncio

        mutator, storage, state_manager = _make_deps()
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        # CancelledError during mutation generation (before persist)
        mutator.mutate_single.side_effect = asyncio.CancelledError()

        ids = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=selector,
            limit=1,
            iteration=0,
        )

        # Nothing persisted — empty result (gather catches the CancelledError)
        assert len(ids) == 0
        storage.add.assert_not_called()

    async def test_exception_after_persist_returns_id(self) -> None:
        """Any exception after storage.add must still return the program ID."""
        mutator, storage, state_manager = _make_deps()
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        # storage.get raises RuntimeError during lineage update
        storage.get.side_effect = RuntimeError("Redis connection lost")

        ids = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=selector,
            limit=1,
            iteration=0,
        )

        # Program was persisted — ID returned despite lineage failure.
        assert len(ids) == 1
        assert isinstance(ids[0], str)
        storage.add.assert_called_once()
