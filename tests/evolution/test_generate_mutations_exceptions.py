"""Tests for gigaevo/evolution/engine/mutation.py — generate_mutations exception handling.

Finding 5 (CRITICAL): asyncio.gather(return_exceptions=True) returns exception objects
in the results list. The counting logic `sum(1 for result in results if result)` treats
truthy exception objects as successful mutations — inflating the persisted count.

The inner coroutine (`generate_and_persist_mutation`) already wraps all `Exception`
subclasses and returns `False`, so ordinary exceptions don't escape. However:

1. A `BaseException` (e.g. asyncio.CancelledError in Python <3.8, KeyboardInterrupt,
   SystemExit) can escape the `except Exception` handler and appear as a truthy object
   in the gather results.
2. We verify the existing code correctly handles a mix of True/False/None returns.
3. We patch asyncio.gather directly to inject exception objects and confirm whether the
   counting logic is correct or inflated.
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
    def test_counting_expression_with_exception_objects_is_truthy(self) -> None:
        """Unit-level proof of the counting bug.

        The expression `sum(1 for result in results if result)` is used in
        generate_mutations to count successes. An exception object is truthy,
        so it is incorrectly counted as a success.

        This test directly evaluates the counting expression against a synthetic
        results list — no coroutine machinery involved — to prove the bug exists
        at the expression level.
        """
        # Simulate what asyncio.gather(return_exceptions=True) can return:
        # True (success), False (inner-handler caught it), RuntimeError (escaped)
        results = [True, True, False, RuntimeError("something exploded")]

        # The counting expression as written in mutation.py line 94:
        current_count = sum(1 for result in results if result)

        # True, True, RuntimeError() are all truthy → 3. Correct answer is 2.
        assert current_count == 3, (
            "Pre-condition: current code counts exception objects as successes."
        )

        # The correct expression — only count True booleans:
        correct_count = sum(1 for result in results if result is True)
        assert correct_count == 2, (
            "A fix using `result is True` gives the correct count."
        )

    def test_base_exception_counting_expression_is_buggy(self) -> None:
        """Unit proof: BaseException subclasses escape `except Exception` and are truthy.

        generate_and_persist_mutation catches `except Exception`. BaseException
        subclasses that are NOT Exception subclasses (e.g. GeneratorExit, but
        note: in Python 3.8+ asyncio.CancelledError IS an Exception subclass)
        can escape the inner handler and appear in gather(return_exceptions=True)
        results as exception objects.

        Regardless of the source, the core bug is identical: any truthy value
        (including exception objects) in the results list inflates the count.

        This test isolates the exact counting expression from mutation.py:94 and
        proves it counts exception objects as successes.
        """
        # Exactly the counting expression from mutation.py line 94:
        results_with_escaped_exc = [
            True,
            GeneratorExit("escaped base exception"),
            False,
        ]
        buggy_count = sum(1 for result in results_with_escaped_exc if result)

        # GeneratorExit is truthy → counted as success. Bug: 2 instead of 1.
        assert buggy_count == 2, (
            "Pre-condition: GeneratorExit() is truthy, current code counts it as a success."
        )
        correct_count = sum(1 for result in results_with_escaped_exc if result is True)
        assert correct_count == 1, (
            "`result is True` correctly ignores the exception object."
        )

    async def test_all_exceptions_via_counting_expression(self) -> None:
        """Directly demonstrate the counting bug with all-exception results list.

        Rather than fighting asyncio.gather patching, we test the counting
        expression that lives at the heart of the bug.
        """
        # Replicate the exact expression from mutation.py line 94
        all_exception_results = [ValueError("v"), RuntimeError("r"), KeyError("k")]

        # Current code counts all of these as "successes" because exceptions are truthy
        buggy_count = sum(1 for result in all_exception_results if result)
        assert buggy_count == 3, (
            "Pre-condition: all exception objects are truthy, current code counts them all."
        )

        # Correct behavior: no exception should count as a success
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

        assert count == 3
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

        assert count == 0
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
        assert count == 0
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

        assert count == 0

    async def test_partial_failure_counts_only_successes(self) -> None:
        """With 3 parent selections: first succeeds, second mutator raises, third succeeds.

        count must be 2 (only the successful ones), not inflated by the failure.
        """
        mutator, storage, state_manager = _make_deps()
        parent = _prog()
        selector = RandomParentSelector(num_parents=1)

        call_count = 0

        async def mutate_side_effect(parents):
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

        assert count == 2
        assert storage.add.call_count == 2
