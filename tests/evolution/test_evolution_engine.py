"""Unit tests for EvolutionEngine: idle detection, mutation, ingestion, refresh."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Every engine call that touches _await_idle must have a timeout — without
# one, a change to ``_has_active_dags`` (e.g., switching from count_by_status
# to get_all_by_status) can make ``_await_idle`` loop forever on unmocked
# AsyncMock methods, silently hanging the test suite.
ENGINE_TEST_TIMEOUT = 5.0  # seconds


def _make_engine() -> SteadyStateEvolutionEngine:
    """Build a minimal EvolutionEngine with all external dependencies mocked."""
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()

    # Safe defaults for ALL status-query methods so _has_active_dags returns
    # False (idle) regardless of implementation.  Without these, changing
    # _has_active_dags from count_by_status to get_all_by_status would make
    # _await_idle loop forever (unmocked AsyncMock returns truthy MagicMock).
    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    storage.snapshot = MagicMock()

    engine = SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=SteadyStateEngineConfig(),
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    # Replace the real ProgramStateManager with a mock so we can assert on
    # set_program_state calls without touching Redis.
    engine.state = AsyncMock()
    return engine


def _prog(state: ProgramState = ProgramState.DONE) -> Program:
    return Program(code="def solve(): return 42", state=state)


# ---------------------------------------------------------------------------
# _ingest_completed_programs
# ---------------------------------------------------------------------------


class TestIngestCompletedPrograms:
    async def test_archive_known_programs_skipped(self) -> None:
        """Programs already in the archive are skipped — strategy.add not called."""
        engine = _make_engine()
        archive_prog = _prog(ProgramState.DONE)
        engine.storage.get_ids_by_status.return_value = [archive_prog.id]
        engine.strategy.get_program_ids.return_value = [archive_prog.id]

        await engine._ingest_completed_programs()

        engine.strategy.add.assert_not_called()
        engine.state.set_program_state.assert_not_called()

    async def test_new_accepted_program_stays_done(self) -> None:
        """A newly accepted program is added to the strategy and stays DONE (no state write)."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        new_prog = _prog(ProgramState.DONE)
        engine.storage.get_ids_by_status.return_value = [new_prog.id]
        engine.storage.mget.return_value = [new_prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.strategy.add.assert_called_once_with(new_prog)
        engine.state.set_program_state.assert_not_called()

    async def test_rejected_by_acceptor_is_discarded(self) -> None:
        """Programs rejected by the acceptor are discarded."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = False

        rej_prog = _prog(ProgramState.DONE)
        engine.storage.get_ids_by_status.return_value = [rej_prog.id]
        engine.storage.mget.return_value = [rej_prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.strategy.add.assert_not_called()
        engine.storage.batch_transition_by_ids.assert_called_once_with(
            [rej_prog.id],
            ProgramState.DONE.value,
            ProgramState.DISCARDED.value,
        )

    async def test_rejected_by_strategy_is_discarded(self) -> None:
        """Programs rejected by strategy.add() are discarded."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = False

        rej_prog = _prog(ProgramState.DONE)
        engine.storage.get_ids_by_status.return_value = [rej_prog.id]
        engine.storage.mget.return_value = [rej_prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.storage.batch_transition_by_ids.assert_called_once_with(
            [rej_prog.id],
            ProgramState.DONE.value,
            ProgramState.DISCARDED.value,
        )

    async def test_empty_done_set_returns_early(self) -> None:
        """No DONE programs → strategy.get_program_ids never called."""
        engine = _make_engine()
        engine.storage.get_ids_by_status.return_value = []

        await engine._ingest_completed_programs()

        engine.strategy.get_program_ids.assert_not_called()

    async def test_mixed_archive_and_new_programs(self) -> None:
        """Archive-known programs are skipped; new programs are evaluated independently."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        archive_prog = _prog(ProgramState.DONE)
        new_prog = _prog(ProgramState.DONE)

        engine.storage.get_ids_by_status.return_value = [archive_prog.id, new_prog.id]
        engine.storage.mget.return_value = [new_prog]
        engine.strategy.get_program_ids.return_value = [archive_prog.id]

        await engine._ingest_completed_programs()

        # Only the new program went through strategy.add
        engine.strategy.add.assert_called_once_with(new_prog)
        engine.state.set_program_state.assert_not_called()

    async def test_strategy_add_exception_doesnt_crash_ingest(self) -> None:
        """strategy.add() raises → exception caught per-item, program discarded, no propagation.

        Regression guard for the per-item exception isolation fix: a failing strategy.add()
        must NOT abort ingestion of remaining programs. The offending program is discarded.
        """
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.side_effect = RuntimeError("archive full")

        prog = _prog(ProgramState.DONE)
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
        engine.strategy.get_program_ids.return_value = []

        # Must NOT raise — per-item exception handler catches it and discards the program
        await engine._ingest_completed_programs()

        # The failed program must be batch-transitioned to DISCARDED
        engine.storage.batch_transition_by_ids.assert_called_once_with(
            [prog.id],
            ProgramState.DONE.value,
            ProgramState.DISCARDED.value,
        )

    async def test_rejected_discard_batch_swallows_exceptions(self) -> None:
        """Batch discard fails → exception caught, metrics still recorded."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = False

        prog = _prog(ProgramState.DONE)
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
        engine.strategy.get_program_ids.return_value = []
        # Make the batch discard fail
        engine.storage.batch_transition_by_ids.side_effect = RuntimeError(
            "Redis timeout"
        )

        # Exception should be caught and logged, not crash
        await engine._ingest_completed_programs()

        # Metrics should still be recorded
        assert engine.metrics.rejected_validation == 1

    async def test_mutation_outcome_called_for_accepted(self) -> None:
        """on_program_ingested called with ACCEPTED outcome for accepted programs."""
        from gigaevo.llm.bandit import MutationOutcome

        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        prog = _prog(ProgramState.DONE)
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.mutation_operator.on_program_ingested.assert_called_once_with(
            prog, engine.storage, outcome=MutationOutcome.ACCEPTED
        )

    async def test_mutation_outcome_called_for_rejected(self) -> None:
        """on_program_ingested called with REJECTED_STRATEGY outcome for rejected programs."""
        from gigaevo.llm.bandit import MutationOutcome

        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = False

        prog = _prog(ProgramState.DONE)
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.mutation_operator.on_program_ingested.assert_called_once_with(
            prog, engine.storage, outcome=MutationOutcome.REJECTED_STRATEGY
        )


# ---------------------------------------------------------------------------
# _await_idle & _has_active_dags
# ---------------------------------------------------------------------------


class TestAwaitIdle:
    async def test_returns_immediately_when_idle(self) -> None:
        """_await_idle returns at once when no QUEUED or RUNNING programs."""
        engine = _make_engine()
        engine.storage.count_by_status.return_value = 0

        await asyncio.wait_for(engine._await_idle(), timeout=ENGINE_TEST_TIMEOUT)

        # Two calls per poll: once for QUEUED, once for RUNNING (via asyncio.gather)
        assert engine.storage.count_by_status.call_count == 2

    async def test_blocks_then_returns_when_counts_drop(self) -> None:
        """_await_idle blocks while programs are active, returns once counts drop to zero."""
        engine = _make_engine()
        engine.config.loop_interval = 0.01  # fast for tests

        # First gather: [queued=3, running=0] → active
        # Second gather: [queued=0, running=0] → idle
        engine.storage.count_by_status.side_effect = [
            3,
            0,  # first poll: QUEUED=3, RUNNING=0
            0,
            0,  # second poll: QUEUED=0, RUNNING=0
        ]

        await asyncio.wait_for(engine._await_idle(), timeout=ENGINE_TEST_TIMEOUT)

        # Must have polled at least twice (2 calls per poll × 2 polls = 4 calls)
        assert engine.storage.count_by_status.call_count >= 4

    async def test_has_active_dags_true_when_queued(self) -> None:
        engine = _make_engine()
        engine.storage.count_by_status.side_effect = [3, 0]

        assert await engine._has_active_dags() is True

    async def test_has_active_dags_true_when_running(self) -> None:
        engine = _make_engine()
        engine.storage.count_by_status.side_effect = [0, 2]

        assert await engine._has_active_dags() is True

    async def test_has_active_dags_false_when_all_zero(self) -> None:
        engine = _make_engine()
        engine.storage.count_by_status.side_effect = [0, 0]

        assert await engine._has_active_dags() is False


# ---------------------------------------------------------------------------
# _select_parents_for_mutation
# ---------------------------------------------------------------------------


class TestSelectParents:
    async def test_returns_parents_from_strategy(self) -> None:
        engine = _make_engine()
        parents = [_prog() for _ in range(3)]
        engine.strategy.select_elites.return_value = parents

        result = await engine._select_parents_for_mutation()

        assert result == parents
        engine.strategy.select_elites.assert_called_once_with(
            total=engine.config.parent_selector.num_parents
        )

    async def test_records_metrics(self) -> None:
        engine = _make_engine()
        engine.strategy.select_elites.return_value = [_prog(), _prog()]

        await engine._select_parents_for_mutation()

        assert engine.metrics.elites_selected == 2


# ---------------------------------------------------------------------------
# generate_mutations (helper function)
# ---------------------------------------------------------------------------


class TestGenerateMutations:
    async def test_generates_and_persists(self) -> None:
        """generate_mutations creates programs from MutationSpecs and persists them."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        storage = AsyncMock()
        state_manager = AsyncMock()
        mutator = AsyncMock()

        parent = _prog(ProgramState.DONE)
        storage.get.return_value = parent

        mutator.mutate_single.return_value = MutationSpec(
            code="def solve(): return 99",
            parents=[parent],
            name="test_mutation",
            metadata={},
        )

        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=RandomParentSelector(num_parents=1),
            limit=3,
            iteration=0,
        )

        assert len(count) == 3
        assert storage.add.call_count == 3
        # Parent lineage updated for each mutation
        assert state_manager.update_program.call_count == 3

    async def test_none_mutation_spec_is_skipped(self) -> None:
        """If mutator returns None, the mutation is not persisted."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        storage = AsyncMock()
        state_manager = AsyncMock()
        mutator = AsyncMock()
        mutator.mutate_single.return_value = None

        parent = _prog()

        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=RandomParentSelector(num_parents=1),
            limit=5,
            iteration=0,
        )

        assert len(count) == 0
        storage.add.assert_not_called()

    async def test_empty_elites_returns_empty(self) -> None:
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        result = await generate_mutations(
            [],
            mutator=AsyncMock(),
            storage=AsyncMock(),
            state_manager=AsyncMock(),
            parent_selector=RandomParentSelector(num_parents=1),
            limit=5,
            iteration=0,
        )

        assert result == []

    async def test_limit_zero_returns_empty(self) -> None:
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        result = await generate_mutations(
            [_prog()],
            mutator=AsyncMock(),
            storage=AsyncMock(),
            state_manager=AsyncMock(),
            parent_selector=RandomParentSelector(num_parents=1),
            limit=0,
            iteration=0,
        )

        assert result == []

    async def test_mutation_exception_doesnt_crash(self) -> None:
        """A failing mutator call is caught; other mutations can still succeed."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        storage = AsyncMock()
        state_manager = AsyncMock()
        mutator = AsyncMock()
        parent = _prog(ProgramState.DONE)
        storage.get.return_value = parent

        # First call raises, second succeeds
        mutator.mutate_single.side_effect = [
            RuntimeError("LLM timeout"),
            MutationSpec(
                code="def solve(): return 1",
                parents=[parent],
                name="ok",
                metadata={},
            ),
        ]

        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=RandomParentSelector(num_parents=1),
            limit=2,
            iteration=0,
        )

        # One succeeded, one failed
        assert len(count) == 1


# ---------------------------------------------------------------------------
# Audit finding 2: Child lineage verification
# ---------------------------------------------------------------------------


class TestChildLineageVerification:
    async def test_child_program_has_parent_ids_in_lineage(self) -> None:
        """When generate_mutations creates a child, its lineage.parents
        should contain the parent program IDs."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        storage = AsyncMock()
        state_manager = AsyncMock()
        mutator = AsyncMock()

        parent = _prog(ProgramState.DONE)
        storage.get.return_value = parent

        mutator.mutate_single.return_value = MutationSpec(
            code="def solve(): return 99",
            parents=[parent],
            name="test_mutation",
            metadata={},
        )

        count = await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=RandomParentSelector(num_parents=1),
            limit=1,
            iteration=0,
        )

        assert len(count) == 1
        # Verify storage.add was called with a Program whose lineage references the parent
        stored_program = storage.add.call_args[0][0]
        assert parent.id in stored_program.lineage.parents

    async def test_child_lineage_generation_increments(self) -> None:
        """Child program's generation should be parent's generation + 1."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        storage = AsyncMock()
        state_manager = AsyncMock()
        mutator = AsyncMock()

        parent = _prog(ProgramState.DONE)
        parent.lineage.generation = 3
        storage.get.return_value = parent

        mutator.mutate_single.return_value = MutationSpec(
            code="def solve(): return 99",
            parents=[parent],
            name="test_mutation",
            metadata={},
        )

        await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=RandomParentSelector(num_parents=1),
            limit=1,
            iteration=0,
        )

        stored_program = storage.add.call_args[0][0]
        assert stored_program.lineage.generation == 4

    async def test_child_lineage_mutation_name_recorded(self) -> None:
        """Child program's lineage.mutation should match the MutationSpec name."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        storage = AsyncMock()
        state_manager = AsyncMock()
        mutator = AsyncMock()

        parent = _prog(ProgramState.DONE)
        storage.get.return_value = parent

        mutator.mutate_single.return_value = MutationSpec(
            code="def solve(): return 99",
            parents=[parent],
            name="crossover_v2",
            metadata={},
        )

        await generate_mutations(
            [parent],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=RandomParentSelector(num_parents=1),
            limit=1,
            iteration=0,
        )

        stored_program = storage.add.call_args[0][0]
        assert stored_program.lineage.mutation == "crossover_v2"

    async def test_multi_parent_child_references_all_parents(self) -> None:
        """When multiple parents are used, child's lineage.parents has all parent IDs."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        storage = AsyncMock()
        state_manager = AsyncMock()
        mutator = AsyncMock()

        parent_a = _prog(ProgramState.DONE)
        parent_b = _prog(ProgramState.DONE)
        storage.get.side_effect = lambda pid: (
            parent_a if pid == parent_a.id else parent_b
        )

        mutator.mutate_single.return_value = MutationSpec(
            code="def solve(): return 99",
            parents=[parent_a, parent_b],
            name="crossover",
            metadata={},
        )

        await generate_mutations(
            [parent_a, parent_b],
            mutator=mutator,
            storage=storage,
            state_manager=state_manager,
            parent_selector=RandomParentSelector(num_parents=2),
            limit=1,
            iteration=0,
        )

        stored_program = storage.add.call_args[0][0]
        assert parent_a.id in stored_program.lineage.parents
        assert parent_b.id in stored_program.lineage.parents
        assert len(stored_program.lineage.parents) == 2
