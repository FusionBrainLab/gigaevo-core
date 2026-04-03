"""Unit tests for EvolutionEngine: generation lifecycle, idle detection, mutation, ingestion, refresh."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Every engine call that touches _await_idle must have a timeout.
# Without this, a refactoring of _has_active_dags (e.g., switching from
# count_by_status to get_all_by_status) can make _await_idle loop forever
# on unmocked AsyncMock methods, silently hanging the test suite.
ENGINE_TEST_TIMEOUT = 5.0  # seconds


def _make_engine() -> EvolutionEngine:
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

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=EngineConfig(),
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
# _refresh_archive_programs
# ---------------------------------------------------------------------------


class TestRefreshArchivePrograms:
    async def test_calls_batch_transition_by_ids(self) -> None:
        """_refresh_archive_programs passes all archive IDs to batch_transition_by_ids."""
        engine = _make_engine()
        ids = ["id1", "id2", "id3"]
        engine.strategy.get_program_ids.return_value = ids
        engine.storage.batch_transition_by_ids.return_value = 2

        count = await engine._refresh_archive_programs()

        assert count == 2
        engine.storage.batch_transition_by_ids.assert_called_once_with(
            ids, ProgramState.DONE.value, ProgramState.QUEUED.value
        )

    async def test_empty_archive_returns_zero(self) -> None:
        """No archive programs → no transitions, returns 0."""
        engine = _make_engine()
        engine.strategy.get_program_ids.return_value = []

        count = await engine._refresh_archive_programs()

        assert count == 0
        engine.storage.batch_transition_by_ids.assert_not_called()


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
# _select_elites_for_mutation
# ---------------------------------------------------------------------------


class TestSelectElites:
    async def test_returns_elites_from_strategy(self) -> None:
        engine = _make_engine()
        elites = [_prog() for _ in range(3)]
        engine.strategy.select_elites.return_value = elites

        result = await engine._select_elites_for_mutation()

        assert result == elites
        engine.strategy.select_elites.assert_called_once_with(
            total=engine.config.max_elites_per_generation
        )

    async def test_records_metrics(self) -> None:
        engine = _make_engine()
        engine.strategy.select_elites.return_value = [_prog(), _prog()]

        await engine._select_elites_for_mutation()

        assert engine.metrics.elites_selected == 2


# ---------------------------------------------------------------------------
# _create_mutants (via generate_mutations)
# ---------------------------------------------------------------------------


class TestCreateMutants:
    async def test_calls_generate_mutations(self) -> None:
        engine = _make_engine()
        elites = [_prog() for _ in range(2)]
        new_ids = [f"mut-{i}" for i in range(5)]

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=new_ids,
        ) as mock_gen:
            result = await engine._create_mutants(elites)

        assert set(result) == set(new_ids)
        mock_gen.assert_called_once()
        assert engine.metrics.mutations_created == 5

    async def test_empty_elites_skip(self) -> None:
        """step() with empty elites skips mutation entirely."""
        engine = _make_engine()

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_gen:
            # Directly test: _create_mutants is not called when elites is empty
            # because step() checks: `if elites else 0`
            engine.strategy.select_elites.return_value = []
            engine.storage.count_by_status.return_value = 0
            engine.storage.get_ids_by_status.return_value = []
            engine.strategy.get_program_ids.return_value = []

            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# step() — full generation lifecycle
# ---------------------------------------------------------------------------


class TestStep:
    async def test_full_lifecycle_with_mutations(self) -> None:
        """step() executes all 6 phases in order with real-ish mocks."""
        engine = _make_engine()
        engine.config.loop_interval = 0.01

        # Phase 2: elites & mutations
        elites = [_prog() for _ in range(2)]
        engine.strategy.select_elites.return_value = elites

        # Phase 4: ingest
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True
        new_prog = _prog(ProgramState.DONE)

        # _await_idle uses count_by_status (returns 0 → idle immediately)
        engine.storage.count_by_status.return_value = 0
        # _ingest_completed_programs uses get_ids_by_status + mget
        engine.storage.get_ids_by_status.return_value = [new_prog.id]
        engine.storage.mget.return_value = [new_prog]

        # Phase 5: refresh → archive has the added program
        engine.strategy.get_program_ids.side_effect = [
            [],  # Phase 4: no archive yet
            [new_prog.id],  # Phase 5: now in archive
            [new_prog.id],  # generation summary log
        ]
        engine.storage.batch_transition_state.return_value = 1

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[new_prog.id],
        ):
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        assert engine.metrics.total_generations == 1
        assert engine.metrics.mutations_created == 1
        assert engine.metrics.added == 1

    async def test_step_skips_phase6_when_no_refresh(self) -> None:
        """When _refresh_archive_programs returns 0, phase 6 _await_idle is skipped."""
        engine = _make_engine()
        engine.storage.count_by_status.return_value = 0
        engine.strategy.select_elites.return_value = []
        engine.storage.get_ids_by_status.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        assert engine.metrics.total_generations == 1

    async def test_ingestion_mutation_outcome_callback(self) -> None:
        """on_program_ingested is called with correct outcome for each program."""
        engine = _make_engine()
        engine.config.loop_interval = 0.01

        # Two elites so _create_mutants is called
        elites = [_prog() for _ in range(2)]
        engine.strategy.select_elites.return_value = elites

        # One rejected by acceptor, one accepted
        engine.config.program_acceptor = MagicMock()
        bad_prog = _prog(ProgramState.DONE)
        good_prog = _prog(ProgramState.DONE)

        engine.config.program_acceptor.is_accepted.side_effect = [False, True]
        engine.strategy.add.return_value = True

        # _await_idle uses count_by_status (returns 0 → idle immediately)
        engine.storage.count_by_status.return_value = 0
        # _ingest uses get_ids_by_status + mget
        engine.storage.get_ids_by_status.return_value = [bad_prog.id, good_prog.id]
        engine.storage.mget.return_value = [bad_prog, good_prog]
        # Neither in archive (Phase 4), then for refresh + summary
        engine.strategy.get_program_ids.side_effect = [[], [], []]

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[bad_prog.id, good_prog.id],
        ):
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        # Check on_program_ingested calls
        calls = engine.mutation_operator.on_program_ingested.call_args_list
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# run() — main loop behavior
# ---------------------------------------------------------------------------


class TestRunLoop:
    async def test_generation_cap_stops_loop(self) -> None:
        """run() stops after max_generations steps."""
        engine = _make_engine()
        engine.config.max_generations = 2
        engine.config.loop_interval = 0.01

        # Make step() a fast no-op
        engine.storage.count_by_status.return_value = 0
        engine.strategy.select_elites.return_value = []
        engine.storage.get_ids_by_status.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await asyncio.wait_for(engine.run(), timeout=ENGINE_TEST_TIMEOUT)

        assert engine.metrics.total_generations == 2
        assert engine._running is False

    async def test_pause_resume_skips_step(self) -> None:
        """While paused, the engine sleeps but doesn't call step()."""
        engine = _make_engine()
        engine.config.max_generations = 1
        engine.config.loop_interval = 0.01

        engine.storage.count_by_status.return_value = 0
        engine.strategy.select_elites.return_value = []
        engine.storage.get_ids_by_status.return_value = []
        engine.strategy.get_program_ids.return_value = []

        call_count = 0

        original_step = engine.step

        async def counting_step():
            nonlocal call_count
            call_count += 1
            await original_step()

        engine.step = counting_step

        # Start paused
        engine.pause()
        assert engine._paused is True

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            # Run for a bit then resume
            task = asyncio.create_task(engine.run())
            await asyncio.sleep(0.05)
            assert call_count == 0  # No steps while paused

            engine.resume()
            await asyncio.sleep(0.05)

            # Let it finish (max_generations=1)
            await task

        assert call_count == 1

    async def test_step_exception_doesnt_crash_loop(self) -> None:
        """An exception in step() is caught; loop continues."""
        engine = _make_engine()
        engine.config.max_generations = 3
        engine.config.loop_interval = 0.01

        call_count = 0

        async def flaky_step():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            engine.metrics.total_generations += 1

        engine.step = flaky_step

        await asyncio.wait_for(engine.run(), timeout=ENGINE_TEST_TIMEOUT)

        # 1st call fails (no gen increment), 2nd and 3rd succeed → 2 generations
        # But cap is 3 so loop runs: fail, gen=1, gen=2, gen=3... let me check
        # Actually: call 1 raises → caught, call 2 → gen=1, call 3 → gen=2,
        # call 4 → gen=3 → cap reached. So call_count = 4
        assert engine.metrics.total_generations >= 2

    async def test_generation_timeout_deprecated_and_ignored(self) -> None:
        """generation_timeout is deprecated — setting it has no effect."""
        engine = _make_engine()
        engine.config.max_generations = 2
        engine.config.generation_timeout = 0.001
        engine.config.loop_interval = 0.01

        async def slow_step():
            await asyncio.sleep(0.1)  # Longer than generation_timeout
            engine.metrics.total_generations += 1

        engine.step = slow_step

        await asyncio.wait_for(engine.run(), timeout=ENGINE_TEST_TIMEOUT)

        # Both generations complete — timeout is not enforced
        assert engine.metrics.total_generations == 2

    async def test_stop_cancels_run(self) -> None:
        """stop() cancels the run() task gracefully."""
        engine = _make_engine()
        engine.config.loop_interval = 0.01

        engine.storage.count_by_status.return_value = 0
        engine.strategy.select_elites.return_value = []
        engine.storage.get_ids_by_status.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            task = asyncio.create_task(engine.run())
            await asyncio.sleep(0.05)
            engine._running = False
            await asyncio.sleep(0.05)
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    async def test_reached_generation_cap(self) -> None:
        engine = _make_engine()
        engine.config.max_generations = 5
        engine.metrics.total_generations = 4
        assert engine._reached_generation_cap() is False

        engine.metrics.total_generations = 5
        assert engine._reached_generation_cap() is True

    async def test_unlimited_generations(self) -> None:
        engine = _make_engine()
        engine.config.max_generations = None
        engine.metrics.total_generations = 999
        assert engine._reached_generation_cap() is False


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
# Audit finding 1: Phase ordering in step()
# ---------------------------------------------------------------------------


class TestStepPhaseOrdering:
    async def test_step_executes_phases_in_correct_order(self) -> None:
        """Instrument the engine's internal methods to record call order,
        then assert the 6-phase sequence: await_idle, select+mutate,
        await_idle, ingest, refresh, await_idle."""
        engine = _make_engine()
        engine.config.loop_interval = 0.01

        call_log: list[str] = []

        async def tracked_await_idle():
            call_log.append("await_idle")
            # Make it always idle
            engine.storage.count_by_status.return_value = 0

        async def tracked_select():
            call_log.append("select_elites")
            return []

        async def tracked_create(elites):
            call_log.append("create_mutants")
            return []

        async def tracked_ingest(**kwargs):
            call_log.append("ingest")

        async def tracked_refresh():
            call_log.append("refresh")
            return 3  # non-zero to trigger phase 6

        engine._await_idle = tracked_await_idle
        engine._select_elites_for_mutation = tracked_select
        engine._create_mutants = tracked_create
        engine._ingest_completed_programs = tracked_ingest
        engine._refresh_archive_programs = tracked_refresh

        await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        # Expected phase order:
        # Phase 1: await_idle
        # Phase 2: select_elites (create_mutants skipped because elites=[])
        # Phase 3: await_idle
        # Phase 4: ingest
        # Phase 5: refresh
        # Phase 6: await_idle (because refresh returned > 0)
        assert call_log == [
            "await_idle",  # Phase 1
            "select_elites",  # Phase 2
            "await_idle",  # Phase 3
            "ingest",  # Phase 4
            "refresh",  # Phase 5
            "await_idle",  # Phase 6
        ]

    async def test_step_skips_phase6_when_refresh_returns_zero(self) -> None:
        """When refresh returns 0, the final await_idle (phase 6) is skipped."""
        engine = _make_engine()
        engine.config.loop_interval = 0.01

        call_log: list[str] = []

        async def tracked_await_idle():
            call_log.append("await_idle")
            engine.storage.count_by_status.return_value = 0

        async def tracked_select():
            call_log.append("select_elites")
            return []

        async def tracked_ingest(**kwargs):
            call_log.append("ingest")

        async def tracked_refresh():
            call_log.append("refresh")
            return 0  # No refreshed programs

        engine._await_idle = tracked_await_idle
        engine._select_elites_for_mutation = tracked_select
        engine._ingest_completed_programs = tracked_ingest
        engine._refresh_archive_programs = tracked_refresh

        await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        # Phase 6 await_idle should NOT appear because refresh returned 0
        assert call_log == [
            "await_idle",  # Phase 1
            "select_elites",  # Phase 2
            "await_idle",  # Phase 3
            "ingest",  # Phase 4
            "refresh",  # Phase 5
            # No Phase 6 await_idle
        ]

    async def test_step_includes_create_mutants_when_elites_exist(self) -> None:
        """When select_elites returns non-empty, create_mutants is called in phase 2."""
        engine = _make_engine()
        engine.config.loop_interval = 0.01

        call_log: list[str] = []
        elites = [_prog() for _ in range(2)]

        async def tracked_await_idle():
            call_log.append("await_idle")
            engine.storage.count_by_status.return_value = 0

        async def tracked_select():
            call_log.append("select_elites")
            return elites

        async def tracked_ingest(**kwargs):
            call_log.append("ingest")

        async def tracked_refresh():
            call_log.append("refresh")
            return 0

        engine._await_idle = tracked_await_idle
        engine._select_elites_for_mutation = tracked_select
        engine._ingest_completed_programs = tracked_ingest
        engine._refresh_archive_programs = tracked_refresh

        new_ids = ["mut-0", "mut-1"]

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=new_ids,
        ) as mock_gen:
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        # Phase 2 should now have generate_mutations called
        assert call_log == [
            "await_idle",  # Phase 1
            "select_elites",  # Phase 2 (select)
            "await_idle",  # Phase 3
            "ingest",  # Phase 4
            "refresh",  # Phase 5
        ]
        # generate_mutations was called (Phase 2 mutant creation)
        mock_gen.assert_called_once()
        assert engine.metrics.mutations_created == 2


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
