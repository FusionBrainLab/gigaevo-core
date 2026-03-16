"""Tests for complex/untested logic paths in EvolutionEngine.

Covers:
- pre_step_hook execution before each step
- restore_state from storage
- _ingest_completed_programs with per-program exception isolation (multi-program batch)
- _refresh_archive_programs with gather exception handling
- run() with generation_timeout triggering TimeoutError mid-step
- step() with pre_step_hook that raises
- _has_active_dags ghost filtering (uses get_all_by_status not count_by_status)
- _has_active_dags log throttling (only logs when counts change)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _engine(**overrides) -> EvolutionEngine:
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=EngineConfig(
            **{k: v for k, v in overrides.items() if k in EngineConfig.model_fields}
        ),
        writer=writer,
        metrics_tracker=metrics_tracker,
        pre_step_hook=overrides.get("pre_step_hook"),
    )
    engine.state = AsyncMock()
    return engine


def _prog(state=ProgramState.DONE):
    return Program(code="def solve(): return 42", state=state)


# ===================================================================
# Category A: pre_step_hook
# ===================================================================


class TestPreStepHook:
    """core.py L186-187: pre_step_hook is called before each step if set."""

    async def test_hook_called_before_each_step(self):
        """pre_step_hook is called exactly once per step, before any phase."""
        hook_calls = []

        async def hook():
            hook_calls.append("called")

        engine = _engine(pre_step_hook=hook)
        engine.storage.get_all_by_status.return_value = []
        engine.strategy.select_elites.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=0,
        ):
            await engine.step()

        assert hook_calls == ["called"]
        assert engine.metrics.total_generations == 1

    async def test_hook_called_before_phases(self):
        """Hook is called before phase 1 (await_idle)."""
        call_order = []

        async def hook():
            call_order.append("hook")

        engine = _engine(pre_step_hook=hook)

        async def tracked_await_idle():
            call_order.append("await_idle")

        async def tracked_select():
            call_order.append("select")
            return []

        async def tracked_ingest():
            call_order.append("ingest")

        async def tracked_refresh():
            call_order.append("refresh")
            return 0

        engine._await_idle = tracked_await_idle
        engine._select_elites_for_mutation = tracked_select
        engine._ingest_completed_programs = tracked_ingest
        engine._refresh_archive_programs = tracked_refresh

        await engine.step()

        # Hook must be first
        assert call_order[0] == "hook"

    async def test_no_hook_no_error(self):
        """When pre_step_hook is None, step() works normally."""
        engine = _engine()
        engine.storage.get_all_by_status.return_value = []
        engine.strategy.select_elites.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=0,
        ):
            await engine.step()

        assert engine.metrics.total_generations == 1

    async def test_hook_exception_propagates(self):
        """If pre_step_hook raises, step() propagates the exception.
        The run() loop catches it and continues."""

        async def bad_hook():
            raise RuntimeError("hook failed")

        engine = _engine(pre_step_hook=bad_hook)

        with pytest.raises(RuntimeError, match="hook failed"):
            await engine.step()


# ===================================================================
# Category B: restore_state
# ===================================================================


class TestRestoreState:
    """core.py L396-401: restore total_generations from storage."""

    async def test_restore_existing_generation_count(self):
        engine = _engine()
        engine.storage.load_run_state.return_value = 17

        await engine.restore_state()

        assert engine.metrics.total_generations == 17

    async def test_restore_no_saved_state(self):
        """When no saved state, total_generations stays at 0."""
        engine = _engine()
        engine.storage.load_run_state.return_value = None

        await engine.restore_state()

        assert engine.metrics.total_generations == 0


# ===================================================================
# Category C: Ingest multi-program exception isolation
# ===================================================================


class TestIngestMultiProgramIsolation:
    """core.py L316-326: per-program exception isolation in _ingest_completed_programs.
    If strategy.add raises for one program, remaining programs are still processed."""

    async def test_first_program_fails_second_still_ingested(self):
        """strategy.add raises for first program, but second is accepted."""
        engine = _engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True

        prog_bad = _prog()
        prog_good = _prog()

        # First call raises, second succeeds
        engine.strategy.add.side_effect = [
            RuntimeError("corrupted metrics"),
            True,
        ]

        engine.storage.get_all_by_status.return_value = [prog_bad, prog_good]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        # Bad program discarded, good program accepted
        assert engine.metrics.added == 1
        # state.set_program_state called for discarding bad program
        discard_calls = [
            c
            for c in engine.state.set_program_state.call_args_list
            if c[0][1] == ProgramState.DISCARDED
        ]
        assert len(discard_calls) >= 1

    async def test_all_programs_fail_none_added(self):
        """All programs fail during ingestion -> none added, all discarded."""
        engine = _engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.side_effect = RuntimeError("always fails")

        progs = [_prog() for _ in range(3)]
        engine.storage.get_all_by_status.return_value = progs
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        assert engine.metrics.added == 0


# ===================================================================
# Category D: Ingest mutation outcome callbacks
# ===================================================================


class TestIngestMutationOutcomeCallbacks:
    """core.py L283-309: on_program_ingested called with correct outcome."""

    async def test_rejected_acceptor_outcome(self):
        """Programs rejected by acceptor get REJECTED_ACCEPTOR callback."""
        engine = _engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = False

        prog = _prog()
        engine.storage.get_all_by_status.return_value = [prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.mutation_operator.on_program_ingested.assert_called_once_with(
            prog, engine.storage, outcome=MutationOutcome.REJECTED_ACCEPTOR
        )

    async def test_accepted_outcome(self):
        engine = _engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        prog = _prog()
        engine.storage.get_all_by_status.return_value = [prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.mutation_operator.on_program_ingested.assert_called_once_with(
            prog, engine.storage, outcome=MutationOutcome.ACCEPTED
        )

    async def test_rejected_strategy_outcome(self):
        engine = _engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = False

        prog = _prog()
        engine.storage.get_all_by_status.return_value = [prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.mutation_operator.on_program_ingested.assert_called_once_with(
            prog, engine.storage, outcome=MutationOutcome.REJECTED_STRATEGY
        )


# ===================================================================
# Category E: _has_active_dags log throttling
# ===================================================================


class TestHasActiveDagsLogThrottle:
    """core.py L379-391: _has_active_dags only logs when counts change."""

    async def test_repeated_same_counts_dont_update_cached(self):
        """When counts are the same twice, _last_pending_dags_counts stays."""
        engine = _engine()
        fake = _prog(ProgramState.QUEUED)

        # First call: QUEUED has 1 program
        engine.storage.get_all_by_status.side_effect = [[fake], []]
        result1 = await engine._has_active_dags()
        assert result1 is True
        assert engine._last_pending_dags_counts == (1, 0)

        # Second call: same counts
        engine.storage.get_all_by_status.side_effect = [[fake], []]
        result2 = await engine._has_active_dags()
        assert result2 is True
        assert engine._last_pending_dags_counts == (1, 0)

    async def test_counts_change_updates_cached(self):
        """When counts change, cached value is updated."""
        engine = _engine()
        fake1 = _prog(ProgramState.QUEUED)
        fake2 = _prog(ProgramState.RUNNING)

        engine.storage.get_all_by_status.side_effect = [[fake1], []]
        await engine._has_active_dags()
        assert engine._last_pending_dags_counts == (1, 0)

        engine.storage.get_all_by_status.side_effect = [[], [fake2]]
        await engine._has_active_dags()
        assert engine._last_pending_dags_counts == (0, 1)

    async def test_idle_resets_cached(self):
        """When no active dags, cached counts are reset to None."""
        engine = _engine()
        engine._last_pending_dags_counts = (2, 3)

        engine.storage.get_all_by_status.return_value = []
        result = await engine._has_active_dags()
        assert result is False
        assert engine._last_pending_dags_counts is None


# ===================================================================
# Category F: _refresh_archive_programs gather exception handling
# ===================================================================


class TestRefreshGatherExceptionHandling:
    """core.py L356-357: gather with return_exceptions=True for refresh transitions."""

    async def test_refresh_state_failure_doesnt_crash(self):
        """If one program's state transition fails, others still complete."""
        engine = _engine()

        progs = [_prog(ProgramState.DONE) for _ in range(3)]
        engine.strategy.get_program_ids.return_value = [p.id for p in progs]
        engine.storage.mget.return_value = progs

        # Make the second state transition fail
        call_count = 0

        async def flaky_set_state(prog, state):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Redis timeout")

        engine.state.set_program_state = flaky_set_state

        count = await engine._refresh_archive_programs()
        assert count == 3  # All were submitted
        assert call_count == 3  # All attempted (gather doesn't short-circuit)

    async def test_refresh_only_done_not_queued(self):
        """Only DONE programs get refreshed, not QUEUED ones."""
        engine = _engine()

        done_prog = _prog(ProgramState.DONE)
        queued_prog = _prog(ProgramState.QUEUED)

        engine.strategy.get_program_ids.return_value = [done_prog.id, queued_prog.id]
        engine.storage.mget.return_value = [done_prog, queued_prog]

        count = await engine._refresh_archive_programs()

        assert count == 1  # Only DONE program
        engine.state.set_program_state.assert_called_once_with(
            done_prog, ProgramState.QUEUED
        )


# ===================================================================
# Category G: run() step timeout
# ===================================================================


class TestRunStepTimeout:
    """core.py L157-167: generation_timeout wraps step() with wait_for."""

    async def test_timeout_doesnt_crash_loop(self):
        """Step timeout is caught, loop continues to next generation."""
        engine = _engine(max_generations=3, generation_timeout=0.001)
        engine.config.loop_interval = 0.01

        call_count = 0

        async def slow_then_fast_step():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(10)  # Triggers timeout
            engine.metrics.total_generations += 1

        engine.step = slow_then_fast_step

        await engine.run()

        # First step timed out, subsequent steps completed
        assert engine.metrics.total_generations >= 2


# ===================================================================
# Category H: step() saves generation counter
# ===================================================================


class TestStepGenerationPersistence:
    """core.py L215-218: step() saves total_generations to storage."""

    async def test_generation_counter_saved_after_step(self):
        engine = _engine()
        engine.storage.get_all_by_status.return_value = []
        engine.strategy.select_elites.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=0,
        ):
            await engine.step()

        engine.storage.save_run_state.assert_called_once_with(
            "engine:total_generations", 1
        )

    async def test_generation_counter_increments_each_step(self):
        engine = _engine()
        engine.storage.get_all_by_status.return_value = []
        engine.strategy.select_elites.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=0,
        ):
            await engine.step()
            await engine.step()
            await engine.step()

        assert engine.metrics.total_generations == 3
        # Last call should save generation 3
        last_call = engine.storage.save_run_state.call_args_list[-1]
        assert last_call[0] == ("engine:total_generations", 3)


# ===================================================================
# Category I: Ingest with mixed archive and new programs
# ===================================================================


class TestIngestMixedBatch:
    """core.py L271-315: batch with archive programs, accepted, rejected by acceptor,
    and rejected by strategy all in one call."""

    async def test_full_mixed_batch(self):
        """Archive + accepted + rejected_acceptor + rejected_strategy."""
        engine = _engine()
        engine.config.program_acceptor = MagicMock()

        archive_prog = _prog()
        accepted_prog = _prog()
        rej_acceptor_prog = _prog()
        rej_strategy_prog = _prog()

        engine.storage.get_all_by_status.return_value = [
            archive_prog,
            accepted_prog,
            rej_acceptor_prog,
            rej_strategy_prog,
        ]
        engine.strategy.get_program_ids.return_value = [archive_prog.id]

        # Acceptor: reject only rej_acceptor_prog
        engine.config.program_acceptor.is_accepted.side_effect = [
            True,  # accepted_prog
            False,  # rej_acceptor_prog
            True,  # rej_strategy_prog
        ]
        # Strategy: accept first, reject second
        engine.strategy.add.side_effect = [True, False]

        await engine._ingest_completed_programs()

        assert engine.metrics.added == 1
        assert engine.metrics.rejected_validation == 1
        assert engine.metrics.rejected_strategy == 1

        # on_program_ingested called 3 times (not for archive prog)
        assert engine.mutation_operator.on_program_ingested.call_count == 3
