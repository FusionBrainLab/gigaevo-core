"""Tests for complex/untested logic paths in EvolutionEngine.

Covers:
- pre_step_hook execution before each step
- restore_state from storage
- _ingest_completed_programs with per-program exception isolation (multi-program batch)
- _refresh_archive_programs with gather exception handling
- generation_timeout is deprecated (no-op, field accepted but ignored)
- step() with pre_step_hook that raises
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

# Every engine call that touches _await_idle must have a timeout.
# Without this, a refactoring of _has_active_dags (e.g., switching from
# count_by_status to get_all_by_status) can make _await_idle loop forever
# on unmocked AsyncMock methods, silently hanging the test suite.
ENGINE_TEST_TIMEOUT = 5.0  # seconds


def _engine(**overrides) -> EvolutionEngine:
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
        engine.storage.count_by_status.return_value = 0
        engine.storage.get_ids_by_status.return_value = []
        engine.strategy.select_elites.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

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

        async def tracked_ingest(**kwargs):
            call_order.append("ingest")

        async def tracked_refresh():
            call_order.append("refresh")
            return 0

        engine._await_idle = tracked_await_idle
        engine._select_elites_for_mutation = tracked_select
        engine._ingest_completed_programs = tracked_ingest
        engine._refresh_archive_programs = tracked_refresh

        await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        # Hook must be first
        assert call_order[0] == "hook"

    async def test_no_hook_no_error(self):
        """When pre_step_hook is None, step() works normally."""
        engine = _engine()
        engine.storage.count_by_status.return_value = 0
        engine.storage.get_ids_by_status.return_value = []
        engine.strategy.select_elites.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        assert engine.metrics.total_generations == 1

    async def test_hook_exception_propagates(self):
        """If pre_step_hook raises, step() propagates the exception.
        The run() loop catches it and continues."""

        async def bad_hook():
            raise RuntimeError("hook failed")

        engine = _engine(pre_step_hook=bad_hook)

        with pytest.raises(RuntimeError, match="hook failed"):
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)


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

        engine.storage.get_ids_by_status.return_value = [prog_bad.id, prog_good.id]
        engine.storage.mget.return_value = [prog_bad, prog_good]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        # Bad program discarded, good program accepted
        assert engine.metrics.added == 1
        # batch_transition_by_ids called to discard bad program
        engine.storage.batch_transition_by_ids.assert_called_once_with(
            [prog_bad.id],
            ProgramState.DONE.value,
            ProgramState.DISCARDED.value,
        )

    async def test_all_programs_fail_none_added(self):
        """All programs fail during ingestion -> none added, all discarded."""
        engine = _engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.side_effect = RuntimeError("always fails")

        progs = [_prog() for _ in range(3)]
        engine.storage.get_ids_by_status.return_value = [p.id for p in progs]
        engine.storage.mget.return_value = progs
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
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
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
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
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
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
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

        # First call: QUEUED=1, RUNNING=0
        engine.storage.count_by_status.side_effect = [1, 0]
        result1 = await engine._has_active_dags()
        assert result1 is True
        assert engine._last_pending_dags_counts == (1, 0)

        # Second call: same counts
        engine.storage.count_by_status.side_effect = [1, 0]
        result2 = await engine._has_active_dags()
        assert result2 is True
        assert engine._last_pending_dags_counts == (1, 0)

    async def test_counts_change_updates_cached(self):
        """When counts change, cached value is updated."""
        engine = _engine()

        engine.storage.count_by_status.side_effect = [1, 0]
        await engine._has_active_dags()
        assert engine._last_pending_dags_counts == (1, 0)

        engine.storage.count_by_status.side_effect = [0, 1]
        await engine._has_active_dags()
        assert engine._last_pending_dags_counts == (0, 1)

    async def test_idle_resets_cached(self):
        """When no active dags, cached counts are reset to None."""
        engine = _engine()
        engine._last_pending_dags_counts = (2, 3)

        engine.storage.count_by_status.side_effect = [0, 0]
        result = await engine._has_active_dags()
        assert result is False
        assert engine._last_pending_dags_counts is None


# ===================================================================
# Category F: _refresh_archive_programs gather exception handling
# ===================================================================


class TestRefreshBatchTransition:
    """_refresh_archive_programs uses batch_transition_by_ids for efficiency."""

    async def test_refresh_passes_ids_to_batch_transition(self):
        """batch_transition_by_ids is called with all archive IDs."""
        engine = _engine()

        ids = ["p1", "p2", "p3"]
        engine.strategy.get_program_ids.return_value = ids
        engine.storage.batch_transition_by_ids.return_value = 3

        count = await engine._refresh_archive_programs()
        assert count == 3
        engine.storage.batch_transition_by_ids.assert_called_once_with(
            ids, ProgramState.DONE.value, ProgramState.QUEUED.value
        )

    async def test_refresh_handles_batch_transition_error(self):
        """_refresh_archive_programs doesn't crash when batch_transition_by_ids raises."""
        engine = _engine()

        engine.strategy.get_program_ids.return_value = ["p1", "p2"]
        engine.storage.batch_transition_by_ids.side_effect = RuntimeError(
            "Redis timeout"
        )

        # Should not raise — returns 0 gracefully
        count = await engine._refresh_archive_programs()
        assert count == 0


# ===================================================================
# Category G: generation_timeout is deprecated (no-op)
# ===================================================================


class TestRunStepTimeout:
    """generation_timeout is deprecated and ignored. Individual program
    timeouts are handled by dag_timeout / stage_timeout."""

    async def test_generation_timeout_field_accepted_but_ignored(self):
        """Setting generation_timeout doesn't crash — it's just ignored."""
        engine = _engine(max_generations=2, generation_timeout=0.001)
        engine.config.loop_interval = 0.01

        async def fast_step():
            await asyncio.sleep(0.1)  # Longer than generation_timeout
            engine.metrics.total_generations += 1

        engine.step = fast_step

        await asyncio.wait_for(engine.run(), timeout=ENGINE_TEST_TIMEOUT)

        # Both generations complete despite generation_timeout=0.001
        assert engine.metrics.total_generations == 2


# ===================================================================
# Category H: step() saves generation counter
# ===================================================================


class TestStepGenerationPersistence:
    """core.py L215-218: step() saves total_generations to storage."""

    async def test_generation_counter_saved_after_step(self):
        engine = _engine()
        engine.storage.count_by_status.return_value = 0
        engine.storage.get_ids_by_status.return_value = []
        engine.strategy.select_elites.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        engine.storage.save_run_state.assert_called_once_with(
            "engine:total_generations", 1
        )

    async def test_generation_counter_increments_each_step(self):
        engine = _engine()
        engine.storage.count_by_status.return_value = 0
        engine.storage.get_ids_by_status.return_value = []
        engine.strategy.select_elites.return_value = []
        engine.strategy.get_program_ids.return_value = []

        with patch(
            "gigaevo.evolution.engine.core.generate_mutations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)
            await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

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

        all_ids = [
            archive_prog.id,
            accepted_prog.id,
            rej_acceptor_prog.id,
            rej_strategy_prog.id,
        ]
        engine.storage.get_ids_by_status.return_value = all_ids
        # mget only returns non-archive programs (archive filtered out by ID)
        engine.storage.mget.return_value = [
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
