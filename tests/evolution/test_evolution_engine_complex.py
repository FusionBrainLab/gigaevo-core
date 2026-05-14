"""Tests for complex/untested logic paths in EvolutionEngine.

Covers:
- restore_state from storage
- _ingest_completed_programs with per-program exception isolation (multi-program batch)
- _has_active_dags log throttling (only logs when counts change)
- Ingest with mixed archive and new programs
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from gigaevo.evolution.engine.config import EngineConfig, SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import MaxMutantsStopper
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# Every engine call that touches _await_idle must have a timeout.
# Without this, a refactoring of _has_active_dags (e.g., switching from
# count_by_status to get_all_by_status) can make _await_idle loop forever
# on unmocked AsyncMock methods, silently hanging the test suite.
ENGINE_TEST_TIMEOUT = 5.0  # seconds


def _engine(**overrides) -> SteadyStateEvolutionEngine:
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

    max_mutants = overrides.pop("max_mutants", None)
    config_kwargs = {
        k: v for k, v in overrides.items() if k in EngineConfig.model_fields
    }
    if max_mutants is not None:
        config_kwargs["stopper"] = MaxMutantsStopper(max_mutants)

    engine = SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=SteadyStateEngineConfig(**config_kwargs),
        writer=writer,
        metrics_tracker=metrics_tracker,
        pre_step_hook=overrides.get("pre_step_hook"),
        post_step_hook=overrides.get("post_step_hook"),
    )
    engine.state = AsyncMock()
    return engine


def _prog(state=ProgramState.DONE):
    return Program(code="def solve(): return 42", state=state)


# ===================================================================
# Category B: restore_state
# ===================================================================


class TestRestoreState:
    """restore_state reads total_mutants from the engine snapshot."""

    async def test_restore_existing_generation_count(self):
        engine = _engine()
        from gigaevo.evolution.engine.snapshot import EngineSnapshot

        engine.storage.load_run_state_str.return_value = EngineSnapshot(
            total_mutants=17
        ).model_dump_json()

        await engine.restore_state()

        assert engine.metrics.iteration == 17

    async def test_restore_no_saved_state(self):
        """When no saved snapshot, total_mutants stays at 0."""
        engine = _engine()
        engine.storage.load_run_state_str.return_value = None

        await engine.restore_state()

        assert engine.metrics.iteration == 0


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
