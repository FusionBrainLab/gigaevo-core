"""Unit tests for :class:`SteadyStateEvolutionEngine`.

The engine composes :func:`dispatcher_loop` + :func:`ingestor_loop` +
:class:`ParentRefresher`. Tests here exercise the engine-level wiring;
per-module behavior (mutant_task, dispatcher, ingestor) is covered in
``test_mutant_task.py``, ``test_dispatcher.py``, ``test_ingestor.py``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import EngineConfig, SteadyStateEngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.refresh import ParentRefresher
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import EvolutionStopper, MaxMutantsStopper
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

SS_TEST_TIMEOUT = 5.0


def _make_ss_engine(
    *,
    max_in_flight: int = 4,
    max_mutants: int | None = None,
    loop_interval: float = 0.01,
) -> SteadyStateEvolutionEngine:
    """Build a minimal SteadyStateEvolutionEngine with mocked dependencies."""
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()
    metrics_tracker.format_best_summary.return_value = ""

    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    storage.snapshot = MagicMock()
    strategy.get_program_ids.return_value = []

    stopper = (
        MaxMutantsStopper(max_mutants)
        if max_mutants is not None
        else EvolutionStopper()
    )
    config = SteadyStateEngineConfig(
        max_in_flight=max_in_flight,
        stopper=stopper,
        loop_interval=loop_interval,
    )

    engine = SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=config,
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    engine.state = AsyncMock()
    return engine


def _prog(state: ProgramState = ProgramState.DONE) -> Program:
    return Program(code="def solve(): return 42", state=state)


# ---------------------------------------------------------------------------
# Construction & interface
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_steady_state_config(self) -> None:
        """Passing plain EngineConfig still works — SteadyStateEngineConfig is now an alias."""
        storage = AsyncMock()
        writer = MagicMock()
        writer.bind.return_value = writer
        # Under the unified config, EngineConfig itself is not a SteadyStateEngineConfig,
        # so we still type-check the alias.
        with pytest.raises(TypeError, match="SteadyStateEngineConfig"):
            SteadyStateEvolutionEngine(
                storage=storage,
                strategy=AsyncMock(),
                mutation_operator=AsyncMock(),
                config=EngineConfig(),
                writer=writer,
                metrics_tracker=MagicMock(),
            )

    def test_is_subclass(self) -> None:
        engine = _make_ss_engine()
        assert isinstance(engine, EvolutionEngine)

    def test_has_parent_refresher(self) -> None:
        """Every SteadyStateEvolutionEngine wires a ParentRefresher at __init__ time."""
        engine = _make_ss_engine()
        assert isinstance(engine._parent_refresher, ParentRefresher)


# ---------------------------------------------------------------------------
# Backpressure semaphore
# ---------------------------------------------------------------------------


class TestBackpressure:
    async def test_semaphore_limits_in_flight(self) -> None:
        """With max_in_flight=2, the 3rd acquire blocks until a slot frees."""
        engine = _make_ss_engine(max_in_flight=2)

        await engine._producer_sema.acquire()
        await engine._producer_sema.acquire()

        acquired = False

        async def try_acquire():
            nonlocal acquired
            await engine._producer_sema.acquire()
            acquired = True

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not acquired, "Should be blocked — all slots taken"

        engine._producer_sema.release()
        await asyncio.sleep(0.05)
        assert acquired, "Should have acquired after release"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Mutant cap — drives dispatcher_loop exit through _reached_mutant_cap
# ---------------------------------------------------------------------------


class TestGenerationCap:
    async def test_cap_stops_dispatcher_loop(self) -> None:
        """dispatcher_loop exits when the stopper says total_mutants cap is hit."""
        from gigaevo.evolution.engine.dispatcher import dispatcher_loop

        engine = _make_ss_engine(max_mutants=1)
        engine._running = True
        engine.metrics.mutations_created = 1  # already at cap

        await asyncio.wait_for(dispatcher_loop(engine), timeout=SS_TEST_TIMEOUT)


# ---------------------------------------------------------------------------
# Resume / snapshot survival
# ---------------------------------------------------------------------------


class TestRestore:
    async def test_restore_hydrates_total_mutants(self) -> None:
        """restore_state lifts both stop-counter and ordinal from the snapshot."""
        from gigaevo.evolution.engine.snapshot import EngineSnapshot

        engine = _make_ss_engine()
        snap = EngineSnapshot(total_mutants=42, next_iteration=50, programs_processed=7)
        engine.storage.load_run_state_str = AsyncMock(
            return_value=snap.model_dump_json()
        )
        await engine.restore_state()
        assert engine.metrics.mutations_created == 42
        assert engine.metrics.iteration == 50
        assert engine.metrics.programs_processed == 7
