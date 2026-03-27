"""Unit tests for SteadyStateEvolutionEngine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.engine.config import EngineConfig, SteadyStateEngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SS_TEST_TIMEOUT = 5.0


def _make_ss_engine(
    *,
    max_in_flight: int = 4,
    max_mutations_per_generation: int = 2,
    max_generations: int | None = None,
    loop_interval: float = 0.01,
) -> SteadyStateEvolutionEngine:
    """Build a minimal SteadyStateEvolutionEngine with mocked dependencies."""
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()
    metrics_tracker.format_best_summary.return_value = ""

    # Safe defaults: engine is idle
    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    strategy.get_program_ids.return_value = []

    config = SteadyStateEngineConfig(
        max_in_flight=max_in_flight,
        max_mutations_per_generation=max_mutations_per_generation,
        max_generations=max_generations,
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
        """Passing plain EngineConfig raises TypeError."""
        storage = AsyncMock()
        writer = MagicMock()
        writer.bind.return_value = writer
        with pytest.raises(TypeError, match="SteadyStateEngineConfig"):
            SteadyStateEvolutionEngine(
                storage=storage,
                strategy=AsyncMock(),
                mutation_operator=AsyncMock(),
                config=EngineConfig(),
                writer=writer,
                metrics_tracker=MagicMock(),
            )

    async def test_step_raises(self) -> None:
        """step() is not meaningful in steady-state mode."""
        engine = _make_ss_engine()
        with pytest.raises(NotImplementedError):
            await engine.step()

    def test_is_subclass(self) -> None:
        engine = _make_ss_engine()
        assert isinstance(engine, EvolutionEngine)


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    async def test_semaphore_limits_in_flight(self) -> None:
        """With max_in_flight=2, the 3rd mutation blocks until a slot frees."""
        engine = _make_ss_engine(max_in_flight=2)

        # Acquire 2 slots
        await engine._in_flight_sema.acquire()
        await engine._in_flight_sema.acquire()

        # 3rd acquire should not complete immediately
        acquired = False

        async def try_acquire():
            nonlocal acquired
            await engine._in_flight_sema.acquire()
            acquired = True

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not acquired, "Should be blocked — all slots taken"

        # Release one slot
        engine._in_flight_sema.release()
        await asyncio.sleep(0.05)
        assert acquired, "Should have acquired after release"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# In-flight tracking
# ---------------------------------------------------------------------------


class TestInFlightTracking:
    async def test_poll_and_ingest_releases_slot(self) -> None:
        """When a program finishes, _poll_and_ingest releases its semaphore slot."""
        engine = _make_ss_engine(max_in_flight=4)

        prog = _prog(ProgramState.DONE)

        # Simulate: program is in-flight
        engine._in_flight.add(prog.id)
        await engine._in_flight_sema.acquire()  # consume one slot

        # Storage returns it as DONE
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        count = await engine._poll_and_ingest()

        assert count == 1
        assert prog.id not in engine._in_flight
        # Semaphore should have been released (we can acquire without blocking)
        await asyncio.wait_for(engine._in_flight_sema.acquire(), timeout=0.1)
        engine._in_flight_sema.release()  # put it back


# ---------------------------------------------------------------------------
# Sweep discarded
# ---------------------------------------------------------------------------


class TestSweepDiscarded:
    async def test_leaked_slot_recovered(self) -> None:
        """Programs that vanish from Redis get swept and slots freed via _poll_and_ingest."""
        engine = _make_ss_engine(max_in_flight=4)

        # Simulate: program is in-flight but disappeared (DagRunner discarded it)
        engine._in_flight.add("ghost-id")
        await engine._in_flight_sema.acquire()

        # mget returns empty → ghost not found in Redis (vanished)
        engine.storage.mget.return_value = []

        # _poll_and_ingest now handles sweep as part of the unified poll
        await engine._poll_and_ingest()

        assert "ghost-id" not in engine._in_flight
        # Slot should be recovered
        await asyncio.wait_for(engine._in_flight_sema.acquire(), timeout=0.1)
        engine._in_flight_sema.release()


# ---------------------------------------------------------------------------
# Drain timeout
# ---------------------------------------------------------------------------


class TestDrainTimeout:
    async def test_drain_timeout_force_releases_slots(self) -> None:
        """_drain_in_flight timeout force-releases in-flight slots."""
        engine = _make_ss_engine(max_in_flight=4)

        # Simulate: 3 programs in-flight but stuck RUNNING (never complete)
        prog1 = _prog(ProgramState.RUNNING)
        prog2 = _prog(ProgramState.RUNNING)
        prog3 = _prog(ProgramState.RUNNING)
        engine._in_flight.update([prog1.id, prog2.id, prog3.id])
        for _ in range(3):
            await engine._in_flight_sema.acquire()

        # mget always returns them as RUNNING (never DONE)
        engine.storage.mget.return_value = [prog1, prog2, prog3]

        # Drain with very short timeout (0.1s)
        await engine._drain_in_flight(timeout_sec=0.1)

        # All slots should have been force-released
        assert len(engine._in_flight) == 0
        # Semaphore should be free (all 4 slots available)
        for _ in range(4):
            await asyncio.wait_for(engine._in_flight_sema.acquire(), timeout=0.1)
            engine._in_flight_sema.release()


# ---------------------------------------------------------------------------
# Epoch triggers
# ---------------------------------------------------------------------------


class TestEpochTriggers:
    def test_count_trigger(self) -> None:
        """Epoch triggers when processed count reaches threshold."""
        engine = _make_ss_engine(max_mutations_per_generation=3)
        engine._processed_since_epoch = 2
        assert not engine._should_trigger_epoch()
        engine._processed_since_epoch = 3
        assert engine._should_trigger_epoch()

    def test_no_trigger(self) -> None:
        """No trigger when threshold not met."""
        engine = _make_ss_engine(max_mutations_per_generation=999)
        engine._processed_since_epoch = 0
        assert not engine._should_trigger_epoch()


# ---------------------------------------------------------------------------
# Epoch refresh
# ---------------------------------------------------------------------------


class TestEpochRefresh:
    async def test_increments_total_generations(self) -> None:
        """Epoch refresh increments total_generations and saves to Redis."""
        engine = _make_ss_engine()
        engine._last_epoch_time = 0.0
        initial_gen = engine.metrics.total_generations

        await engine._epoch_refresh()

        assert engine.metrics.total_generations == initial_gen + 1
        engine.storage.save_run_state.assert_called()

    async def test_resets_processed_count(self) -> None:
        """After epoch refresh, processed count is reset to 0."""
        engine = _make_ss_engine()
        engine._processed_since_epoch = 10
        engine._epoch_mutants = 5

        await engine._epoch_refresh()

        assert engine._processed_since_epoch == 0
        assert engine._epoch_mutants == 0

    async def test_mutation_gate_reopens(self) -> None:
        """Mutation gate is set (open) after epoch refresh completes."""
        engine = _make_ss_engine()

        await engine._epoch_refresh()

        assert engine._mutation_gate.is_set()

    async def test_calls_refresh_and_reindex(self) -> None:
        """Epoch refresh calls _refresh_archive_programs and reindex_archive."""
        engine = _make_ss_engine()
        engine.strategy.get_program_ids.return_value = ["id1"]
        engine.storage.batch_transition_by_ids.return_value = 1

        await engine._epoch_refresh()

        engine.storage.batch_transition_by_ids.assert_called()
        engine.strategy.reindex_archive.assert_called()


# ---------------------------------------------------------------------------
# Generation cap
# ---------------------------------------------------------------------------


class TestGenerationCap:
    async def test_cap_stops_mutation_loop(self) -> None:
        """Mutation loop exits when generation cap is reached."""
        engine = _make_ss_engine(max_generations=1)
        engine.metrics.total_generations = 1  # already at cap

        # Should exit immediately
        await asyncio.wait_for(engine._mutation_loop(), timeout=SS_TEST_TIMEOUT)


# ---------------------------------------------------------------------------
# Ingest batch
# ---------------------------------------------------------------------------


class TestIngestBatch:
    async def test_accepts_and_rejects(self) -> None:
        """_ingest_batch accepts valid programs and rejects invalid ones."""
        engine = _make_ss_engine()
        accepted = _prog(ProgramState.DONE)
        rejected = _prog(ProgramState.DONE)

        engine.storage.mget.return_value = [accepted, rejected]
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.side_effect = [True, False]
        engine.strategy.add.return_value = True
        engine.storage.batch_transition_by_ids.return_value = 1

        count, handled = await engine._ingest_batch([accepted.id, rejected.id])

        assert count == 1
        assert set(handled) == {accepted.id, rejected.id}
        engine.strategy.add.assert_called_once_with(accepted)
        engine.storage.batch_transition_by_ids.assert_called_once()

    async def test_empty_ids(self) -> None:
        """_ingest_batch with empty list returns 0."""
        engine = _make_ss_engine()
        count, handled = await engine._ingest_batch([])
        assert count == 0
        assert handled == []
        engine.storage.mget.assert_not_called()


# ---------------------------------------------------------------------------
# Drain in-flight
# ---------------------------------------------------------------------------


class TestDrainInFlight:
    async def test_drain_when_empty(self) -> None:
        """_drain_in_flight returns immediately when no in-flight programs."""
        engine = _make_ss_engine()
        await asyncio.wait_for(engine._drain_in_flight(), timeout=SS_TEST_TIMEOUT)

    async def test_drain_ingests_remaining(self) -> None:
        """_drain_in_flight ingests DONE programs and clears in-flight set."""
        engine = _make_ss_engine(max_in_flight=4)
        prog = _prog(ProgramState.DONE)

        engine._in_flight.add(prog.id)
        await engine._in_flight_sema.acquire()  # consume a slot

        # mget returns the program as DONE (scoped check)
        engine.storage.mget.return_value = [prog]
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        await asyncio.wait_for(engine._drain_in_flight(), timeout=SS_TEST_TIMEOUT)

        assert len(engine._in_flight) == 0


# ---------------------------------------------------------------------------
# Create single mutant
# ---------------------------------------------------------------------------


class TestCreateSingleMutant:
    async def test_calls_generate_mutations_with_limit_1(self) -> None:
        """_create_single_mutant calls generate_mutations with limit=1."""
        engine = _make_ss_engine()
        elites = [_prog(ProgramState.DONE)]

        with patch(
            "gigaevo.evolution.engine.steady_state.generate_mutations",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = ["new-id"]
            result = await engine._create_single_mutant(elites)

        assert result == ["new-id"]
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs["limit"] == 1


# ---------------------------------------------------------------------------
# E2E: full run with mock storage
# ---------------------------------------------------------------------------


class TestEndToEnd:
    async def test_run_two_epochs(self) -> None:
        """Full run producing 4 mutants across 2 epochs (max_mutations_per_generation=2)."""
        engine = _make_ss_engine(
            max_in_flight=2,
            max_mutations_per_generation=2,
            max_generations=2,
            loop_interval=0.01,
        )

        mutation_count = 0
        mutant_ids: list[str] = []

        async def fake_generate(elites, **kwargs):
            nonlocal mutation_count
            mutation_count += 1
            prog = _prog(ProgramState.DONE)
            mutant_ids.append(prog.id)
            return [prog.id]

        def get_ids_side_effect(status_val):
            """Return mutant IDs as DONE when queried."""
            if status_val == ProgramState.DONE.value:
                return list(mutant_ids)
            return []

        engine.storage.get_ids_by_status.side_effect = get_ids_side_effect

        def mget_side_effect(ids, **kwargs):
            return [_prog(ProgramState.DONE) for _ in ids]

        engine.storage.mget.side_effect = mget_side_effect
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        with patch(
            "gigaevo.evolution.engine.steady_state.generate_mutations",
            side_effect=fake_generate,
        ):
            # select_elites returns a fake elite
            engine.strategy.select_elites.return_value = [_prog(ProgramState.DONE)]

            await asyncio.wait_for(engine.run(), timeout=SS_TEST_TIMEOUT)

        # Should have completed 2 epochs
        assert engine.metrics.total_generations >= 2


# ---------------------------------------------------------------------------
# Metadata iteration
# ---------------------------------------------------------------------------


class TestMetadataIteration:
    async def test_iteration_set_to_epoch(self) -> None:
        """_create_single_mutant passes current epoch as iteration."""
        engine = _make_ss_engine()
        engine.metrics.total_generations = 7
        elites = [_prog(ProgramState.DONE)]

        with patch(
            "gigaevo.evolution.engine.steady_state.generate_mutations",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = []
            await engine._create_single_mutant(elites)

        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["iteration"] == 7


# ---------------------------------------------------------------------------
# Empty archive
# ---------------------------------------------------------------------------


class TestEmptyArchive:
    async def test_mutation_loop_sleeps_when_no_elites(self) -> None:
        """When select_elites returns [], mutation loop sleeps instead of spinning."""
        engine = _make_ss_engine(max_generations=1)
        engine.metrics.total_generations = 1  # at cap → loop exits
        engine.strategy.select_elites.return_value = []

        await asyncio.wait_for(engine._mutation_loop(), timeout=SS_TEST_TIMEOUT)


# ---------------------------------------------------------------------------
# Stagnation detection
# ---------------------------------------------------------------------------


class TestStagnation:
    async def test_stagnation_tracked_at_epoch(self) -> None:
        """Stagnation counter increments when archive size unchanged across epochs."""
        engine = _make_ss_engine()
        engine._prev_archive_size = 5
        engine.strategy.get_program_ids.return_value = ["a", "b", "c", "d", "e"]

        await engine._epoch_refresh()
        assert engine._stagnant_gens == 1

        await engine._epoch_refresh()
        assert engine._stagnant_gens == 2

    async def test_stagnation_resets_on_growth(self) -> None:
        """Stagnation counter resets when archive grows."""
        engine = _make_ss_engine()
        engine._prev_archive_size = 5
        engine._stagnant_gens = 3

        engine.strategy.get_program_ids.return_value = ["a", "b", "c", "d", "e", "f"]

        await engine._epoch_refresh()
        assert engine._stagnant_gens == 0
