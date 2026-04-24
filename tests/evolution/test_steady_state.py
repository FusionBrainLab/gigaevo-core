"""Unit tests for SteadyStateEvolutionEngine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.engine.config import EngineConfig, SteadyStateEngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import EvolutionStopper, MaxGenerationsStopper
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
    storage.snapshot = MagicMock()
    strategy.get_program_ids.return_value = []

    stopper = (
        MaxGenerationsStopper(max_generations)
        if max_generations is not None
        else EvolutionStopper()
    )
    config = SteadyStateEngineConfig(
        max_in_flight=max_in_flight,
        max_mutations_per_generation=max_mutations_per_generation,
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

    async def test_persists_programs_processed(self) -> None:
        """Epoch refresh persists programs_processed via the engine snapshot."""
        engine = _make_ss_engine()
        engine.metrics.programs_processed = 37

        await engine._epoch_refresh()

        # Verify the snapshot write carried programs_processed=37.
        calls = engine.storage.save_run_state.call_args_list
        snap_calls = [c for c in calls if c.args[0] == "engine:snapshot"]
        assert snap_calls, "_epoch_refresh must persist engine:snapshot"
        assert any('"programs_processed":37' in c.args[1] for c in snap_calls)


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


# ---------------------------------------------------------------------------
# Bucketed refresh order (generation-bucketed archive refresh)
# ---------------------------------------------------------------------------


def _prog_with_gen(generation: int, state: ProgramState = ProgramState.DONE) -> Program:
    """Build a Program whose lineage.generation equals `generation`.

    Program construction uses create_child to stamp lineage.generation; here we
    assign state and verify metadata via the lineage attribute.
    """
    p = Program(code=f"def solve_{generation}(): return {generation}", state=state)
    # Lineage is frozen-ish on Program; assign via direct attribute write which
    # Pydantic allows because Lineage is a mutable submodel.
    p.lineage.generation = generation
    return p


class TestRefreshOrderConfig:
    """refresh_order config option (fifo vs. generation_bucketed)."""

    def test_default_is_fifo(self) -> None:
        """Default refresh_order preserves prior behaviour (fifo)."""
        config = SteadyStateEngineConfig(
            max_in_flight=1, max_mutations_per_generation=1
        )
        assert config.refresh_order == "fifo"

    def test_accepts_generation_bucketed(self) -> None:
        """refresh_order='generation_bucketed' is a valid setting."""
        config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
        )
        assert config.refresh_order == "generation_bucketed"

    def test_rejects_unknown_value(self) -> None:
        """Unknown refresh_order values are rejected by pydantic."""
        with pytest.raises(Exception):  # pydantic.ValidationError
            SteadyStateEngineConfig(
                max_in_flight=1,
                max_mutations_per_generation=1,
                refresh_order="random",  # type: ignore[arg-type]
            )


class TestBucketedRefresh:
    """Bucketed-by-generation archive refresh eliminates cross-program race."""

    async def test_fifo_refresh_flips_all_at_once(self) -> None:
        """fifo mode flips all ids in one batch_transition_by_ids call."""
        engine = _make_ss_engine()
        # default refresh_order is fifo
        engine.strategy.get_program_ids.return_value = ["a", "b", "c"]
        engine.storage.batch_transition_by_ids.return_value = 3

        refreshed = await engine._refresh_archive_programs()

        assert refreshed == 3
        # Exactly one flip batch
        assert engine.storage.batch_transition_by_ids.call_count == 1
        flipped_ids = engine.storage.batch_transition_by_ids.call_args.args[0]
        assert set(flipped_ids) == {"a", "b", "c"}

    async def test_bucketed_flips_in_generation_order(self) -> None:
        """Bucketed mode flips programs bucket-by-bucket in ascending generation order.

        This is the race-fix: the bucket for gen N must fully drain before the
        bucket for gen N+1 starts, so a child at gen N+1 never reads tracker
        data before its parent at gen N has written it.
        """
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
        )

        # Three programs across two generations: parent gen=1, children gen=2.
        p_gen1 = _prog_with_gen(1)
        c1_gen2 = _prog_with_gen(2)
        c2_gen2 = _prog_with_gen(2)

        engine.strategy.get_program_ids.return_value = [
            c1_gen2.id,
            p_gen1.id,  # deliberately shuffled — bucketing must re-order
            c2_gen2.id,
        ]
        engine.storage.mget.return_value = [c1_gen2, p_gen1, c2_gen2]

        flip_order: list[frozenset[str]] = []

        async def fake_flip(ids, old, new):
            flip_order.append(frozenset(ids))
            return len(ids)

        engine.storage.batch_transition_by_ids.side_effect = fake_flip

        # _await_idle is called between buckets — fake it so we don't block
        engine._await_idle = AsyncMock()  # type: ignore[method-assign]

        refreshed = await engine._refresh_archive_programs()

        assert refreshed == 3
        # Must be 2 flips (one per generation bucket)
        assert len(flip_order) == 2, f"expected 2 buckets, got {len(flip_order)}"
        # First bucket = gen 1 (parents)
        assert flip_order[0] == {p_gen1.id}
        # Second bucket = gen 2 (children)
        assert flip_order[1] == {c1_gen2.id, c2_gen2.id}
        # _await_idle called between buckets (at least once)
        assert engine._await_idle.await_count >= 1

    async def test_bucketed_awaits_idle_between_buckets(self) -> None:
        """Bucketed mode MUST await idle after each bucket so next bucket sees drained state."""
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
        )

        p1 = _prog_with_gen(1)
        p2 = _prog_with_gen(2)
        p3 = _prog_with_gen(3)

        engine.strategy.get_program_ids.return_value = [p1.id, p2.id, p3.id]
        engine.storage.mget.return_value = [p1, p2, p3]
        engine.storage.batch_transition_by_ids.return_value = 1

        call_sequence: list[str] = []

        async def fake_flip(ids, old, new):
            call_sequence.append(f"flip:{len(ids)}")
            return len(ids)

        async def fake_await_idle():
            call_sequence.append("idle")

        engine.storage.batch_transition_by_ids.side_effect = fake_flip
        engine._await_idle = fake_await_idle  # type: ignore[method-assign]

        await engine._refresh_archive_programs()

        # Pattern: flip, idle, flip, idle, flip (last idle may or may not — we check interleaving)
        flip_positions = [
            i for i, s in enumerate(call_sequence) if s.startswith("flip")
        ]
        idle_positions = [i for i, s in enumerate(call_sequence) if s == "idle"]
        assert len(flip_positions) == 3
        # Between every two flips there is at least one idle call
        for i in range(len(flip_positions) - 1):
            between = [
                p
                for p in idle_positions
                if flip_positions[i] < p < flip_positions[i + 1]
            ]
            assert between, (
                f"no await_idle between bucket {i} and {i + 1}: {call_sequence}"
            )

    async def test_bucketed_empty_ids_returns_zero(self) -> None:
        """Bucketed mode with no ids returns 0 and makes no calls."""
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
        )
        engine.strategy.get_program_ids.return_value = []

        refreshed = await engine._refresh_archive_programs()

        assert refreshed == 0
        engine.storage.batch_transition_by_ids.assert_not_called()

    async def test_bucketed_single_generation_single_batch(self) -> None:
        """All programs in same generation → single bucket, single flip."""
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
        )

        progs = [_prog_with_gen(5) for _ in range(3)]
        engine.strategy.get_program_ids.return_value = [p.id for p in progs]
        engine.storage.mget.return_value = progs
        engine.storage.batch_transition_by_ids.return_value = 3
        engine._await_idle = AsyncMock()  # type: ignore[method-assign]

        refreshed = await engine._refresh_archive_programs()

        assert refreshed == 3
        assert engine.storage.batch_transition_by_ids.call_count == 1


class TestRefreshPassesConfig:
    """refresh_passes config: how many times the bucketed refresh repeats."""

    def test_default_is_one(self) -> None:
        config = SteadyStateEngineConfig(
            max_in_flight=1, max_mutations_per_generation=1
        )
        assert config.refresh_passes == 1

    def test_accepts_two(self) -> None:
        config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_passes=2,
        )
        assert config.refresh_passes == 2

    def test_rejects_zero(self) -> None:
        with pytest.raises(Exception):
            SteadyStateEngineConfig(
                max_in_flight=1,
                max_mutations_per_generation=1,
                refresh_passes=0,
            )

    def test_rejects_negative(self) -> None:
        with pytest.raises(Exception):
            SteadyStateEngineConfig(
                max_in_flight=1,
                max_mutations_per_generation=1,
                refresh_passes=-1,
            )


class TestMultiPassRefresh:
    """refresh_passes=N loops bucketed flow N times with await_idle between passes.

    On D side the filtered LineageStage cache-invalidates per pass via a
    class-level refresh token.  The engine bumps the token before each pass
    so LineageStage re-runs with fresh cross-program tracker data in pass 2.
    """

    async def test_refresh_passes_two_loops_bucketed_twice(self) -> None:
        """refresh_passes=2 ⇒ bucketed flow repeats twice, same gen order each pass."""
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
            refresh_passes=2,
        )

        p1 = _prog_with_gen(1)
        p2 = _prog_with_gen(2)

        engine.strategy.get_program_ids.return_value = [p1.id, p2.id]
        engine.storage.mget.return_value = [p1, p2]

        flip_history: list[frozenset[str]] = []

        async def fake_flip(ids, old, new):
            flip_history.append(frozenset(ids))
            return len(ids)

        engine.storage.batch_transition_by_ids.side_effect = fake_flip
        engine._await_idle = AsyncMock()  # type: ignore[method-assign]

        total = await engine._refresh_archive_programs()

        # Two passes × two buckets = 4 flips
        assert len(flip_history) == 4, (
            f"expected 4 flips (2 passes × 2 buckets), got {len(flip_history)}"
        )
        # Pattern: (pass1: gen1, gen2) then (pass2: gen1, gen2)
        assert flip_history[0] == {p1.id}
        assert flip_history[1] == {p2.id}
        assert flip_history[2] == {p1.id}
        assert flip_history[3] == {p2.id}
        # Total flipped = 2+2+2+2 = count reported by fake_flip
        # (bucketed_refresh sums counts across all passes)
        assert total == 4
        # _await_idle called between buckets within each pass AND between passes
        assert (
            engine._await_idle.await_count >= 3
        )  # ≥1 per between-bucket, ≥1 between-passes

    async def test_refresh_passes_two_bumps_token_before_each_pass(self) -> None:
        """Each refresh pass bumps SharedBenchmarkFilteredLineageStage token.

        Token bumps are the cache-invalidation mechanism: with refresh_passes=2,
        LineageStage sees distinct cache keys on pass 1 and pass 2.
        """
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
            refresh_passes=2,
        )
        engine.strategy.get_program_ids.return_value = [_prog_with_gen(1).id]
        engine.storage.mget.return_value = [_prog_with_gen(1)]
        engine.storage.batch_transition_by_ids.return_value = 1
        engine._await_idle = AsyncMock()  # type: ignore[method-assign]

        initial_token = engine._snapshot.refresh_pass
        await engine._refresh_archive_programs()
        final_token = engine._snapshot.refresh_pass

        # Two passes ⇒ counter bumped by at least 2
        assert final_token - initial_token >= 2, (
            f"expected refresh_pass bumped ≥2 with refresh_passes=2, "
            f"got {final_token - initial_token}"
        )

    async def test_refresh_passes_one_bumps_token_once(self) -> None:
        """refresh_passes=1 (default) still bumps the counter once per refresh."""
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
            refresh_passes=1,
        )
        engine.strategy.get_program_ids.return_value = [_prog_with_gen(1).id]
        engine.storage.mget.return_value = [_prog_with_gen(1)]
        engine.storage.batch_transition_by_ids.return_value = 1
        engine._await_idle = AsyncMock()  # type: ignore[method-assign]

        initial_token = engine._snapshot.refresh_pass
        await engine._refresh_archive_programs()
        final_token = engine._snapshot.refresh_pass

        assert final_token - initial_token == 1

    async def test_fifo_refresh_passes_two_loops_twice(self) -> None:
        """fifo mode with refresh_passes=2 flips twice (whole-archive each pass)."""
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="fifo",
            refresh_passes=2,
        )
        engine.strategy.get_program_ids.return_value = ["a", "b"]
        engine.storage.batch_transition_by_ids.return_value = 2
        engine._await_idle = AsyncMock()  # type: ignore[method-assign]

        await engine._refresh_archive_programs()
        # Two passes for fifo ⇒ two flips of the whole archive
        assert engine.storage.batch_transition_by_ids.call_count == 2


# ---------------------------------------------------------------------------
# Two-pass refresh semantic: stage actually reads updated tracker state
#
# The existing TestMultiPassRefresh proves mechanics (counter bumps, flow
# repeats).  This class proves the end-to-end SEMANTIC: when the engine
# runs _refresh_archive_programs with refresh_passes=2, the downstream
# SharedBenchmarkFilteredLineageStage reads tracker data that pass 1 wrote,
# not the stale pre-refresh values.  That is the whole point of the
# two-pass design on D — closing the cross-program tracker race so
# LineageStage's narrative reflects fresh opponent metrics.
# ---------------------------------------------------------------------------


class TestTwoPassRefreshSemantic:
    """End-to-end: engine's 2-pass refresh actually changes the stage's read.

    Timeline enforced by the test:
      T0: tracker seeded (child fitness vs g1 = 0.3)
      T0: stage.preprocess(child) → evidence reports child_fitness=0.3
      T0→T2: engine runs refresh_archive_programs(refresh_passes=2).
            A hook on _refresh_archive_programs_one_pass mutates the tracker
            during pass 1 to write fitness=0.5 — simulating what
            DGTrackerStage would re-write under the refreshed HoF.
      T2: stage.preprocess(child) → evidence reports child_fitness=0.5
      T2: compute_hash(params) suffix advanced by +2 (one bump per pass)

    If the engine didn't bump the token between passes, a real pipeline's
    cache layer would return pass-1's stale output in pass 2 — the stage
    would never "see" the fresh tracker data and the two-pass design
    would be a no-op.  The hash assertion proves the cache key would miss.
    """

    @pytest.fixture(autouse=True)
    def _reset_token(self):
        from gigaevo.evolution.engine.snapshot import (
            _reset_current_snapshot_for_tests,
        )

        _reset_current_snapshot_for_tests()
        yield
        _reset_current_snapshot_for_tests()

    async def test_stage_reads_fresh_tracker_after_two_pass_refresh(self) -> None:
        import fakeredis.aioredis
        import pytest as _pytest

        from gigaevo.adversarial.dg_tracker import DGImprovementTracker
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )
        from gigaevo.programs.metrics.context import (
            VALIDITY_KEY,
            MetricsContext,
            MetricSpec,
        )
        from gigaevo.programs.stages.common import CacheOnlyInput

        # ---- Tracker (fakeredis-backed) -------------------------------
        tracker = DGImprovementTracker(host="localhost", port=6379, db=0, prefix="tpr")
        tracker._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

        metrics_ctx = MetricsContext(
            specs={
                "fitness": MetricSpec(
                    description="main",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
                VALIDITY_KEY: MetricSpec(
                    description="validity",
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            }
        )

        # ---- Programs --------------------------------------------------
        p_parent = _prog_with_gen(1)
        p_child = _prog_with_gen(2)
        p_child.lineage.parents = [p_parent.id]

        await tracker.record_metrics(
            p_parent.id, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0}
        )
        await tracker.record_metrics(
            p_child.id, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0}
        )

        # ---- Stage (__new__ bypass; __init__ needs LLM we don't want) -
        stage_storage = AsyncMock()

        async def _fake_mget(ids):
            return [p_parent] if p_parent.id in ids else []

        stage_storage.mget.side_effect = _fake_mget

        stage = SharedBenchmarkFilteredLineageStage.__new__(
            SharedBenchmarkFilteredLineageStage
        )
        stage._tracker = tracker
        stage._min_shared = 1
        stage._inject_shared_evidence = True
        stage._metrics_context = metrics_ctx
        stage.storage = stage_storage

        params = CacheOnlyInput(cache_on="probe")

        # ---- Phase 1: pre-refresh stage read ---------------------------
        from gigaevo.evolution.engine.snapshot import get_current_snapshot

        pre = await stage.preprocess(p_child, params)
        initial_token = get_current_snapshot().refresh_pass
        hash_pre = SharedBenchmarkFilteredLineageStage.compute_hash(params)

        assert isinstance(pre, dict), "parent should survive the shared filter"
        assert pre["evidence"][0].shared_child_metrics["fitness"] == _pytest.approx(
            0.3
        ), "pre-refresh preprocess must read the seed tracker state"

        # ---- Phase 2: engine 2-pass refresh; mutate tracker in pass 1 --
        engine = _make_ss_engine()
        engine._ss_config = SteadyStateEngineConfig(
            max_in_flight=1,
            max_mutations_per_generation=1,
            refresh_order="generation_bucketed",
            refresh_passes=2,
        )
        engine.strategy.get_program_ids.return_value = [p_parent.id, p_child.id]
        engine.storage.mget.return_value = [p_parent, p_child]
        engine.storage.batch_transition_by_ids.return_value = 1
        engine._await_idle = AsyncMock()  # type: ignore[method-assign]

        # Hook one_pass so during pass-1 we mutate the tracker.  This
        # simulates what DGTrackerStage does in a real pass 1: it re-runs
        # against the refreshed HoF and rewrites per-(D,G) metrics.
        call_count = {"n": 0}
        original_one_pass = engine._refresh_archive_programs_one_pass

        async def _one_pass_with_tracker_mutation() -> int:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Pass 1: simulate DGTrackerStage re-write.  Child's fitness
                # against g1 moves from 0.3 (stale) → 0.5 (fresh HoF eval).
                await tracker.record_metrics(
                    p_child.id, "g1", {"fitness": 0.5, VALIDITY_KEY: 1.0}
                )
            return await original_one_pass()

        engine._refresh_archive_programs_one_pass = _one_pass_with_tracker_mutation  # type: ignore[method-assign]

        await engine._refresh_archive_programs()

        # ---- Phase 3: post-refresh stage read --------------------------
        post = await stage.preprocess(p_child, params)
        final_token = get_current_snapshot().refresh_pass
        hash_post = SharedBenchmarkFilteredLineageStage.compute_hash(params)

        assert isinstance(post, dict)
        assert post["evidence"][0].shared_child_metrics["fitness"] == _pytest.approx(
            0.5
        ), (
            "Post-refresh preprocess must read the tracker state written "
            "during pass 1; instead it read stale pre-refresh data — the "
            "two-pass refresh was a no-op semantically."
        )

        # Two bumps ⇒ cache key distinguishable from pre-refresh probe.
        assert final_token == initial_token + 2, (
            f"refresh_passes=2 should bump token exactly twice "
            f"(initial={initial_token}, final={final_token})"
        )
        assert hash_post != hash_pre, (
            "compute_hash unchanged after 2-pass refresh — a real cache "
            "would cache-HIT and return pass-1 stale output on pass 2."
        )
        assert hash_post.endswith(f":rp{initial_token + 2}")

        # Call order sanity: one_pass hook invoked exactly once per pass.
        assert call_count["n"] == 2, (
            f"expected _refresh_archive_programs_one_pass called 2x, "
            f"got {call_count['n']}"
        )
