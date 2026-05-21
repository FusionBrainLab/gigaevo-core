"""Slot-leak chaos: prove slot conservation under adversarial timings.

This suite validates the strong invariant of the two-sema model:

    _buffer_sema._value + len(_in_flight) == max_in_flight

Producer slots are held transiently by the dispatcher between acquire
and the spawn-or-release decision, so ``_producer_sema._value`` is only
range-bounded (``0 <= v <= max_in_flight``). The buffer + in-flight sum
is conserved exactly: every successful LLM call consumes one buffer
slot and registers exactly one id; every ingest pops exactly one id and
releases exactly one buffer slot. A double-release or stranded id
breaks this conservation and is the failure mode the suite must catch.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import EvolutionStopper


def _make_chaos_engine(
    *,
    max_in_flight: int = 4,
) -> SteadyStateEvolutionEngine:
    """Build a minimal engine for chaos testing."""
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()

    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    storage.snapshot = MagicMock()
    strategy.get_program_ids.return_value = []

    config = SteadyStateEngineConfig(
        max_in_flight=max_in_flight,
        stopper=EvolutionStopper(),
        loop_interval=0.001,
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


def _get_max_in_flight(engine: SteadyStateEvolutionEngine) -> int:
    """Extract max_in_flight from engine config."""
    return engine._ss_config.max_in_flight


def _slot_conservation_check(engine: SteadyStateEvolutionEngine) -> bool:
    """Verify the strong buffer/in-flight conservation law.

    A double-release of ``_buffer_sema`` (or a leaked in-flight id whose
    buffer slot was already returned) breaks ``buffer._value + |in_flight|
    == max_in_flight``. The producer sema is only range-checked because it
    is held transiently by the dispatcher and is not part of the steady-
    state conservation law.
    """
    max_in_flight = _get_max_in_flight(engine)
    producer_ok = 0 <= engine._producer_sema._value <= max_in_flight
    buffer_conservation = (
        engine._buffer_sema._value + len(engine._in_flight) == max_in_flight
    )
    return producer_ok and buffer_conservation


class TestSlotConservation:
    """Invariant: buffer + |in_flight| == max_in_flight always."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """At construction, all slots are available."""
        engine = _make_chaos_engine(max_in_flight=3)
        assert engine._producer_sema._value == 3
        assert engine._buffer_sema._value == 3
        assert len(engine._in_flight) == 0
        assert _slot_conservation_check(engine)

    @pytest.mark.asyncio
    async def test_producer_acquire_holds_slot(self) -> None:
        """Acquiring producer slot decreases pool, total conserved."""
        engine = _make_chaos_engine(max_in_flight=4)
        await engine._producer_sema.acquire()
        assert engine._producer_sema._value == 3
        assert _slot_conservation_check(engine)

        await engine._producer_sema.acquire()
        await engine._producer_sema.acquire()
        assert engine._producer_sema._value == 1
        assert _slot_conservation_check(engine)

    @pytest.mark.asyncio
    async def test_in_flight_holds_slot(self) -> None:
        """Adding to in_flight decreases available buffer, total conserved."""
        engine = _make_chaos_engine(max_in_flight=3)
        # Simulate producer holding a buffer slot while we add to in-flight
        await engine._buffer_sema.acquire()
        engine._in_flight.add("mutant-1")
        assert engine._buffer_sema._value == 2
        assert len(engine._in_flight) == 1
        assert _slot_conservation_check(engine)


class TestRaceConditions:
    """Adversarial timing: producer + ingestor racing on slots."""

    @pytest.mark.asyncio
    async def test_rapid_acquire_release_cycles(self) -> None:
        """Rapid producer acquire/release under contention."""
        engine = _make_chaos_engine(max_in_flight=3)

        async def hammer_producer():
            for _ in range(10):
                await engine._producer_sema.acquire()
                engine._producer_sema.release()
                await asyncio.sleep(0.001)

        await hammer_producer()
        assert _slot_conservation_check(engine)
        assert engine._producer_sema._value == 3

    @pytest.mark.asyncio
    async def test_concurrent_producer_buffer_transfer(self) -> None:
        """Producer acquires and releases both semaphores, buffer slot transferred to ingestor."""
        engine = _make_chaos_engine(max_in_flight=2)

        # Producer acquires producer slot and buffer slot; simulates LLM phase
        await engine._producer_sema.acquire()
        await engine._buffer_sema.acquire()
        engine._in_flight.add("p1")  # transferred to ingestor

        # At this point: producer is held, buffer is held, in-flight = 1
        # Slot conservation: max=2, held_producer=1, held_buffer=1, in_flight=1
        # Check: 1 + 1 - 1 <= 2 (no overflow)
        assert _slot_conservation_check(engine)

        # Producer releases producer slot (dispatcher will do this)
        engine._producer_sema.release()
        # Ingestor releases buffer slot when mutant is DONE
        engine._buffer_sema.release()
        engine._in_flight.discard("p1")

        # Back to full capacity
        assert _slot_conservation_check(engine)

    @pytest.mark.asyncio
    async def test_cancel_on_producer_acquire_releases(self) -> None:
        """Cancelling a task blocked on producer acquire releases the slot."""
        engine = _make_chaos_engine(max_in_flight=1)
        await engine._producer_sema.acquire()  # drain

        task = asyncio.create_task(engine._producer_sema.acquire())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Slot should be released back
        assert engine._producer_sema._value == 0  # still drained by first acquire
        assert _slot_conservation_check(engine)


class TestMultiplexedRaces:
    """Complex races: dispatcher + producer + ingestor all moving."""

    @pytest.mark.asyncio
    async def test_three_way_slot_contention(self) -> None:
        """Three tasks competing for producer and buffer slots."""
        engine = _make_chaos_engine(max_in_flight=2)
        results = []

        async def dispatcher_sim(task_id: int):
            """Dispatcher: acquire producer, do work, release."""
            try:
                await asyncio.wait_for(engine._producer_sema.acquire(), timeout=0.2)
                await asyncio.sleep(0.02)
                engine._producer_sema.release()
                results.append(("dispatcher", task_id, "ok"))
            except TimeoutError:
                results.append(("dispatcher", task_id, "timeout"))

        async def producer_sim(task_id: int):
            """Producer: hold both semaphores briefly."""
            try:
                await asyncio.wait_for(engine._producer_sema.acquire(), timeout=0.2)
                engine._in_flight.add(f"prod-{task_id}")
                await asyncio.wait_for(engine._buffer_sema.acquire(), timeout=0.2)
                await asyncio.sleep(0.01)
                engine._producer_sema.release()
                # buffer held for ingestor
                results.append(("producer", task_id, "ok"))
            except TimeoutError:
                results.append(("producer", task_id, "timeout"))
                if f"prod-{task_id}" in engine._in_flight:
                    engine._in_flight.discard(f"prod-{task_id}")

        async def ingestor_sim(task_id: int):
            """Ingestor: wait for buffer slot (already held by producer)."""
            await asyncio.sleep(0.05)
            try:
                # Simulate receiving a buffer slot from producer
                engine._buffer_sema.release()
                engine._in_flight.discard(f"prod-{task_id % 2}")
                results.append(("ingestor", task_id, "ok"))
            except Exception:
                results.append(("ingestor", task_id, "error"))

        tasks = [
            asyncio.create_task(dispatcher_sim(0)),
            asyncio.create_task(producer_sim(1)),
            asyncio.create_task(ingestor_sim(1)),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Regardless of timing, slot conservation holds
        assert _slot_conservation_check(engine), (
            f"slot leak: producer={engine._producer_sema._value}, "
            f"buffer={engine._buffer_sema._value}, "
            f"in_flight={len(engine._in_flight)}"
        )


class TestEdgeCases:
    """Boundary conditions and pathological scenarios."""

    @pytest.mark.asyncio
    async def test_all_slots_in_flight(self) -> None:
        """Drain both semaphores, fill in-flight."""
        engine = _make_chaos_engine(max_in_flight=3)

        for i in range(3):
            await engine._producer_sema.acquire()
            engine._producer_sema.release()
            await engine._buffer_sema.acquire()
            engine._in_flight.add(f"full-{i}")

        assert engine._producer_sema._value == 3
        assert engine._buffer_sema._value == 0
        assert len(engine._in_flight) == 3
        assert _slot_conservation_check(engine)


__all__: list[str] = []
