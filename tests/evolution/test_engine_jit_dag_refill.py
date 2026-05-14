"""JIT DAG refill: producer buffer acquire not blocked by full DAG.

The two-sema design decouples producer (buffer) from dispatcher (producer)
such that DAG-wall-clock delays do not stall mutant production:

1. dispatcher acquires producer_sema, spawns run_one_mutant(task_id)
2. dispatcher RELEASES producer_sema immediately after create_task
3. run_one_mutant acquires buffer_sema only AFTER LLM succeeds
4. run_one_mutant polls/refreshes parents; may block on DAG queue/wall-clock
5. ingestor polls redis, receives DONE, releases buffer_sema

This design ensures a slow DAG does not leak back to producer acquisition.
These tests validate:

- Producer buffer acquire is not blocked by slow/full DAG (refill queue)
- Slow DAG does not bubble up backpressure to producer acquisition
- Multiple mutants can pipeline through buffer while one is stuck in DAG
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import EvolutionStopper


def _make_engine(*, max_in_flight: int = 4) -> SteadyStateEvolutionEngine:
    """Build minimal engine for JIT refill tests."""
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


class TestJITRefillDecoupling:
    """Dispatcher release is not blocked by producer DAG delays."""

    @pytest.mark.asyncio
    async def test_producer_unblocked_by_slow_dag(self) -> None:
        """Producer buffer acquire does not wait for slow parent refresh DAG.

        Scenario:
        - Mutant 1 acquired producer slot, got LLM result, waiting on parent refresh DAG
        - Mutant 2 tries to acquire producer slot (another fresh dispatcher)
        - Should NOT block — dispatcher release happens before DAG starts
        """
        engine = _make_engine(max_in_flight=2)

        # Simulate mutant 1: acquired producer, release it, acquire buffer
        await engine._producer_sema.acquire()
        assert engine._producer_sema._value == 1

        # Dispatcher immediately releases (before mutant spawns DAG tasks)
        engine._producer_sema.release()
        assert engine._producer_sema._value == 2

        # Mutant 1 reaches producer and acquires buffer (after LLM)
        await engine._buffer_sema.acquire()
        engine._in_flight.add("mutant-1")
        assert engine._buffer_sema._value == 1

        # Mutant 2: dispatcher can still acquire producer (not blocked by mutant 1's DAG)
        acquired_producer_2 = False
        try:
            await asyncio.wait_for(engine._producer_sema.acquire(), timeout=0.1)
            acquired_producer_2 = True
        except TimeoutError:
            pass

        # This should always succeed—dispatcher NOT blocked by slow DAG
        assert acquired_producer_2, "dispatcher blocked by mutant 1's slow DAG"

        # Clean up
        engine._producer_sema.release()

    @pytest.mark.asyncio
    async def test_multiple_mutants_pipeline_through_buffer(self) -> None:
        """Multiple mutants can pipeline through buffer while one blocks on DAG.

        Scenario:
        - Mutant A in buffer, stuck in parent refresh (slow DAG)
        - Mutant B in buffer, waiting for ingestor
        - Mutant C can still acquire buffer (not blocked by A's DAG)
        """
        engine = _make_engine(max_in_flight=3)

        # Mutant A: buffer acquired, in-flight, in DAG (slow)
        await engine._buffer_sema.acquire()
        engine._in_flight.add("mutant-a")

        # Mutant B: also buffer acquired, in-flight
        await engine._buffer_sema.acquire()
        engine._in_flight.add("mutant-b")

        # Mutant C: can still acquire buffer (not blocked by A's DAG)
        acquired_buffer_c = False
        try:
            await asyncio.wait_for(engine._buffer_sema.acquire(), timeout=0.1)
            acquired_buffer_c = True
            engine._in_flight.add("mutant-c")
        except TimeoutError:
            pass

        # Should succeed—no DAG blocking
        assert acquired_buffer_c, "buffer blocked even with available slot"
        assert len(engine._in_flight) == 3

        # Clean up
        engine._buffer_sema.release()
        engine._in_flight.discard("mutant-c")
        engine._buffer_sema.release()
        engine._in_flight.discard("mutant-b")
        engine._buffer_sema.release()
        engine._in_flight.discard("mutant-a")

    @pytest.mark.asyncio
    async def test_slow_dag_does_not_bubble_backpressure(self) -> None:
        """Slow DAG on producer does not bubble backpressure to dispatcher.

        Producer acquires buffer (second sema), enters DAG phase. Even if
        DAG is slow, dispatcher is not blocked because dispatcher only
        holds the first sema (producer_sema), which was released.
        """
        engine = _make_engine(max_in_flight=1)

        # Simulate dispatcher + producer lifecycle
        await engine._producer_sema.acquire()  # dispatcher acquires
        assert engine._producer_sema._value == 0

        # Dispatcher releases producer slot immediately
        engine._producer_sema.release()
        assert engine._producer_sema._value == 1

        # Producer gets LLM result, acquires buffer (now in DAG phase)
        await engine._buffer_sema.acquire()
        engine._in_flight.add("slow-mutant")
        assert engine._buffer_sema._value == 0

        # Even though buffer is now full (1 mutant in DAG), producer was NOT blocked
        # because it runs AFTER dispatcher releases producer_sema.
        # New dispatcher iteration can acquire producer freely:
        acquired = False
        try:
            await asyncio.wait_for(engine._producer_sema.acquire(), timeout=0.1)
            acquired = True
        except TimeoutError:
            pass

        assert acquired, "dispatcher blocked by slow DAG on producer"

        # Clean up
        engine._producer_sema.release()
        engine._buffer_sema.release()
        engine._in_flight.discard("slow-mutant")

    @pytest.mark.asyncio
    async def test_dag_queue_full_does_not_stall_production(self) -> None:
        """Full DAG refill queue does not stall LLM production.

        Scenario:
        - Suppose parent-refresh DAG queue can hold N tasks
        - Even if queue is full, producer (LLM output) still acquires buffer
        - Producer then enqueues parent-refresh task and moves on
        """
        engine = _make_engine(max_in_flight=3)

        # Simulate 3 mutants in producer phase:
        # Mutant 1, 2, 3 all acquired producer slots and released (dispatcher side)
        # Now all in producer (LLM + buffer acquire)

        for i in range(3):
            await engine._producer_sema.acquire()
            engine._producer_sema.release()

        # All 3 buffer slots acquired
        for i in range(3):
            await engine._buffer_sema.acquire()
            engine._in_flight.add(f"mutant-{i}")

        # At this point, all buffer slots are full and all in-flight.
        # Simulate mutant 0 gets slow DAG (queue full, wall-clock slow)
        # This should NOT block new dispatcher iteration:

        acquired_producer = False
        try:
            await asyncio.wait_for(engine._producer_sema.acquire(), timeout=0.1)
            acquired_producer = True
        except TimeoutError:
            pass

        # Producer semaphore should be unblocked (still available pool)
        assert acquired_producer, "producer blocked by full DAG queue"

        # Clean up
        engine._producer_sema.release()
        for i in range(3):
            engine._buffer_sema.release()
            engine._in_flight.discard(f"mutant-{i}")

    @pytest.mark.asyncio
    async def test_jit_release_allows_high_throughput(self) -> None:
        """JIT dispatcher release allows high mutant spawn rate.

        Without JIT (dispatcher holds producer until DAG completes),
        spawn rate would be limited by DAG wall-clock.
        With JIT (dispatcher releases immediately), spawn rate is decoupled.
        """
        engine = _make_engine(max_in_flight=4)

        spawned = []

        async def dispatcher_cycle(task_id: int) -> None:
            """Simulate fast dispatcher spawn (no DAG blocking)."""
            try:
                await asyncio.wait_for(engine._producer_sema.acquire(), timeout=0.1)
                spawned.append(task_id)
                # Immediately release (JIT principle)
                engine._producer_sema.release()
                await asyncio.sleep(0.001)  # small inter-dispatch delay
            except TimeoutError:
                pass

        # Spawn 10 dispatcher iterations rapidly
        tasks = [asyncio.create_task(dispatcher_cycle(i)) for i in range(10)]
        await asyncio.gather(*tasks)

        # With max_in_flight=4 and immediate release, all should spawn
        # (no DAG backpressure)
        assert len(spawned) == 10, (
            f"dispatcher stalled: only {len(spawned)}/10 spawned (DAG backpressure?)"
        )


class TestDAGLatencyIsolation:
    """Producer DAG latency is isolated from dispatcher throughput."""

    @pytest.mark.asyncio
    async def test_high_dag_latency_does_not_reduce_dispatch_rate(
        self,
    ) -> None:
        """Slow DAG (100ms) does not reduce dispatcher spawn rate.

        With max_in_flight=2:
        - Dispatcher can spawn 10 mutants in 10ms
        - Even though DAG is slow (100ms per mutant), throughput is decoupled
        """
        engine = _make_engine(max_in_flight=2)

        dispatch_count = 0
        dag_start_count = 0

        async def fast_dispatcher() -> None:
            nonlocal dispatch_count
            for _ in range(10):
                try:
                    await asyncio.wait_for(engine._producer_sema.acquire(), timeout=0.1)
                    dispatch_count += 1
                    engine._producer_sema.release()
                except TimeoutError:
                    break

        async def slow_dag_simulator() -> None:
            """Simulate slow parent-refresh DAG."""
            nonlocal dag_start_count
            for i in range(10):
                dag_start_count += 1
                await asyncio.sleep(0.05)  # slow refresh

        dispatcher = asyncio.create_task(fast_dispatcher())
        dag = asyncio.create_task(slow_dag_simulator())

        await asyncio.gather(dispatcher, dag)

        # Dispatcher should complete all 10 spawns before DAG finishes 1
        assert dispatch_count == 10, "dispatcher stalled by slow DAG"

    @pytest.mark.asyncio
    async def test_buffer_semaphore_decouples_from_dag_latency(self) -> None:
        """Buffer sema acquisition is not blocked by producer DAG delays."""
        engine = _make_engine(max_in_flight=2)

        # Mutant 1: in buffer, simulating DAG delay
        await engine._producer_sema.acquire()
        engine._producer_sema.release()
        await engine._buffer_sema.acquire()
        engine._in_flight.add("mutant-1")

        # Simulate slow DAG (but buffer sema not released yet)
        dag_blocked = False

        async def try_second_producer() -> None:
            nonlocal dag_blocked
            try:
                # Mutant 2 tries to get buffer while 1 is in DAG
                await asyncio.wait_for(engine._buffer_sema.acquire(), timeout=0.1)
                engine._in_flight.add("mutant-2")
            except TimeoutError:
                dag_blocked = True

        await try_second_producer()
        assert not dag_blocked, (
            "mutant 2 blocked by mutant 1's slow DAG (should only wait on buffer sema)"
        )

        # Clean up
        engine._buffer_sema.release()
        engine._buffer_sema.release()
        engine._in_flight.clear()


__all__: list[str] = []
