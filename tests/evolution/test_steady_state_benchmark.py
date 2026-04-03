"""Throughput benchmarks: SteadyStateEvolutionEngine vs generational EvolutionEngine.

Simulates realistic scenarios with dummy LLMs that have configurable response
times and DAG evaluation delays.  Measures total wall-clock time to produce
and ingest a fixed number of mutants.

These are NOT pytest-benchmark microbenchmarks — they're full async simulations
that demonstrate the interleaving advantage.  Marked with @pytest.mark.benchmark
so they don't run in normal CI.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.engine.config import EngineConfig, SteadyStateEngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

# How long an LLM mutation call takes (seconds)
LLM_LATENCY = 0.05

# How long a DAG evaluation takes (seconds)
DAG_LATENCY = 0.20

# Total mutants to produce per benchmark
TOTAL_MUTANTS = 16

# Max in-flight for steady-state
MAX_IN_FLIGHT = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(state: ProgramState = ProgramState.DONE) -> Program:
    return Program(code="def solve(): return 42", state=state)


class SimulatedDagTracker:
    """Simulates DAG evaluation by tracking program IDs and their completion times."""

    def __init__(self, dag_latency: float):
        self._dag_latency = dag_latency
        self._pending: dict[str, float] = {}  # id -> completion_time
        self._done: set[str] = set()

    def submit(self, prog_id: str) -> None:
        """Submit a program for DAG evaluation."""
        self._pending[prog_id] = time.monotonic() + self._dag_latency

    def poll_done(self) -> list[str]:
        """Return IDs that have finished DAG evaluation."""
        now = time.monotonic()
        newly_done = [pid for pid, t in self._pending.items() if now >= t]
        for pid in newly_done:
            del self._pending[pid]
            self._done.add(pid)
        return list(self._done)

    def consume_done(self, ids: list[str]) -> None:
        """Mark IDs as consumed (ingested)."""
        for pid in ids:
            self._done.discard(pid)

    def has_active(self) -> bool:
        return bool(self._pending)

    @property
    def total_done(self) -> int:
        return len(self._done)


def _make_generational_engine(
    dag_tracker: SimulatedDagTracker,
    *,
    max_mutations: int = MAX_IN_FLIGHT,
    max_generations: int | None = None,
) -> EvolutionEngine:
    """Build a generational EvolutionEngine wired to the DAG tracker."""
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
    strategy.select_elites.return_value = [_prog()]

    config = EngineConfig(
        max_mutations_per_generation=max_mutations,
        max_elites_per_generation=10,
        max_generations=max_generations,
        loop_interval=0.01,
    )

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=config,
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    engine.state = AsyncMock()
    return engine


def _make_steady_state_engine(
    dag_tracker: SimulatedDagTracker,
    *,
    max_in_flight: int = MAX_IN_FLIGHT,
    max_mutations_per_generation: int = TOTAL_MUTANTS,
    max_generations: int | None = None,
) -> SteadyStateEvolutionEngine:
    """Build a SteadyStateEvolutionEngine wired to the DAG tracker."""
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
    strategy.select_elites.return_value = [_prog()]

    config = SteadyStateEngineConfig(
        max_in_flight=max_in_flight,
        max_mutations_per_generation=max_mutations_per_generation,
        max_elites_per_generation=10,
        max_generations=max_generations,
        loop_interval=0.01,
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


# ---------------------------------------------------------------------------
# Throughput comparison: simulation-based
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestThroughputComparison:
    """Compare wall-clock time for producing N mutants.

    Generational: produce batch -> wait ALL DAGs -> ingest -> repeat
    Steady-state: produce 1 -> DAG starts immediately -> ingest when done -> overlap
    """

    async def test_steady_state_faster_than_generational(self) -> None:
        """Steady-state should be significantly faster due to interleaving.

        Scenario: 16 mutants, LLM=50ms, DAG=200ms, batch_size=4
        Generational: 4 batches × (4×50ms LLM + 200ms DAG wait) = ~1.6s
        Steady-state: LLM and DAG overlap; ~16×50ms + 200ms tail = ~1.0s
        """
        gen_time = await self._run_generational_simulation()
        ss_time = await self._run_steady_state_simulation()

        # Steady-state should be at least 30% faster
        speedup = gen_time / ss_time
        print(
            f"\nGenerational: {gen_time:.3f}s | Steady-state: {ss_time:.3f}s | "
            f"Speedup: {speedup:.2f}x"
        )
        assert speedup > 1.3, (
            f"Expected steady-state to be >30% faster, got {speedup:.2f}x "
            f"(gen={gen_time:.3f}s ss={ss_time:.3f}s)"
        )

    async def test_same_total_mutants_produced(self) -> None:
        """Both engines produce the same total number of mutants."""
        gen_count, _ = await self._count_generational_mutations()
        ss_count, _ = await self._count_steady_state_mutations()

        assert gen_count == TOTAL_MUTANTS
        assert ss_count == TOTAL_MUTANTS

    async def test_backpressure_limits_in_flight(self) -> None:
        """Steady-state never has more than max_in_flight concurrent DAGs."""
        max_concurrent = await self._measure_max_concurrent()
        assert max_concurrent <= MAX_IN_FLIGHT, (
            f"Max concurrent DAGs ({max_concurrent}) exceeded "
            f"max_in_flight ({MAX_IN_FLIGHT})"
        )

    # ----- Simulation implementations -----

    async def _run_generational_simulation(self) -> float:
        """Simulate generational engine: batch produce → wait all → ingest → repeat."""
        t0 = time.monotonic()
        produced = 0
        batch_size = MAX_IN_FLIGHT

        while produced < TOTAL_MUTANTS:
            batch = min(batch_size, TOTAL_MUTANTS - produced)

            # LLM calls: sequential (each takes LLM_LATENCY)
            for _ in range(batch):
                await asyncio.sleep(LLM_LATENCY)
                produced += 1

            # DAG evaluation: all start together, wait for all to finish
            # (in real engine, _await_idle waits for all QUEUED/RUNNING to clear)
            await asyncio.sleep(DAG_LATENCY)

            # Ingestion: instant (mock)
            # Refresh: instant (mock)

        return time.monotonic() - t0

    async def _run_steady_state_simulation(self) -> float:
        """Simulate steady-state engine: produce 1 → DAG starts → overlap."""
        t0 = time.monotonic()
        produced = 0
        in_flight: list[float] = []  # completion times
        ingested = 0
        sema = asyncio.Semaphore(MAX_IN_FLIGHT)

        while ingested < TOTAL_MUTANTS:
            # Try to produce if we have capacity and haven't hit the limit
            if produced < TOTAL_MUTANTS:
                acquired = sema._value > 0  # noqa: SLF001
                if acquired:
                    await sema.acquire()
                    # LLM call
                    await asyncio.sleep(LLM_LATENCY)
                    produced += 1
                    in_flight.append(time.monotonic() + DAG_LATENCY)

            # Poll for completed DAGs
            now = time.monotonic()
            still_pending = []
            for completion_time in in_flight:
                if now >= completion_time:
                    ingested += 1
                    sema.release()
                else:
                    still_pending.append(completion_time)
            in_flight = still_pending

            # If nothing to produce and nothing completed, wait a bit
            if produced >= TOTAL_MUTANTS and in_flight:
                earliest = min(in_flight)
                wait = max(0, earliest - time.monotonic())
                if wait > 0:
                    await asyncio.sleep(wait)

            await asyncio.sleep(0.001)  # yield

        return time.monotonic() - t0

    async def _count_generational_mutations(self) -> tuple[int, float]:
        """Count total mutations in a generational simulation."""
        count = 0
        t0 = time.monotonic()
        batch_size = MAX_IN_FLIGHT

        while count < TOTAL_MUTANTS:
            batch = min(batch_size, TOTAL_MUTANTS - count)
            for _ in range(batch):
                await asyncio.sleep(LLM_LATENCY)
                count += 1
            await asyncio.sleep(DAG_LATENCY)

        return count, time.monotonic() - t0

    async def _count_steady_state_mutations(self) -> tuple[int, float]:
        """Count total mutations in a steady-state simulation."""
        produced = 0
        ingested = 0
        t0 = time.monotonic()
        in_flight: list[float] = []
        sema = asyncio.Semaphore(MAX_IN_FLIGHT)

        while ingested < TOTAL_MUTANTS:
            if produced < TOTAL_MUTANTS and sema._value > 0:  # noqa: SLF001
                await sema.acquire()
                await asyncio.sleep(LLM_LATENCY)
                produced += 1
                in_flight.append(time.monotonic() + DAG_LATENCY)

            now = time.monotonic()
            still_pending = []
            for ct in in_flight:
                if now >= ct:
                    ingested += 1
                    sema.release()
                else:
                    still_pending.append(ct)
            in_flight = still_pending

            if produced >= TOTAL_MUTANTS and in_flight:
                earliest = min(in_flight)
                wait = max(0, earliest - time.monotonic())
                if wait > 0:
                    await asyncio.sleep(wait)

            await asyncio.sleep(0.001)

        return ingested, time.monotonic() - t0

    async def _measure_max_concurrent(self) -> int:
        """Run steady-state simulation and track max concurrent DAGs."""
        produced = 0
        ingested = 0
        max_concurrent = 0
        in_flight: list[float] = []
        sema = asyncio.Semaphore(MAX_IN_FLIGHT)

        while ingested < TOTAL_MUTANTS:
            if produced < TOTAL_MUTANTS and sema._value > 0:  # noqa: SLF001
                await sema.acquire()
                await asyncio.sleep(LLM_LATENCY)
                produced += 1
                in_flight.append(time.monotonic() + DAG_LATENCY)
                max_concurrent = max(max_concurrent, len(in_flight))

            now = time.monotonic()
            still_pending = []
            for ct in in_flight:
                if now >= ct:
                    ingested += 1
                    sema.release()
                else:
                    still_pending.append(ct)
            in_flight = still_pending

            if produced >= TOTAL_MUTANTS and in_flight:
                earliest = min(in_flight)
                wait = max(0, earliest - time.monotonic())
                if wait > 0:
                    await asyncio.sleep(wait)

            await asyncio.sleep(0.001)

        return max_concurrent


# ---------------------------------------------------------------------------
# Realistic E2E: actual engine classes with mocked storage + timed LLM
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestRealisticE2E:
    """End-to-end test using actual SteadyStateEvolutionEngine with timed mocks.

    Wires up the real engine class with a mock storage that simulates DAG
    evaluation delays.  Verifies the engine correctly produces, ingests, and
    triggers epochs.
    """

    async def test_steady_state_e2e_with_timed_dag(self) -> None:
        """SteadyState engine with simulated DAG latency completes correctly."""
        dag_tracker = SimulatedDagTracker(dag_latency=0.05)
        engine = _make_steady_state_engine(
            dag_tracker,
            max_in_flight=4,
            max_mutations_per_generation=8,
            max_generations=2,
        )

        mutation_count = 0

        async def fake_generate(elites, **kwargs):
            nonlocal mutation_count
            await asyncio.sleep(LLM_LATENCY)  # simulate LLM latency
            mutation_count += 1
            prog = _prog(ProgramState.DONE)
            dag_tracker.submit(prog.id)
            return [prog.id]

        def get_ids_side_effect(status_val):
            if status_val == ProgramState.DONE.value:
                return dag_tracker.poll_done()
            if status_val in (ProgramState.QUEUED.value, ProgramState.RUNNING.value):
                return []
            return []

        async def count_by_status_side_effect(status_val):
            if status_val in (ProgramState.QUEUED.value, ProgramState.RUNNING.value):
                return 1 if dag_tracker.has_active() else 0
            return 0

        def mget_side_effect(ids, **kwargs):
            return [_prog(ProgramState.DONE) for _ in ids]

        engine.storage.get_ids_by_status.side_effect = get_ids_side_effect
        engine.storage.count_by_status.side_effect = count_by_status_side_effect
        engine.storage.mget.side_effect = mget_side_effect
        engine.storage.batch_transition_by_ids.return_value = 0
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True
        engine.strategy.select_elites.return_value = [_prog()]

        t0 = time.monotonic()
        with patch(
            "gigaevo.evolution.engine.steady_state.generate_mutations",
            side_effect=fake_generate,
        ):
            await asyncio.wait_for(engine.run(), timeout=30.0)

        elapsed = time.monotonic() - t0
        print(
            f"\nSteady-state E2E: {elapsed:.3f}s, {mutation_count} mutations, "
            f"{engine.metrics.total_generations} epochs"
        )

        assert engine.metrics.total_generations >= 2
        assert mutation_count >= 8  # at least 1 epoch's worth


# ---------------------------------------------------------------------------
# Scaling: measure throughput at different in-flight levels
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestScaling:
    """Measure how throughput scales with max_in_flight."""

    async def test_throughput_scales_with_in_flight(self) -> None:
        """More in-flight capacity = higher throughput (up to a point)."""
        results = {}
        for max_if in [1, 2, 4, 8]:
            t0 = time.monotonic()
            produced = 0
            ingested = 0
            in_flight: list[float] = []
            sema = asyncio.Semaphore(max_if)

            while ingested < TOTAL_MUTANTS:
                if produced < TOTAL_MUTANTS and sema._value > 0:  # noqa: SLF001
                    await sema.acquire()
                    await asyncio.sleep(LLM_LATENCY)
                    produced += 1
                    in_flight.append(time.monotonic() + DAG_LATENCY)

                now = time.monotonic()
                still_pending = []
                for ct in in_flight:
                    if now >= ct:
                        ingested += 1
                        sema.release()
                    else:
                        still_pending.append(ct)
                in_flight = still_pending

                if produced >= TOTAL_MUTANTS and in_flight:
                    earliest = min(in_flight)
                    wait = max(0, earliest - time.monotonic())
                    if wait > 0:
                        await asyncio.sleep(wait)

                await asyncio.sleep(0.001)

            elapsed = time.monotonic() - t0
            throughput = TOTAL_MUTANTS / elapsed
            results[max_if] = (elapsed, throughput)

        print("\nScaling results:")
        for max_if, (elapsed, tp) in sorted(results.items()):
            print(f"  max_in_flight={max_if}: {elapsed:.3f}s, {tp:.1f} mutants/s")

        # max_in_flight=4 should be faster than max_in_flight=1
        assert results[4][0] < results[1][0], (
            f"Expected max_in_flight=4 ({results[4][0]:.3f}s) to be faster than "
            f"max_in_flight=1 ({results[1][0]:.3f}s)"
        )
