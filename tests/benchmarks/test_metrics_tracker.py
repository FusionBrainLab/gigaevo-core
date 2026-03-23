"""Benchmark: MetricsTracker frontier recomputation.

The MetricsTracker maintains cumulative frontier series for every metric.
When NO_CACHE stages re-evaluate programs, the full frontier must be
recomputed from all valid programs. This is the dominant cost in the
tracker's hot path and scales with archive size.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import fakeredis
import pytest

from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.metrics_tracker import MetricsTracker
from tests.benchmarks.conftest import (
    BenchmarkTimer,
    make_metrics_context,
    make_storage,
)

pytestmark = pytest.mark.benchmark


def _make_counting_writer() -> MagicMock:
    """Writer that counts calls but does minimal work."""
    writer = MagicMock()
    writer.bind.return_value = writer
    return writer


def _build_tracker(
    writer: MagicMock | None = None,
) -> MetricsTracker:
    """Create a MetricsTracker with in-memory storage (never polled)."""
    server = fakeredis.FakeServer()
    storage = make_storage(server=server)
    if writer is None:
        writer = _make_counting_writer()
    return MetricsTracker(
        storage=storage,
        metrics_context=make_metrics_context(),
        writer=writer,
        interval=999.0,  # never auto-polls
    )


def _populate_tracker(tracker: MetricsTracker, n: int) -> None:
    """Directly populate _valid_programs with n entries (bypass storage)."""
    for i in range(n):
        pid = f"prog_{i:06d}"
        iteration = i
        fitness = 0.5 + (i / n) * 0.5  # 0.5 → 1.0
        tracker._valid_programs[pid] = (iteration, {"fitness": fitness})
        tracker._best_valid["fitness"] = (fitness, iteration)
        tracker._seen_ids.add(pid)
        tracker._seen_fitness[pid] = (("fitness", fitness),)
        tracker._valid_count += 1


class TestFrontierRecomputation:
    """Time _recompute_and_write_frontier() at various archive sizes."""

    @pytest.fixture(params=[500, 2000, 5000])
    def tracker_size(self, request):
        return request.param

    def test_recompute_frontier(self, tracker_size: int) -> None:
        tracker = _build_tracker()
        _populate_tracker(tracker, tracker_size)

        with BenchmarkTimer() as t:
            tracker._recompute_and_write_frontier("fitness")

        print(f"BENCHMARK: frontier_recompute N={tracker_size}: {t.elapsed_ms:.2f}ms")


class TestIncrementalUpdate:
    """Time _process_program() for a single new program (common case)."""

    @pytest.fixture(params=[500, 2000])
    def bg_size(self, request):
        return request.param

    async def test_incremental_process(self, bg_size: int) -> None:
        tracker = _build_tracker()
        _populate_tracker(tracker, bg_size)

        # Create a new program to process
        new_prog = Program(
            code="def run_code(): return 99",
            state=ProgramState.DONE,
            metrics={"fitness": 0.99, "is_valid": 1.0},
            metadata={"iteration": bg_size + 1},
        )
        new_prog.lineage.generation = bg_size // 5

        with BenchmarkTimer() as t:
            for _ in range(100):
                # Reset so it processes again
                tracker._seen_ids.discard(new_prog.id)
                tracker._valid_programs.pop(new_prog.id, None)
                await tracker._process_program(new_prog)

        avg_ms = t.elapsed_ms / 100
        print(
            f"BENCHMARK: incremental_process bg={bg_size}: "
            f"{avg_ms:.3f}ms/program ({t.elapsed_ms:.1f}ms for 100)"
        )


class TestRefreshChangedFitness:
    """Time _refresh_changed_fitness() when K programs have changed metrics.

    This simulates NO_CACHE stages updating metrics after archive refresh.
    """

    async def test_refresh_with_changes(self) -> None:
        n = 500
        server = fakeredis.FakeServer()
        storage = make_storage(server=server)
        writer = _make_counting_writer()
        tracker = MetricsTracker(
            storage=storage,
            metrics_context=make_metrics_context(),
            writer=writer,
            interval=999.0,
        )

        # Add programs to storage and tracker
        for i in range(n):
            p = Program(
                code=f"def run_code(): return {i}",
                state=ProgramState.DONE,
                metrics={"fitness": 0.5 + i * 0.001, "is_valid": 1.0},
                metadata={"iteration": i},
            )
            p.lineage.generation = i // 5
            await storage.add(p)
            tracker._seen_ids.add(p.id)
            tracker._seen_fitness[p.id] = (("fitness", 0.5 + i * 0.001),)
            tracker._valid_programs[p.id] = (i, {"fitness": 0.5 + i * 0.001})
            tracker._valid_count += 1

            # Simulate 10% of programs having changed metrics in storage
            if i % 10 == 0:
                p.metrics["fitness"] = 0.5 + i * 0.001 + 0.1
                await storage.update(p)

        with BenchmarkTimer() as t:
            await tracker._refresh_changed_fitness()

        print(
            f"BENCHMARK: refresh_changed_fitness N={n} (10% changed): "
            f"{t.elapsed_ms:.1f}ms"
        )
