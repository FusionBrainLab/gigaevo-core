"""Benchmark: EvolutionaryStatisticsCollector scaling at O(N) programs.

Isolates the get_all() + compute overhead that runs once per DAG in the
refresh phase. Uses production-weight programs (~50KB each) to expose
serialization costs.
"""

from __future__ import annotations

import fakeredis
import pytest

from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.programs.stages.collector import EvolutionaryStatisticsCollector
from tests.benchmarks.conftest import (
    BenchmarkTimer,
    cleanup_storage,
    make_metrics_context,
    make_storage,
    populate_archive,
)
from tests.integration.test_mini_run import _make_island_config

pytestmark = pytest.mark.benchmark


async def _setup(n: int, redis_url: str | None = None):
    """Create storage + strategy + N heavy programs."""
    server = None if redis_url else fakeredis.FakeServer()
    storage = make_storage(server=server, redis_url=redis_url)
    config = _make_island_config()
    strategy = MapElitesMultiIsland(
        island_configs=[config],
        program_storage=storage,
    )
    programs = await populate_archive(storage, strategy, n, heavy=True)
    return storage, strategy, programs


class TestGetAllScaling:
    """Time storage.get_all() for N heavy programs."""

    async def test_get_all_scaling(
        self, archive_size: int, redis_url: str | None
    ) -> None:
        storage, strategy, programs = await _setup(archive_size, redis_url)

        with BenchmarkTimer() as t:
            result = await storage.get_all()

        assert len(result) == archive_size
        backend = "redis" if redis_url else "fakeredis"
        print(f"BENCHMARK: get_all N={archive_size} ({backend}): {t.elapsed_ms:.1f}ms")
        await cleanup_storage(storage)


class TestCollectorComputeScaling:
    """Time full collector.compute() (get_all + stats) with heavy programs."""

    async def test_collector_compute_scaling(
        self, archive_size: int, redis_url: str | None
    ) -> None:
        storage, strategy, programs = await _setup(archive_size, redis_url)
        metrics_ctx = make_metrics_context()
        collector = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=metrics_ctx,
            timeout=60.0,
        )
        collector.attach_inputs({})
        target = programs[0]

        with BenchmarkTimer() as t:
            result = await collector.compute(target)

        assert result.total_program_count == archive_size
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: collector.compute N={archive_size} ({backend}): "
            f"{t.elapsed_ms:.1f}ms"
        )
        await cleanup_storage(storage)

    async def test_collector_repeated_calls(
        self, archive_size: int, redis_url: str | None
    ) -> None:
        """Simulate refresh phase: K=20 sequential compute() calls."""
        storage, strategy, programs = await _setup(archive_size, redis_url)
        metrics_ctx = make_metrics_context()
        collector = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=metrics_ctx,
            timeout=60.0,
        )
        collector.attach_inputs({})
        target = programs[0]
        k = 20

        with BenchmarkTimer() as t:
            for _ in range(k):
                result = await collector.compute(target)

        assert result.total_program_count == archive_size
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: collector x{k} (refresh sim) N={archive_size} ({backend}): "
            f"{t.elapsed_ms:.0f}ms"
        )
        await cleanup_storage(storage)
