"""Benchmark: MAP-Elites strategy operations at scale.

These operations run every generation: select_elites picks parents,
island.add() ingests new programs, and reindex_archive() rebuilds
the archive after refresh. All scale with archive size.
"""

from __future__ import annotations

import fakeredis
import pytest

from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.benchmarks.conftest import (
    BenchmarkTimer,
    cleanup_storage,
    make_storage,
    populate_archive,
)
from tests.integration.test_mini_run import _make_code, _make_island_config

pytestmark = pytest.mark.benchmark


async def _setup(n: int, redis_url: str | None = None):
    """Create storage + strategy + N programs in archive."""
    server = None if redis_url else fakeredis.FakeServer()
    storage = make_storage(server=server, redis_url=redis_url)
    config = _make_island_config()
    strategy = MapElitesMultiIsland(
        island_configs=[config],
        program_storage=storage,
    )
    programs = await populate_archive(storage, strategy, n, heavy=False)
    return storage, strategy, programs


class TestSelectElites:
    """Time select_elites() with various archive sizes."""

    async def test_select_elites(
        self, archive_size: int, redis_url: str | None
    ) -> None:
        storage, strategy, programs = await _setup(archive_size, redis_url)

        k = 50
        with BenchmarkTimer() as t:
            for _ in range(k):
                await strategy.select_elites(total=3)

        avg_ms = t.elapsed_ms / k
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: select_elites(3) N={archive_size} ({backend}): "
            f"{avg_ms:.2f}ms/call ({t.elapsed_ms:.0f}ms for {k} calls)"
        )
        await cleanup_storage(storage)


class TestIslandAdd:
    """Time island.add() into a full archive."""

    async def test_island_add(self, archive_size: int, redis_url: str | None) -> None:
        storage, strategy, programs = await _setup(archive_size, redis_url)
        island = strategy.islands["main"]

        # Create new programs to add (some will replace, some won't)
        new_programs = []
        for i in range(50):
            fitness = 0.5 + i * 0.1  # some better, some worse than archive
            x = float(i % 10)
            code = _make_code(fitness, x)
            p = Program(code=code, state=ProgramState.DONE)
            p.add_metrics({"fitness": fitness, "x": x})
            await storage.add(p)
            new_programs.append(p)

        with BenchmarkTimer() as t:
            for p in new_programs:
                await island.add(p)

        avg_ms = t.elapsed_ms / len(new_programs)
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: island.add N={archive_size} ({backend}): "
            f"{avg_ms:.2f}ms/add ({t.elapsed_ms:.1f}ms for {len(new_programs)} adds)"
        )
        await cleanup_storage(storage)


class TestReindexArchive:
    """Time reindex_archive() — runs every refresh cycle."""

    async def test_reindex_archive(
        self, archive_size: int, redis_url: str | None
    ) -> None:
        storage, strategy, programs = await _setup(archive_size, redis_url)
        island = strategy.islands["main"]

        with BenchmarkTimer() as t:
            await island.reindex_archive()

        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: reindex_archive N={archive_size} ({backend}): "
            f"{t.elapsed_ms:.1f}ms"
        )
        await cleanup_storage(storage)
