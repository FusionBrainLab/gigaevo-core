"""Benchmark: Full engine.step() wall time — the headline metric.

Measures end-to-end generation time with pre-populated archive of
production-weight programs (~50KB each). This is the number to optimize:
tests pass + numbers go down = the system got faster.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import fakeredis
import pytest

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.collector import EvolutionaryStatisticsCollector
from gigaevo.runner.dag_blueprint import DAGBlueprint
from gigaevo.runner.dag_runner import DagRunner, DagRunnerConfig
from tests.benchmarks.conftest import (
    cleanup_storage,
    make_metrics_context,
    make_storage,
    populate_archive,
)
from tests.integration.test_mini_run import (
    SEED_CODE,
    FormatMockStage,
    IncrementMutationOperator,
    ValidateMockStage,
    _make_island_config,
    _reset_counter,
)

pytestmark = pytest.mark.benchmark


def _make_null_writer() -> MagicMock:
    writer = MagicMock()
    writer.bind.return_value = writer
    return writer


def _make_metrics_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop():
        pass

    tracker.stop = _stop
    return tracker


async def _run_engine(
    archive_size,
    max_generations=1,
    max_mutations=3,
    include_collector=False,
    redis_url: str | None = None,
):
    """Build system, pre-populate with heavy programs, run engine."""
    _reset_counter()
    server = None if redis_url else fakeredis.FakeServer()
    storage = make_storage(server=server, redis_url=redis_url)
    config = _make_island_config()
    strategy = MapElitesMultiIsland(
        island_configs=[config],
        program_storage=storage,
    )
    writer = _make_null_writer()

    # Build blueprint — optionally with real collector using this storage
    if include_collector:
        metrics_ctx = make_metrics_context()
        blueprint = DAGBlueprint(
            nodes={
                "validate": lambda: ValidateMockStage(timeout=10.0),
                "format": lambda: FormatMockStage(timeout=10.0),
                "collector": lambda: EvolutionaryStatisticsCollector(
                    storage=storage,
                    metrics_context=metrics_ctx,
                    timeout=60.0,
                ),
            },
            data_flow_edges=[],
            max_parallel_stages=4,
            dag_timeout=60.0,
        )
    else:
        blueprint = DAGBlueprint(
            nodes={
                "validate": lambda: ValidateMockStage(timeout=10.0),
                "format": lambda: FormatMockStage(timeout=10.0),
            },
            data_flow_edges=[],
            max_parallel_stages=4,
            dag_timeout=30.0,
        )

    dag_runner = DagRunner(
        storage=storage,
        dag_blueprint=blueprint,
        config=DagRunnerConfig(
            poll_interval=0.01,
            max_concurrent_dags=4,
            dag_timeout=60.0,
        ),
        writer=writer,
    )

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=IncrementMutationOperator(),
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=1,
            max_mutations_per_generation=max_mutations,
            generation_timeout=120.0,
            max_generations=max_generations,
        ),
        writer=writer,
        metrics_tracker=_make_metrics_tracker(),
    )

    # Pre-populate archive with heavy programs
    await populate_archive(storage, strategy, archive_size, heavy=True)

    # Add a seed for the engine to process
    seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
    await storage.add(seed)

    dag_runner.start()
    engine.start()

    start = time.perf_counter()
    try:
        await asyncio.wait_for(engine.task, timeout=300.0)
    except TimeoutError:
        pytest.fail(
            f"Engine did not finish {max_generations} gen(s) within 300s "
            f"(archive_size={archive_size})"
        )
    elapsed_s = time.perf_counter() - start

    await dag_runner.stop()  # Also closes storage
    # For real Redis, clean up keys
    if redis_url:
        cleanup_storage_obj = make_storage(redis_url=redis_url)
        await cleanup_storage(cleanup_storage_obj)
    return engine, elapsed_s


class TestGenerationWallTime:
    """Single generation wall time with N heavy archive programs."""

    async def test_generation_wall_time(
        self, archive_size: int, redis_url: str | None
    ) -> None:
        engine, elapsed_s = await _run_engine(archive_size, redis_url=redis_url)

        assert engine.metrics.total_generations == 1
        progs_per_s = (archive_size + 3) / elapsed_s
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: generation_wall_time N={archive_size} ({backend}): "
            f"{elapsed_s:.3f}s ({progs_per_s:.1f} prog/s)"
        )


class TestGenerationWithCollector:
    """Generation with real EvolutionaryStatisticsCollector in pipeline."""

    async def test_generation_with_collector(
        self, archive_size: int, redis_url: str | None
    ) -> None:
        engine, elapsed_s = await _run_engine(
            archive_size, include_collector=True, redis_url=redis_url
        )

        assert engine.metrics.total_generations == 1
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: generation_with_collector N={archive_size} ({backend}): "
            f"{elapsed_s:.3f}s"
        )


class TestThreeGenerations:
    """3 generations to show per-gen cost stabilization."""

    async def test_3_generations(
        self, archive_size: int, redis_url: str | None
    ) -> None:
        engine, elapsed_s = await _run_engine(
            archive_size, max_generations=3, redis_url=redis_url
        )

        assert engine.metrics.total_generations == 3
        per_gen = elapsed_s / 3
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: 3_generations N={archive_size} ({backend}): "
            f"{elapsed_s:.3f}s total, {per_gen:.3f}s/gen"
        )
