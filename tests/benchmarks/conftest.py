"""Shared fixtures for throughput benchmarks.

By default uses fakeredis (no server needed). Pass --redis-url to benchmark
against a real Redis instance.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock
import uuid

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_blueprint import DAGBlueprint
from gigaevo.runner.dag_runner import DagRunner, DagRunnerConfig
from tests.integration.test_mini_run import (
    FormatMockStage,
    IncrementMutationOperator,
    ValidateMockStage,
    _make_code,
    _make_island_config,
    _reset_counter,
)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# pytest CLI option
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--redis-url",
        default=None,
        help="Redis URL for benchmarks (e.g. redis://localhost:6379/15). "
        "If omitted, uses fakeredis.",
    )


@pytest.fixture(scope="session")
def redis_url(request):
    """Returns the --redis-url value or None (fakeredis)."""
    return request.config.getoption("--redis-url")


# ---------------------------------------------------------------------------
# Archive size fixture
# ---------------------------------------------------------------------------


@pytest.fixture(params=[500, 2000, 5000])
def archive_size(request):
    """Parameterized archive size for scaling benchmarks."""
    return request.param


# ---------------------------------------------------------------------------
# Heavy program factory
# ---------------------------------------------------------------------------

# ~50KB metadata string (realistic LLM mutation context)
_BIG_CONTEXT = "mutation context with lots of LLM reasoning " * 1200

# Nested lineage dict
_LINEAGE_SUMMARY = {
    f"gen_{i}": {"parent": f"prog_{i - 1}", "score": float(i) / 100} for i in range(50)
}

_STAGE_NAMES = [
    "validate",
    "execute",
    "complexity",
    "optuna_1",
    "optuna_2",
    "metrics_a",
    "metrics_b",
    "cache_check",
]


def make_heavy_program(fitness: float, x: float) -> Program:
    """Create a program with realistic heavy payload (~50KB metadata, 8 stage results).

    Uses the same code format as _make_code() so ValidateMockStage can extract metrics.
    """
    code = _make_code(fitness, x)
    p = Program(
        code=code,
        state=ProgramState.DONE,
        metrics={"fitness": fitness, "x": x},
        metadata={
            "mutation_context": _BIG_CONTEXT,
            "lineage_summary": _LINEAGE_SUMMARY,
            "extra": list(range(500)),
        },
    )
    for stage_name in _STAGE_NAMES:
        p.stage_results[stage_name] = ProgramStageResult(
            status=StageState.COMPLETED,
            output={"values": list(range(200)), "label": stage_name * 10},
        )
    return p


# ---------------------------------------------------------------------------
# Storage factory
# ---------------------------------------------------------------------------


def make_storage(
    server: fakeredis.FakeServer | None = None,
    redis_url: str | None = None,
) -> RedisProgramStorage:
    """Create RedisProgramStorage backed by fakeredis or real Redis.

    Pass server= for fakeredis, redis_url= for real Redis. Exactly one must be set.
    """
    prefix = f"bench_{uuid.uuid4().hex[:8]}"

    if redis_url is not None:
        config = RedisProgramStorageConfig(redis_url=redis_url, key_prefix=prefix)
        return RedisProgramStorage(config)

    if server is None:
        server = fakeredis.FakeServer()

    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix=prefix
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


async def cleanup_storage(storage: RedisProgramStorage) -> None:
    """Delete all keys with this storage's prefix, then close.

    Safe for both fakeredis and real Redis.
    """
    try:
        prefix = storage._keys._prefix

        async def _cleanup(r):
            keys = [k async for k in r.scan_iter(f"{prefix}:*", count=500)]
            if keys:
                await r.delete(*keys)

        await storage._conn.execute("cleanup", _cleanup)
    except Exception:
        pass
    try:
        await storage.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Archive population
# ---------------------------------------------------------------------------


async def populate_archive(
    storage: RedisProgramStorage,
    strategy: MapElitesMultiIsland,
    n: int,
    *,
    heavy: bool = True,
) -> list[Program]:
    """Pre-populate storage with n DONE programs and add to MAP-Elites archive.

    Args:
        heavy: Use production-weight programs (~50KB each) instead of toy ones.
    """
    programs = []
    for i in range(n):
        fitness = 1.0 + i * 0.01
        x = (i % 10) + 0.1 * (i // 10)

        if heavy:
            p = make_heavy_program(fitness, x)
        else:
            code = _make_code(fitness, x)
            p = Program(code=code, state=ProgramState.DONE)
            p.add_metrics({"fitness": fitness, "x": x})

        if programs:
            p.lineage.parents = [programs[-1].id]
        p.lineage.generation = (i // 5) + 1
        await storage.add(p)
        programs.append(p)
    # Add to MAP-Elites archive via island
    island = strategy.islands["main"]
    for p in programs:
        await island.add(p)
    return programs


# ---------------------------------------------------------------------------
# Blueprint factories
# ---------------------------------------------------------------------------


def make_benchmark_blueprint() -> DAGBlueprint:
    """2-stage pipeline: ValidateMock + FormatMock."""
    return DAGBlueprint(
        nodes={
            "validate": lambda: ValidateMockStage(timeout=10.0),
            "format": lambda: FormatMockStage(timeout=10.0),
        },
        data_flow_edges=[],
        max_parallel_stages=4,
        dag_timeout=30.0,
    )


def make_metrics_context() -> MetricsContext:
    """Simple metrics context for benchmarks."""
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Fitness score",
                is_primary=True,
                higher_is_better=True,
            ),
        }
    )


# ---------------------------------------------------------------------------
# Null mocks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Full system builder
# ---------------------------------------------------------------------------


def build_system(
    server: fakeredis.FakeServer | None,
    blueprint: DAGBlueprint,
    max_generations: int,
    max_mutations: int = 3,
    max_concurrent_dags: int = 4,
    redis_url: str | None = None,
) -> tuple[RedisProgramStorage, DagRunner, EvolutionEngine, MapElitesMultiIsland]:
    """Wire storage + strategy + DagRunner + EvolutionEngine."""
    _reset_counter()
    storage = make_storage(server=server, redis_url=redis_url)
    config = _make_island_config()
    strategy = MapElitesMultiIsland(
        island_configs=[config],
        program_storage=storage,
    )
    writer = _make_null_writer()

    dag_runner = DagRunner(
        storage=storage,
        dag_blueprint=blueprint,
        config=DagRunnerConfig(
            poll_interval=0.01,
            max_concurrent_dags=max_concurrent_dags,
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
            generation_timeout=60.0,
            max_generations=max_generations,
        ),
        writer=writer,
        metrics_tracker=_make_metrics_tracker(),
    )

    return storage, dag_runner, engine, strategy


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class BenchmarkTimer:
    """Simple context-manager timer for benchmark measurements."""

    def __init__(self):
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
