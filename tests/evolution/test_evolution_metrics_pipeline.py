"""Integration test: evolution loop with a full metrics pipeline.

Combines the HalvingMutationOperator from test_resume_e2e.py with a
FakeDagRunner that runs EnsureMetrics + NormalizeMetrics, verifying that:

1. Fitness values from the metrics pipeline are correctly read by the archive.
2. Normalized metrics (score_norm) accumulate on programs in the archive.
3. The evolution makes progress (lower values → higher fitness over generations).
4. NormalizeMetrics operates correctly when metrics come from the pipeline
   (not hardcoded), i.e. the full EnsureMetrics → NormalizeMetrics chain works.

Setup
-----
- Mutation:    FloatHalvingOperator — return N → return N/2  (same as resume_e2e)
- Validation:  FakeMetricsDagRunner — runs EnsureMetrics then NormalizeMetrics
               fitness = (MAX - value) / MAX  so fitness ∈ (0, 1], higher = better
               x = halving depth (0, 1, 2, ...)  — each depth gets its own bin
- BehaviorSpace: x ∈ [0, 5), 5 bins, no eviction
- Fitness metric: "fitness" ∈ [0, 1]

Expected: after 5 generations the archive has 5 programs (values 1024, 512, 256, 128, 64),
each with computed fitness and score_norm metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.elite_selectors import ScalarTournamentEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.metrics import EnsureMetricsStage, NormalizeMetricsStage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED_VALUE = 1024.0
SEED_CODE = f"def entrypoint():\n    return {SEED_VALUE}"
_VALUE_RE = re.compile(r"return\s+([\d.]+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Metrics context: fitness in [0, 1], behavior dimension x (halving depth)
# ---------------------------------------------------------------------------


def _make_metrics_ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="normalized fitness (higher = better)",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
                sentinel_value=-1.0,
            ),
            "x": MetricSpec(
                description="halving depth",
                is_primary=False,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=5.0,
                sentinel_value=-1.0,
            ),
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_value(code: str) -> float:
    m = _VALUE_RE.search(code)
    if m is None:
        raise ValueError(f"Cannot extract return value from code:\n{code}")
    return float(m.group(1))


def _halved_code(value: float) -> str:
    return f"def entrypoint():\n    return {value / 2.0}"


def _compute_raw_metrics(code: str) -> dict[str, float]:
    """Compute fitness and halving depth from code."""
    value = _extract_value(code)
    depth = math.log2(SEED_VALUE / value) if value > 0 else 0.0
    # fitness: higher value → lower fitness (0 → 1 scale)
    fitness = (SEED_VALUE - value) / SEED_VALUE
    return {"fitness": fitness, "x": depth}


def _archive_values(programs: list[Program]) -> set[float]:
    return {_extract_value(p.code) for p in programs}


# ---------------------------------------------------------------------------
# Mutation operator: halves the return value
# ---------------------------------------------------------------------------


class FloatHalvingOperator(MutationOperator):
    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        parent = selected_parents[0]
        value = _extract_value(parent.code)
        return MutationSpec(
            code=_halved_code(value),
            parents=selected_parents,
            name="halving",
        )


# ---------------------------------------------------------------------------
# Fake DAG runner that runs EnsureMetrics + NormalizeMetrics
# ---------------------------------------------------------------------------


class FakeMetricsDagRunner:
    """Evaluates QUEUED programs by running the full metrics pipeline.

    This simulates a real DAG runner that:
    1. Computes raw metrics (like a validate.py call)
    2. Runs EnsureMetricsStage to validate/clamp them
    3. Runs NormalizeMetricsStage to normalize to [0,1]
    """

    def __init__(
        self,
        storage: RedisProgramStorage,
        state_manager: ProgramStateManager,
        metrics_ctx: MetricsContext,
    ):
        self._storage = storage
        self._sm = state_manager
        self._ctx = metrics_ctx
        self._task: asyncio.Task | None = None

        sentinels = metrics_ctx.get_sentinels()
        self._ensure_stage = EnsureMetricsStage(
            metrics_factory=sentinels,
            metrics_context=metrics_ctx,
            timeout=5.0,
        )
        self._ensure_stage.__class__.cache_handler = NO_CACHE

        self._normalize_stage = NormalizeMetricsStage(
            metrics_context=metrics_ctx,
            timeout=5.0,
        )
        self._normalize_stage.__class__.cache_handler = NO_CACHE

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="fake-metrics-dag-runner")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            queued = await self._storage.get_all_by_status(ProgramState.QUEUED.value)
            for prog in queued:
                await self._evaluate(prog)
            await asyncio.sleep(0.005)

    async def _evaluate(self, prog: Program) -> None:
        await self._sm.set_program_state(prog, ProgramState.RUNNING)

        # Step 1: compute raw metrics (simulates validate.py)
        raw_metrics = _compute_raw_metrics(prog.code)

        # Step 2: inject raw metrics as candidate via attach_inputs, then run EnsureMetrics
        self._ensure_stage.attach_inputs(
            {"candidate": FloatDictContainer(data=raw_metrics)}
        )
        await self._ensure_stage.compute(prog)

        # Step 3: run NormalizeMetrics (reads from prog.metrics populated by EnsureMetrics)
        # NormalizeMetrics has VoidInput — no inputs to attach
        await self._normalize_stage.compute(prog)

        # RUNNING → DONE
        await self._sm.set_program_state(prog, ProgramState.DONE)


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def _make_fakeredis_storage(server: fakeredis.FakeServer) -> RedisProgramStorage:
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix="test"
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_island_config(fitness_key: str = "fitness") -> IslandConfig:
    behavior_space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=5.0, num_bins=5, type="linear")}
    )
    return IslandConfig(
        island_id="test",
        behavior_space=behavior_space,
        max_size=None,
        archive_selector=SumArchiveSelector(fitness_keys=[fitness_key]),
        archive_remover=None,
        elite_selector=ScalarTournamentEliteSelector(
            fitness_key=fitness_key,
            fitness_key_higher_is_better=True,
            tournament_size=99,
        ),
        migrant_selector=RandomMigrantSelector(),
    )


def _make_null_writer() -> MagicMock:
    writer = MagicMock()
    writer.bind.return_value = writer
    return writer


def _build_engine(
    storage: RedisProgramStorage, max_generations: int
) -> EvolutionEngine:
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop():
        pass

    tracker.stop = _stop

    return EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=FloatHalvingOperator(),
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=1,
            max_mutations_per_generation=1,
            generation_timeout=30.0,
            max_generations=max_generations,
        ),
        writer=_make_null_writer(),
        metrics_tracker=tracker,
    )


async def _run_with_metrics(
    storage: RedisProgramStorage,
    metrics_ctx: MetricsContext,
    max_generations: int,
) -> EvolutionEngine:
    engine = _build_engine(storage, max_generations)
    state_manager = ProgramStateManager(storage)
    dag_runner = FakeMetricsDagRunner(storage, state_manager, metrics_ctx)

    dag_runner.start()
    engine.start()
    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except TimeoutError:
        pytest.fail(f"Engine did not finish {max_generations} gens within 30s")
    finally:
        await dag_runner.stop()
        await storage.close()

    return engine


async def _get_archive_programs(server: fakeredis.FakeServer) -> list[Program]:
    storage = _make_fakeredis_storage(server)
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    programs = await strategy.islands["test"].get_elites()
    await storage.close()
    return programs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvolutionWithMetricsPipeline:
    async def test_archive_has_normalized_metrics_after_evolution(self) -> None:
        """After 5 generations, all archive programs have fitness_norm from the metrics pipeline."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        metrics_ctx = _make_metrics_ctx()

        # Add seed program
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_with_metrics(storage, metrics_ctx, max_generations=5)
        assert engine.metrics.total_generations == 5

        programs = await _get_archive_programs(server)
        assert len(programs) == 5

        # Every archived program must have metrics from the pipeline
        for prog in programs:
            assert "fitness" in prog.metrics, (
                f"Program {prog.id[:8]} missing 'fitness' metric"
            )
            assert "fitness_norm" in prog.metrics, (
                f"Program {prog.id[:8]} missing 'fitness_norm' from NormalizeMetrics"
            )
            assert "x_norm" in prog.metrics, (
                f"Program {prog.id[:8]} missing 'x_norm' from NormalizeMetrics"
            )
            # fitness ∈ [0, 1] → fitness_norm must also be in [0, 1]
            assert 0.0 <= prog.metrics["fitness_norm"] <= 1.0, (
                f"fitness_norm={prog.metrics['fitness_norm']} out of [0,1]"
            )

    async def test_fitness_increases_with_halving_depth(self) -> None:
        """More-halved programs have strictly higher fitness values.

        fitness = (SEED - value) / SEED
        value 1024 → fitness 0.0
        value  512 → fitness 0.5
        value  256 → fitness 0.75
        ...
        """
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        metrics_ctx = _make_metrics_ctx()

        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        await _run_with_metrics(storage, metrics_ctx, max_generations=5)
        programs = await _get_archive_programs(server)

        # Sort by return value (decreasing) — larger value = less halved = lower fitness
        by_value = sorted(programs, key=lambda p: _extract_value(p.code), reverse=True)
        fitness_values = [p.metrics["fitness"] for p in by_value]

        # Fitness must be strictly increasing as value decreases (more halving = better)
        for i in range(len(fitness_values) - 1):
            assert fitness_values[i] < fitness_values[i + 1], (
                f"Fitness not monotonically increasing with halving depth: {fitness_values}"
            )

    async def test_archive_values_match_halving_trajectory(self) -> None:
        """The archive must contain exactly the expected set of return values."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        metrics_ctx = _make_metrics_ctx()

        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        await _run_with_metrics(storage, metrics_ctx, max_generations=5)
        programs = await _get_archive_programs(server)
        values = _archive_values(programs)

        assert values == {1024.0, 512.0, 256.0, 128.0, 64.0}, (
            f"Unexpected archive values: {sorted(values)}"
        )

    async def test_normalized_score_aggregate_present(self) -> None:
        """The 'normalized_score' aggregate metric is present on all archive programs."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        metrics_ctx = _make_metrics_ctx()

        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        await _run_with_metrics(storage, metrics_ctx, max_generations=3)
        programs = await _get_archive_programs(server)

        for prog in programs:
            assert "normalized_score" in prog.metrics, (
                f"Program {prog.id[:8]} missing 'normalized_score' aggregate metric"
            )
            assert 0.0 <= prog.metrics["normalized_score"] <= 1.0
