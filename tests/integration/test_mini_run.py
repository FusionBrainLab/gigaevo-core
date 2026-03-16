"""Mini run.py — full integration test with real DagRunner, real EvolutionEngine,
real MapElitesMultiIsland, and fakeredis.  Every external dependency (LLM, exec
runners, network) is replaced by lightweight mock stages and a deterministic
mutation operator.

This is a "children's version" of run.py: it exercises the exact same code paths
(DagRunner scheduling, DAG execution, state transitions, engine generation loop,
archive ingestion, refresh cycles) but completes in <5 seconds with zero I/O.

Pipeline (2 stages):
    ValidateMock ──→ FormatMock
    (exec code)       (extract metrics → program.metrics)

Mutation operator:
    Deterministic — bumps fitness by 1.0, assigns unique x per call.

Archive:
    MAP-Elites with 10 linear bins on x ∈ [0, 10).
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.elite_selectors import ScalarTournamentEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.core_types import VoidInput, VoidOutput
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.runner.dag_blueprint import DAGBlueprint
from gigaevo.runner.dag_runner import DagRunner, DagRunnerConfig

# ---------------------------------------------------------------------------
# Mock stages
# ---------------------------------------------------------------------------

_RETURN_RE = re.compile(
    r'return\s*\{\s*"fitness":\s*([\d.]+)\s*,\s*"x":\s*([\d.]+)\s*\}',
    re.MULTILINE,
)


def _extract_metrics(code: str) -> dict[str, float]:
    m = _RETURN_RE.search(code)
    if m is None:
        raise ValueError(f"Cannot extract metrics from code:\n{code}")
    return {"fitness": float(m.group(1)), "x": float(m.group(2))}


def _make_code(fitness: float, x: float) -> str:
    return f'def entrypoint():\n    return {{"fitness": {fitness}, "x": {x}}}'


class ValidateMockStage(Stage):
    """Executes program code and extracts metrics → side-effect on program."""

    InputsModel = VoidInput
    OutputModel = VoidOutput

    async def compute(self, program: Program) -> None:
        metrics = _extract_metrics(program.code)
        program.add_metrics(metrics)


class FormatMockStage(Stage):
    """Dummy format stage (no-op, all work done in validate)."""

    InputsModel = VoidInput
    OutputModel = VoidOutput

    async def compute(self, program: Program) -> None:
        pass  # Metrics already populated by validate


# ---------------------------------------------------------------------------
# Deterministic mutation operator
# ---------------------------------------------------------------------------

_CALL_COUNTER = 0


def _reset_counter() -> None:
    global _CALL_COUNTER
    _CALL_COUNTER = 0


class IncrementMutationOperator(MutationOperator):
    """Bumps fitness by 1.0, assigns unique x per call."""

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        global _CALL_COUNTER
        parent = selected_parents[0]
        parent_metrics = _extract_metrics(parent.code)
        new_fitness = parent_metrics["fitness"] + 1.0
        new_x = 0.5 + _CALL_COUNTER
        _CALL_COUNTER += 1
        return MutationSpec(
            code=_make_code(new_fitness, new_x),
            parents=selected_parents,
            name="increment",
        )


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------

SEED_CODE = _make_code(fitness=1.0, x=0.0)


def _make_storage(server: fakeredis.FakeServer) -> RedisProgramStorage:
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix="minirun"
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_blueprint() -> DAGBlueprint:
    return DAGBlueprint(
        nodes={
            "validate": lambda: ValidateMockStage(timeout=10.0),
            "format": lambda: FormatMockStage(timeout=10.0),
        },
        data_flow_edges=[],  # No data flow; both stages are independent
        max_parallel_stages=4,
        dag_timeout=30.0,
    )


def _make_island_config(*, max_size: int | None = None) -> IslandConfig:
    from gigaevo.evolution.strategies.removers import FitnessArchiveRemover

    return IslandConfig(
        island_id="main",
        behavior_space=BehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=10, type="linear"
                )
            }
        ),
        max_size=max_size,
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=(
            FitnessArchiveRemover(
                fitness_key="fitness", fitness_key_higher_is_better=True
            )
            if max_size is not None
            else None
        ),
        elite_selector=ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=99,
        ),
        migrant_selector=RandomMigrantSelector(),
    )


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


def _build(
    server: fakeredis.FakeServer,
    max_generations: int,
    *,
    max_elites: int = 1,
    max_mutations: int = 1,
    mutation_operator: MutationOperator | None = None,
    island_config: IslandConfig | None = None,
) -> tuple[RedisProgramStorage, DagRunner, EvolutionEngine, MapElitesMultiIsland]:
    """Wire all components together — mirrors run.py's instantiate step."""
    storage = _make_storage(server)

    config = island_config or _make_island_config()
    strategy = MapElitesMultiIsland(
        island_configs=[config],
        program_storage=storage,
    )

    writer = _make_null_writer()

    dag_runner = DagRunner(
        storage=storage,
        dag_blueprint=_make_blueprint(),
        config=DagRunnerConfig(
            poll_interval=0.01,
            max_concurrent_dags=4,
            dag_timeout=30.0,
        ),
        writer=writer,
    )

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=mutation_operator or IncrementMutationOperator(),
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=max_elites,
            max_mutations_per_generation=max_mutations,
            generation_timeout=30.0,
            max_generations=max_generations,
        ),
        writer=writer,
        metrics_tracker=_make_metrics_tracker(),
    )

    return storage, dag_runner, engine, strategy


async def _run_mini(
    server: fakeredis.FakeServer,
    max_generations: int,
    *,
    max_elites: int = 1,
    max_mutations: int = 1,
    mutation_operator: MutationOperator | None = None,
    island_config: IslandConfig | None = None,
) -> tuple[EvolutionEngine, MapElitesMultiIsland, DagRunner]:
    """Seed, wire, run, and return the engine + strategy + runner."""
    storage, dag_runner, engine, strategy = _build(
        server,
        max_generations,
        max_elites=max_elites,
        max_mutations=max_mutations,
        mutation_operator=mutation_operator,
        island_config=island_config,
    )

    # Load seed
    seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
    await storage.add(seed)

    # Start (same order as run.py)
    dag_runner.start()
    engine.start()

    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Engine did not finish {max_generations} gens within 30s")
    finally:
        await dag_runner.stop()
        await storage.close()

    return engine, strategy, dag_runner


async def _get_archive(server: fakeredis.FakeServer) -> list[Program]:
    storage = _make_storage(server)
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    programs = await strategy.islands["main"].get_elites()
    await storage.close()
    return programs


async def _get_all_programs(server: fakeredis.FakeServer) -> list[Program]:
    storage = _make_storage(server)
    done = await storage.get_all_by_status(ProgramState.DONE.value)
    queued = await storage.get_all_by_status(ProgramState.QUEUED.value)
    running = await storage.get_all_by_status(ProgramState.RUNNING.value)
    discarded = await storage.get_all_by_status(ProgramState.DISCARDED.value)
    await storage.close()
    return done + queued + running + discarded


# ---------------------------------------------------------------------------
# Tests — Full mini-run.py integration
# ---------------------------------------------------------------------------


class TestMiniRunBasic:
    """Core sanity: engine completes, archive grows, metrics populated."""

    async def test_5gen_completes(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        engine, strategy, runner = await _run_mini(server, max_generations=5)

        assert engine.metrics.total_generations == 5

    async def test_archive_grows(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=5)

        programs = await _get_archive(server)
        assert len(programs) >= 3, (
            f"Expected >=3 archive entries after 5 gens, got {len(programs)}"
        )

    async def test_all_archive_programs_have_metrics(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=4)

        programs = await _get_archive(server)
        for p in programs:
            assert "fitness" in p.metrics, f"{p.id[:8]} missing 'fitness'"
            assert "x" in p.metrics, f"{p.id[:8]} missing 'x'"

    async def test_best_fitness_exceeds_seed(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=5)

        programs = await _get_archive(server)
        best = max(p.metrics["fitness"] for p in programs)
        assert best > 1.0, f"Best fitness {best} should exceed seed fitness 1.0"


class TestMiniRunDagRunner:
    """Verify the real DagRunner processed programs correctly."""

    async def test_dag_runner_started_dags(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        _, _, runner = await _run_mini(server, max_generations=3)

        assert runner._metrics.dag_runs_started > 0

    async def test_dag_runner_completed_dags(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        _, _, runner = await _run_mini(server, max_generations=3)

        assert runner._metrics.dag_runs_completed > 0

    async def test_dag_runner_no_errors(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        _, _, runner = await _run_mini(server, max_generations=3)

        assert runner._metrics.dag_errors == 0, (
            f"DagRunner had {runner._metrics.dag_errors} errors "
            f"(timeouts={runner._metrics.dag_timeouts}, "
            f"builds={runner._metrics.dag_build_failures})"
        )

    async def test_dag_runner_no_orphans(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        _, _, runner = await _run_mini(server, max_generations=3)

        assert runner._metrics.orphaned_programs_discarded == 0


class TestMiniRunStateHygiene:
    """No programs stuck in transient states after run."""

    async def test_no_queued_after_run(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=3)

        storage = _make_storage(server)
        queued = await storage.get_all_by_status(ProgramState.QUEUED.value)
        await storage.close()
        assert len(queued) == 0, f"{len(queued)} programs stuck QUEUED"

    async def test_no_running_after_run(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=3)

        storage = _make_storage(server)
        running = await storage.get_all_by_status(ProgramState.RUNNING.value)
        await storage.close()
        assert len(running) == 0, f"{len(running)} programs stuck RUNNING"

    async def test_all_archive_programs_done(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=3)

        programs = await _get_archive(server)
        for p in programs:
            assert p.state == ProgramState.DONE, (
                f"Archive program {p.id[:8]} in {p.state}, expected DONE"
            )


class TestMiniRunLineage:
    """Lineage chains are correctly wired through the real pipeline."""

    async def test_mutants_have_parents(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=4)

        seed_code = SEED_CODE
        all_progs = await _get_all_programs(server)

        non_seed = [p for p in all_progs if p.code != seed_code]
        for p in non_seed:
            assert p.lineage.parents, f"{p.id[:8]} has empty parents"

    async def test_mutation_name_recorded(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=3)

        all_progs = await _get_all_programs(server)
        non_seed = [p for p in all_progs if p.code != SEED_CODE]
        for p in non_seed:
            assert p.lineage.mutation == "increment", (
                f"{p.id[:8]} has mutation '{p.lineage.mutation}'"
            )

    async def test_generation_depth_grows(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=5)

        all_progs = await _get_all_programs(server)
        non_seed = [p for p in all_progs if p.code != SEED_CODE]
        max_gen = max((p.lineage.generation for p in non_seed), default=0)
        assert max_gen > 0, "No program has lineage.generation > 0"


class TestMiniRunEngineMetrics:
    """Engine metrics track correctly across generations."""

    async def test_generation_counter(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        engine, _, _ = await _run_mini(server, max_generations=4)
        assert engine.metrics.total_generations == 4

    async def test_mutations_created(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        engine, _, _ = await _run_mini(server, max_generations=3)
        assert engine.metrics.mutations_created >= 2

    async def test_programs_processed(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        engine, _, _ = await _run_mini(server, max_generations=3)
        assert engine.metrics.programs_processed >= 1

    async def test_elites_selected(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        engine, _, _ = await _run_mini(server, max_generations=4)
        assert engine.metrics.elites_selected >= 2

    async def test_refresh_submitted(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        engine, _, _ = await _run_mini(server, max_generations=3)
        assert engine.metrics.submitted_for_refresh >= 1


class TestMiniRunMultipleMutations:
    """Multiple mutations per generation with real DagRunner."""

    async def test_multi_mutants_per_gen(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        engine, _, _ = await _run_mini(server, max_generations=3, max_mutations=3)

        assert engine.metrics.mutations_created >= 3
        programs = await _get_archive(server)
        assert len(programs) >= 3

    async def test_multi_mutants_all_have_lineage(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=3, max_mutations=3)

        all_progs = await _get_all_programs(server)
        non_seed = [p for p in all_progs if p.code != SEED_CODE]
        for p in non_seed:
            assert p.lineage.parents, f"{p.id[:8]} missing parents"
            assert p.lineage.mutation == "increment"


class TestMiniRunStageResults:
    """Verify stage_results are populated by the real DAG execution."""

    async def test_archive_programs_have_stage_results(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=3)

        programs = await _get_archive(server)
        for p in programs:
            assert "validate" in p.stage_results, (
                f"{p.id[:8]} missing 'validate' stage result"
            )
            assert "format" in p.stage_results, (
                f"{p.id[:8]} missing 'format' stage result"
            )

    async def test_stage_results_show_completed(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=3)

        from gigaevo.programs.core_types import StageState

        programs = await _get_archive(server)
        for p in programs:
            for stage_name in ("validate", "format"):
                result = p.stage_results[stage_name]
                assert result.status == StageState.COMPLETED, (
                    f"{p.id[:8]}/{stage_name} status={result.status}, expected COMPLETED"
                )


class TestMiniRunResume:
    """Verify run-state persistence allows resume (same as run.py --resume)."""

    async def test_persisted_generation_counter(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        await _run_mini(server, max_generations=3)

        # Build fresh engine on same Redis, restore state
        storage, _, engine2, strategy2 = _build(server, max_generations=10)
        await engine2.restore_state()
        await strategy2.restore_state()

        assert engine2.metrics.total_generations == 3
        assert strategy2.generation > 0
        await storage.close()


class TestMiniRunStrategyGeneration:
    """Strategy generation counter advances correctly."""

    async def test_strategy_generation_advances(self) -> None:
        _reset_counter()
        server = fakeredis.FakeServer()
        _, strategy, _ = await _run_mini(server, max_generations=4)

        assert strategy.generation >= 2, (
            f"Expected strategy.generation >= 2, got {strategy.generation}"
        )
