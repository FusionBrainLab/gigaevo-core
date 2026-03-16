"""Integration tests for complex scenarios not covered elsewhere.

Test groups:
  1. Diamond DAG with partial middle-node failure
  2. Evolution engine survives when all mutations fail
  3. Evolution engine survives when mutation operator raises for every elite
  4. Stale cache correctness after mid-chain failure on rerun
"""

from __future__ import annotations

import asyncio
import contextlib
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
from gigaevo.programs.core_types import StageIO, StageState, VoidInput
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from tests.conftest import NullWriter

# ===========================================================================
# Part 1: Diamond DAG with partial middle-node failure
# ===========================================================================

# -- Stage I/O types --


class IntOutput(StageIO):
    value: int = 0


class IntInput(StageIO):
    data: IntOutput


class OptIntInput(StageIO):
    data: IntOutput | None = None


class DualMandatoryInput(StageIO):
    left: IntOutput
    right: IntOutput


class DualOptInput(StageIO):
    left: IntOutput | None = None
    right: IntOutput | None = None


# -- Stage classes --


class ProduceN(Stage):
    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    def __init__(self, *, timeout: float = 5.0, value: int = 1):
        super().__init__(timeout=timeout)
        self._value = value

    async def compute(self, program: Program) -> IntOutput:
        return IntOutput(value=self._value)


class FailStage(Stage):
    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        raise RuntimeError("intentional failure")


class IncrStage(Stage):
    InputsModel = IntInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        return IntOutput(value=self.params.data.value + 1)


class OptIncrStage(Stage):
    InputsModel = OptIntInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        if self.params.data is not None:
            return IntOutput(value=self.params.data.value + 1)
        return IntOutput(value=-1)


class SumDualOptStage(Stage):
    InputsModel = DualOptInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        left_val = self.params.left.value if self.params.left is not None else 0
        right_val = self.params.right.value if self.params.right is not None else 0
        return IntOutput(value=left_val + right_val)


class SumDualMandatoryStage(Stage):
    InputsModel = DualMandatoryInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        return IntOutput(value=self.params.left.value + self.params.right.value)


def _make_dag(nodes, edges, state_manager, *, exec_deps=None, **kwargs):
    return DAG(
        nodes=nodes,
        data_flow_edges=edges,
        execution_order_deps=exec_deps,
        state_manager=state_manager,
        writer=NullWriter(),
        **kwargs,
    )


class TestDiamondPartialFailure:
    """Diamond DAG A→{B,C}→D where one middle node fails.

    Topology:
        A (source) ──┬──→ B (fails)  ──┬──→ D (join)
                      └──→ C (succeeds) ┘

    Tests vary D's input type (mandatory vs optional) to cover:
    - Mandatory inputs from both B and C → D skipped when B fails
    - Optional inputs from both B and C → D runs with B=None
    - Mandatory from C, optional from B → D runs with B=None
    """

    async def test_diamond_B_fails_D_mandatory_both_skipped(
        self, state_manager, make_program
    ):
        """A→{B,C}→D with D taking mandatory data from both B and C.

        B fails → D cannot get its mandatory 'left' input → D is SKIPPED.
        C still runs and completes.
        """
        dag = _make_dag(
            {
                "a": ProduceN(value=10),
                "b": FailStage(timeout=5.0),
                "c": IncrStage(timeout=5.0),
                "d": SumDualMandatoryStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "c", "data"),
                DataFlowEdge.create("b", "d", "left"),  # mandatory, B fails
                DataFlowEdge.create("c", "d", "right"),  # mandatory, C succeeds
            ],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        # D skipped because mandatory 'left' from B is unavailable
        assert prog.stage_results["d"].status == StageState.SKIPPED

    async def test_diamond_B_fails_D_optional_both_runs(
        self, state_manager, make_program
    ):
        """A→{B,C}→D with D taking optional data from both B and C.

        B fails → D still runs with left=None, right=C's output.
        """
        dag = _make_dag(
            {
                "a": ProduceN(value=10),
                "b": FailStage(timeout=5.0),
                "c": IncrStage(timeout=5.0),
                "d": SumDualOptStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "c", "data"),
                DataFlowEdge.create("b", "d", "left"),  # optional, B fails → None
                DataFlowEdge.create("c", "d", "right"),  # optional, C succeeds
            ],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        assert prog.stage_results["d"].status == StageState.COMPLETED
        # left=0 (B failed, None → 0), right=11 (C: 10+1)
        assert prog.stage_results["d"].output.value == 11

    async def test_diamond_C_fails_B_succeeds_D_optional(
        self, state_manager, make_program
    ):
        """Mirror case: C fails, B succeeds, D optional from both.

        D runs with left=B's value, right=None.
        """
        dag = _make_dag(
            {
                "a": ProduceN(value=5),
                "b": IncrStage(timeout=5.0),
                "c": FailStage(timeout=5.0),
                "d": SumDualOptStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "d", "left"),  # optional, B succeeds
                DataFlowEdge.create("c", "d", "right"),  # optional, C fails
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.FAILED
        assert prog.stage_results["d"].status == StageState.COMPLETED
        # left=B(6), right=None(C failed) → 6+0=6
        assert prog.stage_results["d"].output.value == 6

    async def test_diamond_both_middle_fail_D_optional(
        self, state_manager, make_program
    ):
        """Both B and C fail → D runs with both inputs as None (both optional)."""
        dag = _make_dag(
            {
                "a": ProduceN(value=1),
                "b": FailStage(timeout=5.0),
                "c": FailStage(timeout=5.0),
                "d": SumDualOptStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("b", "d", "left"),
                DataFlowEdge.create("c", "d", "right"),
            ],
            state_manager,
            exec_deps={
                "b": [ExecutionOrderDependency.on_success("a")],
                "c": [ExecutionOrderDependency.on_success("a")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.FAILED
        assert prog.stage_results["d"].status == StageState.COMPLETED
        # Both None → 0+0=0
        assert prog.stage_results["d"].output.value == 0

    async def test_diamond_both_middle_fail_D_mandatory_skipped(
        self, state_manager, make_program
    ):
        """Both B and C fail → D with mandatory inputs is SKIPPED."""
        dag = _make_dag(
            {
                "a": ProduceN(value=1),
                "b": FailStage(timeout=5.0),
                "c": FailStage(timeout=5.0),
                "d": SumDualMandatoryStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("b", "d", "left"),
                DataFlowEdge.create("c", "d", "right"),
            ],
            state_manager,
            exec_deps={
                "b": [ExecutionOrderDependency.on_success("a")],
                "c": [ExecutionOrderDependency.on_success("a")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.FAILED
        assert prog.stage_results["d"].status == StageState.SKIPPED

    async def test_diamond_all_stages_reach_final_state(
        self, state_manager, make_program
    ):
        """Every stage in a diamond with failure reaches a final state (no PENDING/RUNNING)."""
        from gigaevo.programs.core_types import FINAL_STATES

        dag = _make_dag(
            {
                "a": ProduceN(value=1),
                "b": FailStage(timeout=5.0),
                "c": IncrStage(timeout=5.0),
                "d": SumDualOptStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "c", "data"),
                DataFlowEdge.create("b", "d", "left"),
                DataFlowEdge.create("c", "d", "right"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        for name, result in prog.stage_results.items():
            assert result.status in FINAL_STATES, (
                f"Stage '{name}' stuck in non-final state {result.status}"
            )


# ===========================================================================
# Part 2: Evolution engine survives all-mutations-fail scenarios
# ===========================================================================

# -- Helpers --

_VALUE_RE = re.compile(r'return\s*\{\s*"fitness":\s*([\d.]+)', re.MULTILINE)

SEED_CODE = 'def entrypoint():\n    return {"fitness": 1.0, "x": 0.0}'


def _make_code(fitness: float, x: float) -> str:
    return f'def entrypoint():\n    return {{"fitness": {fitness}, "x": {x}}}'


def _extract_metrics(code: str) -> dict[str, float]:
    m = re.search(
        r'return\s*\{\s*"fitness":\s*([\d.]+)\s*,\s*"x":\s*([\d.]+)\s*\}',
        code,
        re.MULTILINE,
    )
    if m is None:
        raise ValueError(f"Cannot extract metrics from code:\n{code}")
    return {"fitness": float(m.group(1)), "x": float(m.group(2))}


class FakeDagRunner:
    def __init__(self, storage, state_manager):
        self._storage = storage
        self._sm = state_manager
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._loop(), name="fake-dag-runner")

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self):
        while True:
            queued = await self._storage.get_all_by_status(ProgramState.QUEUED.value)
            for prog in queued:
                await self._evaluate(prog)
            await asyncio.sleep(0.005)

    async def _evaluate(self, prog):
        await self._sm.set_program_state(prog, ProgramState.RUNNING)
        metrics = _extract_metrics(prog.code)
        prog.add_metrics(metrics)
        await self._sm.set_program_state(prog, ProgramState.DONE)


def _make_fakeredis_storage(server):
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix="test"
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_island_config():
    behavior_space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=10, type="linear")}
    )
    return IslandConfig(
        island_id="test",
        behavior_space=behavior_space,
        max_size=None,
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=None,
        elite_selector=ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=99,
        ),
        migrant_selector=RandomMigrantSelector(),
    )


def _make_null_writer():
    writer = MagicMock()
    writer.bind.return_value = writer
    return writer


def _make_metrics_tracker():
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop():
        pass

    tracker.stop = _stop
    return tracker


def _build_engine(storage, max_generations, *, mutation_operator):
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    return EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=mutation_operator,
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=1,
            max_mutations_per_generation=1,
            generation_timeout=30.0,
            max_generations=max_generations,
        ),
        writer=_make_null_writer(),
        metrics_tracker=_make_metrics_tracker(),
    )


async def _run_engine(storage, max_generations, *, mutation_operator):
    engine = _build_engine(
        storage, max_generations, mutation_operator=mutation_operator
    )
    sm = ProgramStateManager(storage)
    runner = FakeDagRunner(storage, sm)

    runner.start()
    engine.start()
    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except TimeoutError:
        pytest.fail(f"Engine did not finish {max_generations} gens within 30s")
    finally:
        await runner.stop()
        await storage.close()

    return engine


class AlwaysNoneMutationOperator(MutationOperator):
    """Returns None for every mutation attempt — simulates LLM producing garbage."""

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        return None


class AlwaysRaisingMutationOperator(MutationOperator):
    """Raises on every mutation attempt — simulates LLM timeout/crash."""

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        raise RuntimeError("LLM unavailable")


class FailFirstThenSucceedOperator(MutationOperator):
    """Fails for the first N calls, then produces valid mutations."""

    def __init__(self, fail_count: int = 3):
        self._calls = 0
        self._fail_count = fail_count

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise RuntimeError(f"Transient failure #{self._calls}")
        parent = selected_parents[0]
        parent_metrics = _extract_metrics(parent.code)
        return MutationSpec(
            code=_make_code(parent_metrics["fitness"] + 1.0, 1.5),
            parents=selected_parents,
            name="recovered",
        )


class TestAllMutationsReturnNone:
    """Engine survives when mutation operator returns None for every attempt."""

    async def test_engine_completes_with_none_mutations(self) -> None:
        """Generation counter still advances when all mutations return None."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_engine(
            storage, max_generations=3, mutation_operator=AlwaysNoneMutationOperator()
        )

        assert engine.metrics.total_generations == 3
        # No mutations were persisted
        assert engine.metrics.mutations_created == 0

    async def test_archive_unchanged_with_none_mutations(self) -> None:
        """Archive contains only the seed after gens with no successful mutations."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        await _run_engine(
            storage, max_generations=3, mutation_operator=AlwaysNoneMutationOperator()
        )

        check_storage = _make_fakeredis_storage(server)
        strategy = MapElitesMultiIsland(
            island_configs=[_make_island_config()], program_storage=check_storage
        )
        programs = await strategy.islands["test"].get_elites()
        await check_storage.close()

        assert len(programs) == 1
        assert programs[0].metrics["fitness"] == 1.0


class TestAllMutationsRaise:
    """Engine survives when mutation operator raises for every attempt."""

    async def test_engine_completes_despite_all_raises(self) -> None:
        """Generation counter advances even when every mutate_single raises."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_engine(
            storage,
            max_generations=3,
            mutation_operator=AlwaysRaisingMutationOperator(),
        )

        assert engine.metrics.total_generations == 3
        assert engine.metrics.mutations_created == 0

    async def test_no_queued_or_running_after_all_raises(self) -> None:
        """No programs stuck in transient states after raise-only run."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        await _run_engine(
            storage,
            max_generations=3,
            mutation_operator=AlwaysRaisingMutationOperator(),
        )

        check_storage = _make_fakeredis_storage(server)
        queued = await check_storage.get_all_by_status(ProgramState.QUEUED.value)
        running = await check_storage.get_all_by_status(ProgramState.RUNNING.value)
        await check_storage.close()

        assert len(queued) == 0
        assert len(running) == 0


class TestTransientMutationFailure:
    """Engine recovers after initial mutation failures then succeeds."""

    async def test_engine_recovers_from_transient_failures(self) -> None:
        """After 3 failing mutation calls, the operator succeeds.

        We run 5 gens. Gen 1 has empty archive (no mutation). Gens 2-5
        each call mutate_single once. Calls 1-3 fail (gens 2-4), call 4
        succeeds (gen 5). The engine should complete all 5 gens and have
        at least 1 mutation created.
        """
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_engine(
            storage,
            max_generations=5,
            mutation_operator=FailFirstThenSucceedOperator(fail_count=3),
        )

        assert engine.metrics.total_generations == 5
        # At least 1 mutation should have succeeded (gen 5)
        assert engine.metrics.mutations_created >= 1

    async def test_recovered_mutant_in_archive(self) -> None:
        """The successfully mutated program ends up in the archive."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        await _run_engine(
            storage,
            max_generations=5,
            mutation_operator=FailFirstThenSucceedOperator(fail_count=3),
        )

        check_storage = _make_fakeredis_storage(server)
        strategy = MapElitesMultiIsland(
            island_configs=[_make_island_config()], program_storage=check_storage
        )
        programs = await strategy.islands["test"].get_elites()
        await check_storage.close()

        # The recovered mutant (fitness=2.0, x=1.5) lands in a different bin
        # than the seed (fitness=1.0, x=0.0) → archive has both, OR the mutant
        # (x=0.5) shares bin 0 with seed and replaces it.
        # Either way, the mutant with fitness=2.0 must be present.
        fitnesses = {p.metrics["fitness"] for p in programs}
        assert 2.0 in fitnesses, (
            f"Expected recovered mutant (fitness=2.0) in archive, got: {fitnesses}"
        )


# ===========================================================================
# Part 3: Stale cache + mid-chain failure on rerun
# ===========================================================================


class CountingStage(Stage):
    """Counts how many times compute() is called (class-level per subclass)."""

    InputsModel = VoidInput
    OutputModel = IntOutput
    call_count: int = 0

    async def compute(self, program: Program) -> IntOutput:
        self.__class__.call_count += 1
        return IntOutput(value=self.__class__.call_count)


class CountA(CountingStage):
    call_count: int = 0


class CountB(CountingStage):
    call_count: int = 0


class CountC(CountingStage):
    call_count: int = 0


class FailOnSecondRun(Stage):
    """Succeeds on the first call, fails on subsequent calls."""

    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE
    call_count: int = 0

    async def compute(self, program: Program) -> IntOutput:
        FailOnSecondRun.call_count += 1
        if FailOnSecondRun.call_count > 1:
            raise RuntimeError("fail on second run")
        return IntOutput(value=42)


@pytest.fixture(autouse=True)
def _reset_counting_stages():
    CountA.call_count = 0
    CountB.call_count = 0
    CountC.call_count = 0
    FailOnSecondRun.call_count = 0
    yield
    CountA.call_count = 0
    CountB.call_count = 0
    CountC.call_count = 0
    FailOnSecondRun.call_count = 0


class TestStaleCacheMidChainFailure:
    """Chain A→B→C; first run succeeds (cached); second run B fails (NO_CACHE).

    On the second run, B's NO_CACHE forces re-execution. If B fails,
    C (which has mandatory data dep on B) should be SKIPPED — not served
    from the stale cache of the first run.
    """

    async def test_downstream_skipped_not_stale_cached(
        self, state_manager, make_program
    ):
        """C is SKIPPED (not COMPLETED from stale cache) when B fails on rerun."""
        prog = make_program()

        # First run: all succeed
        dag1 = _make_dag(
            {
                "a": ProduceN(value=10),
                "b": FailOnSecondRun(timeout=5.0),
                "c": IncrStage(timeout=5.0),
            },
            [DataFlowEdge.create("b", "c", "data")],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        await dag1.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED

        # Second run: B fails (NO_CACHE), C should be SKIPPED
        dag2 = _make_dag(
            {
                "a": ProduceN(value=10),
                "b": FailOnSecondRun(timeout=5.0),
                "c": IncrStage(timeout=5.0),
            },
            [DataFlowEdge.create("b", "c", "data")],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        await dag2.run(prog)

        assert prog.stage_results["b"].status == StageState.FAILED
        # Critical assertion: C must NOT be served from stale cache
        assert prog.stage_results["c"].status == StageState.SKIPPED

    async def test_always_after_stage_still_runs_on_rerun_failure(
        self, state_manager, make_program
    ):
        """A cleanup stage with always_after(B) runs even when B fails on rerun."""
        prog = make_program()

        def make_nodes():
            return {
                "a": ProduceN(value=10),
                "b": FailOnSecondRun(timeout=5.0),
                "cleanup": ProduceN(value=99),
            }

        exec_deps = {"cleanup": [ExecutionOrderDependency.always_after("b")]}

        # First run: all succeed
        dag1 = _make_dag(make_nodes(), [], state_manager, exec_deps=exec_deps)
        await dag1.run(prog)
        assert prog.stage_results["cleanup"].status == StageState.COMPLETED

        # Second run: B fails, cleanup should still run
        dag2 = _make_dag(make_nodes(), [], state_manager, exec_deps=exec_deps)
        await dag2.run(prog)
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["cleanup"].status == StageState.COMPLETED
