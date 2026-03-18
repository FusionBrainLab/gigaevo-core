"""Advanced integration tests: cycle detection, deep chains, concurrent isolation,
acceptor-rejects-all stagnation, and wide fan-out/fan-in DAGs.

Test groups:
  1. DAG cycle detection variants (data-flow, self-loop, indirect 3-node, mixed)
  2. Deep chain (6 stages) with mid-chain failure cascade
  3. Concurrent program isolation through same DAG topology
  4. Evolution engine with acceptor that rejects everything (stagnation)
  5. Wide fan-out/fan-in DAG (1→8→1)
  6. DAG with interleaved exec-order + data-flow forming a complex topology
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
from gigaevo.programs.core_types import (
    FINAL_STATES,
    StageIO,
    StageState,
    VoidInput,
)
from gigaevo.programs.dag.automata import (
    DAGValidator,
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from tests.conftest import NullWriter

# ===========================================================================
# Shared I/O types
# ===========================================================================


class IntOutput(StageIO):
    value: int = 0


class IntInput(StageIO):
    data: IntOutput


class OptIntInput(StageIO):
    data: IntOutput | None = None


class MultiIntInput(StageIO):
    """Accepts up to 8 optional integer inputs."""

    i0: IntOutput | None = None
    i1: IntOutput | None = None
    i2: IntOutput | None = None
    i3: IntOutput | None = None
    i4: IntOutput | None = None
    i5: IntOutput | None = None
    i6: IntOutput | None = None
    i7: IntOutput | None = None


# ===========================================================================
# Shared stage classes
# ===========================================================================


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
    """Fails with no inputs (source-position failure)."""

    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        raise RuntimeError("intentional failure")


class FailIncrStage(Stage):
    """Accepts mandatory IntInput then fails (mid-chain failure)."""

    InputsModel = IntInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        raise RuntimeError("intentional mid-chain failure")


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


class TagStage(Stage):
    """Writes a unique tag metric to the program, proving which program ran."""

    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    def __init__(self, *, timeout: float = 5.0, tag: str, value: int = 1):
        super().__init__(timeout=timeout)
        self._tag = tag
        self._value = value

    async def compute(self, program: Program) -> IntOutput:
        program.add_metrics({f"tag_{self._tag}": float(self._value)})
        return IntOutput(value=self._value)


class SumAllStage(Stage):
    """Sums all non-None inputs from MultiIntInput."""

    InputsModel = MultiIntInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        total = 0
        for field_name in ["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7"]:
            val = getattr(self.params, field_name, None)
            if val is not None:
                total += val.value
        return IntOutput(value=total)


def _make_dag(nodes, edges, state_manager, *, exec_deps=None, **kwargs):
    return DAG(
        nodes=nodes,
        data_flow_edges=edges,
        execution_order_deps=exec_deps,
        state_manager=state_manager,
        writer=NullWriter(),
        **kwargs,
    )


# ===========================================================================
# Group 1: DAG Cycle Detection Variants
# ===========================================================================


class TestCycleDetectionVariants:
    """Verify cycle detection catches all topological cycle patterns."""

    def test_self_loop_in_exec_deps(self):
        """A stage with an exec dep on itself is a self-loop cycle."""
        errors = DAGValidator.validate_structure(
            {"a": ProduceN},
            [],
            {"a": [ExecutionOrderDependency(stage_name="a", condition="always")]},
        )
        assert any("cycle" in e.lower() for e in errors), (
            f"Self-loop not detected. Errors: {errors}"
        )

    def test_self_loop_in_data_flow(self):
        """A data-flow edge from a stage to itself is a self-loop cycle."""
        errors = DAGValidator.validate_structure(
            {"a": IncrStage},
            [DataFlowEdge.create("a", "a", "data")],
        )
        assert any("cycle" in e.lower() for e in errors), (
            f"Self-loop via data-flow not detected. Errors: {errors}"
        )

    def test_indirect_3node_cycle_via_exec_deps(self):
        """A→B→C→A cycle through exec deps must be detected."""
        errors = DAGValidator.validate_structure(
            {"a": ProduceN, "b": ProduceN, "c": ProduceN},
            [],
            {
                "b": [ExecutionOrderDependency(stage_name="a", condition="success")],
                "c": [ExecutionOrderDependency(stage_name="b", condition="success")],
                "a": [ExecutionOrderDependency(stage_name="c", condition="success")],
            },
        )
        assert any("cycle" in e.lower() for e in errors), (
            f"3-node cycle not detected. Errors: {errors}"
        )

    def test_indirect_3node_cycle_via_data_flow(self):
        """A→B→C→A cycle through data-flow edges must be detected."""
        errors = DAGValidator.validate_structure(
            {"a": IncrStage, "b": IncrStage, "c": IncrStage},
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "c", "data"),
                DataFlowEdge.create("c", "a", "data"),
            ],
        )
        assert any("cycle" in e.lower() for e in errors), (
            f"3-node data-flow cycle not detected. Errors: {errors}"
        )

    def test_mixed_cycle_data_flow_plus_exec_dep(self):
        """A cycle formed by a data-flow A→B and exec dep B→A must be caught."""
        errors = DAGValidator.validate_structure(
            {"a": ProduceN, "b": IncrStage},
            [DataFlowEdge.create("a", "b", "data")],
            {"a": [ExecutionOrderDependency(stage_name="b", condition="always")]},
        )
        assert any("cycle" in e.lower() for e in errors), (
            f"Mixed data-flow/exec-dep cycle not detected. Errors: {errors}"
        )

    def test_no_false_positive_on_diamond(self):
        """A diamond (A→B, A→C, B→D, C→D) is acyclic — no errors."""
        errors = DAGValidator.validate_structure(
            {"a": ProduceN, "b": ProduceN, "c": ProduceN, "d": ProduceN},
            [],
            {
                "b": [ExecutionOrderDependency(stage_name="a", condition="success")],
                "c": [ExecutionOrderDependency(stage_name="a", condition="success")],
                "d": [
                    ExecutionOrderDependency(stage_name="b", condition="success"),
                    ExecutionOrderDependency(stage_name="c", condition="success"),
                ],
            },
        )
        cycle_errors = [e for e in errors if "cycle" in e.lower()]
        assert not cycle_errors, (
            f"Diamond falsely detected as cycle. Errors: {cycle_errors}"
        )

    def test_dag_constructor_raises_on_cycle(self, state_manager):
        """DAG() raises ValueError when given cyclic dependencies."""
        with pytest.raises(ValueError, match="(?i)cycle"):
            _make_dag(
                {"a": ProduceN(), "b": ProduceN()},
                [],
                state_manager,
                exec_deps={
                    "b": [ExecutionOrderDependency(stage_name="a", condition="always")],
                    "a": [ExecutionOrderDependency(stage_name="b", condition="always")],
                },
            )


# ===========================================================================
# Group 2: Deep Chain with Mid-Chain Failure Cascade
# ===========================================================================


class TestDeepChainFailureCascade:
    """6-stage linear chain: A→B→C→D→E→F. Failure at different points."""

    def _build_chain(self, state_manager, *, fail_at: str | None = None):
        """Build a 6-stage chain where `fail_at` stage fails.

        'a' is the source (VoidInput → IntOutput). All others accept IntInput.
        Use FailIncrStage for mid-chain failures (accepts IntInput then fails).
        """
        nodes = {}
        for name in "abcdef":
            if name == fail_at and name == "a":
                nodes[name] = FailStage(timeout=5.0)
            elif name == fail_at:
                nodes[name] = FailIncrStage(timeout=5.0)
            elif name == "a":
                nodes[name] = ProduceN(value=1)
            else:
                nodes[name] = IncrStage(timeout=5.0)

        edges = [
            DataFlowEdge.create("a", "b", "data"),
            DataFlowEdge.create("b", "c", "data"),
            DataFlowEdge.create("c", "d", "data"),
            DataFlowEdge.create("d", "e", "data"),
            DataFlowEdge.create("e", "f", "data"),
        ]
        return _make_dag(nodes, edges, state_manager)

    async def test_no_failure_full_chain_completes(self, state_manager, make_program):
        """All 6 stages complete: values are 1, 2, 3, 4, 5, 6."""
        dag = self._build_chain(state_manager)
        prog = make_program()
        await dag.run(prog)

        for name in "abcdef":
            assert prog.stage_results[name].status == StageState.COMPLETED, (
                f"Stage '{name}' not COMPLETED"
            )

        expected = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
        for name, val in expected.items():
            assert prog.stage_results[name].output.value == val, (
                f"Stage '{name}' output={prog.stage_results[name].output.value}, expected={val}"
            )

    async def test_fail_at_b_cascades_to_cdef(self, state_manager, make_program):
        """B fails → C, D, E, F all SKIPPED (mandatory chain)."""
        dag = self._build_chain(state_manager, fail_at="b")
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        for name in "cdef":
            assert prog.stage_results[name].status == StageState.SKIPPED, (
                f"Stage '{name}' should be SKIPPED after B failure, got {prog.stage_results[name].status}"
            )

    async def test_fail_at_d_cascades_to_ef_only(self, state_manager, make_program):
        """D fails → E, F SKIPPED; A, B, C complete normally."""
        dag = self._build_chain(state_manager, fail_at="d")
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        assert prog.stage_results["d"].status == StageState.FAILED
        assert prog.stage_results["e"].status == StageState.SKIPPED
        assert prog.stage_results["f"].status == StageState.SKIPPED

    async def test_fail_at_f_only_f_fails(self, state_manager, make_program):
        """F (tail) fails → only F is FAILED; A-E complete."""
        dag = self._build_chain(state_manager, fail_at="f")
        prog = make_program()
        await dag.run(prog)

        for name in "abcde":
            assert prog.stage_results[name].status == StageState.COMPLETED
        assert prog.stage_results["f"].status == StageState.FAILED

    async def test_all_stages_reach_final_state(self, state_manager, make_program):
        """All stages reach a final state regardless of where the failure is."""
        for fail_at in [None, "b", "c", "d", "e", "f"]:
            dag = self._build_chain(state_manager, fail_at=fail_at)
            prog = make_program()
            await dag.run(prog)

            for name in "abcdef":
                assert prog.stage_results[name].status in FINAL_STATES, (
                    f"Stage '{name}' stuck in {prog.stage_results[name].status} "
                    f"when fail_at={fail_at}"
                )


# ===========================================================================
# Group 3: Concurrent Program Isolation
# ===========================================================================


class TestConcurrentProgramIsolation:
    """Two programs run through DAGs with the same topology concurrently.
    Verify no metric/output cross-contamination.
    """

    async def test_two_programs_no_metric_crosstalk(self, state_manager, make_program):
        """Two programs run through identical DAG topologies in parallel.
        Each has a TagStage writing a unique metric; neither should see the other's metric.
        """
        prog1 = make_program(code="def f(): return 1")
        prog2 = make_program(code="def f(): return 2")

        dag1 = _make_dag(
            {"tag": TagStage(tag="prog1", value=100)},
            [],
            state_manager,
        )
        dag2 = _make_dag(
            {"tag": TagStage(tag="prog2", value=200)},
            [],
            state_manager,
        )

        await asyncio.gather(dag1.run(prog1), dag2.run(prog2))

        # prog1 should only have its own tag
        assert prog1.metrics.get("tag_prog1") == 100.0
        assert "tag_prog2" not in prog1.metrics

        # prog2 should only have its own tag
        assert prog2.metrics.get("tag_prog2") == 200.0
        assert "tag_prog1" not in prog2.metrics

    async def test_two_programs_independent_stage_results(
        self, state_manager, make_program
    ):
        """Two programs run through a chain; each gets its own output values."""
        prog1 = make_program()
        prog2 = make_program()

        dag1 = _make_dag(
            {"a": ProduceN(value=10), "b": IncrStage(timeout=5.0)},
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        dag2 = _make_dag(
            {"a": ProduceN(value=99), "b": IncrStage(timeout=5.0)},
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )

        await asyncio.gather(dag1.run(prog1), dag2.run(prog2))

        assert prog1.stage_results["b"].output.value == 11  # 10 + 1
        assert prog2.stage_results["b"].output.value == 100  # 99 + 1

    async def test_failure_in_one_program_doesnt_affect_other(
        self, state_manager, make_program
    ):
        """One program's DAG fails; the other succeeds independently."""
        prog_ok = make_program()
        prog_fail = make_program()

        dag_ok = _make_dag(
            {"a": ProduceN(value=5)},
            [],
            state_manager,
        )
        dag_fail = _make_dag(
            {"a": FailStage(timeout=5.0)},
            [],
            state_manager,
        )

        await asyncio.gather(dag_ok.run(prog_ok), dag_fail.run(prog_fail))

        assert prog_ok.stage_results["a"].status == StageState.COMPLETED
        assert prog_ok.stage_results["a"].output.value == 5

        assert prog_fail.stage_results["a"].status == StageState.FAILED


# ===========================================================================
# Group 4: Engine with Acceptor Rejecting Everything (Stagnation)
# ===========================================================================


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


SEED_CODE = 'def entrypoint():\n    return {"fitness": 1.0, "x": 0.0}'

_CALL_COUNTER = 0


def _reset_counter():
    global _CALL_COUNTER
    _CALL_COUNTER = 0


def _make_code(fitness, x):
    return f'def entrypoint():\n    return {{"fitness": {fitness}, "x": {x}}}'


def _extract_metrics(code):
    m = re.search(
        r'return\s*\{\s*"fitness":\s*([\d.]+)\s*,\s*"x":\s*([\d.]+)\s*\}',
        code,
        re.MULTILINE,
    )
    if m is None:
        raise ValueError(f"Cannot extract metrics from code:\n{code}")
    return {"fitness": float(m.group(1)), "x": float(m.group(2))}


class IncrementOperator(MutationOperator):
    async def mutate_single(self, selected_parents):
        global _CALL_COUNTER
        parent = selected_parents[0]
        metrics = _extract_metrics(parent.code)
        _CALL_COUNTER += 1
        return MutationSpec(
            code=_make_code(metrics["fitness"] + 1.0, 0.5 + _CALL_COUNTER),
            parents=selected_parents,
            name="increment",
        )


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


async def _run_engine(storage, max_generations, *, mutation_operator, acceptor=None):
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    config = EngineConfig(
        loop_interval=0.005,
        max_elites_per_generation=1,
        max_mutations_per_generation=1,
        generation_timeout=30.0,
        max_generations=max_generations,
    )
    if acceptor is not None:
        config.program_acceptor = acceptor

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=mutation_operator,
        config=config,
        writer=_make_null_writer(),
        metrics_tracker=_make_metrics_tracker(),
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


class RejectAllAcceptor:
    """Acceptor that rejects every program."""

    def is_accepted(self, program):
        return False


class TestAcceptorRejectsAll:
    """Engine survives when acceptor rejects everything — consecutive stagnation."""

    async def test_engine_completes_with_all_rejected(self):
        """Generation counter advances even when acceptor rejects all programs."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_engine(
            storage,
            max_generations=4,
            mutation_operator=IncrementOperator(),
            acceptor=RejectAllAcceptor(),
        )

        assert engine.metrics.total_generations == 4

    async def test_archive_stays_empty_when_all_rejected(self):
        """With reject-all acceptor, no programs enter the archive."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_engine(
            storage,
            max_generations=3,
            mutation_operator=IncrementOperator(),
            acceptor=RejectAllAcceptor(),
        )

        assert engine.metrics.added == 0
        assert engine.metrics.rejected_validation >= 1

    async def test_no_mutations_after_seed_rejected(self):
        """Since the seed itself is rejected, archive stays empty → no elites
        → no mutations in subsequent generations."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_engine(
            storage,
            max_generations=3,
            mutation_operator=IncrementOperator(),
            acceptor=RejectAllAcceptor(),
        )

        # Gen 1 ingests seed (rejected) → archive empty
        # Gen 2-3: no elites → no mutations
        assert engine.metrics.mutations_created == 0

    async def test_no_stranded_programs_after_stagnation(self):
        """No QUEUED or RUNNING programs after a stagnated run."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        await _run_engine(
            storage,
            max_generations=3,
            mutation_operator=IncrementOperator(),
            acceptor=RejectAllAcceptor(),
        )

        check = _make_fakeredis_storage(server)
        queued = await check.get_all_by_status(ProgramState.QUEUED.value)
        running = await check.get_all_by_status(ProgramState.RUNNING.value)
        await check.close()

        assert len(queued) == 0
        assert len(running) == 0


# ===========================================================================
# Group 5: Wide Fan-Out / Fan-In DAG (1→8→1)
# ===========================================================================


class TestWideFanOutFanIn:
    """Wide DAG: one source fans out to 8 parallel stages, then joins into one."""

    async def test_fanout_8_all_succeed(self, state_manager, make_program):
        """1 source → 8 parallel producers → 1 aggregator. All succeed."""
        branches = {f"b{i}": ProduceN(value=i + 1) for i in range(8)}
        nodes = {"src": ProduceN(value=0), **branches, "agg": SumAllStage(timeout=5.0)}

        edges = [DataFlowEdge.create(f"b{i}", "agg", f"i{i}") for i in range(8)]
        exec_deps = {
            f"b{i}": [ExecutionOrderDependency(stage_name="src", condition="success")]
            for i in range(8)
        }

        dag = _make_dag(nodes, edges, state_manager, exec_deps=exec_deps)
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["src"].status == StageState.COMPLETED
        for i in range(8):
            assert prog.stage_results[f"b{i}"].status == StageState.COMPLETED

        assert prog.stage_results["agg"].status == StageState.COMPLETED
        # Sum of 1+2+3+4+5+6+7+8 = 36
        assert prog.stage_results["agg"].output.value == 36

    async def test_fanout_8_one_fails_agg_partial(self, state_manager, make_program):
        """1→8→1 where branch 3 fails. Aggregator runs with 7 inputs (optional)."""
        branches = {}
        for i in range(8):
            if i == 3:
                branches[f"b{i}"] = FailStage(timeout=5.0)
            else:
                branches[f"b{i}"] = ProduceN(value=i + 1)

        nodes = {"src": ProduceN(value=0), **branches, "agg": SumAllStage(timeout=5.0)}

        edges = [DataFlowEdge.create(f"b{i}", "agg", f"i{i}") for i in range(8)]
        exec_deps = {
            f"b{i}": [ExecutionOrderDependency(stage_name="src", condition="success")]
            for i in range(8)
        }

        dag = _make_dag(nodes, edges, state_manager, exec_deps=exec_deps)
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["b3"].status == StageState.FAILED
        assert prog.stage_results["agg"].status == StageState.COMPLETED
        # Sum without branch 3 (value=4): 1+2+3+5+6+7+8 = 32
        assert prog.stage_results["agg"].output.value == 32

    async def test_fanout_8_source_fails_all_branches_skipped(
        self, state_manager, make_program
    ):
        """Source fails → all 8 branches skipped (on_success dep) → aggregator
        runs with all None inputs."""
        branches = {f"b{i}": ProduceN(value=i + 1) for i in range(8)}
        nodes = {
            "src": FailStage(timeout=5.0),
            **branches,
            "agg": SumAllStage(timeout=5.0),
        }

        edges = [DataFlowEdge.create(f"b{i}", "agg", f"i{i}") for i in range(8)]
        exec_deps = {
            f"b{i}": [ExecutionOrderDependency(stage_name="src", condition="success")]
            for i in range(8)
        }

        dag = _make_dag(nodes, edges, state_manager, exec_deps=exec_deps)
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["src"].status == StageState.FAILED
        for i in range(8):
            assert prog.stage_results[f"b{i}"].status == StageState.SKIPPED
        assert prog.stage_results["agg"].status == StageState.COMPLETED
        assert prog.stage_results["agg"].output.value == 0

    async def test_fanout_all_stages_reach_final_state(
        self, state_manager, make_program
    ):
        """All 10 stages (1 src + 8 branches + 1 agg) reach a final state."""
        branches = {}
        for i in range(8):
            if i % 3 == 0:
                branches[f"b{i}"] = FailStage(timeout=5.0)
            else:
                branches[f"b{i}"] = ProduceN(value=i + 1)

        nodes = {"src": ProduceN(value=0), **branches, "agg": SumAllStage(timeout=5.0)}
        edges = [DataFlowEdge.create(f"b{i}", "agg", f"i{i}") for i in range(8)]
        exec_deps = {
            f"b{i}": [ExecutionOrderDependency(stage_name="src", condition="success")]
            for i in range(8)
        }

        dag = _make_dag(nodes, edges, state_manager, exec_deps=exec_deps)
        prog = make_program()
        await dag.run(prog)

        for name, result in prog.stage_results.items():
            assert result.status in FINAL_STATES, (
                f"Stage '{name}' stuck in {result.status}"
            )


# ===========================================================================
# Group 6: Complex Topology with Interleaved Dependencies
# ===========================================================================


class TestComplexTopology:
    """A 7-node DAG with mixed exec-order and data-flow deps:

    A ──data──→ B ──data──→ D ──data──→ F
    A ──exec──→ C ──data──→ D
    A ──exec──→ E ──exec──→ F
    C ──exec──→ G (always_after C)
    B ──data──→ G (optional)
    """

    async def test_complex_topology_happy_path(self, state_manager, make_program):
        """All nodes succeed; verify output values propagate correctly."""
        dag = _make_dag(
            {
                "a": ProduceN(value=1),
                "b": IncrStage(timeout=5.0),  # a→b data: 1+1=2
                "c": ProduceN(value=10),
                "d": IncrStage(timeout=5.0),  # b→d data: 2+1=3
                "e": ProduceN(value=20),
                "f": IncrStage(timeout=5.0),  # d→f data: 3+1=4
                "g": OptIncrStage(timeout=5.0),  # b→g optional data: 2+1=3
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "d", "data"),
                DataFlowEdge.create("d", "f", "data"),
                DataFlowEdge.create("b", "g", "data"),
            ],
            state_manager,
            exec_deps={
                "c": [ExecutionOrderDependency(stage_name="a", condition="success")],
                "e": [ExecutionOrderDependency(stage_name="a", condition="success")],
                "f": [ExecutionOrderDependency(stage_name="e", condition="success")],
                "g": [ExecutionOrderDependency.always_after("c")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        for name in "abcdefg":
            assert prog.stage_results[name].status == StageState.COMPLETED, (
                f"Stage '{name}' not COMPLETED"
            )

        assert prog.stage_results["a"].output.value == 1
        assert prog.stage_results["b"].output.value == 2
        assert prog.stage_results["d"].output.value == 3
        assert prog.stage_results["f"].output.value == 4
        assert prog.stage_results["g"].output.value == 3  # b(2) + 1

    async def test_complex_topology_c_fails_g_still_runs(
        self, state_manager, make_program
    ):
        """C fails but G has always_after(C) → G still runs."""
        dag = _make_dag(
            {
                "a": ProduceN(value=1),
                "b": IncrStage(timeout=5.0),
                "c": FailStage(timeout=5.0),  # C fails
                "d": IncrStage(timeout=5.0),
                "e": ProduceN(value=20),
                "f": IncrStage(timeout=5.0),
                "g": OptIncrStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "d", "data"),
                DataFlowEdge.create("d", "f", "data"),
                DataFlowEdge.create("b", "g", "data"),
            ],
            state_manager,
            exec_deps={
                "c": [ExecutionOrderDependency(stage_name="a", condition="success")],
                "e": [ExecutionOrderDependency(stage_name="a", condition="success")],
                "f": [ExecutionOrderDependency(stage_name="e", condition="success")],
                "g": [ExecutionOrderDependency.always_after("c")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["c"].status == StageState.FAILED
        # G has always_after(C) + optional data from B → runs with B's output
        assert prog.stage_results["g"].status == StageState.COMPLETED
        assert prog.stage_results["g"].output.value == 3  # b(2) + 1
        # D and F still complete (they don't depend on C)
        assert prog.stage_results["d"].status == StageState.COMPLETED
        assert prog.stage_results["f"].status == StageState.COMPLETED
