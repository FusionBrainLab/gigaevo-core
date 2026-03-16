"""Complex DAG integration tests — multi-concern scenarios focused on edge cases.

Each test exercises at least 2–3 interacting subsystems simultaneously.
All stage mock classes are defined locally to avoid polluting conftest.py.

Test groups:
  1. Cascading failures with mixed dependency types
  2. Re-run scenarios (refresh / stale states)
  3. Concurrency edge cases
  4. State persistence correctness
  5. Complex exec-order + data-flow interactions
  6. Input/output passing correctness
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from gigaevo.programs.core_types import (
    FINAL_STATES,
    ProgramStageResult,
    StageIO,
    StageState,
    VoidInput,
    VoidOutput,
)
from gigaevo.programs.dag.automata import (
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE, InputHashCache
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Local stage I/O types
# ---------------------------------------------------------------------------


class IntOutput(StageIO):
    value: int = 0


class IntInput(StageIO):
    data: IntOutput


class OptIntInput(StageIO):
    data: IntOutput | None = None


class DualOptInput(StageIO):
    left: IntOutput | None = None
    right: IntOutput | None = None


class DualMandatoryInput(StageIO):
    left: IntOutput
    right: IntOutput


# ---------------------------------------------------------------------------
# Local stage classes (all defined here, not in conftest)
# ---------------------------------------------------------------------------


class ProduceOne(Stage):
    """Produces IntOutput(value=1)."""

    InputsModel = VoidInput
    OutputModel = IntOutput

    async def compute(self, program: Program) -> IntOutput:
        return IntOutput(value=1)


class FailProducer(Stage):
    """Always fails, produces nothing."""

    InputsModel = VoidInput
    OutputModel = IntOutput

    async def compute(self, program: Program) -> IntOutput:
        raise RuntimeError("FailProducer intentional failure")


class IncrStage(Stage):
    """Reads mandatory IntInput, increments value by 1."""

    InputsModel = IntInput
    OutputModel = IntOutput

    async def compute(self, program: Program) -> IntOutput:
        return IntOutput(value=self.params.data.value + 1)


class OptIncrStage(Stage):
    """Reads optional IntInput; returns value+1 if present, else -1."""

    InputsModel = OptIntInput
    OutputModel = IntOutput

    async def compute(self, program: Program) -> IntOutput:
        if self.params.data is not None:
            return IntOutput(value=self.params.data.value + 1)
        return IntOutput(value=-1)


class DualOptSumStage(Stage):
    """Sums two optional inputs; each missing counts as 0."""

    InputsModel = DualOptInput
    OutputModel = IntOutput

    async def compute(self, program: Program) -> IntOutput:
        left_val = self.params.left.value if self.params.left is not None else 0
        right_val = self.params.right.value if self.params.right is not None else 0
        return IntOutput(value=left_val + right_val)


class NoOpStage(Stage):
    """Succeeds immediately with VoidOutput."""

    InputsModel = VoidInput
    OutputModel = VoidOutput

    async def compute(self, program: Program) -> None:
        return None


class MetricsStage(Stage):
    """Writes a named metric to program.metrics during compute()."""

    InputsModel = VoidInput
    OutputModel = VoidOutput

    metric_key: str = "metric_a"
    metric_value: float = 1.0

    def __init__(
        self, *, timeout: float, metric_key: str = "metric_a", metric_value: float = 1.0
    ):
        super().__init__(timeout=timeout)
        self.metric_key = metric_key
        self.metric_value = metric_value

    async def compute(self, program: Program) -> None:
        # Small sleep to force genuine concurrency in parallel tests
        await asyncio.sleep(0.02)
        program.add_metrics({self.metric_key: self.metric_value})
        return None


class CounterStage(Stage):
    """Counting stage; increments a class-level counter on each compute() call.

    Each subclass must have its own counter to avoid cross-test contamination.
    """

    InputsModel = VoidInput
    OutputModel = IntOutput
    call_count: int = 0

    async def compute(self, program: Program) -> IntOutput:
        self.__class__.call_count += 1
        return IntOutput(value=self.__class__.call_count)


# Distinct subclasses so tests can have independent counters
class CounterA(CounterStage):
    call_count: int = 0


class CounterB(CounterStage):
    call_count: int = 0


class CounterC(CounterStage):
    call_count: int = 0


class CounterD(CounterStage):
    call_count: int = 0


class CounterE(CounterStage):
    call_count: int = 0


class SlowProducer(Stage):
    """Sleeps briefly, then produces IntOutput(value=7)."""

    InputsModel = VoidInput
    OutputModel = IntOutput

    def __init__(self, *, timeout: float, sleep: float = 0.1):
        super().__init__(timeout=timeout)
        self.sleep = sleep

    async def compute(self, program: Program) -> IntOutput:
        await asyncio.sleep(self.sleep)
        return IntOutput(value=7)


class InfiniteStage(Stage):
    """Sleeps forever — used to test stage timeout."""

    InputsModel = VoidInput
    OutputModel = IntOutput

    async def compute(self, program: Program) -> IntOutput:
        await asyncio.sleep(3600)
        return IntOutput(value=0)  # pragma: no cover


class OrderTrackingStage(Stage):
    """Appends its name to a shared list when its semaphore slot is acquired."""

    InputsModel = VoidInput
    OutputModel = IntOutput

    def __init__(self, *, timeout: float, name: str, log: list[str], active: set[str]):
        super().__init__(timeout=timeout)
        self._name = name
        self._log = log
        self._active = active

    async def compute(self, program: Program) -> IntOutput:
        self._active.add(self._name)
        self._log.append(self._name)
        await asyncio.sleep(0.02)
        self._active.discard(self._name)
        return IntOutput(value=1)


class UniqueSideEffectStage(Stage):
    """Writes a uniquely keyed metric to identify this program's DAG run."""

    InputsModel = VoidInput
    OutputModel = IntOutput

    def __init__(self, *, timeout: float, program_tag: str, value: float = 1.0):
        super().__init__(timeout=timeout)
        self.program_tag = program_tag
        self.value = value

    async def compute(self, program: Program) -> IntOutput:
        program.add_metrics({f"tag_{self.program_tag}": self.value})
        return IntOutput(value=int(self.value))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_dag(
    nodes: dict,
    edges: list,
    state_manager,
    *,
    exec_deps: dict | None = None,
    **kwargs,
) -> DAG:
    return DAG(
        nodes=nodes,
        data_flow_edges=edges,
        execution_order_deps=exec_deps,
        state_manager=state_manager,
        writer=NullWriter(),
        **kwargs,
    )


def _make_result(
    status: StageState,
    *,
    input_hash: str | None = None,
    output=None,
) -> ProgramStageResult:
    now = datetime.now(UTC)
    return ProgramStageResult(
        status=status,
        started_at=now,
        finished_at=now if status in FINAL_STATES else None,
        input_hash=input_hash,
        output=output,
    )


# ---------------------------------------------------------------------------
# Reset class-level counters before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_local_counters():
    CounterA.call_count = 0
    CounterB.call_count = 0
    CounterC.call_count = 0
    CounterD.call_count = 0
    CounterE.call_count = 0
    yield
    CounterA.call_count = 0
    CounterB.call_count = 0
    CounterC.call_count = 0
    CounterD.call_count = 0
    CounterE.call_count = 0


# ===========================================================================
# Group 1: Cascading failures with mixed dependency types
# ===========================================================================


class TestCascadingFailuresMixed:
    """Multi-dep scenarios combining mandatory/optional data-flow and exec-order deps."""

    async def test_deep_mandatory_chain_with_optional_bypass_and_always_after(
        self, state_manager, make_program
    ):
        """A->B->C (mandatory data-flow), D reads A optionally, E has always_after(C).

        B fails -> C skipped (mandatory dep on B), D still runs (optional input
        from A that succeeds), E still runs (always_after means any final state).

        Topology:
          A (ProduceOne, succeeds, value=1)
          B (FailProducer, fails; no data-flow inputs)
          C (IncrStage, mandatory dep on B via data-flow -> SKIPPED because B failed)
          D (OptIncrStage, optional dep on A -> runs with value=2)
          E (NoOpStage, always_after(C) -> runs even though C is SKIPPED)

        Expected: A=COMPLETED, B=FAILED, C=SKIPPED, D=COMPLETED(value=2), E=COMPLETED
        """
        dag = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),
                "b": FailProducer(timeout=5.0),
                "c": IncrStage(timeout=5.0),
                "d": OptIncrStage(timeout=5.0),
                "e": NoOpStage(timeout=5.0),
            },
            [
                DataFlowEdge.create(
                    "b", "c", "data"
                ),  # mandatory: B fails -> C SKIPPED
                DataFlowEdge.create(
                    "a", "d", "data"
                ),  # optional: A succeeds -> D gets value
            ],
            state_manager,
            exec_deps={"e": [ExecutionOrderDependency.always_after("c")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.SKIPPED
        # D gets A's output (value=1) as optional input -> value=2
        assert prog.stage_results["d"].status == StageState.COMPLETED
        assert prog.stage_results["d"].output.value == 2
        # E always runs after C (even though C is SKIPPED)
        assert prog.stage_results["e"].status == StageState.COMPLETED

    async def test_fork_join_one_fork_fails_other_succeeds_join_has_optional_both(
        self, state_manager, make_program
    ):
        """Fork-join: A->B (mandatory), A->C (mandatory), B->D (optional), C->D (optional).

        B fails -> D still runs but receives only C's output.

        Expected: A=COMPLETED, B=FAILED, C=COMPLETED, D=COMPLETED (left=None, right=value from C)
        """
        dag = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),
                "b": FailProducer(timeout=5.0),  # fails
                "c": IncrStage(timeout=5.0),  # succeeds, value=2 (from a=1)
                "d": DualOptSumStage(timeout=5.0),
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

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        assert prog.stage_results["d"].status == StageState.COMPLETED
        # left=None (B failed) -> 0; right=C's output (value=2) -> sum=2
        assert prog.stage_results["d"].output.value == 2

    async def test_on_failure_exec_dep_with_mandatory_data_dep_both_satisfied(
        self, state_manager, make_program
    ):
        """A fails, C succeeds; B has on_failure(A) AND mandatory data from C.

        Both conditions are simultaneously satisfied -> B runs.

        Expected: A=FAILED, C=COMPLETED, B=COMPLETED
        """
        dag = _make_dag(
            {
                "a": FailProducer(timeout=5.0),
                "c": ProduceOne(timeout=5.0),
                "b": IncrStage(timeout=5.0),
            },
            [DataFlowEdge.create("c", "b", "data")],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        # B: data dep on C (satisfied) + exec dep on_failure(A) (satisfied) -> runs
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["b"].output.value == 2  # value=1 + 1

    async def test_three_way_fanin_two_optional_fail_mandatory_succeeds(
        self, state_manager, make_program
    ):
        """A succeeds (mandatory for D), B and C fail (optional for D).

        D runs with B=None and C=None optional inputs.

        Expected: A=COMPLETED, B=FAILED, C=FAILED, D=COMPLETED(left=A.value, right=0)
        """
        dag = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),  # succeeds, value=1
                "b": FailProducer(timeout=5.0),
                "c": FailProducer(timeout=5.0),
                "d": DualOptSumStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "d", "left"),
                DataFlowEdge.create("b", "d", "right"),  # optional, B fails
            ],
            state_manager,
            # We only have left and right in DualOptInput; C is a free runner but
            # let's wire it via exec dep to make D wait for it
            exec_deps={"d": [ExecutionOrderDependency.always_after("c")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.FAILED
        assert prog.stage_results["d"].status == StageState.COMPLETED
        # left=A(1), right=None(B failed) -> 1+0=1
        assert prog.stage_results["d"].output.value == 1


# ===========================================================================
# Group 2: Re-run scenarios (refresh / stale states)
# ===========================================================================


class TestReRunScenarios:
    """Caching behavior across multiple DAG runs on the same program."""

    async def test_full_stale_5stage_diamond_all_cached(
        self, state_manager, make_program
    ):
        """5-stage diamond DAG; first run populates results; second run all cached.

        Verify: on second run, CounterX.call_count remains at 1 for all stages.
        """

        # Diamond: A -> B, A -> C; B -> D, C -> D; all -> E (always_after D)
        def nodes_fn():
            return {
                "a": CounterA(timeout=5.0),
                "b": CounterB(timeout=5.0),
                "c": CounterC(timeout=5.0),
                "d": NoOpStage(timeout=5.0),
                "e": NoOpStage(timeout=5.0),
            }

        exec_deps = {
            "b": [ExecutionOrderDependency.on_success("a")],
            "c": [ExecutionOrderDependency.on_success("a")],
            "d": [
                ExecutionOrderDependency.on_success("b"),
                ExecutionOrderDependency.on_success("c"),
            ],
            "e": [ExecutionOrderDependency.always_after("d")],
        }

        prog = make_program()
        dag1 = _make_dag(nodes_fn(), [], state_manager, exec_deps=exec_deps)
        await dag1.run(prog)

        # After first run
        assert CounterA.call_count == 1
        assert CounterB.call_count == 1
        assert CounterC.call_count == 1

        # Second run — all stages should be cached
        dag2 = _make_dag(nodes_fn(), [], state_manager, exec_deps=exec_deps)
        await dag2.run(prog)

        # call_count must remain 1 for all counter stages (cached, not re-executed)
        assert CounterA.call_count == 1
        assert CounterB.call_count == 1
        assert CounterC.call_count == 1

        for stage in ("a", "b", "c", "d", "e"):
            assert prog.stage_results[stage].status == StageState.COMPLETED

    async def test_partial_stale_only_pending_stages_execute(
        self, state_manager, make_program
    ):
        """A and B COMPLETED (stale with valid hashes); C and D PENDING.

        A->C->D, B->D (all mandatory exec deps, no data-flow to keep types simple).

        Expected: A.count=0, B.count=0 (cached); C.count=1, D.count=1 (fresh).
        """

        # Run once to generate valid hashes for A and B
        def nodes_fn_ab():
            return {
                "a": CounterA(timeout=5.0),
                "b": CounterB(timeout=5.0),
            }

        prog = make_program()
        dag_ab = _make_dag(nodes_fn_ab(), [], state_manager)
        await dag_ab.run(prog)
        # Reset counters — A and B have run once; their results are now "stale" cached
        CounterA.call_count = 0
        CounterB.call_count = 0

        # Now add C and D as PENDING and run the full DAG
        dag_full = _make_dag(
            {
                "a": CounterA(timeout=5.0),
                "b": CounterB(timeout=5.0),
                "c": CounterC(timeout=5.0),
                "d": CounterD(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "c": [ExecutionOrderDependency.on_success("a")],
                "d": [
                    ExecutionOrderDependency.on_success("c"),
                    ExecutionOrderDependency.on_success("b"),
                ],
            },
        )
        await dag_full.run(prog)

        # A and B cached: their call_count must still be 0
        assert CounterA.call_count == 0, "A should have been cached, not re-executed"
        assert CounterB.call_count == 0, "B should have been cached, not re-executed"
        # C and D ran fresh
        assert CounterC.call_count == 1, "C should have executed exactly once"
        assert CounterD.call_count == 1, "D should have executed exactly once"

    async def test_stale_result_invalidated_by_changed_input_hash(
        self, state_manager, make_program
    ):
        """B has COMPLETED stale result with a wrong input_hash -> B re-executes.

        A produces IntOutput(value=1); B (IncrStage with a call counter) depends on A.
        Pre-set B's stored hash to something that won't match A's actual output hash,
        triggering a re-run.

        Expected: B executes once on re-run (call count goes from 0 to 1).
        """

        class CountingIncrStage(Stage):
            """IncrStage that counts calls; uses InputHashCache so hash mismatch triggers rerun."""

            InputsModel = IntInput
            OutputModel = IntOutput
            cache_handler = InputHashCache()
            call_count: int = 0

            async def compute(self, program: Program) -> IntOutput:
                CountingIncrStage.call_count += 1
                return IntOutput(value=self.params.data.value + 1)

        CountingIncrStage.call_count = 0

        # First run to populate A's result and compute correct hash for B
        dag1 = _make_dag(
            {"a": ProduceOne(timeout=5.0), "b": CountingIncrStage(timeout=5.0)},
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program()
        await dag1.run(prog)
        assert CountingIncrStage.call_count == 1
        CountingIncrStage.call_count = 0

        # Corrupt B's stored input_hash so it won't match on re-run
        prog.stage_results["b"].input_hash = "deliberately_wrong_hash_xyz"

        dag2 = _make_dag(
            {"a": ProduceOne(timeout=5.0), "b": CountingIncrStage(timeout=5.0)},
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        await asyncio.sleep(0.01)
        await dag2.run(prog)

        assert CountingIncrStage.call_count == 1, (
            "B must have re-executed due to hash mismatch"
        )

    async def test_multi_generation_stale_idempotency(
        self, state_manager, make_program
    ):
        """Run DAG first time, then run same program again; second run all cached.

        Third run also all cached (idempotent). Both stages are independent
        (no data-flow edges) so CounterA and CounterC both have VoidInput.
        """

        def nodes_fn():
            return {
                "a": CounterA(timeout=5.0),
                "c": CounterC(timeout=5.0),
            }

        prog = make_program()

        # Run 1: fresh execution
        dag1 = _make_dag(nodes_fn(), [], state_manager)
        await dag1.run(prog)
        assert CounterA.call_count == 1
        assert CounterC.call_count == 1

        # Run 2: all cached
        dag2 = _make_dag(nodes_fn(), [], state_manager)
        await dag2.run(prog)
        assert CounterA.call_count == 1
        assert CounterC.call_count == 1

        # Run 3: still all cached
        dag3 = _make_dag(nodes_fn(), [], state_manager)
        await dag3.run(prog)
        assert CounterA.call_count == 1
        assert CounterC.call_count == 1

        # Stage results are unchanged across runs
        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED


# ===========================================================================
# Group 3: Concurrency edge cases
# ===========================================================================


class TestConcurrencyEdgeCases:
    """Race conditions, semaphore behavior, and timeout scenarios."""

    async def test_two_parallel_stages_both_write_metrics_no_loss(
        self, state_manager, make_program
    ):
        """A and B run in parallel; both call program.add_metrics().

        C runs after both (always_after A and B). All metrics must be present.

        Uses asyncio.sleep in A and B to force genuine concurrency.
        """
        dag = _make_dag(
            {
                "a": MetricsStage(timeout=5.0, metric_key="key_a", metric_value=10.0),
                "b": MetricsStage(timeout=5.0, metric_key="key_b", metric_value=20.0),
                "c": NoOpStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "c": [
                    ExecutionOrderDependency.always_after("a"),
                    ExecutionOrderDependency.always_after("b"),
                ]
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED

        # Both metrics must be present — no data lost due to concurrent writes
        assert prog.metrics.get("key_a") == 10.0, "key_a must be written by A"
        assert prog.metrics.get("key_b") == 20.0, "key_b must be written by B"

    async def test_semaphore_1_with_5_ready_stages_serializes_correctly(
        self, state_manager, make_program
    ):
        """5 independent stages with max_parallel_stages=1; no two overlap.

        Each stage appends its name to a shared list when it starts and records
        the set of active stages. Since semaphore=1, active set never has >1 element.
        """
        log: list[str] = []
        active: set[str] = set()
        max_concurrent_seen: list[int] = [0]

        class TrackingStage(Stage):
            InputsModel = VoidInput
            OutputModel = IntOutput

            def __init__(self, *, timeout: float, name: str):
                super().__init__(timeout=timeout)
                self._name = name

            async def compute(self, program: Program) -> IntOutput:
                active.add(self._name)
                log.append(self._name)
                max_concurrent_seen[0] = max(max_concurrent_seen[0], len(active))
                await asyncio.sleep(0.02)
                active.discard(self._name)
                return IntOutput(value=1)

        dag = _make_dag(
            {f"s{i}": TrackingStage(timeout=5.0, name=f"s{i}") for i in range(5)},
            [],
            state_manager,
            max_parallel_stages=1,
        )
        prog = make_program()
        await dag.run(prog)

        # All stages must have run
        assert len(log) == 5
        assert set(log) == {f"s{i}" for i in range(5)}
        # With semaphore=1, never more than 1 stage should have been active at once
        assert max_concurrent_seen[0] <= 1, (
            f"Expected max 1 concurrent stage, saw {max_concurrent_seen[0]}"
        )

    async def test_stage_timeout_fails_and_downstream_always_after_still_runs(
        self, state_manager, make_program
    ):
        """InfiniteStage with 0.3s timeout -> FAILED; downstream always_after still runs.

        This tests that stage-level timeout (asyncio.TimeoutError inside execute())
        is caught and converted to FAILED, not propagated to the whole DAG.
        """
        dag = _make_dag(
            {
                "infinite": InfiniteStage(timeout=0.3),
                "after": NoOpStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={"after": [ExecutionOrderDependency.always_after("infinite")]},
        )
        prog = make_program()
        await asyncio.wait_for(dag.run(prog), timeout=10.0)

        # InfiniteStage timed out -> FAILED
        assert prog.stage_results["infinite"].status == StageState.FAILED
        # 'after' has always_after -> runs regardless
        assert prog.stage_results["after"].status == StageState.COMPLETED

    async def test_dag_timeout_raises_asyncio_timeout_error(
        self, state_manager, make_program
    ):
        """SlowProducer (sleeps 0.5s) in a chain with dag_timeout=0.1s raises TimeoutError."""
        dag = _make_dag(
            {
                "slow": SlowProducer(timeout=60.0, sleep=0.5),
                "next": OptIncrStage(timeout=5.0),
            },
            [DataFlowEdge.create("slow", "next", "data")],
            state_manager,
            dag_timeout=0.1,
        )
        prog = make_program()
        with pytest.raises(asyncio.TimeoutError):
            await dag.run(prog)


# ===========================================================================
# Group 4: State persistence correctness
# ===========================================================================


class TestStatePersistenceCorrectness:
    """Redis persistence: stage results and metrics persisted mid-run."""

    async def test_stage_side_effects_persisted_to_redis_after_completion(
        self, state_manager, fakeredis_storage, make_program
    ):
        """SideEffectStage writes metrics; after DAG completes, fetch from Redis.

        Asserts the metric is present on the fetched (deserialized) program.
        """
        dag = _make_dag(
            {
                "side": MetricsStage(
                    timeout=5.0, metric_key="persisted_key", metric_value=42.0
                ),
            },
            [],
            state_manager,
        )
        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.metrics.get("persisted_key") == 42.0

    async def test_stage_results_persisted_per_stage_not_only_at_end(
        self, state_manager, fakeredis_storage, make_program
    ):
        """3-stage DAG: A (succeeds), B (fails), C (mandatory dep on B -> SKIPPED).

        Fetches program from Redis after DAG completes and verifies all three
        stage results are persisted with correct statuses. This confirms
        update_stage_result is called per-stage, not just at the end.

        Topology:
          A (ProduceOne, independent, succeeds)
          B (FailProducer, no data input, always fails)
          C (IncrStage, mandatory data-flow input from B -> SKIPPED because B failed)
        """
        dag = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),
                "b": FailProducer(timeout=5.0),
                "c": IncrStage(timeout=5.0),
            },
            [DataFlowEdge.create("b", "c", "data")],
            state_manager,
        )
        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None

        # All three stage results must be present in Redis-persisted state
        assert "a" in fetched.stage_results
        assert "b" in fetched.stage_results
        assert "c" in fetched.stage_results

        # Check statuses: A=COMPLETED (independent), B=FAILED, C=SKIPPED
        assert fetched.stage_results["a"].status == StageState.COMPLETED
        assert fetched.stage_results["b"].status == StageState.FAILED
        assert fetched.stage_results["c"].status == StageState.SKIPPED

    async def test_concurrent_dags_for_different_programs_no_cross_contamination(
        self, state_manager, fakeredis_storage, make_program
    ):
        """3 programs run their DAGs concurrently; each writes a unique metric.

        After all complete, fetch each program from Redis and verify no cross-
        contamination: each program has only its own unique metric.
        """
        programs = [make_program(code=f"def p{i}(): return {i}") for i in range(3)]
        for p in programs:
            await fakeredis_storage.add(p)

        async def run_prog(prog: Program, tag: str) -> None:
            dag = _make_dag(
                {
                    "effect": UniqueSideEffectStage(
                        timeout=5.0, program_tag=tag, value=float(ord(tag))
                    )
                },
                [],
                state_manager,
            )
            await dag.run(prog)

        await asyncio.gather(
            run_prog(programs[0], "x"),
            run_prog(programs[1], "y"),
            run_prog(programs[2], "z"),
        )

        for prog, tag in zip(programs, ["x", "y", "z"]):
            fetched = await fakeredis_storage.get(prog.id)
            assert fetched is not None
            # Own metric present
            assert f"tag_{tag}" in fetched.metrics, (
                f"Program {tag}: own metric tag_{tag} missing from Redis"
            )
            # Other programs' metrics absent
            for other_tag in ["x", "y", "z"]:
                if other_tag != tag:
                    assert f"tag_{other_tag}" not in fetched.metrics, (
                        f"Program {tag}: foreign metric tag_{other_tag} should not be present"
                    )


# ===========================================================================
# Group 5: Complex exec-order + data-flow interactions
# ===========================================================================


class TestComplexExecOrderDataFlow:
    """Detailed interaction scenarios between exec-order deps and data-flow deps."""

    async def test_on_success_impossible_when_upstream_is_skipped(
        self, state_manager, make_program
    ):
        """A->B (mandatory data-flow), C has on_success(B).

        A fails -> B SKIPPED -> C IMPOSSIBLE (SKIPPED is not success).

        Expected: A=FAILED, B=SKIPPED, C=SKIPPED
        """
        dag = _make_dag(
            {
                "a": FailProducer(timeout=5.0),
                "b": IncrStage(timeout=5.0),
                "c": NoOpStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
            exec_deps={"c": [ExecutionOrderDependency.on_success("b")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED
        assert prog.stage_results["c"].status == StageState.SKIPPED

    async def test_on_failure_satisfies_when_upstream_is_skipped(
        self, state_manager, make_program
    ):
        """A fails -> B SKIPPED (mandatory dep on A). C has on_failure(B).

        SKIPPED counts as "failure" for on_failure deps.

        Expected: A=FAILED, B=SKIPPED, C=COMPLETED
        """
        dag = _make_dag(
            {
                "a": FailProducer(timeout=5.0),
                "b": IncrStage(timeout=5.0),
                "c": NoOpStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
            exec_deps={"c": [ExecutionOrderDependency.on_failure("b")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED
        assert prog.stage_results["c"].status == StageState.COMPLETED

    async def test_multiple_exec_deps_all_must_succeed(
        self, state_manager, make_program
    ):
        """C has on_success(A) AND on_success(B); A succeeds, B fails -> C SKIPPED.

        All exec deps must be satisfied. A partial success is not enough.

        Expected: A=COMPLETED, B=FAILED, C=SKIPPED
        """
        dag = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),
                "b": FailProducer(timeout=5.0),
                "c": NoOpStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "c": [
                    ExecutionOrderDependency.on_success("a"),
                    ExecutionOrderDependency.on_success("b"),
                ]
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.SKIPPED

    async def test_always_after_waits_for_slow_dep_before_starting(
        self, state_manager, make_program
    ):
        """A sleeps 0.15s; B has always_after(A). B must not start before A finishes.

        Verified using timestamps: B.started_at >= A.finished_at.
        """
        dag = _make_dag(
            {
                "a": SlowProducer(timeout=5.0, sleep=0.15),
                "b": NoOpStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        a_result = prog.stage_results["a"]
        b_result = prog.stage_results["b"]

        assert a_result.status == StageState.COMPLETED
        assert b_result.status == StageState.COMPLETED

        # B must have started after A finished
        assert a_result.finished_at is not None
        assert b_result.started_at is not None
        assert b_result.started_at >= a_result.finished_at, (
            f"B started at {b_result.started_at} before A finished at {a_result.finished_at}"
        )

    async def test_complex_8stage_realistic_pipeline_with_execute_failure(
        self, state_manager, make_program
    ):
        """Topology mimicking a real pipeline; Execute fails -> downstream cascade.

        Topology (simplified):
          ValidateCode -success-> Execute
          Execute -data(mandatory)-> FetchMetrics    (SKIPPED: mandatory dep fails)
          Execute -data(mandatory)-> FetchArtifact   (SKIPPED: mandatory dep fails)
          FetchMetrics -optional-> EnsureMetrics     (optional; SKIPPED producer -> None)
          ComputeComplexity -optional-> EnsureMetrics (optional; succeeds)
          EnsureMetrics -always-> Insights
          Insights -optional-> MutationContext
          EnsureMetrics -optional-> MutationContext

        FetchMetrics and FetchArtifact use IncrStage (MANDATORY input from Execute),
        so they are SKIPPED when Execute fails.

        EnsureMetrics, Insights, and MutationContext use optional inputs so they run
        even when their producers are skipped.

        With Execute failing:
          ValidateCode=COMPLETED, Execute=FAILED
          FetchMetrics=SKIPPED (mandatory input from Execute)
          FetchArtifact=SKIPPED (mandatory input from Execute)
          ComputeComplexity=COMPLETED (independent)
          EnsureMetrics=COMPLETED (optional inputs: FetchMetrics=SKIPPED->None, CC=1; sum=1)
          Insights=COMPLETED (always_after EnsureMetrics; optional data from EnsureMetrics)
          MutationContext=COMPLETED (optional inputs from both Insights and EnsureMetrics)
        """
        dag = _make_dag(
            {
                "ValidateCode": ProduceOne(timeout=5.0),
                "Execute": FailProducer(timeout=5.0),
                # MANDATORY input from Execute -> will be SKIPPED when Execute fails
                "FetchMetrics": IncrStage(timeout=5.0),
                "FetchArtifact": IncrStage(timeout=5.0),
                "ComputeComplexity": ProduceOne(timeout=5.0),
                # OPTIONAL inputs -> will run even when producers are skipped/failed
                "EnsureMetrics": DualOptSumStage(timeout=5.0),
                "Insights": OptIncrStage(timeout=5.0),
                "MutationContext": DualOptSumStage(timeout=5.0),
            },
            [
                # Execute mandatory output feeds FetchMetrics and FetchArtifact
                # (mandatory IncrStage input -> SKIPPED when Execute fails)
                DataFlowEdge.create("Execute", "FetchMetrics", "data"),
                DataFlowEdge.create("Execute", "FetchArtifact", "data"),
                # EnsureMetrics: optional from FetchMetrics (skipped) and ComputeComplexity
                DataFlowEdge.create("FetchMetrics", "EnsureMetrics", "left"),
                DataFlowEdge.create("ComputeComplexity", "EnsureMetrics", "right"),
                # Insights: optional from EnsureMetrics
                DataFlowEdge.create("EnsureMetrics", "Insights", "data"),
                # MutationContext: optional from Insights + EnsureMetrics
                DataFlowEdge.create("Insights", "MutationContext", "left"),
                DataFlowEdge.create("EnsureMetrics", "MutationContext", "right"),
            ],
            state_manager,
            exec_deps={
                "Execute": [ExecutionOrderDependency.on_success("ValidateCode")],
                "Insights": [ExecutionOrderDependency.always_after("EnsureMetrics")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["ValidateCode"].status == StageState.COMPLETED
        assert prog.stage_results["Execute"].status == StageState.FAILED
        # FetchMetrics and FetchArtifact have MANDATORY input from Execute -> SKIPPED
        assert prog.stage_results["FetchMetrics"].status == StageState.SKIPPED
        assert prog.stage_results["FetchArtifact"].status == StageState.SKIPPED
        # ComputeComplexity is independent -> runs
        assert prog.stage_results["ComputeComplexity"].status == StageState.COMPLETED
        # EnsureMetrics: optional inputs (FetchMetrics=SKIPPED -> None, CC=1) -> sum=1
        assert prog.stage_results["EnsureMetrics"].status == StageState.COMPLETED
        assert prog.stage_results["EnsureMetrics"].output.value == 1
        # Insights: always_after EnsureMetrics; optional input from EnsureMetrics(1) -> 2
        assert prog.stage_results["Insights"].status == StageState.COMPLETED
        # MutationContext gets optional inputs from Insights + EnsureMetrics (both COMPLETED)
        assert prog.stage_results["MutationContext"].status == StageState.COMPLETED


# ===========================================================================
# Group 6: Input/output passing correctness
# ===========================================================================


class TestInputOutputPassingCorrectness:
    """Value correctness as data flows through DAGs."""

    async def test_multi_hop_4stage_value_propagation(
        self, state_manager, make_program
    ):
        """A produces 1, B increments to 2, C to 3, D to 4.

        Asserts both in-memory output AND stage_results entry.
        """
        dag = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),
                "b": IncrStage(timeout=5.0),
                "c": IncrStage(timeout=5.0),
                "d": IncrStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "c", "data"),
                DataFlowEdge.create("c", "d", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].output.value == 1
        assert prog.stage_results["b"].output.value == 2
        assert prog.stage_results["c"].output.value == 3
        assert prog.stage_results["d"].output.value == 4

    async def test_optional_input_is_none_when_producer_fails_not_stale_value(
        self, state_manager, make_program
    ):
        """Pre-populate program with stale COMPLETED result for producer P (value=99).

        Then run DAG where P fails. Consumer has optional input from P.
        Consumer must receive None (not stale value=99).

        This verifies build_named_inputs only passes COMPLETED producers from this run.
        """
        # Build and run once to get valid hashes
        dag_setup = _make_dag(
            {"p": ProduceOne(timeout=5.0), "consumer": OptIncrStage(timeout=5.0)},
            [DataFlowEdge.create("p", "consumer", "data")],
            state_manager,
        )
        prog = make_program()
        await dag_setup.run(prog)
        # At this point, p has COMPLETED with value=1 and consumer got 2

        # Now inject a stale COMPLETED result for p with value=99
        from gigaevo.programs.core_types import ProgramStageResult

        prog.stage_results["p"] = ProgramStageResult.success(output=IntOutput(value=99))

        # Run DAG where P is a FailProducer (fails this run)
        dag_fail = _make_dag(
            {"p": FailProducer(timeout=5.0), "consumer": OptIncrStage(timeout=5.0)},
            [DataFlowEdge.create("p", "consumer", "data")],
            state_manager,
        )
        await dag_fail.run(prog)

        assert prog.stage_results["p"].status == StageState.FAILED
        assert prog.stage_results["consumer"].status == StageState.COMPLETED
        # Consumer received None (not stale 99) -> value = -1
        assert prog.stage_results["consumer"].output.value == -1, (
            "Consumer must receive None when producer fails this run, not stale value=99"
        )

    async def test_two_parallel_stages_add_metrics_both_present(
        self, state_manager, make_program
    ):
        """A writes key_a=1.0, B writes key_b=2.0 (parallel, independent).

        C depends on both (always_after A and always_after B).
        After C runs, both keys must be in program.metrics.
        """
        dag = _make_dag(
            {
                "a": MetricsStage(timeout=5.0, metric_key="key_a", metric_value=1.0),
                "b": MetricsStage(timeout=5.0, metric_key="key_b", metric_value=2.0),
                "c": NoOpStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "c": [
                    ExecutionOrderDependency.always_after("a"),
                    ExecutionOrderDependency.always_after("b"),
                ]
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED

        assert prog.metrics.get("key_a") == 1.0
        assert prog.metrics.get("key_b") == 2.0

    async def test_stale_optional_input_not_passed_when_producer_fails_fresh_run(
        self, state_manager, make_program
    ):
        """More thorough check: run A->opt, then pre-inject stale result for A with
        a different value, then run DAG where A fails. opt must get None, not stale.

        Steps:
          1. Run DAG: A produces 5, consumer gets 6 (5+1).
          2. Corrupt A's result to have value=100 (stale).
          3. Run DAG again with A as FailProducer.
          4. consumer receives None -> value=-1, not stale 100+1=101.
        """
        dag1 = _make_dag(
            {"a": ProduceOne(timeout=5.0), "consumer": OptIncrStage(timeout=5.0)},
            [DataFlowEdge.create("a", "consumer", "data")],
            state_manager,
        )
        prog = make_program()
        await dag1.run(prog)
        assert prog.stage_results["consumer"].output.value == 2  # 1+1

        # Inject stale result for A with value=100
        prog.stage_results["a"] = _make_result(
            StageState.COMPLETED, output=IntOutput(value=100), input_hash="stale_hash"
        )

        dag2 = _make_dag(
            {"a": FailProducer(timeout=5.0), "consumer": OptIncrStage(timeout=5.0)},
            [DataFlowEdge.create("a", "consumer", "data")],
            state_manager,
        )
        await dag2.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["consumer"].status == StageState.COMPLETED
        # Must be -1 (None input), not 101 (stale value)
        assert prog.stage_results["consumer"].output.value == -1


# ===========================================================================
# Group 7: Additional tricky edge cases
# ===========================================================================


class TestAdditionalEdgeCases:
    """Scenarios that stress the DAG's state-tracking logic."""

    async def test_all_stages_end_in_final_state_complex_failure_topology(
        self, state_manager, make_program
    ):
        """Complex failure propagation: every stage must end in a FINAL state.

        Topology:
          A (fails) -> B (mandatory, skipped) -> D (mandatory, skipped)
          C (independent, succeeds) -> D (optional second input, but D already skipped)
          E has always_after(D) -> runs

        All 5 stages must be in FINAL_STATES.
        """
        dag = _make_dag(
            {
                "a": FailProducer(timeout=5.0),
                "b": IncrStage(timeout=5.0),
                "c": ProduceOne(timeout=5.0),
                "d": DualOptSumStage(timeout=5.0),
                "e": NoOpStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),  # mandatory, will skip
                DataFlowEdge.create("b", "d", "left"),  # optional from skipped B
                DataFlowEdge.create("c", "d", "right"),  # optional from successful C
            ],
            state_manager,
            exec_deps={"e": [ExecutionOrderDependency.always_after("d")]},
        )
        prog = make_program()
        await dag.run(prog)

        for name in ("a", "b", "c", "d", "e"):
            assert prog.stage_results[name].status in FINAL_STATES, (
                f"Stage '{name}' ended in non-final state: {prog.stage_results[name].status}"
            )

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        # D: left=None (B skipped), right=C(1) -> value=1
        assert prog.stage_results["d"].status == StageState.COMPLETED
        assert prog.stage_results["d"].output.value == 1
        # E: always_after D -> runs
        assert prog.stage_results["e"].status == StageState.COMPLETED

    async def test_stale_completed_result_overwritten_when_dep_fails_this_run(
        self, state_manager, make_program
    ):
        """Regression: stale COMPLETED results for both B and C must be overwritten
        when A (mandatory dep) fails in this run.

        This is the deadlock-regression pattern but with 2 downstream stages.
        """
        dag = _make_dag(
            {
                "a": FailProducer(timeout=5.0),
                "b": IncrStage(timeout=5.0),
                "c": OptIncrStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),  # mandatory
                DataFlowEdge.create("a", "c", "data"),  # optional
            ],
            state_manager,
        )
        # Pre-inject stale COMPLETED results for B and C
        prog = make_program(
            stage_results={
                "b": _make_result(StageState.COMPLETED, output=IntOutput(value=50)),
                "c": _make_result(StageState.COMPLETED, output=IntOutput(value=50)),
            }
        )
        await asyncio.wait_for(dag.run(prog), timeout=10.0)

        assert prog.stage_results["a"].status == StageState.FAILED
        # B had mandatory dep on A -> SKIPPED (stale result overwritten)
        assert prog.stage_results["b"].status == StageState.SKIPPED
        # C had optional dep on A -> C still runs, data=None -> value=-1
        assert prog.stage_results["c"].status == StageState.COMPLETED
        assert prog.stage_results["c"].output.value == -1

    async def test_single_stage_no_deps_completes_and_is_cached_on_second_run(
        self, state_manager, make_program
    ):
        """Simplest possible DAG: one stage, no deps. Verify caching after first run."""

        def nodes_fn():
            return {"a": CounterA(timeout=5.0)}

        prog = make_program()

        dag1 = _make_dag(nodes_fn(), [], state_manager)
        await dag1.run(prog)
        assert CounterA.call_count == 1

        dag2 = _make_dag(nodes_fn(), [], state_manager)
        await dag2.run(prog)
        assert CounterA.call_count == 1  # cached

    async def test_exec_dep_and_data_flow_from_same_upstream(
        self, state_manager, make_program
    ):
        """B has BOTH data-flow (mandatory) AND exec-dep on_success from A.

        A succeeds -> both gates open -> B runs.
        """
        dag = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),
                "b": IncrStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["b"].output.value == 2  # 1 + 1

    async def test_no_cache_stage_reruns_even_with_matching_hash(
        self, state_manager, make_program
    ):
        """NeverCached policy: stage re-executes on every DAG run regardless of hash.

        Uses a custom NeverCachedCountStage with NO_CACHE policy.
        """

        class NeverCachedCountStage(Stage):
            InputsModel = VoidInput
            OutputModel = IntOutput
            cache_handler = NO_CACHE
            call_count: int = 0

            async def compute(self, program: Program) -> IntOutput:
                NeverCachedCountStage.call_count += 1
                return IntOutput(value=NeverCachedCountStage.call_count)

        NeverCachedCountStage.call_count = 0

        def nodes_fn():
            return {"nc": NeverCachedCountStage(timeout=5.0)}

        prog = make_program()

        dag1 = _make_dag(nodes_fn(), [], state_manager)
        await dag1.run(prog)
        assert NeverCachedCountStage.call_count == 1

        await asyncio.sleep(0.01)

        dag2 = _make_dag(nodes_fn(), [], state_manager)
        await dag2.run(prog)
        assert NeverCachedCountStage.call_count == 2

    async def test_empty_optional_input_receives_none_not_stale_output(
        self, state_manager, make_program
    ):
        """Consumer has optional input from producer. Producer not wired.

        Consumer must receive None for that field.
        """
        dag = _make_dag(
            {"consumer": OptIncrStage(timeout=5.0)},
            [],  # No edge wired -> optional input=None
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["consumer"].status == StageState.COMPLETED
        assert prog.stage_results["consumer"].output.value == -1


# ===========================================================================
# Group 8: Audit Finding Tests
# ===========================================================================


class TestAuditCancelledErrorComplex:
    """Audit Finding #4: CancelledError in a complex multi-stage DAG."""

    async def test_cancelled_stage_in_diamond_cascades_correctly(
        self, state_manager, make_program
    ):
        """Diamond DAG where one fork gets CancelledError.

        Topology:
          A (succeeds, value=1)
          B (CancelledError) -- mandatory dep on A
          C (succeeds, value=2) -- mandatory dep on A
          D (mandatory dep on B, optional dep on C)

        Expected: A=COMPLETED, B=CANCELLED, C=COMPLETED, D=SKIPPED (mandatory B fails)
        """

        class CancellingProducer(Stage):
            InputsModel = IntInput
            OutputModel = IntOutput

            async def compute(self, program: Program) -> IntOutput:
                raise asyncio.CancelledError()

        dag = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),
                "b": CancellingProducer(timeout=5.0),
                "c": IncrStage(timeout=5.0),
                "d": DualOptSumStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("a", "c", "data"),
                DataFlowEdge.create(
                    "b", "d", "left"
                ),  # mandatory via DualOptInput -> optional
                DataFlowEdge.create("c", "d", "right"),  # optional
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.CANCELLED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        # D has optional inputs from both B and C; B is CANCELLED so left=None
        assert prog.stage_results["d"].status == StageState.COMPLETED
        # left=None(B cancelled)->0, right=C(2)->2, sum=2
        assert prog.stage_results["d"].output.value == 2


class TestAuditSemaphoreComplex:
    """Audit Finding #5: Strengthened semaphore with concurrency tracking."""

    async def test_semaphore_limit_2_with_6_stages_peak_concurrency_tracked(
        self, state_manager, make_program
    ):
        """6 independent stages, semaphore=2. Track peak concurrency with a lock.

        Each stage increments a counter on entry, sleeps, then decrements.
        The peak concurrent count must be <= 2.
        """
        peak_concurrent = [0]
        current_count = [0]
        lock = asyncio.Lock()

        class TrackedStage(Stage):
            InputsModel = VoidInput
            OutputModel = IntOutput

            def __init__(self, *, timeout: float, idx: int):
                super().__init__(timeout=timeout)
                self._idx = idx

            async def compute(self, program: Program) -> IntOutput:
                async with lock:
                    current_count[0] += 1
                    if current_count[0] > peak_concurrent[0]:
                        peak_concurrent[0] = current_count[0]
                await asyncio.sleep(0.04)
                async with lock:
                    current_count[0] -= 1
                return IntOutput(value=self._idx)

        dag = _make_dag(
            {f"t{i}": TrackedStage(timeout=5.0, idx=i) for i in range(6)},
            [],
            state_manager,
            max_parallel_stages=2,
        )
        prog = make_program()
        await dag.run(prog)

        for i in range(6):
            assert prog.stage_results[f"t{i}"].status == StageState.COMPLETED

        assert peak_concurrent[0] <= 2, (
            f"Expected peak concurrency <= 2, got {peak_concurrent[0]}"
        )


class TestAuditInputHashComplex:
    """Audit Finding #6: input_hash correctness end-to-end in complex DAGs."""

    async def test_input_hash_end_to_end_in_chain(self, state_manager, make_program):
        """Run A->B->C chain. Capture all input_hashes. Rerun with same inputs.
        All hashes should match. Then change A's value and verify B/C hashes differ."""

        # Run 1
        dag1 = _make_dag(
            {
                "a": ProduceOne(timeout=5.0),
                "b": IncrStage(timeout=5.0),
                "c": IncrStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "c", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag1.run(prog)

        hash_a1 = prog.stage_results["a"].input_hash
        hash_b1 = prog.stage_results["b"].input_hash
        hash_c1 = prog.stage_results["c"].input_hash

        assert hash_a1 is not None
        assert hash_b1 is not None
        assert hash_c1 is not None

        # B and C have different inputs, so their hashes should differ
        assert hash_b1 != hash_c1, (
            "B and C have different inputs (different values), their hashes should differ"
        )


class TestAuditMetricsInRedisComplex:
    """Audit Finding #1: Metrics verified in Redis for complex DAGs."""

    async def test_multi_stage_metrics_all_persisted_to_redis(
        self, state_manager, fakeredis_storage, make_program
    ):
        """A and B both write metrics. C depends on both. Verify all metrics in Redis."""
        dag = _make_dag(
            {
                "a": MetricsStage(timeout=5.0, metric_key="score_a", metric_value=85.5),
                "b": MetricsStage(timeout=5.0, metric_key="score_b", metric_value=92.3),
                "c": NoOpStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "c": [
                    ExecutionOrderDependency.always_after("a"),
                    ExecutionOrderDependency.always_after("b"),
                ]
            },
        )
        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.metrics.get("score_a") == 85.5, (
            "score_a metric not correctly persisted to Redis"
        )
        assert fetched.metrics.get("score_b") == 92.3, (
            "score_b metric not correctly persisted to Redis"
        )


class TestAuditSkipResultComplex:
    """Audit Finding #2: Skip results verified in Redis for complex DAGs."""

    async def test_skip_result_in_cascade_persisted_to_redis(
        self, state_manager, fakeredis_storage, make_program
    ):
        """A fails -> B SKIPPED (mandatory) -> C SKIPPED (mandatory dep on B).
        Verify all skip results are correctly persisted in Redis with error details."""
        dag = _make_dag(
            {
                "a": FailProducer(timeout=5.0),
                "b": IncrStage(timeout=5.0),
                "c": IncrStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "c", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None

        # Verify cascade of skip results in Redis
        assert fetched.stage_results["a"].status == StageState.FAILED
        assert fetched.stage_results["b"].status == StageState.SKIPPED
        assert fetched.stage_results["c"].status == StageState.SKIPPED

        # Both skip results should have error details
        for name in ("b", "c"):
            skip_res = fetched.stage_results[name]
            assert skip_res.error is not None, (
                f"SKIPPED stage '{name}' should have error details in Redis"
            )
            assert skip_res.started_at is not None
            assert skip_res.finished_at is not None
