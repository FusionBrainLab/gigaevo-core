"""Integration tests: full metrics pipeline DAG (EnsureMetrics → NormalizeMetrics).

These tests exercise the two-stage metrics pipeline as it runs in production:
  - EnsureMetricsStage populates missing/invalid metrics with sentinels, clamps values
  - NormalizeMetricsStage normalizes bounded metrics to [0, 1]
  - DAG DataFlowEdge carries the FloatDictContainer output from EnsureMetrics into
    NormalizeMetrics (via the "candidate" input field)

Key scenarios tested:
  1. Happy path: good upstream metrics → EnsureMetrics passes them through → NormalizeMetrics
     produces correct normalized values
  2. Missing upstream metrics → EnsureMetrics fills sentinels → NormalizeMetrics skips
     sentinel values that are outside bounds (or handles them via clamp)
  3. NormalizeMetrics running WITHOUT EnsureMetrics (DAG misconfiguration):
     verifies the fixed skip behavior (no KeyError crash)
  4. Boundary values (at/beyond bounds) are correctly clamped in normalized space
  5. Multi-metric pipeline: primary + secondary metrics flow through together
"""

from __future__ import annotations

import pytest

from gigaevo.programs.core_types import StageState, VoidInput
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.metrics import EnsureMetricsStage, NormalizeMetricsStage
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> MetricsContext:
    """Score [0, 100] higher-is-better primary; cost [0, 50] lower-is-better secondary."""
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="primary score",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=100.0,
                sentinel_value=-1.0,
            ),
            "cost": MetricSpec(
                description="secondary cost",
                is_primary=False,
                higher_is_better=False,
                lower_bound=0.0,
                upper_bound=50.0,
                sentinel_value=1e5,
            ),
        }
    )


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


def _ensure(ctx: MetricsContext) -> EnsureMetricsStage:
    s = EnsureMetricsStage(
        metrics_factory=ctx.get_sentinels(),
        metrics_context=ctx,
        timeout=5.0,
    )
    s.__class__.cache_handler = NO_CACHE
    return s


def _normalize(
    ctx: MetricsContext, aggregate_key: str = "normalized_score"
) -> NormalizeMetricsStage:
    s = NormalizeMetricsStage(
        metrics_context=ctx,
        aggregate_key=aggregate_key,
        timeout=5.0,
    )
    s.__class__.cache_handler = NO_CACHE
    return s


# ---------------------------------------------------------------------------
# A tiny FakeInput stage that injects metrics into the pipeline via output
# (simulates what validate.py + FetchArtifact produce in the real pipeline).
# ---------------------------------------------------------------------------


class MetricsProducerOutput(FloatDictContainer):
    pass


class MetricsProducerStage(Stage):
    """Injects a fixed dict as FloatDictContainer for downstream stages."""

    InputsModel = VoidInput
    OutputModel = MetricsProducerOutput
    cache_handler = NO_CACHE

    def __init__(self, metrics: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self._metrics = metrics

    async def compute(self, program: Program) -> MetricsProducerOutput:
        return MetricsProducerOutput(data=self._metrics)


# ---------------------------------------------------------------------------
# 1. Happy path: full EnsureMetrics → NormalizeMetrics pipeline via DAG
# ---------------------------------------------------------------------------


class TestFullMetricsPipelineDAG:
    async def test_good_metrics_flow_through_pipeline(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """EnsureMetrics → NormalizeMetrics in a DAG with good upstream metrics."""
        ctx = _make_ctx()
        producer = MetricsProducerStage({"score": 75.0, "cost": 25.0}, timeout=5.0)
        ensure = _ensure(ctx)
        normalize = _normalize(ctx)

        dag = DAG(
            nodes={"producer": producer, "ensure": ensure, "normalize": normalize},
            data_flow_edges=[
                DataFlowEdge.create("producer", "ensure", "candidate"),
            ],
            execution_order_deps={
                "normalize": [ExecutionOrderDependency.on_success("ensure")]
            },
            state_manager=state_manager,
            writer=NullWriter(),
        )

        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        # EnsureMetrics should have stored clamped metrics
        assert prog.metrics["score"] == pytest.approx(75.0)
        assert prog.metrics["cost"] == pytest.approx(25.0)

        # NormalizeMetrics should have produced normalized values
        # score=75 in [0,100] higher_is_better → 0.75
        assert prog.metrics["score_norm"] == pytest.approx(0.75)
        # cost=25 in [0,50] lower_is_better → (25/50)=0.5 flipped → 0.5
        assert prog.metrics["cost_norm"] == pytest.approx(0.5)
        # Aggregate: mean(0.75, 0.5) = 0.625
        assert prog.metrics["normalized_score"] == pytest.approx(0.625)

    async def test_pipeline_stage_results_recorded(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """Both stages record their results in program.stage_results after DAG run."""
        ctx = _make_ctx()
        producer = MetricsProducerStage({"score": 50.0, "cost": 10.0}, timeout=5.0)
        ensure = _ensure(ctx)
        normalize = _normalize(ctx)

        dag = DAG(
            nodes={"producer": producer, "ensure": ensure, "normalize": normalize},
            data_flow_edges=[DataFlowEdge.create("producer", "ensure", "candidate")],
            execution_order_deps={
                "normalize": [ExecutionOrderDependency.on_success("ensure")]
            },
            state_manager=state_manager,
            writer=NullWriter(),
        )

        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        assert prog.stage_results["ensure"].status == StageState.COMPLETED
        assert prog.stage_results["normalize"].status == StageState.COMPLETED

    async def test_metrics_at_boundaries_clamped(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """Values outside [lo, hi] are clamped by EnsureMetrics then normalized to 0/1."""
        ctx = _make_ctx()
        # score=200 > 100 (upper bound) → clamped to 100 → normalized to 1.0
        # cost=-10 < 0 (lower bound) → clamped to 0 → lower_is_better → flipped to 1.0
        producer = MetricsProducerStage({"score": 200.0, "cost": -10.0}, timeout=5.0)
        ensure = _ensure(ctx)
        normalize = _normalize(ctx)

        dag = DAG(
            nodes={"producer": producer, "ensure": ensure, "normalize": normalize},
            data_flow_edges=[DataFlowEdge.create("producer", "ensure", "candidate")],
            execution_order_deps={
                "normalize": [ExecutionOrderDependency.on_success("ensure")]
            },
            state_manager=state_manager,
            writer=NullWriter(),
        )

        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        assert prog.metrics["score_norm"] == pytest.approx(1.0)
        assert prog.metrics["cost_norm"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. NormalizeMetrics WITHOUT EnsureMetrics (DAG misconfiguration)
#    Fixed: missing metrics are silently skipped (no KeyError crash)
# ---------------------------------------------------------------------------


class TestNormalizeWithoutEnsure:
    async def test_normalize_alone_skips_missing_metrics(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """NormalizeMetrics without EnsureMetrics: missing keys are skipped, not KeyError.

        This is a regression test for the bug fix: program.metrics.get(key) with
        continue replaces the bare program.metrics[key] that raised KeyError.
        """
        ctx = _make_ctx()
        normalize = _normalize(ctx)

        dag = DAG(
            nodes={"normalize": normalize},
            data_flow_edges=[],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=NullWriter(),
        )

        # Program has NO metrics at all (EnsureMetrics never ran)
        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)  # Must NOT raise KeyError

        # Normalize stage completed (no crash)
        assert prog.stage_results["normalize"].status == StageState.COMPLETED
        # No normalized keys added (all skipped)
        assert "score_norm" not in prog.metrics
        assert "cost_norm" not in prog.metrics

    async def test_normalize_partial_metrics_skips_missing(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """With only some metrics present, NormalizeMetrics normalizes what it can."""
        ctx = _make_ctx()
        normalize = _normalize(ctx)

        dag = DAG(
            nodes={"normalize": normalize},
            data_flow_edges=[],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=NullWriter(),
        )

        # Program has only 'score', missing 'cost'
        prog = make_program(metrics={"score": 80.0})
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        assert prog.stage_results["normalize"].status == StageState.COMPLETED
        # score was present → normalized
        assert "score_norm" in prog.metrics
        assert prog.metrics["score_norm"] == pytest.approx(0.8)
        # cost was absent → silently skipped
        assert "cost_norm" not in prog.metrics
        # Aggregate is based on whatever was normalized (just score_norm)
        assert prog.metrics["normalized_score"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 3. Sentinel value handling
# ---------------------------------------------------------------------------


class TestSentinelValueHandling:
    async def test_sentinel_value_preserved_not_clamped(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """EnsureMetrics preserves sentinel values (not clamped to [lo, hi])."""
        ctx = _make_ctx()
        # score sentinel = -1.0, which is outside [0, 100] but must not be clamped
        producer = MetricsProducerStage({"score": -1.0, "cost": 1e5}, timeout=5.0)
        ensure = _ensure(ctx)

        dag = DAG(
            nodes={"producer": producer, "ensure": ensure},
            data_flow_edges=[DataFlowEdge.create("producer", "ensure", "candidate")],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=NullWriter(),
        )

        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        # Sentinel values are preserved by EnsureMetrics
        assert prog.metrics["score"] == pytest.approx(-1.0)
        assert prog.metrics["cost"] == pytest.approx(1e5)

    async def test_pipeline_with_sentinel_input(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """Sentinel → EnsureMetrics preserves it → NormalizeMetrics skips it.

        Regression (H1 fix): NormalizeMetricsStage must NOT normalize sentinel
        values.  Prior to the fix, score=-1.0 (sentinel) would be normalized
        to 0.0 via clamp((-1-0)/100, 0, 1), making a failed run
        indistinguishable from a zero-score run in MAP-Elites selection.
        """
        ctx = _make_ctx()
        # score sentinel = -1.0 → EnsureMetrics preserves it
        # NormalizeMetrics must skip it (no score_norm key emitted)
        producer = MetricsProducerStage({"score": -1.0, "cost": 10.0}, timeout=5.0)
        ensure = _ensure(ctx)
        normalize = _normalize(ctx)

        dag = DAG(
            nodes={"producer": producer, "ensure": ensure, "normalize": normalize},
            data_flow_edges=[DataFlowEdge.create("producer", "ensure", "candidate")],
            execution_order_deps={
                "normalize": [ExecutionOrderDependency.on_success("ensure")]
            },
            state_manager=state_manager,
            writer=NullWriter(),
        )

        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        # score=-1.0 is the sentinel → NormalizeMetrics skips it, no score_norm key
        assert "score_norm" not in prog.metrics, (
            "Sentinel was normalised to 0.0 (H1 regression)"
        )
        # cost=10.0 is not a sentinel → still normalised normally
        assert "cost_norm" in prog.metrics
