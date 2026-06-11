"""Integration tests: metrics pipeline DAG around EnsureMetricsStage.

These tests exercise the metrics stage as it runs in production:
  - EnsureMetricsStage populates missing/invalid metrics with sentinels, clamps values
  - DAG DataFlowEdge carries the FloatDictContainer output from an upstream producer
    into EnsureMetrics (via the "candidate" input field)

Key scenarios tested:
  1. Happy path: good upstream metrics → EnsureMetrics passes them through
  2. Boundary values (at/beyond bounds) are correctly clamped
  3. Sentinel values are preserved (not clamped)
"""

from __future__ import annotations

import pytest

from gigaevo.programs.core_types import StageState, VoidInput
from gigaevo.programs.dag.automata import DataFlowEdge
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.metrics import EnsureMetricsStage
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


def _ensure(ctx: MetricsContext) -> EnsureMetricsStage:
    s = EnsureMetricsStage(
        metrics_factory=ctx.get_sentinels(),
        metrics_context=ctx,
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
# 1. Happy path: producer → EnsureMetrics pipeline via DAG
# ---------------------------------------------------------------------------


class TestMetricsPipelineDAG:
    async def test_good_metrics_flow_through_pipeline(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """Producer → EnsureMetrics in a DAG with good upstream metrics."""
        ctx = _make_ctx()
        producer = MetricsProducerStage({"score": 75.0, "cost": 25.0}, timeout=5.0)
        ensure = _ensure(ctx)

        dag = DAG(
            nodes={"producer": producer, "ensure": ensure},
            data_flow_edges=[
                DataFlowEdge.create("producer", "ensure", "candidate"),
            ],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=NullWriter(),
        )

        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        assert prog.metrics["score"] == pytest.approx(75.0)
        assert prog.metrics["cost"] == pytest.approx(25.0)
        assert prog.stage_results["ensure"].status == StageState.COMPLETED

    async def test_metrics_at_boundaries_clamped(
        self, state_manager, fakeredis_storage, make_program
    ) -> None:
        """Values outside [lo, hi] are clamped by EnsureMetrics."""
        ctx = _make_ctx()
        # score=200 > 100 (upper bound) → clamped to 100
        # cost=-10 < 0 (lower bound) → clamped to 0
        producer = MetricsProducerStage({"score": 200.0, "cost": -10.0}, timeout=5.0)
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

        assert prog.metrics["score"] == pytest.approx(100.0)
        assert prog.metrics["cost"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Sentinel value handling
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
