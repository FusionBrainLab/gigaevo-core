"""Tests for EnsureMetricsStage and NormalizeMetricsStage."""

from __future__ import annotations

import pytest

from gigaevo.programs.core_types import StageState
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.metrics import EnsureMetricsStage, NormalizeMetricsStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(specs: dict | None = None) -> MetricsContext:
    if specs is not None:
        return MetricsContext(specs=specs)
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
            ),
        }
    )


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


def _make_ensure_stage(
    ctx: MetricsContext | None = None,
    factory: dict[str, float] | None = None,
) -> EnsureMetricsStage:
    ctx = ctx or _make_ctx()
    factory = factory or ctx.get_sentinels()
    stage = EnsureMetricsStage(
        metrics_factory=factory,
        metrics_context=ctx,
        timeout=5.0,
    )
    # Override cache handler so tests always execute
    stage.__class__.cache_handler = NO_CACHE
    return stage


def _make_normalize_stage(ctx: MetricsContext | None = None) -> NormalizeMetricsStage:
    ctx = ctx or _make_ctx()
    stage = NormalizeMetricsStage(
        metrics_context=ctx,
        timeout=5.0,
    )
    stage.__class__.cache_handler = NO_CACHE
    return stage


# ---------------------------------------------------------------------------
# TestEnsureMetricsStage
# ---------------------------------------------------------------------------


class TestEnsureMetricsStage:
    async def test_candidate_metrics_used_when_provided(self):
        """Candidate input with score=80 → output has score=80."""
        stage = _make_ensure_stage()
        stage.attach_inputs(
            {"candidate": FloatDictContainer(data={"score": 80.0, "cost": 10.0})}
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert prog.metrics["score"] == 80.0
        assert prog.metrics["cost"] == 10.0

    async def test_factory_fallback_when_no_candidate(self):
        """No candidate input → factory metrics used (sentinels)."""
        ctx = _make_ctx()
        stage = _make_ensure_stage(ctx=ctx)
        stage.attach_inputs({"candidate": None})
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        # Factory returns sentinels
        assert prog.metrics["score"] == ctx.get_sentinels()["score"]

    async def test_callable_factory(self):
        """Factory is a lambda → called correctly."""
        ctx = _make_ctx()

        def factory_fn():
            return {"score": 42.0, "cost": 5.0}

        stage = EnsureMetricsStage(
            metrics_factory=factory_fn,
            metrics_context=ctx,
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"candidate": None})
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert prog.metrics["score"] == 42.0

    async def test_non_finite_raises(self):
        """Candidate with score=inf → stage FAILED."""
        stage = _make_ensure_stage()
        stage.attach_inputs(
            {
                "candidate": FloatDictContainer(
                    data={"score": float("inf"), "cost": 10.0}
                )
            }
        )
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "finite" in result.error.message.lower()

    async def test_nan_raises(self):
        """Candidate with score=NaN → stage FAILED."""
        stage = _make_ensure_stage()
        stage.attach_inputs(
            {
                "candidate": FloatDictContainer(
                    data={"score": float("nan"), "cost": 10.0}
                )
            }
        )
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED

    async def test_sentinel_preserved_not_clamped(self):
        """score=-1.0 (sentinel) → preserved, not clamped to lo=0."""
        stage = _make_ensure_stage()
        stage.attach_inputs(
            {"candidate": FloatDictContainer(data={"score": -1.0, "cost": 10.0})}
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert prog.metrics["score"] == -1.0  # sentinel, not clamped to 0

    async def test_value_clamped_to_upper_bound(self):
        """score=200 → clamped to hi=100."""
        stage = _make_ensure_stage()
        stage.attach_inputs(
            {"candidate": FloatDictContainer(data={"score": 200.0, "cost": 10.0})}
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert prog.metrics["score"] == 100.0

    async def test_value_clamped_to_lower_bound(self):
        """score=-50 (not sentinel) → clamped to lo=0."""
        # Use a context where sentinel is very different from -50
        ctx = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="s",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=100.0,
                    sentinel_value=-999.0,
                ),
            }
        )
        stage = _make_ensure_stage(ctx=ctx, factory=ctx.get_sentinels())
        stage.attach_inputs({"candidate": FloatDictContainer(data={"score": -50.0})})
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert prog.metrics["score"] == 0.0

    async def test_missing_required_key_raises(self):
        """Candidate missing 'score' key → stage FAILED."""
        stage = _make_ensure_stage()
        # Only provide 'cost', not 'score'
        stage.attach_inputs({"candidate": FloatDictContainer(data={"cost": 10.0})})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert (
            "Missing" in result.error.message
            or "missing" in result.error.message.lower()
        )

    async def test_sentinel_metrics_written_as_safety_net(self):
        """Even when candidate processing fails, sentinel values are on program."""
        stage = _make_ensure_stage()
        # Trigger failure with inf
        stage.attach_inputs(
            {
                "candidate": FloatDictContainer(
                    data={"score": float("inf"), "cost": 10.0}
                )
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.FAILED
        # Sentinel metrics should still be written (safety net)
        assert "score" in prog.metrics
        assert "cost" in prog.metrics


# ---------------------------------------------------------------------------
# TestNormalizeMetricsStage
# ---------------------------------------------------------------------------


class TestNormalizeMetricsStage:
    async def test_normalizes_to_01(self):
        """score=50, lo=0, hi=100 → score_norm=0.5."""
        stage = _make_normalize_stage()
        stage.attach_inputs({})
        prog = _prog()
        prog.add_metrics({"score": 50.0, "cost": 25.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert prog.metrics["score_norm"] == pytest.approx(0.5)

    async def test_flips_lower_is_better(self):
        """cost=25, lo=0, hi=50, higher_is_better=False → cost_norm=0.5."""
        stage = _make_normalize_stage()
        stage.attach_inputs({})
        prog = _prog()
        prog.add_metrics({"score": 50.0, "cost": 25.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        # cost: ratio = (25-0)/(50-0) = 0.5, flipped: 1.0 - 0.5 = 0.5
        assert prog.metrics["cost_norm"] == pytest.approx(0.5)

    async def test_lower_is_better_high_value(self):
        """cost=0 (best), lo=0, hi=50, higher_is_better=False → cost_norm=1.0."""
        stage = _make_normalize_stage()
        stage.attach_inputs({})
        prog = _prog()
        prog.add_metrics({"score": 50.0, "cost": 0.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        # cost: ratio = 0/50 = 0.0, flipped: 1.0 - 0.0 = 1.0
        assert prog.metrics["cost_norm"] == pytest.approx(1.0)

    async def test_clamp_above_1(self):
        """score=150, lo=0, hi=100 → score_norm=1.0."""
        stage = _make_normalize_stage()
        stage.attach_inputs({})
        prog = _prog()
        prog.add_metrics({"score": 150.0, "cost": 25.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert prog.metrics["score_norm"] == pytest.approx(1.0)

    async def test_clamp_below_0(self):
        """score=-10, lo=0, hi=100 → score_norm=0.0."""
        stage = _make_normalize_stage()
        stage.attach_inputs({})
        prog = _prog()
        prog.add_metrics({"score": -10.0, "cost": 25.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert prog.metrics["score_norm"] == pytest.approx(0.0)

    async def test_aggregate_key_computed(self):
        """2 metrics normalized → aggregate = mean of both."""
        stage = _make_normalize_stage()
        stage.attach_inputs({})
        prog = _prog()
        prog.add_metrics({"score": 50.0, "cost": 25.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        # score_norm=0.5, cost_norm=0.5 → aggregate = 0.5
        assert "normalized_score" in prog.metrics
        expected_agg = (prog.metrics["score_norm"] + prog.metrics["cost_norm"]) / 2
        assert prog.metrics["normalized_score"] == pytest.approx(expected_agg)

    async def test_missing_metric_key_silently_skipped(self):
        """Program.metrics missing 'score' → silently skipped, stage succeeds.

        Regression test: previous implementation used bare dict[key] access which
        raised KeyError. Fixed to use .get(key) with continue, so missing metrics
        are skipped rather than crashing the stage.
        """
        stage = _make_normalize_stage()
        stage.attach_inputs({})
        prog = _prog()
        # Only cost, no score
        prog.add_metrics({"cost": 25.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        # cost was present → normalized
        assert "cost_norm" in prog.metrics
        # score was absent → silently skipped
        assert "score_norm" not in prog.metrics

    async def test_no_bounds_skipped(self):
        """Metric with lo=None, hi=None → not in output."""
        ctx = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="primary",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=100.0,
                ),
                "unbounded": MetricSpec(
                    description="no bounds",
                    is_primary=False,
                    higher_is_better=True,
                    lower_bound=None,
                    upper_bound=None,
                ),
            }
        )
        stage = _make_normalize_stage(ctx=ctx)
        stage.attach_inputs({})
        prog = _prog()
        prog.add_metrics({"score": 50.0, "unbounded": 99.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert "score_norm" in prog.metrics
        assert "unbounded_norm" not in prog.metrics

    async def test_equal_bounds_skipped(self):
        """Metric with lo=10, hi=10 → not normalized (hi <= lo)."""
        ctx = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="primary",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=100.0,
                ),
                "flat": MetricSpec(
                    description="equal bounds",
                    is_primary=False,
                    higher_is_better=True,
                    lower_bound=10.0,
                    upper_bound=10.0,
                ),
            }
        )
        stage = _make_normalize_stage(ctx=ctx)
        stage.attach_inputs({})
        prog = _prog()
        prog.add_metrics({"score": 50.0, "flat": 10.0})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert "flat_norm" not in prog.metrics
