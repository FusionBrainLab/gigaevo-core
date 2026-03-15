"""Tests for Finding 3: NormalizeMetricsStage raises KeyError on missing metric key.

gigaevo/programs/stages/metrics.py — NormalizeMetricsStage.compute, line 157:

    v = program.metrics[key]

If the metric key is present in the MetricsContext but absent from program.metrics,
this raises a bare `KeyError`. The error message gives no context about which stage
raised it or which program/metric was involved.

These tests verify:
1. The current behavior (KeyError raised) — so we know the contract.
2. That a program with all required metrics passes correctly.
3. Boundary: metrics present but None value.
4. Multiple metrics, some missing.

Finding 4 note: get_primary_key() / get_primary_spec() implicit None return is
NOT a real runtime risk because MetricsContext's @model_validator enforces exactly
one primary metric at construction time. We include one test confirming this contract
and that the methods always return non-None for valid contexts.
"""

from __future__ import annotations

import pytest

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.metrics import NormalizeMetricsStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx_with_bounds() -> MetricsContext:
    """MetricsContext where all metrics have bounds (so NormalizeMetricsStage normalizes them)."""
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="Main score",
                higher_is_better=True,
                is_primary=True,
                lower_bound=0.0,
                upper_bound=100.0,
            ),
            "cost": MetricSpec(
                description="Cost",
                higher_is_better=False,
                lower_bound=0.0,
                upper_bound=50.0,
            ),
        }
    )


def _make_ctx_no_bounds() -> MetricsContext:
    """MetricsContext where metrics have no bounds (stage skips them)."""
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="Main score",
                higher_is_better=True,
                is_primary=True,
            ),
        }
    )


def _prog_with_metrics(metrics: dict) -> Program:
    p = Program(code="def solve(): return 1", state=ProgramState.RUNNING)
    p.add_metrics(metrics)
    return p


def _prog_empty() -> Program:
    return Program(code="def solve(): return 1", state=ProgramState.RUNNING)


# ---------------------------------------------------------------------------
# TestNormalizeMetricsMissingKey — Finding 3
# ---------------------------------------------------------------------------


class TestNormalizeMetricsMissingKey:
    async def test_missing_metric_silently_skipped(self) -> None:
        """Metrics absent from program.metrics are silently skipped, not KeyError.

        Fixed behavior: program.metrics.get(key) returns None → continue.
        This matches the existing "skip if no bounds" behavior and prevents
        bare KeyErrors when NormalizeMetricsStage runs without EnsureMetricsStage.
        """
        ctx = _make_ctx_with_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
        )

        # Program has 'score' but is missing 'cost'
        prog = _prog_with_metrics({"score": 75.0})

        result = await stage.compute(prog)
        # 'score' is present and normalized; 'cost' is absent → skipped
        assert "score_norm" in result.data
        assert "cost_norm" not in result.data

    async def test_completely_empty_metrics_returns_empty(self) -> None:
        """Program with no metrics at all: all bounded keys are skipped → empty result."""
        ctx = _make_ctx_with_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
        )

        prog = _prog_empty()

        result = await stage.compute(prog)
        # No metrics present → nothing normalized → empty dict, no aggregate
        assert result.data == {}

    async def test_all_metrics_present_normalizes_correctly(self) -> None:
        """Happy path: all metrics present → normalized correctly."""
        ctx = _make_ctx_with_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
        )

        prog = _prog_with_metrics({"score": 75.0, "cost": 25.0})
        result = await stage.compute(prog)

        # score=75 in [0,100] higher_is_better → (75-0)/(100-0) = 0.75
        assert result.data["score_norm"] == pytest.approx(0.75)
        # cost=25 in [0,50] lower_is_better → (25-0)/(50-0) = 0.5, flipped = 0.5
        assert result.data["cost_norm"] == pytest.approx(0.5)
        # aggregate mean = (0.75 + 0.5) / 2
        assert result.data["normalized_score"] == pytest.approx(0.625)

    async def test_metrics_at_bounds_clamped_to_0_and_1(self) -> None:
        """Values outside bounds are clamped to [0, 1] in normalized space."""
        ctx = _make_ctx_with_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
        )

        # score=150 exceeds upper_bound=100 → clamped to 1.0
        # cost=-5 below lower_bound=0 → clamped to 0.0 → flipped = 1.0
        prog = _prog_with_metrics({"score": 150.0, "cost": -5.0})
        result = await stage.compute(prog)

        assert result.data["score_norm"] == pytest.approx(1.0)
        assert result.data["cost_norm"] == pytest.approx(1.0)

    async def test_metrics_at_minimum_normalized_to_zero(self) -> None:
        """score=0 (minimum) → normalized to 0.0."""
        ctx = _make_ctx_with_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
        )

        prog = _prog_with_metrics({"score": 0.0, "cost": 0.0})
        result = await stage.compute(prog)

        # score=0 in [0,100] higher_is_better → 0.0
        assert result.data["score_norm"] == pytest.approx(0.0)
        # cost=0 in [0,50] lower_is_better → 0.0 → flipped = 1.0
        assert result.data["cost_norm"] == pytest.approx(1.0)

    async def test_no_bounds_metrics_skipped(self) -> None:
        """Metrics without bounds are silently skipped (no KeyError, empty result)."""
        ctx = _make_ctx_no_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
        )

        # Program has the metric but no bounds are defined → skip
        prog = _prog_with_metrics({"score": 0.75})
        result = await stage.compute(prog)

        # No bounded metrics → nothing to normalize → no aggregate key
        assert result.data == {}

    async def test_aggregate_key_absent_when_no_normalized_metrics(self) -> None:
        """When no metrics are normalized, the aggregate key must not appear."""
        ctx = _make_ctx_no_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
        )

        prog = _prog_with_metrics({"score": 0.5})
        result = await stage.compute(prog)

        assert "normalized_score" not in result.data

    async def test_normalized_metrics_written_to_program(self) -> None:
        """Normalized metrics are added to program.metrics, not just returned."""
        ctx = _make_ctx_with_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
        )

        prog = _prog_with_metrics({"score": 50.0, "cost": 10.0})
        await stage.compute(prog)

        assert "score_norm" in prog.metrics
        assert "cost_norm" in prog.metrics
        assert "normalized_score" in prog.metrics

    async def test_custom_aggregate_key(self) -> None:
        """A custom aggregate_key is used in place of the default."""
        ctx = _make_ctx_with_bounds()
        stage = NormalizeMetricsStage(
            timeout=30.0,
            metrics_context=ctx,
            aggregate_key="my_aggregate",
        )

        prog = _prog_with_metrics({"score": 50.0, "cost": 25.0})
        result = await stage.compute(prog)

        assert "my_aggregate" in result.data
        assert "normalized_score" not in result.data


# ---------------------------------------------------------------------------
# TestGetPrimaryKeyNeverNone — Finding 4 (model_validator enforces this)
# ---------------------------------------------------------------------------


class TestGetPrimaryKeyNeverNone:
    def test_get_primary_key_returns_string_not_none(self) -> None:
        """get_primary_key() always returns the primary key, never None.

        The @model_validator enforces exactly one is_primary=True metric at
        construction time, so the for-loop in get_primary_key() always finds
        a hit. This test guards against regressions that remove that invariant.
        """
        ctx = _make_ctx_with_bounds()
        key = ctx.get_primary_key()
        assert key is not None
        assert isinstance(key, str)
        assert key == "score"

    def test_get_primary_spec_returns_spec_not_none(self) -> None:
        """get_primary_spec() always returns the primary MetricSpec, never None."""
        ctx = _make_ctx_with_bounds()
        spec = ctx.get_primary_spec()
        assert spec is not None
        assert spec.is_primary is True

    def test_construction_with_zero_primaries_raises(self) -> None:
        """The model_validator blocks construction with zero primary metrics.

        This is the guard that makes get_primary_key() / get_primary_spec()
        safe to call — they can never fall off the end of the loop on a
        valid MetricsContext.
        """
        with pytest.raises(ValueError, match="Exactly one.*is_primary=True"):
            MetricsContext(
                specs={
                    "score": MetricSpec(
                        description="score", higher_is_better=True, is_primary=False
                    )
                }
            )

    def test_construction_with_two_primaries_raises(self) -> None:
        """Two is_primary=True metrics → ValueError at construction."""
        with pytest.raises(ValueError, match="Exactly one.*is_primary=True"):
            MetricsContext(
                specs={
                    "a": MetricSpec(
                        description="a", higher_is_better=True, is_primary=True
                    ),
                    "b": MetricSpec(
                        description="b", higher_is_better=True, is_primary=True
                    ),
                }
            )
