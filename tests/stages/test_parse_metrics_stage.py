"""Tests for ParseMetricsStage — composes program.metrics from primitives."""

from __future__ import annotations

import pytest

from gigaevo.programs.metrics.aggregators import (
    ConfigurableAggregator,
    ConstantSpec,
    IntrinsicSpec,
    ReduceSpec,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.python_executors.execution import ParseMetricsStage


def _ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness", higher_is_better=True, is_primary=True
            ),
            "is_valid": MetricSpec(description="validity", higher_is_better=True),
        }
    )


def _agg() -> ConfigurableAggregator:
    return ConfigurableAggregator(
        outputs={
            "is_valid": ConstantSpec(value=1.0),
            "n_opponents": ReduceSpec(op="count"),
            "fitness": ReduceSpec(op="mean", field="score"),
            "quality": IntrinsicSpec(key="quality", default=0.0),
        },
        invalid_defaults={
            "is_valid": 0.0,
            "n_opponents": 0.0,
            "fitness": -1.0,
            "quality": 0.0,
        },
        metrics_context=_ctx(),
    )


@pytest.mark.asyncio
async def test_aggregator_required_raises_on_none():
    with pytest.raises(ValueError, match="aggregator required"):
        ParseMetricsStage(aggregator=None, timeout=10)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_composes_metrics_from_per_opp(dummy_program):
    stage = ParseMetricsStage(aggregator=_agg(), timeout=10)
    artifact = {
        "role": "improver",
        "per_opp_metrics": [
            {"score": 0.4, "is_valid": 1.0},
            {"score": 0.6, "is_valid": 1.0},
        ],
    }
    raw = Box(data=({"quality": 0.7}, artifact))
    stage.attach_inputs({"raw_validator_output": raw})
    out = await stage.compute(dummy_program)
    metrics, out_artifact = out.data
    assert metrics["fitness"] == pytest.approx(0.5)
    assert metrics["n_opponents"] == 2.0
    assert metrics["quality"] == 0.7
    assert out_artifact is artifact  # passthrough by reference


@pytest.mark.asyncio
async def test_empty_per_opp_returns_invalid_defaults(dummy_program):
    stage = ParseMetricsStage(aggregator=_agg(), timeout=10)
    raw = Box(data=({}, {"role": "improver", "per_opp_metrics": []}))
    stage.attach_inputs({"raw_validator_output": raw})
    out = await stage.compute(dummy_program)
    metrics, _ = out.data
    assert metrics == {
        "is_valid": 0.0,
        "n_opponents": 0.0,
        "fitness": -1.0,
        "quality": 0.0,
    }


@pytest.mark.asyncio
async def test_missing_per_opp_metrics_key_treated_as_candidate_failure(dummy_program):
    stage = ParseMetricsStage(aggregator=_agg(), timeout=10)
    raw = Box(data=({}, {"role": "improver"}))  # no per_opp_metrics
    stage.attach_inputs({"raw_validator_output": raw})
    out = await stage.compute(dummy_program)
    metrics, _ = out.data
    assert metrics["is_valid"] == 0.0
    assert metrics["fitness"] == -1.0


@pytest.mark.asyncio
async def test_none_artifact_treated_as_candidate_failure(dummy_program):
    stage = ParseMetricsStage(aggregator=_agg(), timeout=10)
    raw = Box(data=({}, None))
    stage.attach_inputs({"raw_validator_output": raw})
    out = await stage.compute(dummy_program)
    metrics, _ = out.data
    assert metrics["is_valid"] == 0.0


@pytest.mark.asyncio
async def test_raises_on_non_tuple_input(dummy_program):
    stage = ParseMetricsStage(aggregator=_agg(), timeout=10)
    raw = Box(data={"not": "a tuple"})
    stage.attach_inputs({"raw_validator_output": raw})
    # Validation happens when accessing params, before compute is called
    with pytest.raises(KeyError, match="tuple"):
        await stage.compute(dummy_program)
