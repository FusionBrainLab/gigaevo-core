"""Validation pipeline tests covering parse_output and metric coercion.

parse_output rejects non-mapping shapes (None / float / int / list) with
ValueError or TypeError; EnsureMetricsStage._coerce_and_clamp raises
ValueError naming the metric key for non-numeric inputs.
"""

from __future__ import annotations

import pytest

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.stages.metrics import EnsureMetricsStage
from gigaevo.programs.stages.python_executors.execution import CallValidatorFunction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(x):
    """Call CallValidatorFunction.parse_output without needing a real file path."""
    inst = CallValidatorFunction.__new__(CallValidatorFunction)
    return inst.parse_output(x)


def _make_coerce_stage() -> EnsureMetricsStage:
    """Minimal EnsureMetricsStage for testing _coerce_and_clamp directly."""
    ctx = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="primary fitness",
                higher_is_better=True,
                is_primary=True,
            )
        }
    )
    stage = EnsureMetricsStage.__new__(EnsureMetricsStage)
    stage.ctx = ctx
    return stage


# ---------------------------------------------------------------------------
# Bug A — parse_output type validation
# ---------------------------------------------------------------------------


def test_parse_output_raises_for_none() -> None:
    """parse_output(None) raises instead of producing (None, None)."""
    with pytest.raises((TypeError, ValueError)):
        _parse(None)


def test_parse_output_raises_for_float() -> None:
    """parse_output(<float>) raises; raw fitness values are not auto-wrapped."""
    with pytest.raises((TypeError, ValueError)):
        _parse(0.85)


def test_parse_output_raises_for_int() -> None:
    """parse_output(1) must raise."""
    with pytest.raises((TypeError, ValueError)):
        _parse(1)


def test_parse_output_raises_for_list() -> None:
    """parse_output([0.5, 0.3]) must raise."""
    with pytest.raises((TypeError, ValueError)):
        _parse([0.5, 0.3])


def test_parse_output_accepts_dict() -> None:
    """parse_output({'fitness': 0.5}) must return ({'fitness': 0.5}, None)."""
    result = _parse({"fitness": 0.5})
    assert result == ({"fitness": 0.5}, None)


def test_parse_output_accepts_tuple() -> None:
    """parse_output(({'fitness': 0.5}, artifact)) must be returned unchanged."""
    artifact = [1, 2, 3]
    result = _parse(({"fitness": 0.5}, artifact))
    assert result == ({"fitness": 0.5}, artifact)


# ---------------------------------------------------------------------------
# Bug B — _coerce_and_clamp TypeError on string values
# ---------------------------------------------------------------------------


def test_coerce_and_clamp_raises_value_error_for_nonnumeric_string() -> None:
    """_coerce_and_clamp raises ValueError naming the metric key on non-numeric input."""
    stage = _make_coerce_stage()
    with pytest.raises(ValueError, match="fitness"):
        stage._coerce_and_clamp("fitness", "high")


def test_coerce_and_clamp_raises_value_error_for_none() -> None:
    """_coerce_and_clamp('fitness', None) must raise ValueError."""
    stage = _make_coerce_stage()
    with pytest.raises((TypeError, ValueError)):
        stage._coerce_and_clamp("fitness", None)


def test_coerce_and_clamp_accepts_valid_float() -> None:
    """Sanity: _coerce_and_clamp('fitness', 0.5) must return 0.5."""
    stage = _make_coerce_stage()
    assert stage._coerce_and_clamp("fitness", 0.5) == 0.5
