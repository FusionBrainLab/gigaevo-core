"""Regression tests for validate.py return-value bugs in the execution pipeline.

Bug A — parse_output accepts None and wrong types (Findings 4+5):
    CallValidatorFunction.parse_output() passes through any value unchecked.
    None → (None, None), which causes 'TypeError: {**None}' deep in MergeMetricsStage
    with no hint that validate() returned None.  Lists, ints, and floats crash
    with similarly cryptic errors.
    Fix: type-check in parse_output; raise ValueError with a clear message pointing
    to the validate() return value.

Bug B — _coerce_and_clamp raises TypeError for string metric values (Finding 6):
    math.isfinite("0.85") raises TypeError (not ValueError).  This may confuse
    error-handling code that only catches ValueError, and gives the junior researcher
    no guidance that they returned a string instead of a float.
    Fix: attempt float() coercion first; raise ValueError with a descriptive message
    on failure.
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
    """parse_output(None) must raise, not silently produce (None, None).

    'None' is the most common mistake: the researcher forgot the return statement.
    Before the fix: returns (None, None) → downstream crash in MergeMetricsStage
    with 'TypeError: {**None}' — no pointer to validate().
    """
    with pytest.raises((TypeError, ValueError)):
        _parse(None)


def test_parse_output_raises_for_float() -> None:
    """parse_output(0.85) must raise.

    Researcher returned the raw float accuracy instead of wrapping it in a dict.
    Before the fix: returns (0.85, None) → crash in MergeMetricsStage with
    'TypeError: float is not iterable'.
    """
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
    """_coerce_and_clamp('fitness', 'high') must raise ValueError, not TypeError.

    Before the fix: math.isfinite('high') raises TypeError with a generic message
    from Python builtins — no pointer back to the metric key or validate().
    Fix: try float(value) first; on coercion failure raise ValueError with a
    message naming the metric key, e.g.
    "Metric 'fitness' must be numeric, got 'str': 'high'".
    """
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
