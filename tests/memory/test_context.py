from pydantic import ValidationError
import pytest

from gigaevo.memory.context import ContextualGain, DecisionContext


def _ctx(**metrics):
    return DecisionContext(parent_metrics=dict(metrics))


def test_decision_context_holds_parent_metrics():
    ctx = _ctx(r2=0.87, complexity=12.0)
    assert ctx.parent_metrics == {"r2": 0.87, "complexity": 12.0}


def test_decision_context_is_frozen():
    ctx = _ctx(r2=0.87)
    with pytest.raises(ValidationError):
        ctx.parent_metrics = {"r2": 0.0}


def test_decision_context_equality_by_value():
    assert _ctx(r2=0.5) == _ctx(r2=0.5)
    assert _ctx(r2=0.5) != _ctx(r2=0.6)


def test_contextual_gain_defaults_to_valid():
    g = ContextualGain(context=_ctx(r2=0.8), gain=0.01)
    assert g.gain == pytest.approx(0.01)
    assert g.invalid is False


def test_contextual_gain_is_frozen():
    g = ContextualGain(context=_ctx(r2=0.8), gain=0.01)
    with pytest.raises(ValidationError):
        g.gain = 0.99
