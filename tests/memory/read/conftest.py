"""Fixtures for the read layer: card/event factories and event capture."""

from __future__ import annotations

import pytest

from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext


@pytest.fixture
def make_card():
    counter = iter(range(10_000))

    def _make_card(**overrides) -> Card:
        n = next(counter)
        params = {
            "id": f"mem-test{n:04d}",
            "kind": CardKind.INSIGHT,
            "description": f"idea-{n} exploits problem structure",
            "explanation_summary": f"works because of invariant-{n}",
        }
        params.update(overrides)
        return Card(**params)

    return _make_card


@pytest.fixture
def make_event():
    def _make_event(
        gain: float,
        *,
        gain_se: float = 0.0,
        invalid: bool = False,
        founding: bool = False,
        unused: bool = False,
        metrics: dict[str, float] | None = None,
    ) -> ContextualGain:
        return ContextualGain(
            context=DecisionContext(parent_metrics=metrics or {}),
            gain=gain,
            gain_se=gain_se,
            invalid=invalid,
            founding=founding,
            unused=unused,
        )

    return _make_event


@pytest.fixture
def captured_events(monkeypatch):
    events: list = []
    for module in (
        "gigaevo.memory.read.reader",
        "gigaevo.memory.read.auction",
    ):
        monkeypatch.setattr(f"{module}.emit_memory_event", events.append)
    return events
