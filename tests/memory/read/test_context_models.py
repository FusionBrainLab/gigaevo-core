from __future__ import annotations

import pytest

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import DecisionContext
from gigaevo.memory.context import BDCellMemoryContext, GlobalMemoryContext


class _Parent:
    id = "parent"
    metrics = {"x": 0.2}


class _Outcome:
    def __init__(
        self,
        *,
        oid: str,
        fitness: float | None,
        base_fitness: float,
        x: float,
        no_card_control: bool = True,
        invalid: bool = False,
    ) -> None:
        self.id = oid
        self.fitness = fitness
        self.base_fitness = base_fitness
        self.base_metrics = {"x": x}
        self.base_selected_ids = ()
        self.no_card_control = no_card_control
        self.invalid = invalid


def _space() -> BehaviorSpace:
    return BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )


def test_global_read_context_uses_primary_parent():
    context = GlobalMemoryContext().read_context([_Parent()])

    assert context == DecisionContext(parent_metrics={"x": 0.2}, parent_id="parent")


@pytest.mark.parametrize(
    "model",
    [
        GlobalMemoryContext(task_key="heilbronn"),
        BDCellMemoryContext(behavior_space=_space(), task_key="heilbronn"),
    ],
)
def test_read_context_stamps_configured_task_key(model):
    assert model.read_context([_Parent()]).task_key == "heilbronn"


@pytest.mark.parametrize(
    "model", [GlobalMemoryContext(), BDCellMemoryContext(behavior_space=_space())]
)
def test_read_context_task_key_defaults_to_empty(model):
    assert model.read_context([_Parent()]).task_key == ""


def test_bd_context_returns_same_cell_events(make_card, make_event):
    model = BDCellMemoryContext(behavior_space=_space())
    card = make_card(
        gain_events=(
            make_event(0.2, metrics={"x": 0.2}),
            make_event(0.4, metrics={"x": 0.8}),
        )
    )

    events = model.evidence_events(card, DecisionContext(parent_metrics={"x": 0.1}))

    assert [event.gain for event in events] == [pytest.approx(0.2)]


def test_bd_evidence_cells_keeps_first_event_per_cell(make_card, make_event):
    model = BDCellMemoryContext(behavior_space=_space())
    card = make_card(
        gain_events=(
            make_event(0.1, metrics={"x": 0.2}),
            make_event(0.2, metrics={"x": 0.3}),
            make_event(0.3, metrics={"x": 0.8}),
        )
    )

    events = model.evidence_cells(card.gain_events)

    assert [event.gain for event in events] == [
        pytest.approx(0.1),
        pytest.approx(0.3),
    ]


def test_bd_evidence_cells_skips_unbinnable_events(make_card, make_event):
    model = BDCellMemoryContext(behavior_space=_space())
    card = make_card(
        gain_events=(
            make_event(0.1, metrics={}),
            make_event(0.2, metrics={"x": 0.8}),
        )
    )

    events = model.evidence_cells(card.gain_events)

    assert [event.gain for event in events] == [pytest.approx(0.2)]


def test_bd_context_falls_back_when_cell_has_only_founding(make_card, make_event):
    model = BDCellMemoryContext(behavior_space=_space())
    card = make_card(
        gain_events=(
            make_event(0.9, founding=True, metrics={"x": 0.2}),
            make_event(0.4, metrics={"x": 0.8}),
        )
    )

    events = model.evidence_events(card, DecisionContext(parent_metrics={"x": 0.1}))

    assert [event.gain for event in events] == [pytest.approx(0.9), pytest.approx(0.4)]


def test_global_no_card_baseline_excludes_invalid_children():
    # A crashed control has no measurable delta; scoring it as 0.0 drags the
    # baseline median toward zero and inflates every card's apparent gain.
    baseline = GlobalMemoryContext().fit_no_card_baseline(
        [
            _Outcome(oid="a", fitness=0.6, base_fitness=0.5, x=0.2),
            _Outcome(oid="crash", fitness=None, base_fitness=0.5, x=0.2, invalid=True),
        ],
        higher_is_better=True,
    )

    assert baseline.baseline_for(
        _Outcome(oid="q", fitness=0.0, base_fitness=0.0, x=0.2)
    ) == pytest.approx(0.1)


def test_bd_no_card_baseline_excludes_invalid_children():
    model = BDCellMemoryContext(behavior_space=_space())
    baseline = model.fit_no_card_baseline(
        [
            _Outcome(oid="a", fitness=0.6, base_fitness=0.5, x=0.2),
            _Outcome(oid="crash", fitness=None, base_fitness=0.5, x=0.2, invalid=True),
        ],
        higher_is_better=True,
    )

    assert baseline.baseline_for(
        _Outcome(oid="q", fitness=0.0, base_fitness=0.0, x=0.2)
    ) == pytest.approx(0.1)


def test_bd_no_card_baseline_uses_cell_with_global_fallback():
    model = BDCellMemoryContext(behavior_space=_space())
    baseline = model.fit_no_card_baseline(
        [
            _Outcome(oid="a", fitness=0.6, base_fitness=0.5, x=0.2),
            _Outcome(oid="b", fitness=0.9, base_fitness=0.5, x=0.8),
        ],
        higher_is_better=True,
    )

    assert baseline.baseline_for(
        _Outcome(oid="q", fitness=0.0, base_fitness=0.0, x=0.2)
    ) == pytest.approx(0.1)
    assert baseline.baseline_for(
        _Outcome(oid="q", fitness=0.0, base_fitness=0.0, x=float("nan"))
    ) == pytest.approx(0.25)
