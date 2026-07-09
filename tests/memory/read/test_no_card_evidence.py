from __future__ import annotations

import pytest

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import DecisionContext
from gigaevo.memory.context import BDCellMemoryContext
from gigaevo.memory.context.no_card import JsonNoCardEvidenceStore


class _Outcome:
    def __init__(
        self,
        *,
        oid: str,
        fitness: float,
        base_fitness: float,
        selected: tuple[str, ...] = (),
        no_card_control: bool = True,
        x: float | None = None,
    ) -> None:
        self.id = oid
        self.fitness = fitness
        self.base_fitness = base_fitness
        self.base_metrics = {"fitness": base_fitness}
        if x is not None:
            self.base_metrics["x"] = x
        self.base_id = "base"
        self.base_selected_ids = selected
        self.no_card_control = no_card_control
        self.invalid = False


def test_json_no_card_evidence_records_controls_and_ignores_selected(tmp_path):
    store = JsonNoCardEvidenceStore(
        path=tmp_path / "no_card.json",
        local_min_effective_n=1.0,
    )

    store.record_outcomes(
        [
            _Outcome(oid="control-a", fitness=0.6, base_fitness=0.5),
            _Outcome(
                oid="selected", fitness=0.9, base_fitness=0.5, selected=("mem-a",)
            ),
        ],
        higher_is_better=True,
    )
    summary = store.summary_for(None)

    assert summary.source == "global_control"
    assert summary.evidence_n == 1.0
    assert summary.baseline == pytest.approx(0.1)
    assert summary.prior.source == "global_control"
    assert summary.prior.support_n == 1.0


def test_json_no_card_evidence_splits_median_ties_neutrally(tmp_path):
    store = JsonNoCardEvidenceStore(
        path=tmp_path / "no_card.json",
        local_min_effective_n=1.0,
    )

    store.record_outcomes(
        [
            _Outcome(oid="control-a", fitness=0.6, base_fitness=0.5),
            _Outcome(oid="control-b", fitness=0.6, base_fitness=0.5),
        ],
        higher_is_better=True,
    )
    summary = store.summary_for(None)

    assert summary.baseline == pytest.approx(0.1)
    assert summary.prior.alpha == pytest.approx(summary.prior.beta)
    assert summary.prior.support_n == pytest.approx(2.0)


def test_json_no_card_evidence_even_controls_use_midpoint_median(tmp_path):
    store = JsonNoCardEvidenceStore(
        path=tmp_path / "no_card.json",
        local_min_effective_n=1.0,
    )

    store.record_outcomes(
        [
            _Outcome(oid="control-a", fitness=0.6, base_fitness=0.5),
            _Outcome(oid="control-b", fitness=0.9, base_fitness=0.5),
        ],
        higher_is_better=True,
    )
    summary = store.summary_for(None)

    assert summary.baseline == pytest.approx(0.25)
    assert summary.prior.alpha == pytest.approx(summary.prior.beta)


def test_json_no_card_evidence_prefers_controls_over_natural_local_empty(
    tmp_path,
):
    space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    store = JsonNoCardEvidenceStore(
        path=tmp_path / "no_card.json",
        context_model=BDCellMemoryContext(behavior_space=space),
    )

    store.record_outcomes(
        [
            _Outcome(
                oid="control-low",
                fitness=0.6,
                base_fitness=0.5,
                no_card_control=True,
                x=0.2,
            ),
            _Outcome(
                oid="natural-high",
                fitness=1.3,
                base_fitness=0.5,
                no_card_control=False,
                x=0.8,
            ),
        ],
        higher_is_better=True,
    )
    summary = store.summary_for(DecisionContext(parent_metrics={"x": 0.8}))

    assert summary.source == "local_shrunk"
    assert summary.baseline == pytest.approx(0.1)


def test_json_no_card_evidence_shrinkage_uses_limited_nonlocal_pseudocounts(
    tmp_path,
):
    space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    store = JsonNoCardEvidenceStore(
        path=tmp_path / "no_card.json",
        context_model=BDCellMemoryContext(behavior_space=space),
        local_min_effective_n=8.0,
        shrink_events=4.0,
        sign_strength_cap=100.0,
    )

    outcomes = [
        _Outcome(
            oid="local",
            fitness=0.7,
            base_fitness=0.5,
            no_card_control=True,
            x=0.8,
        )
    ]
    outcomes.extend(
        _Outcome(
            oid=f"global-{i}",
            fitness=0.6 + 0.01 * i,
            base_fitness=0.5,
            no_card_control=True,
            x=0.2,
        )
        for i in range(20)
    )
    store.record_outcomes(outcomes, higher_is_better=True)
    summary = store.summary_for(DecisionContext(parent_metrics={"x": 0.8}))

    assert summary.source == "local_shrunk"
    assert summary.evidence_n == pytest.approx(5.0)
    assert summary.prior.support_n == pytest.approx(5.0)


def test_json_no_card_evidence_updates_existing_outcome(tmp_path):
    store = JsonNoCardEvidenceStore(path=tmp_path / "no_card.json")

    store.record_outcomes(
        [_Outcome(oid="same", fitness=0.6, base_fitness=0.5)],
        higher_is_better=True,
    )
    store.record_outcomes(
        [_Outcome(oid="same", fitness=0.7, base_fitness=0.5)],
        higher_is_better=True,
    )

    assert store.summary_for(None).baseline == pytest.approx(0.2)
