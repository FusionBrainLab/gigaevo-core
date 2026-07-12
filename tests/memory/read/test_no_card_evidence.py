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
        fitness: float | None,
        base_fitness: float,
        selected: tuple[str, ...] = (),
        no_card_control: bool = True,
        x: float | None = None,
        invalid: bool = False,
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
        self.invalid = invalid


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


def test_json_no_card_evidence_invalid_children_fail_sign_but_skip_median(
    tmp_path,
):
    # Crashed no-card children carry no honest progress magnitude: they must
    # not drag the baseline median toward zero, but they are unconditional
    # sign failures for the abstention prior.
    store = JsonNoCardEvidenceStore(
        path=tmp_path / "no_card.json",
        local_min_effective_n=1.0,
    )

    store.record_outcomes(
        [
            _Outcome(oid="control-a", fitness=0.7, base_fitness=0.5),
            _Outcome(oid="control-b", fitness=0.7, base_fitness=0.5),
            _Outcome(oid="crash-a", fitness=None, base_fitness=0.5, invalid=True),
            _Outcome(oid="crash-b", fitness=None, base_fitness=0.5, invalid=True),
        ],
        higher_is_better=True,
    )
    summary = store.summary_for(None)

    assert summary.baseline == pytest.approx(0.2)
    assert summary.prior.alpha == pytest.approx(4.0)
    assert summary.prior.beta == pytest.approx(6.0)


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


def _stored_ids(store: JsonNoCardEvidenceStore) -> list[str]:
    return [obs.id for obs in store._read()]


def test_json_no_card_evidence_prunes_oldest_naturals_first(tmp_path):
    store = JsonNoCardEvidenceStore(path=tmp_path / "no_card.json", max_observations=3)

    store.record_outcomes(
        [
            _Outcome(
                oid="natural-old", fitness=0.6, base_fitness=0.5, no_card_control=False
            ),
            _Outcome(oid="control-old", fitness=0.6, base_fitness=0.5),
            _Outcome(
                oid="natural-new", fitness=0.7, base_fitness=0.5, no_card_control=False
            ),
        ],
        higher_is_better=True,
    )
    store.record_outcomes(
        [_Outcome(oid="control-new", fitness=0.8, base_fitness=0.5)],
        higher_is_better=True,
    )

    assert _stored_ids(store) == ["control-old", "natural-new", "control-new"]


def test_json_no_card_evidence_prunes_oldest_controls_when_no_naturals(tmp_path):
    store = JsonNoCardEvidenceStore(path=tmp_path / "no_card.json", max_observations=2)

    store.record_outcomes(
        [
            _Outcome(oid=f"control-{i}", fitness=0.5 + 0.1 * i, base_fitness=0.5)
            for i in range(4)
        ],
        higher_is_better=True,
    )

    assert _stored_ids(store) == ["control-2", "control-3"]


def test_json_no_card_evidence_below_cap_keeps_everything(tmp_path):
    store = JsonNoCardEvidenceStore(path=tmp_path / "no_card.json")

    store.record_outcomes(
        [
            _Outcome(
                oid="natural", fitness=0.6, base_fitness=0.5, no_card_control=False
            ),
            _Outcome(oid="control", fitness=0.7, base_fitness=0.5),
        ],
        higher_is_better=True,
    )

    assert sorted(_stored_ids(store)) == ["control", "natural"]
