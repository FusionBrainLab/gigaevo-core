from __future__ import annotations

import pytest

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
    ) -> None:
        self.id = oid
        self.fitness = fitness
        self.base_fitness = base_fitness
        self.base_metrics = {"fitness": base_fitness}
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

    assert summary.source == "local_control"
    assert summary.evidence_n == 1.0
    assert summary.baseline == pytest.approx(0.1)
    assert summary.prior.source == "local_control"
    assert summary.prior.support_n == 1.0


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
