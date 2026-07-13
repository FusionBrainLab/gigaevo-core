"""Write-path regressions over the real local store."""

from __future__ import annotations

from threading import Thread

import pytest

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.read.reputation import (
    BetaBinomialReputation,
    BootstrapReputation,
)
from gigaevo.memory.storage.config import StoreConfig
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.write.admission import CardAdmissionGate, WriteOutcome
from gigaevo.memory.write.eviction import HarmEvictor, NullEvictor
from tests.fakes.embedding import FakeEmbeddingFunction


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    FakeEmbeddingFunction.embedded.clear()
    return LocalMemoryStore(StoreConfig(path=tmp_path / "store"))


def test_sweep_with_same_store_bootstrap_reputation_completes(
    local_store, make_card, make_event
):
    card = make_card(gain_events=tuple(make_event(-0.1) for _ in range(4)))
    local_store.save(card)
    reputation = BootstrapReputation(
        BetaBinomialReputation(), local_store, n_bootstrap=32
    )
    gate = CardAdmissionGate(store=local_store, evictor=HarmEvictor(reputation))
    errors: list[BaseException] = []
    results: list[list[str]] = []

    def sweep() -> None:
        try:
            results.append(gate.sweep())
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=sweep, daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert errors == []
    assert results == [[card.id]]
    assert local_store.get(card.id) is None


def _second_store(tmp_path) -> LocalMemoryStore:
    return LocalMemoryStore(StoreConfig(path=tmp_path / "store"))


class NegativeTotalEvictor:
    def should_evict(self, card: Card) -> bool:
        total = sum(
            float(event.gain) for event in card.gain_events if event.gain is not None
        )
        return total < 0.0

    def eviction_reason(self, card: Card) -> str:
        del card
        return "negative total evidence"

    def sweep(self, cards) -> list[str]:
        return [card.id for card in cards if self.should_evict(card)]


def test_merge_folds_fresh_partner_restamp(
    local_store, tmp_path, make_card, make_event
):
    target = make_card(description="target prose")
    partner = make_card(description="stale partner prose")
    local_store.save(target)
    local_store.save(partner)
    submitted = local_store.get(partner.id).model_copy(
        update={"description": "curated merged prose"}
    )
    restamped = make_event(0.7, task_key="task-b")
    store2 = _second_store(tmp_path)
    store2.update(
        partner.id,
        lambda fresh: fresh.model_copy(update={"gain_events": (restamped,)}),
    )

    result = CardAdmissionGate(store=local_store, evictor=NullEvictor()).merge(
        target.id, submitted
    )

    survivor = local_store.get(target.id)
    assert result.card_id == target.id
    assert survivor.description == "curated merged prose"
    assert survivor.gain_events == (restamped,)
    assert local_store.get(partner.id) is None


def test_bump_provenance_preserves_interleaved_restamp(
    local_store, tmp_path, make_card, make_event
):
    target = make_card(programs=("existing-child",))
    local_store.save(target)
    stale = local_store.get(target.id)
    restamped = make_event(0.4, task_key="task-a")
    store2 = _second_store(tmp_path)
    store2.update(
        target.id,
        lambda fresh: fresh.model_copy(update={"gain_events": (restamped,)}),
    )

    result = CardAdmissionGate(
        store=local_store, evictor=NullEvictor()
    ).bump_provenance(stale.id, "new-child")

    survivor = local_store.get(target.id)
    assert result.card_id == target.id
    assert survivor.programs == ("existing-child", "new-child")
    assert survivor.gain_events == (restamped,)


def test_known_id_readmit_preserves_interleaved_restamp(
    local_store, tmp_path, make_card, make_event
):
    banked = make_card(description="old prose")
    local_store.save(banked)
    stale_reauthor = local_store.get(banked.id).model_copy(
        update={"description": "new prose"}
    )
    restamped = make_event(-0.3, task_key="task-a")
    store2 = _second_store(tmp_path)
    store2.update(
        banked.id,
        lambda fresh: fresh.model_copy(update={"gain_events": (restamped,)}),
    )

    result = CardAdmissionGate(store=local_store, evictor=NullEvictor()).admit(
        stale_reauthor
    )

    survivor = local_store.get(banked.id)
    assert result.card_id == banked.id
    assert survivor.description == "new prose"
    assert survivor.gain_events == (restamped,)


def test_known_id_harm_readmit_uses_fresh_union_that_rescues_card(
    local_store, tmp_path, make_card, make_event
):
    stale_harm = make_event(-1.0, parent_id="stale")
    banked = make_card(description="old prose", gain_events=(stale_harm,))
    local_store.save(banked)
    stale_reauthor = local_store.get(banked.id).model_copy(
        update={"description": "new prose"}
    )
    positive_restamp = make_event(2.0, parent_id="fresh")
    store2 = _second_store(tmp_path)
    store2.update(
        banked.id,
        lambda fresh: fresh.model_copy(update={"gain_events": (positive_restamp,)}),
    )

    result = CardAdmissionGate(store=local_store, evictor=NegativeTotalEvictor()).admit(
        stale_reauthor
    )

    survivor = local_store.get(banked.id)
    assert result.outcome is WriteOutcome.UPDATED
    assert result.card_id == banked.id
    assert survivor.description == "new prose"
    assert survivor.gain_events == (positive_restamp, stale_harm)


def test_known_id_harm_readmit_deletes_when_fresh_union_remains_harmful(
    local_store, tmp_path, make_card, make_event
):
    stale_harm = make_event(-1.0, parent_id="stale")
    banked = make_card(gain_events=(stale_harm,))
    local_store.save(banked)
    stale_reauthor = local_store.get(banked.id)
    weak_positive_restamp = make_event(0.2, parent_id="fresh")
    store2 = _second_store(tmp_path)
    store2.update(
        banked.id,
        lambda fresh: fresh.model_copy(
            update={"gain_events": (weak_positive_restamp,)}
        ),
    )
    gate = CardAdmissionGate(store=local_store, evictor=NegativeTotalEvictor())

    result = gate.admit(stale_reauthor)

    assert result.outcome is WriteOutcome.REJECTED_HARM
    assert result.card_id == ""
    assert local_store.get(banked.id) is None
    assert gate.is_tombstoned(banked.id)


def test_twin_retirement_folds_fresh_restamp_into_successor(
    local_store, tmp_path, make_card, make_event
):
    successor = make_card(
        id="program-new",
        kind=CardKind.PROGRAM,
        program_id="new",
        code_sha256="same-code",
        description="successor prose",
    )
    twin = make_card(
        id="program-old",
        kind=CardKind.PROGRAM,
        program_id="old",
        code_sha256="same-code",
        description="twin prose",
    )
    local_store.save(successor)
    local_store.save(twin)
    stale_twin = local_store.get(twin.id)
    restamped = make_event(0.8, task_key="task-b")
    store2 = _second_store(tmp_path)
    store2.update(
        twin.id,
        lambda fresh: fresh.model_copy(update={"gain_events": (restamped,)}),
    )

    result = CardAdmissionGate(store=local_store, evictor=NullEvictor()).retire_twin(
        stale_twin, successor_id=successor.id
    )

    survivor = local_store.get(successor.id)
    assert result.card_id == successor.id
    assert survivor.description == "successor prose"
    assert survivor.gain_events == (restamped,)
    assert twin.id in survivor.absorbed_ids
    assert local_store.get(twin.id) is None
