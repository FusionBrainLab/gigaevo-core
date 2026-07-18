"""Write-path regressions over the real local store."""

from __future__ import annotations

import pytest

from gigaevo.memory.storage.config import StoreConfig
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.write.admission import CardAdmissionGate, WriteOutcome
from gigaevo.memory.write.eviction import NullEvictor
from tests.fakes.embedding import FakeEmbeddingFunction


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    FakeEmbeddingFunction.embedded.clear()
    return LocalMemoryStore(StoreConfig(path=tmp_path / "store"))


def _second_store(tmp_path) -> LocalMemoryStore:
    return LocalMemoryStore(StoreConfig(path=tmp_path / "store"))


def test_equivalent_update_preserves_interleaved_restamp(
    local_store, tmp_path, make_card, make_event
):
    target = make_card(task_key="task", programs=("existing-child",))
    local_store.save(target)
    stale = local_store.get(target.id)
    restamped = make_event(0.4, task_key="task-a")
    store2 = _second_store(tmp_path)
    store2.update(
        target.id,
        lambda fresh: fresh.model_copy(update={"gain_events": (restamped,)}),
    )

    incoming = make_card(
        id="",
        task_key="task",
        programs=("new-child",),
    )
    result = CardAdmissionGate(
        store=local_store, evictor=NullEvictor()
    ).update_equivalent(stale.id, incoming)

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
    assert survivor.description == "old prose"
    assert survivor.gain_events == (restamped,)


def test_known_id_readmit_unions_fresh_and_incoming_evidence(
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

    result = CardAdmissionGate(store=local_store, evictor=NullEvictor()).admit(
        stale_reauthor
    )

    survivor = local_store.get(banked.id)
    assert result.outcome is WriteOutcome.UPDATED
    assert result.card_id == banked.id
    assert survivor.description == "old prose"
    assert survivor.gain_events == (positive_restamp, stale_harm)
