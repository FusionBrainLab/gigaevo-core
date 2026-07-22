"""MemoryStore contract for LocalMemoryStore (persisted bank + in-memory Chroma).

The shared contract runs against the store; backend-specific sections cover
local persistence, retrieval, and state.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from unittest.mock import Mock

import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.events import memory_event_context
from gigaevo.memory.storage.bank import CardBank
from gigaevo.memory.storage.base import ResearchRequest
import gigaevo.memory.storage.local as local_module
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.storage.state import StoreState


def _make_noop_index(_embed):
    return Mock()


def _append_program_markers(config, card_id, markers, ready, start):
    local_module.VectorIndex = _make_noop_index
    store = LocalMemoryStore(config)
    ready.put(True)
    start.wait()
    for marker in markers:
        store.update(
            card_id,
            lambda card, marker=marker: card.model_copy(
                update={"programs": (*card.programs, marker)}
            ),
        )


def _admit_under_authoring_transaction(config, marker, ready, start):
    local_module.VectorIndex = _make_noop_index
    store = LocalMemoryStore(config)
    ready.put(True)
    start.wait()

    async def admit() -> None:
        async with store.authoring_transaction():
            cards = store.snapshot()
            if not cards:
                await asyncio.sleep(0.1)
                store.save(
                    Card(
                        id=f"mem-{marker}",
                        task_key=marker,
                        description="shared semantic action",
                        programs=(marker,),
                    )
                )
                return
            target = cards[0]
            store.update(
                target.id,
                lambda card: card.model_copy(
                    update={"programs": tuple(dict.fromkeys((*card.programs, marker)))}
                ),
            )

    asyncio.run(admit())


@pytest.fixture
def store(make_store_config):
    with LocalMemoryStore(make_store_config()) as local:
        yield local


def test_save_round_trips(store, make_card):
    card = make_card()
    assert store.save(card) == card.id
    assert store.get(card.id) == card


def test_save_mints_id_when_empty(store, make_card):
    card_id = store.save(make_card(id=""))
    assert card_id.startswith("mem-")
    fetched = store.get(card_id)
    assert fetched is not None
    assert fetched.id == card_id


def test_get_missing_returns_none(store):
    assert store.get("mem-missing0000") is None


def test_delete(store, make_card):
    card = make_card()
    store.save(card)
    assert store.delete(card.id) is True
    assert store.get(card.id) is None
    assert store.delete(card.id) is False


def test_snapshot_is_sorted_and_isolated(store, make_card):
    for card_id in ("mem-ccc0", "mem-aaa0", "mem-bbb0"):
        store.save(make_card(id=card_id))
    snap = store.snapshot()
    assert [card.id for card in snap] == ["mem-aaa0", "mem-bbb0", "mem-ccc0"]
    store.save(make_card())
    assert len(snap) == 3
    assert len(store.snapshot()) == 4


def test_program_card_round_trips(store, make_card):
    card = make_card(
        kind=CardKind.PROGRAM,
        program_id="prog-1",
        code="def f():\n    return 0",
        fitness=0.53,
    )
    store.save(card)
    assert store.get(card.id) == card


class TestLocalStore:
    def test_initializes_ready(self, make_store_config):
        store = LocalMemoryStore(make_store_config())
        assert store.is_ready
        assert store.state is StoreState.READY

    def test_bank_survives_restart(self, make_store_config, make_card):
        config = make_store_config()
        card = make_card()
        LocalMemoryStore(config).save(card)
        assert LocalMemoryStore(config).get(card.id) == card

    def test_startup_rebuilds_index_from_existing_bank(
        self, make_store_config, make_card
    ):
        config = make_store_config()
        card = make_card(description="startup rebuild target")
        bank = CardBank(config.bank_file)
        bank.put(card)
        bank.persist()

        store = LocalMemoryStore(config)

        assert [hit.card.id for hit in store.nearest(card.description, k=1)] == [
            card.id
        ]

    def test_other_store_refreshes_bank_and_its_index(
        self, make_store_config, make_card
    ):
        config = make_store_config()
        store_a = LocalMemoryStore(config)
        store_b = LocalMemoryStore(config)
        card = make_card(description="cross process refresh target")

        store_a.save(card)

        assert store_b.get(card.id) == card
        assert [hit.card.id for hit in store_b.nearest(card.description, k=1)] == [
            card.id
        ]

    def test_update_eviction_removes_card_from_index(
        self, make_store_config, make_card
    ):
        store = LocalMemoryStore(make_store_config())
        card = make_card(description="eviction index target")
        store.save(card)

        assert store.update(card.id, lambda _card: None) == card
        assert store.nearest(card.description, k=1) == []

    def test_same_path_stores_do_not_share_index_state(
        self, make_store_config, make_card
    ):
        config = make_store_config()
        store_a = LocalMemoryStore(config)
        store_b = LocalMemoryStore(config)
        card = make_card(description="isolated index target")
        store_a.save(card)
        assert store_b.get(card.id) == card

        store_b._index.remove([card.id])

        assert store_b.nearest(card.description, k=1) == []
        assert [hit.card.id for hit in store_a.nearest(card.description, k=1)] == [
            card.id
        ]

    def test_shared_bank_instances_do_not_clobber_each_other(
        self, make_store_config, make_card
    ):
        config = make_store_config()
        store_a = LocalMemoryStore(config)
        store_b = LocalMemoryStore(config)
        card_a = make_card(id="mem-shared-a", description="alpha shared bank")
        card_b = make_card(id="mem-shared-b", description="beta shared bank")

        store_a.save(card_a)
        store_b.save(card_b)

        reopened = LocalMemoryStore(config)
        assert {card.id for card in reopened.snapshot()} == {
            "mem-shared-a",
            "mem-shared-b",
        }
        assert store_a.get(card_b.id) == card_b
        assert [hit.card.id for hit in store_a.nearest("beta shared bank", k=2)][
            0
        ] == card_b.id

    def test_atomic_update_preserves_cross_process_writes(
        self, make_store_config, make_card
    ):
        config = make_store_config()
        card = make_card(id="mem-shared-update")
        bank = CardBank(config.bank_file)
        bank.put(card)
        bank.persist()
        context = multiprocessing.get_context("fork")
        ready = context.Queue()
        start = context.Event()
        marker_groups = (
            tuple(f"run-a-{index}" for index in range(8)),
            tuple(f"run-b-{index}" for index in range(8)),
        )
        processes = [
            context.Process(
                target=_append_program_markers,
                args=(config, card.id, markers, ready, start),
            )
            for markers in marker_groups
        ]
        for process in processes:
            process.start()
        for _ in processes:
            assert ready.get(timeout=30) is True
        start.set()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0

        persisted = CardBank(config.bank_file).get(card.id)
        assert persisted is not None
        assert set(persisted.programs) == set(marker_groups[0] + marker_groups[1])

    def test_authoring_transaction_serializes_cross_process_admission(
        self, make_store_config
    ):
        config = make_store_config()
        context = multiprocessing.get_context("fork")
        ready = context.Queue()
        start = context.Event()
        markers = ("run-a", "run-b")
        processes = [
            context.Process(
                target=_admit_under_authoring_transaction,
                args=(config, marker, ready, start),
            )
            for marker in markers
        ]
        for process in processes:
            process.start()
        for _ in processes:
            assert ready.get(timeout=30) is True
        start.set()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0

        persisted = CardBank(config.bank_file).snapshot()
        assert len(persisted) == 1
        assert set(persisted[0].programs) == set(markers)

    def test_update_missing_skips_transform(self, make_store_config):
        store = LocalMemoryStore(make_store_config())
        transformed = False

        def transform(card):
            nonlocal transformed
            transformed = True
            return card

        assert store.update("mem-vanished", transform) is None
        assert transformed is False

    def test_corrupt_bank_raises(self, make_store_config):
        config = make_store_config()
        config.path.mkdir(parents=True, exist_ok=True)
        config.bank_file.write_text("{not json", encoding="utf-8")
        with pytest.raises(MemoryStorageError, match="corrupt card bank"):
            LocalMemoryStore(config)

    def test_rebuild_returns_to_ready(self, make_store_config, make_card):
        store = LocalMemoryStore(make_store_config())
        store.save(make_card())
        store.rebuild()
        assert store.state is StoreState.READY

    def test_rebuild_surfaces_bank_corruption_as_error_state(
        self, make_store_config, make_card
    ):
        config = make_store_config()
        store = LocalMemoryStore(config)
        store.save(make_card())
        config.bank_file.write_text("{corrupt", encoding="utf-8")
        with pytest.raises(MemoryStorageError):
            store.rebuild()
        assert store.state is StoreState.ERROR
        assert not store.is_ready

    def test_save_rolls_back_ram_on_persist_failure(
        self, make_store_config, make_card, monkeypatch
    ):
        store = LocalMemoryStore(make_store_config())
        card = make_card()

        def fail_persist():
            raise RuntimeError("disk full")

        monkeypatch.setattr(store._bank, "persist", fail_persist)
        with pytest.raises(RuntimeError, match="disk full"):
            store.save(card)

        assert store.get(card.id) is None
        assert store.snapshot() == ()

    def test_delete_rolls_back_ram_on_persist_failure(
        self, make_store_config, make_card, monkeypatch
    ):
        store = LocalMemoryStore(make_store_config())
        card = make_card()
        store.save(card)

        def fail_persist():
            raise RuntimeError("disk full")

        monkeypatch.setattr(store._bank, "persist", fail_persist)
        with pytest.raises(RuntimeError, match="disk full"):
            store.delete(card.id)

        assert store.get(card.id) == card

    def test_nearest_ranks_matching_card_first(self, make_store_config, make_card):
        store = LocalMemoryStore(make_store_config())
        target = make_card(description="zebra quantum lattice")
        store.save(target)
        store.save(make_card(description="ordinary gradient descent"))
        hits = store.nearest("zebra quantum lattice", k=2)
        assert hits
        assert hits[0].card.id == target.id
        assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)

    def test_nearest_kind_filter(self, make_store_config, make_card):
        store = LocalMemoryStore(make_store_config())
        store.save(make_card(description="shared topic"))
        exemplar = make_card(
            kind=CardKind.PROGRAM, program_id="prog-1", description="shared topic"
        )
        store.save(exemplar)
        hits = store.nearest("shared topic", k=5, kind=CardKind.PROGRAM)
        assert [hit.card.id for hit in hits] == [exemplar.id]

    def test_nearest_task_filter(self, make_store_config, make_card):
        store = LocalMemoryStore(make_store_config())
        own = make_card(task_key="own", description="shared topic")
        foreign = make_card(task_key="foreign", description="shared topic")
        store.save(own)
        store.save(foreign)

        hits = store.nearest("shared topic", k=5, task_key="own")

        assert [hit.card.id for hit in hits] == [own.id]

    def test_deleted_card_leaves_retrieval(self, make_store_config, make_card):
        store = LocalMemoryStore(make_store_config())
        card = make_card(description="zebra quantum lattice")
        store.save(card)
        store.delete(card.id)
        assert store.nearest("zebra quantum lattice", k=5) == []

    async def test_research_without_llm_is_empty(self, make_store_config):
        store = LocalMemoryStore(make_store_config())
        result = await store.research(ResearchRequest(query="anything"))
        assert result.cards == ()
        assert result.iterations == 0

    async def test_research_event_recorded(self, make_store_config, tmp_path):
        store = LocalMemoryStore(make_store_config())
        events_file = tmp_path / "events.jsonl"
        with memory_event_context(event_path=events_file):
            await store.research(ResearchRequest(query="anything"))
        rows = [json.loads(line) for line in events_file.read_text().splitlines()]
        assert [row["event"] for row in rows] == ["MEMORY_RESEARCH"]
        assert rows[0]["outcome"] == "empty"
