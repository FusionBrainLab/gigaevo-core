"""MemoryStore contract against both backends, plus backend-specific behavior.

The shared contract runs parametrized over LocalMemoryStore (tmpdir bank +
Chroma) and RemoteMemoryStore (httpx MockTransport over an in-memory card
service). Backend-specific sections cover local persistence/retrieval/state
and the remote skeleton's not-yet-implemented surface.
"""

from __future__ import annotations

import json

import httpx
import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.cards import CardKind
from gigaevo.memory.events import memory_event_context
from gigaevo.memory.storage.base import ResearchRequest
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.storage.remote import RemoteMemoryStore
from gigaevo.memory.storage.state import StoreState


class FakeCardService:
    def __init__(self) -> None:
        self.cards: dict[str, dict] = {}
        self.rebuilds = 0
        self.healthy = True

    def client(self) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(self._handle),
            base_url="http://memory.test",
        )

    def _mint(self, card: dict) -> str:
        card_id = card["id"] or f"mem-remote{len(self.cards):04d}"
        card["id"] = card_id
        self.cards[card_id] = card
        return card_id

    def _handle(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        if path == "/health":
            return httpx.Response(200 if self.healthy else 503)
        if path == "/cards" and method == "POST":
            card_id = self._mint(json.loads(request.content))
            return httpx.Response(200, json={"id": card_id})
        if path == "/cards" and method == "GET":
            return httpx.Response(200, json={"cards": list(self.cards.values())})
        if path == "/cards/merge" and method == "POST":
            payload = json.loads(request.content)
            ids = [self._mint(card) for card in payload["cards"]]
            return httpx.Response(200, json={"ids": ids})
        if path == "/rebuild" and method == "POST":
            self.rebuilds += 1
            return httpx.Response(200)
        if path.startswith("/cards/"):
            card_id = path.removeprefix("/cards/")
            if card_id not in self.cards:
                return httpx.Response(404)
            if method == "GET":
                return httpx.Response(200, json=self.cards[card_id])
            if method == "DELETE":
                del self.cards[card_id]
                return httpx.Response(200)
        return httpx.Response(404)


@pytest.fixture(params=["local", "remote"])
def store(request, make_store_config):
    if request.param == "local":
        with LocalMemoryStore(make_store_config()) as local:
            yield local
    else:
        with RemoteMemoryStore(client=FakeCardService().client()) as remote:
            yield remote


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


def test_apply_merges_saves_survivors(store, make_card):
    survivor = make_card(absorbed_ids=("mem-absorbed00",))
    fresh = make_card(id="")
    ids = store.apply_merges([survivor, fresh])
    assert ids[0] == survivor.id
    assert ids[1].startswith("mem-")
    assert store.get(ids[0]) == survivor
    assert store.get(ids[1]) is not None


def test_apply_merges_empty_is_noop(store):
    assert store.apply_merges([]) == []


class TestLocalStore:
    def test_initializes_ready(self, make_store_config):
        store = LocalMemoryStore(make_store_config())
        assert store.is_ready
        assert store.state is StoreState.READY

    def test_external_writer_visible_after_refresh(self, make_store_config, make_card):
        config = make_store_config()
        writer = LocalMemoryStore(config)
        reader = LocalMemoryStore(config)
        card = make_card()
        writer.save(card)
        assert card in reader.snapshot()
        assert reader.get(card.id) == card

    def test_bank_survives_restart(self, make_store_config, make_card):
        config = make_store_config()
        card = make_card()
        LocalMemoryStore(config).save(card)
        assert LocalMemoryStore(config).get(card.id) == card

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


class TestRemoteStore:
    def test_requires_url_or_client(self):
        with pytest.raises(ValueError, match="base_url"):
            RemoteMemoryStore()

    def test_is_ready_reflects_health(self):
        service = FakeCardService()
        store = RemoteMemoryStore(client=service.client())
        assert store.is_ready
        service.healthy = False
        assert not store.is_ready

    def test_is_ready_false_on_transport_error(self):
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = httpx.Client(
            transport=httpx.MockTransport(refuse), base_url="http://memory.test"
        )
        assert not RemoteMemoryStore(client=client).is_ready

    def test_retrieval_not_implemented(self, make_card):
        store = RemoteMemoryStore(client=FakeCardService().client())
        with pytest.raises(NotImplementedError):
            store.nearest("text", k=3)

    async def test_research_not_implemented(self):
        store = RemoteMemoryStore(client=FakeCardService().client())
        with pytest.raises(NotImplementedError):
            await store.research(ResearchRequest(query="anything"))

    def test_rebuild_posts(self):
        service = FakeCardService()
        RemoteMemoryStore(client=service.client()).rebuild()
        assert service.rebuilds == 1
