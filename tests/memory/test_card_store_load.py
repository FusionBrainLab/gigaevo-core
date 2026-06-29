"""CardStore index-file load contract.

A missing index file is the legitimate cold start; an EXISTING file that
fails to parse means the bank's persisted state is corrupt and must fail
fast (``MemoryStorageError``) instead of silently starting empty and then
overwriting the corrupt-but-recoverable file on the next persist.
"""

from __future__ import annotations

import json

import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.models import MemoryCard


def test_missing_index_file_is_clean_cold_start(tmp_path):
    store = CardStore(index_file=tmp_path / "absent.json")

    assert store.cards == {}


def test_corrupt_index_file_raises(tmp_path):
    index = tmp_path / "index.json"
    index.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(MemoryStorageError):
        CardStore(index_file=index)


def _write_index(index, cards: dict, mapping: dict | None = None) -> None:
    index.write_text(
        json.dumps(
            {
                "memory_cards": cards,
                "entity_by_card_id": mapping or {},
                "entity_version_by_entity": {},
            }
        ),
        encoding="utf-8",
    )


def test_valid_index_file_loads(tmp_path):
    index = tmp_path / "index.json"
    _write_index(index, {"mem-1": {"id": "mem-1", "description": "use ridge targets"}})

    store = CardStore(index_file=index)

    assert set(store.cards) == {"mem-1"}


def test_reload_picks_up_writer_additions(tmp_path):
    index = tmp_path / "index.json"
    _write_index(index, {"mem-1": {"id": "mem-1", "description": "use ridge targets"}})
    store = CardStore(index_file=index)
    _write_index(
        index,
        {
            "mem-1": {"id": "mem-1", "description": "use ridge targets"},
            "mem-2": {"id": "mem-2", "description": "clip outliers"},
        },
        {"mem-2": "entity-2"},
    )

    store.reload()

    assert set(store.cards) == {"mem-1", "mem-2"}
    assert store.entity_by_card_id == {"mem-2": "entity-2"}
    assert store.card_id_by_entity == {"entity-2": "mem-2"}


def test_persist_reload_preserves_absorbed_ids(tmp_path):
    # A merged survivor's absorbed_ids re-alias absorbed cards' frozen gain
    # attribution at the next restamp. persist() serializes them, but reload runs
    # through normalize_memory_card — they must survive the roundtrip, or a resume
    # (or reader reload) before the next restamp re-orphans the absorbed events.
    index = tmp_path / "index.json"
    store = CardStore(index_file=index)
    store.put("mem-S", MemoryCard(id="mem-S", absorbed_ids=["mem-P"]))
    store.persist()

    reloaded = CardStore(index_file=index)

    card = reloaded.get("mem-S")
    assert isinstance(card, MemoryCard)
    assert card.absorbed_ids == ["mem-P"]


def test_reload_on_corrupt_index_keeps_last_good_snapshot(tmp_path):
    # A reader hitting a corrupt index mid-run must keep serving its last
    # good state — clearing before the parse would leave it permanently
    # empty while the warn-and-continue caller keeps the run alive.
    index = tmp_path / "index.json"
    _write_index(
        index,
        {"mem-1": {"id": "mem-1", "description": "use ridge targets"}},
        {"mem-1": "entity-1"},
    )
    store = CardStore(index_file=index)
    index.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(MemoryStorageError):
        store.reload()

    assert set(store.cards) == {"mem-1"}
    assert store.entity_by_card_id == {"mem-1": "entity-1"}
    assert store.card_id_by_entity == {"entity-1": "mem-1"}
