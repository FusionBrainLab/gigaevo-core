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


def test_missing_index_file_is_clean_cold_start(tmp_path):
    store = CardStore(index_file=tmp_path / "absent.json")

    assert store.cards == {}


def test_corrupt_index_file_raises(tmp_path):
    index = tmp_path / "index.json"
    index.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(MemoryStorageError):
        CardStore(index_file=index)


def test_valid_index_file_loads(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "memory_cards": {
                    "mem-1": {"id": "mem-1", "description": "use ridge targets"}
                },
                "entity_by_card_id": {},
                "entity_version_by_entity": {},
            }
        ),
        encoding="utf-8",
    )

    store = CardStore(index_file=index)

    assert set(store.cards) == {"mem-1"}
