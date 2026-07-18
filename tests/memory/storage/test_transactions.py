"""Cross-operation card-bank transaction regressions."""

from __future__ import annotations

from threading import Thread

import pytest

from gigaevo.exceptions import StorageError
from gigaevo.memory.storage.local import LocalMemoryStore


def _run_thread(call):
    errors: list[BaseException] = []
    result: list = []

    def run() -> None:
        try:
            result.append(call())
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert errors == []
    return result[0]


def test_update_transform_can_read_same_store(make_store_config, make_card):
    store = LocalMemoryStore(make_store_config())
    card = make_card(programs=("before",))
    store.save(card)

    def transform(fresh):
        assert store.snapshot() == (fresh,)
        assert store.get(card.id) == fresh
        return fresh.model_copy(update={"programs": (*fresh.programs, "after")})

    updated = _run_thread(lambda: store.update(card.id, transform))

    assert updated is not None
    assert updated.programs == ("before", "after")
    assert store.get(card.id) == updated


def test_shared_bank_lock_refuses_exclusive_upgrade(make_store_config, make_card):
    store = LocalMemoryStore(make_store_config())
    card = make_card()
    store.save(card)

    with store._lock, store._bank_file_lock(exclusive=False):
        with pytest.raises(StorageError, match="shared card-bank lock"):
            store.update(card.id, lambda fresh: fresh)

    assert store.get(card.id) == card
