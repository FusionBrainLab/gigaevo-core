"""CardBank snapshot caching: identity until mutation, invalidation on writes."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.cards import ContextualGain, DecisionContext
from gigaevo.memory.storage.bank import AsyncCardBankFileLock, CardBank


def test_snapshot_identity_between_reads(tmp_path, make_card):
    bank = CardBank(tmp_path / "bank.json")
    bank.put(make_card())
    assert bank.snapshot() is bank.snapshot()


def test_put_invalidates_snapshot(tmp_path, make_card):
    bank = CardBank(tmp_path / "bank.json")
    bank.put(make_card())
    before = bank.snapshot()
    card = make_card()
    bank.put(card)
    after = bank.snapshot()
    assert after is not before
    assert card.id in {c.id for c in after}


def test_remove_invalidates_snapshot(tmp_path, make_card):
    bank = CardBank(tmp_path / "bank.json")
    card = make_card()
    bank.put(card)
    before = bank.snapshot()
    assert bank.remove(card.id)
    assert bank.snapshot() == ()
    assert [c.id for c in before] == [card.id]


def test_remove_missing_keeps_snapshot(tmp_path, make_card):
    bank = CardBank(tmp_path / "bank.json")
    bank.put(make_card())
    before = bank.snapshot()
    assert not bank.remove("mem-absent")
    assert bank.snapshot() is before


def test_restore_snapshot_invalidates_cache(tmp_path, make_card):
    bank = CardBank(tmp_path / "bank.json")
    bank.put(make_card())
    bank.snapshot()
    bank.restore_snapshot(())
    assert bank.snapshot() == ()


def test_reload_invalidates_snapshot(tmp_path, make_card):
    path = tmp_path / "bank.json"
    reader = CardBank(path)
    assert reader.snapshot() == ()
    writer = CardBank(path)
    card = make_card()
    writer.put(card)
    writer.persist()
    reader.reload()
    assert [c.id for c in reader.snapshot()] == [card.id]


def test_task_keys_round_trip_through_local_bank(tmp_path, make_card):
    path = tmp_path / "bank.json"
    card = make_card(
        task_key="heilbronn",
        gain_events=(
            ContextualGain(
                context=DecisionContext(task_key="heilbronn", parent_id="parent-1"),
                gain=0.2,
            ),
        ),
    )
    bank = CardBank(path)
    bank.put(card)
    bank.persist()

    (restored,) = CardBank(path).snapshot()

    assert restored.task_key == "heilbronn"
    assert restored.gain_events[0].context.task_key == "heilbronn"


def test_reload_rejects_payload_key_embedded_id_mismatch(tmp_path, make_card):
    path = tmp_path / "bank.json"
    card = make_card()
    path.write_text(
        json.dumps({"cards": {"mem-wrong-key": card.model_dump(mode="json")}}),
        encoding="utf-8",
    )

    with pytest.raises(MemoryStorageError, match="embedded card id"):
        CardBank(path)


@pytest.mark.asyncio
async def test_cancelled_async_lock_waiter_does_not_leak_lock(tmp_path):
    path = tmp_path / "authoring.lock"
    entered = False

    async def wait_for_lock() -> None:
        nonlocal entered
        async with AsyncCardBankFileLock(path, poll_seconds=0.001):
            entered = True

    async with AsyncCardBankFileLock(path):
        waiter = asyncio.create_task(wait_for_lock())
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    assert entered is False
    async with AsyncCardBankFileLock(path):
        pass


def test_reload_if_changed_detects_same_size_same_mtime_atomic_replace(
    tmp_path, make_card
):
    path = tmp_path / "bank.json"
    card = make_card(description="alpha")
    bank = CardBank(path)
    bank.put(card)
    bank.persist()
    before = path.stat()

    replacement = tmp_path / "replacement.json"
    replacement_card = card.model_copy(update={"description": "bravo"})
    replacement.write_text(
        json.dumps(
            {"cards": {card.id: replacement_card.model_dump(mode="json")}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert replacement.stat().st_size == before.st_size
    os.replace(replacement, path)
    assert (path.stat().st_dev, path.stat().st_ino) != (before.st_dev, before.st_ino)

    assert bank.reload_if_changed() is True
    assert bank.get(card.id).description == "bravo"
