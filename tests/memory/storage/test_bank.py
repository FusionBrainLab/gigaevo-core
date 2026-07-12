"""CardBank snapshot caching: identity until mutation, invalidation on writes."""

from __future__ import annotations

from gigaevo.memory.storage.bank import CardBank


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
