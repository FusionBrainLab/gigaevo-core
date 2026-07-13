"""Cross-operation card-bank transaction regressions."""

from __future__ import annotations

from threading import Barrier, Thread

import pytest

from gigaevo.exceptions import MergeAborted, StorageError
from gigaevo.memory.cards import ContextualGain, DecisionContext
from gigaevo.memory.storage.bank import CardBank
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.write.merge import merge_cards


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


def test_merge_retire_uses_fresh_target_evidence(make_store_config, make_card):
    store = LocalMemoryStore(make_store_config())
    target = make_card(id="mem-target")
    partner = make_card(id="mem-partner")
    store.save(target)
    store.save(partner)
    stale_target = store.get(target.id)
    restamped = ContextualGain(
        context=DecisionContext(parent_id="restamped-parent"), gain=0.5
    )
    store.update(
        target.id,
        lambda fresh: fresh.model_copy(
            update={"gain_events": (*fresh.gain_events, restamped)}
        ),
    )
    folded_targets = []

    def fold(fresh_target, fresh_partner):
        folded_targets.append(fresh_target)
        assert fresh_partner == partner
        return merge_cards(fresh_target, fresh_partner, replace_description=True)

    result = store.merge_retire(target.id, partner.id, fold)

    assert result.outcome == "merged"
    assert stale_target is not None
    assert folded_targets[0] != stale_target
    survivor = store.get(target.id)
    assert survivor is not None
    assert restamped in survivor.gain_events
    assert store.get(partner.id) is None


def test_merge_retire_handles_vanished_cards(make_store_config, make_card):
    config = make_store_config()
    store = LocalMemoryStore(config)
    target = make_card(id="mem-target")
    partner = make_card(id="mem-partner")
    store.save(target)
    store.save(partner)
    store.delete(partner.id)
    seen_partners = []

    result = store.merge_retire(
        target.id,
        partner.id,
        lambda fresh, fresh_partner: (
            seen_partners.append(fresh_partner)
            or fresh.model_copy(update={"description": "partner vanished"})
        ),
    )

    assert result.outcome == "merged"
    assert seen_partners == [None]
    survivor = store.get(target.id)
    assert survivor is not None
    assert survivor.description == "partner vanished"

    vanished_target = make_card(id="mem-vanished-target")
    remaining_partner = make_card(id="mem-remaining-partner")
    store.save(vanished_target)
    store.save(remaining_partner)
    store.delete(vanished_target.id)
    before = config.bank_file.read_bytes()
    folded = False

    def forbidden_fold(fresh, fresh_partner):
        nonlocal folded
        folded = True
        return fresh

    missing = store.merge_retire(
        vanished_target.id, remaining_partner.id, forbidden_fold
    )

    assert missing.outcome == "target_missing"
    assert folded is False
    assert config.bank_file.read_bytes() == before
    assert store.get(remaining_partner.id) == remaining_partner


def test_opposite_direction_merges_cannot_delete_both_cards(
    make_store_config, make_card
):
    config = make_store_config()
    store_a = LocalMemoryStore(config)
    store_b = LocalMemoryStore(config)

    for iteration in range(20):
        card_x = make_card(id=f"mem-race-x-{iteration}")
        card_y = make_card(id=f"mem-race-y-{iteration}")
        store_a.save(card_x)
        store_a.save(card_y)
        barrier = Barrier(3)
        errors: list[BaseException] = []

        def merge(store, target_id, partner_id):
            try:
                barrier.wait()
                store.merge_retire(
                    target_id,
                    partner_id,
                    lambda target, partner: merge_cards(
                        target,
                        partner if partner is not None else target,
                        replace_description=False,
                    ),
                )
            except BaseException as exc:
                errors.append(exc)

        workers = (
            Thread(
                target=merge,
                args=(store_a, card_x.id, card_y.id),
                daemon=True,
            ),
            Thread(
                target=merge,
                args=(store_b, card_y.id, card_x.id),
                daemon=True,
            ),
        )
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(timeout=10)
            assert not worker.is_alive()
        assert errors == []
        bank_ids = {card.id for card in CardBank(config.bank_file).snapshot()}
        assert {card_x.id, card_y.id} & bank_ids


def test_harm_fold_removes_cards_and_index_entries(make_store_config, make_card):
    store = LocalMemoryStore(make_store_config())
    target = make_card(id="mem-harm-target", description="harm target alpha")
    partner = make_card(id="mem-harm-partner", description="harm partner beta")
    store.save(target)
    store.save(partner)

    result = store.merge_retire(target.id, partner.id, lambda _target, _partner: None)

    assert result.outcome == "retired"
    assert store.snapshot() == ()
    hits = store._index.query(
        store._config.embed.nearest_scope, "harm target alpha partner beta", 10
    )
    assert hits == []


def test_merge_aborted_leaves_bank_byte_identical(make_store_config, make_card):
    config = make_store_config()
    store = LocalMemoryStore(config)
    target = make_card(id="mem-abort-target")
    partner = make_card(id="mem-abort-partner")
    store.save(target)
    store.save(partner)
    before = config.bank_file.read_bytes()

    def abort(_target, _partner):
        raise MergeAborted

    result = store.merge_retire(target.id, partner.id, abort)

    assert result.outcome == "aborted"
    assert config.bank_file.read_bytes() == before
    assert {card.id for card in store.snapshot()} == {target.id, partner.id}


def test_merge_retire_rejects_alias_cycles(make_store_config, make_card):
    config = make_store_config()
    store = LocalMemoryStore(config)
    target = make_card(id="mem-alias-target")
    partner = make_card(id="mem-alias-partner", absorbed_ids=("mem-alias-target",))
    store.save(target)
    store.save(partner)
    before = config.bank_file.read_bytes()

    reverse = store.merge_retire(target.id, partner.id, lambda fresh, _: fresh)
    identical = store.merge_retire(target.id, target.id, lambda fresh, _: fresh)
    self_absorbing = store.merge_retire(
        target.id,
        "",
        lambda fresh, _: fresh.model_copy(
            update={"absorbed_ids": (*fresh.absorbed_ids, fresh.id)}
        ),
    )

    assert [reverse.outcome, identical.outcome, self_absorbing.outcome] == [
        "aborted",
        "aborted",
        "aborted",
    ]
    assert config.bank_file.read_bytes() == before


def test_merge_retire_validates_survivor_id(make_store_config, make_card):
    config = make_store_config()
    store = LocalMemoryStore(config)
    target = make_card(id="mem-id-target")
    store.save(target)
    before = config.bank_file.read_bytes()

    with pytest.raises(ValueError, match="keep the target id"):
        store.merge_retire(
            target.id,
            "",
            lambda fresh, _: fresh.model_copy(update={"id": "mem-changed-id"}),
        )

    assert config.bank_file.read_bytes() == before
