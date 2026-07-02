"""CardAdmissionGate verdicts and the WriteLedger audit trail."""

from __future__ import annotations

import json

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.write.admission import (
    CardAdmissionGate,
    WriteLedger,
    WriteLedgerRecord,
    WriteOutcome,
)
from gigaevo.memory.write.eviction import NullEvictor


class MarkingEvictor:
    """Evicts exactly the ids in ``harmful``."""

    def __init__(self, harmful: set[str]) -> None:
        self._harmful = harmful

    def should_evict(self, card: Card) -> bool:
        return card.id in self._harmful

    def sweep(self, cards) -> list[str]:
        return [card.id for card in cards if self.should_evict(card)]


def read_rows(path) -> list[WriteLedgerRecord]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [WriteLedgerRecord.model_validate(json.loads(line)) for line in f]


def make_gate(store, tmp_path, evictor=None):
    ledger = WriteLedger(tmp_path / "write_ledger.jsonl")
    gate = CardAdmissionGate(
        store=store, evictor=evictor or NullEvictor(), ledger=ledger
    )
    return gate, ledger


def test_admit_empty_id_mints_and_records_added(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    final_id = gate.admit(make_card(id=""))
    assert final_id.startswith("minted-")
    assert store.get(final_id) is not None
    rows = read_rows(ledger.path)
    assert len(rows) == 1
    assert rows[0].outcome is WriteOutcome.ADDED
    assert rows[0].final_id == final_id


def test_admit_known_id_records_updated(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    card = make_card()
    store.save(card)
    final_id = gate.admit(card.model_copy(update={"description": "revised"}))
    assert final_id == card.id
    assert store.get(card.id).description == "revised"
    assert read_rows(ledger.path)[-1].outcome is WriteOutcome.UPDATED


def test_admit_harmful_known_card_deletes_and_rejects(store, make_card, tmp_path):
    card = make_card()
    store.save(card)
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor({card.id}))
    assert gate.admit(card) == ""
    assert store.get(card.id) is None
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.REJECTED_HARM
    assert row.final_id == ""


def test_admit_harmful_unknown_card_does_not_delete(store, make_card, tmp_path):
    card = make_card()
    gate, _ = make_gate(store, tmp_path, MarkingEvictor({card.id}))
    assert gate.admit(card) == ""
    assert store.deleted_ids == []


def test_merge_missing_target_is_noop(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    assert gate.merge("absent", make_card()) == ""
    assert read_rows(ledger.path) == []


def test_merge_program_target_is_noop(store, make_card, tmp_path):
    target = make_card(
        kind=CardKind.PROGRAM, program_id="p1", code="x = 1", fitness=0.5
    )
    store.save(target)
    gate, _ = make_gate(store, tmp_path)
    assert gate.merge(target.id, make_card()) == ""


def test_merge_success_records_submitted_card_id(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    incoming = make_card()
    gate, ledger = make_gate(store, tmp_path)
    assert gate.merge(target.id, incoming) == target.id
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.MERGED
    assert row.incoming_id == incoming.id
    assert row.final_id == target.id
    assert row.merge_targets == (target.id,)
    assert store.get(target.id).description == incoming.description


def test_merge_harmful_union_deletes_target(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    incoming = make_card()
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor({target.id}))
    assert gate.merge(target.id, incoming) == ""
    assert store.get(target.id) is None
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.REJECTED_HARM
    assert row.incoming_id == incoming.id


def test_merge_store_failure_records_nothing(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    store.fail_merges = True
    gate, ledger = make_gate(store, tmp_path)
    assert gate.merge(target.id, make_card()) == ""
    assert read_rows(ledger.path) == []


def test_bump_provenance_appends_child_once(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    gate, ledger = make_gate(store, tmp_path)

    assert gate.bump_provenance(target.id, "child-1") == target.id
    assert store.get(target.id).programs == ("child-1",)

    saves_before = len(store.saved_ids)
    assert gate.bump_provenance(target.id, "child-1") == target.id
    assert len(store.saved_ids) == saves_before
    assert store.get(target.id).programs == ("child-1",)

    rows = read_rows(ledger.path)
    assert [r.outcome for r in rows] == [WriteOutcome.UPDATED, WriteOutcome.UPDATED]


def test_bump_provenance_missing_or_program_target_is_noop(store, make_card, tmp_path):
    program = make_card(
        kind=CardKind.PROGRAM, program_id="p1", code="x = 1", fitness=0.5
    )
    store.save(program)
    gate, _ = make_gate(store, tmp_path)
    assert gate.bump_provenance("absent", "child-1") == ""
    assert gate.bump_provenance(program.id, "child-1") == ""


def test_sweep_deletes_evicted_and_records(store, make_card, tmp_path):
    bad = make_card()
    good = make_card()
    store.save(bad)
    store.save(good)
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor({bad.id}))
    assert gate.sweep() == [bad.id]
    assert store.get(bad.id) is None
    assert store.get(good.id) is not None
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.EVICTED
    assert row.incoming_id == bad.id


def test_ledger_record_never_raises_on_unwritable_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    ledger = WriteLedger(blocker / "ledger.jsonl")
    ledger.record(incoming_id="a", final_id="a", outcome=WriteOutcome.ADDED)
