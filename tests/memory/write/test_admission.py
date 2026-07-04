"""CardAdmissionGate verdicts and the WriteLedger audit trail."""

from __future__ import annotations

import json

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.write.admission import (
    CardAdmissionGate,
    WriteLedger,
    WriteLedgerRecord,
    WriteOutcome,
    WriteResult,
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
    result = gate.admit(make_card(id=""))
    assert result.outcome is WriteOutcome.ADDED
    assert result.landed
    assert result.card_id.startswith("minted-")
    assert store.get(result.card_id) is not None
    rows = read_rows(ledger.path)
    assert len(rows) == 1
    assert rows[0].outcome is WriteOutcome.ADDED
    assert rows[0].final_id == result.card_id


def test_admit_known_id_records_updated(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    card = make_card()
    store.save(card)
    result = gate.admit(card.model_copy(update={"description": "revised"}))
    assert result.outcome is WriteOutcome.UPDATED
    assert result.card_id == card.id
    assert store.get(card.id).description == "revised"
    assert read_rows(ledger.path)[-1].outcome is WriteOutcome.UPDATED


def test_admit_harmful_known_card_deletes_and_rejects(store, make_card, tmp_path):
    card = make_card()
    store.save(card)
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor({card.id}))
    result = gate.admit(card)
    assert result.rejected_harm
    assert result.card_id == ""
    assert not result.landed
    assert store.get(card.id) is None
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.REJECTED_HARM
    assert row.final_id == ""


def test_admit_harmful_unknown_card_does_not_delete(store, make_card, tmp_path):
    card = make_card()
    gate, _ = make_gate(store, tmp_path, MarkingEvictor({card.id}))
    assert gate.admit(card).rejected_harm
    assert store.deleted_ids == []


def test_merge_missing_target_is_benign_noop(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    result = gate.merge("absent", make_card())
    assert result.outcome is WriteOutcome.DISCARDED
    assert not result.landed
    assert not result.rejected_harm
    assert read_rows(ledger.path) == []


def test_merge_program_target_is_benign_noop(store, make_card, tmp_path):
    target = make_card(
        kind=CardKind.PROGRAM, program_id="p1", code="x = 1", fitness=0.5
    )
    store.save(target)
    gate, _ = make_gate(store, tmp_path)
    result = gate.merge(target.id, make_card())
    assert result.outcome is WriteOutcome.DISCARDED
    assert not result.rejected_harm


def test_merge_success_records_submitted_card_id(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    incoming = make_card()
    gate, ledger = make_gate(store, tmp_path)
    result = gate.merge(target.id, incoming)
    assert result.outcome is WriteOutcome.MERGED
    assert result.card_id == target.id
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
    result = gate.merge(target.id, incoming)
    assert result.rejected_harm
    assert result.card_id == ""
    assert store.get(target.id) is None
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.REJECTED_HARM
    assert row.incoming_id == incoming.id


def test_merge_store_failure_is_benign_noop(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    store.fail_merges = True
    gate, ledger = make_gate(store, tmp_path)
    result = gate.merge(target.id, make_card())
    assert result.outcome is WriteOutcome.DISCARDED
    assert not result.landed
    assert not result.rejected_harm
    assert read_rows(ledger.path) == []


def test_bump_provenance_appends_child_once(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    gate, ledger = make_gate(store, tmp_path)

    assert gate.bump_provenance(target.id, "child-1").card_id == target.id
    assert store.get(target.id).programs == ("child-1",)

    saves_before = len(store.saved_ids)
    assert gate.bump_provenance(target.id, "child-1").card_id == target.id
    assert len(store.saved_ids) == saves_before
    assert store.get(target.id).programs == ("child-1",)

    rows = read_rows(ledger.path)
    assert [r.outcome for r in rows] == [WriteOutcome.UPDATED, WriteOutcome.UPDATED]


def test_bump_provenance_missing_or_program_target_is_benign_noop(
    store, make_card, tmp_path
):
    program = make_card(
        kind=CardKind.PROGRAM, program_id="p1", code="x = 1", fitness=0.5
    )
    store.save(program)
    gate, _ = make_gate(store, tmp_path)
    assert gate.bump_provenance("absent", "child-1").outcome is WriteOutcome.DISCARDED
    assert gate.bump_provenance(program.id, "child-1").outcome is WriteOutcome.DISCARDED


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


def test_write_result_benign_noop_is_discarded_only():
    # Only DISCARDED is safe to re-author as fresh. Every other non-landed
    # verdict is a harm-driven deletion the librarian must drop — EVICTED must
    # never read as a benign no-op or a swept-harmful card resurrects as NEW.
    landed = {WriteOutcome.ADDED, WriteOutcome.UPDATED, WriteOutcome.MERGED}
    for outcome in WriteOutcome:
        card_id = "x" if outcome in landed else ""
        result = WriteResult(outcome=outcome, card_id=card_id)
        assert result.benign_noop is (outcome is WriteOutcome.DISCARDED)
        assert result.rejected_harm is (outcome is WriteOutcome.REJECTED_HARM)
        assert result.landed is (outcome in landed)
    assert not WriteResult(outcome=WriteOutcome.EVICTED).benign_noop


def test_ledger_record_never_raises_on_unwritable_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    ledger = WriteLedger(blocker / "ledger.jsonl")
    ledger.record(incoming_id="a", final_id="a", outcome=WriteOutcome.ADDED)
