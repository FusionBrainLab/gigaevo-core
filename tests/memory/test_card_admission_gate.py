"""Behavior tests for CardAdmissionGate: harm gate + ledger, no dedup."""

from __future__ import annotations

import json
from pathlib import Path

from gigaevo.memory.core.admission_gate import CardAdmissionGate
from gigaevo.memory.core.write_ledger import WriteLedger
from gigaevo.memory.shared_memory.models import MemoryCard

_IDX = Path("/tmp/claude-1000/_gate_test_idx.json")


class _FakeStore:
    """Backend twin: exposes the methods the gate touches."""

    def __init__(self) -> None:
        from gigaevo.memory.shared_memory.card_store import CardStore

        self.card_store = CardStore(index_file=_IDX)
        self.card_store.cards.clear()
        self.saved: list[MemoryCard] = []
        self.deleted: list[str] = []

    def save_card_direct(self, card: MemoryCard) -> str:
        self.card_store.cards[card.id] = card
        self.saved.append(card)
        return card.id

    def delete(self, card_id: str) -> bool:
        self.deleted.append(card_id)
        return self.card_store.cards.pop(card_id, None) is not None

    def apply_merges(self, merges: list[tuple[str, MemoryCard]]) -> list[str]:
        ids = []
        for target_id, card in merges:
            self.card_store.cards[target_id] = card
            ids.append(target_id)
        return ids


class _NeverHarmful:
    def should_evict(self, card: MemoryCard) -> bool:
        return False

    def sweep(self, bank):  # noqa: ANN001
        return []


class _AlwaysHarmful:
    def should_evict(self, card: MemoryCard) -> bool:
        return True

    def sweep(self, bank):  # noqa: ANN001
        return list(bank)


def _card(cid: str = "mem-1", desc: str = "spectral gap widening") -> MemoryCard:
    return MemoryCard(id=cid, description=desc, keywords=["spectral"])


class TestAdmit:
    def test_clean_new_card_is_added_and_stored(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        final_id = gate.admit(_card("mem-1"))
        assert final_id == "mem-1"
        assert "mem-1" in store.card_store.cards

    def test_confidently_harmful_card_is_rejected_and_not_stored(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_AlwaysHarmful())
        final_id = gate.admit(_card("mem-2"))
        assert final_id == ""
        assert "mem-2" not in store.card_store.cards

    def test_known_id_is_updated_in_place(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        gate.admit(_card("mem-3", "old"))
        final_id = gate.admit(_card("mem-3", "new prose"))
        assert final_id == "mem-3"
        assert store.card_store.cards["mem-3"].description == "new prose"


class TestMerge:
    def test_merge_applies_to_target_and_returns_its_id(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        gate.admit(_card("mem-A", "original"))
        merged = _card("mem-A", "synthesized union")
        final_id = gate.merge("mem-A", merged)
        assert final_id == "mem-A"
        assert store.card_store.cards["mem-A"].description == "synthesized union"

    def test_merge_into_missing_target_returns_empty(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        assert gate.merge("nope", _card("nope")) == ""

    def test_merge_stamps_target_id_onto_idless_incoming_card(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        gate.admit(_card("mem-A", "original"))
        final_id = gate.merge("mem-A", _card("", "synthesized union"))
        assert final_id == "mem-A"
        assert store.card_store.cards["mem-A"].description == "synthesized union"

    def test_bump_provenance_appends_child_and_returns_target(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        gate.admit(_card("mem-D"))
        gate.bump_provenance("mem-D", "child-7")
        final_id = gate.bump_provenance("mem-D", "child-7")
        assert final_id == "mem-D"
        assert store.card_store.cards["mem-D"].programs == ["child-7"]

    def test_bump_provenance_missing_target_returns_empty(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        assert gate.bump_provenance("ghost", "child-7") == ""


class TestSweep:
    def test_sweep_evicts_confidently_harmful_cards(self) -> None:
        store = _FakeStore()
        CardAdmissionGate(store=store, evictor=_NeverHarmful()).admit(_card("mem-h"))
        gate = CardAdmissionGate(store=store, evictor=_AlwaysHarmful())
        evicted = gate.sweep()
        assert evicted == ["mem-h"]
        assert "mem-h" not in store.card_store.cards


class TestLedger:
    def _rows(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def test_add_then_update_emit_distinct_outcome_rows(self, tmp_path) -> None:
        store = _FakeStore()
        ledger_path = tmp_path / "write_ledger.jsonl"
        gate = CardAdmissionGate(
            store=store, evictor=_NeverHarmful(), ledger=WriteLedger(ledger_path)
        )
        gate.admit(_card("mem-L", "first"))
        gate.admit(_card("mem-L", "second"))
        outcomes = [r["outcome"] for r in self._rows(ledger_path)]
        assert outcomes == ["added", "updated"]

    def test_rejected_harm_row_has_empty_final_id(self, tmp_path) -> None:
        store = _FakeStore()
        ledger_path = tmp_path / "write_ledger.jsonl"
        gate = CardAdmissionGate(
            store=store, evictor=_AlwaysHarmful(), ledger=WriteLedger(ledger_path)
        )
        gate.admit(_card("mem-X"))
        (row,) = self._rows(ledger_path)
        assert row["outcome"] == "rejected_harm"
        assert row["final_id"] == ""
