"""Behavior tests for CardAdmissionGate: harm gate + ledger, no dedup."""

from __future__ import annotations

import json
from pathlib import Path

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.admission_gate import CardAdmissionGate
from gigaevo.memory.core.write_ledger import WriteLedger
from gigaevo.memory.shared_memory.models import MemoryCard, ProgramCard

_IDX = Path("/tmp/claude-1000/_gate_test_idx.json")


def _gain(value: float) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(parent_metrics={"f": value}), gain=value
    )


class _FakeStore:
    """Backend twin: exposes the methods the gate touches."""

    def __init__(self) -> None:
        from gigaevo.memory.shared_memory.card_store import CardStore

        self.card_store = CardStore(index_file=_IDX)
        self.card_store.cards.clear()
        self.saved: list[MemoryCard] = []
        self.deleted: list[str] = []

    def get_card(self, card_id: str) -> MemoryCard | None:
        return self.card_store.cards.get(card_id)

    def all_cards_snapshot(self) -> dict[str, MemoryCard]:
        return dict(self.card_store.cards)

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

    def test_merge_preserves_target_evidence_and_unions_provenance(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        store.save_card_direct(
            MemoryCard(
                id="mem-A",
                description="original",
                keywords=["k1"],
                programs=["p1"],
                gain_events=[_gain(0.1)],
            )
        )
        incoming = MemoryCard(
            id="", description="union prose", keywords=["k2"], programs=["p2"]
        )
        final_id = gate.merge("mem-A", incoming)
        survivor = store.card_store.cards["mem-A"]
        assert final_id == "mem-A"
        assert survivor.description == "union prose"
        assert survivor.programs == ["p1", "p2"]
        # Provenance (programs, gain_events) is unioned/preserved, but keywords
        # follow the prose: a librarian merge carries an agent-curated union
        # keyword set, so the merge takes it verbatim rather than re-unioning the
        # target's old list (which would re-bloat the survivor).
        assert survivor.keywords == ["k2"]
        assert survivor.gain_events == [_gain(0.1)]

    def test_merge_resulting_in_harmful_card_evicts_target(self) -> None:
        store = _FakeStore()
        store.save_card_direct(_card("mem-A", "original"))
        gate = CardAdmissionGate(store=store, evictor=_AlwaysHarmful())
        final_id = gate.merge("mem-A", _card("", "union prose"))
        assert final_id == ""
        assert "mem-A" not in store.card_store.cards

    def test_bump_provenance_does_not_mutate_original_target(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        gate.admit(_card("mem-D"))
        original = store.card_store.cards["mem-D"]
        gate.bump_provenance("mem-D", "child-7")
        assert original.programs == []
        assert store.card_store.cards["mem-D"].programs == ["child-7"]

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

    def test_bump_provenance_on_program_card_target_returns_empty(self) -> None:
        store = _FakeStore()
        gate = CardAdmissionGate(store=store, evictor=_NeverHarmful())
        store.save_card_direct(
            ProgramCard(id="program-7", program_id="7", description="exemplar")
        )
        assert gate.bump_provenance("program-7", "child-7") == ""


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

    def test_harmful_merge_emits_rejected_harm_row(self, tmp_path) -> None:
        store = _FakeStore()
        ledger_path = tmp_path / "write_ledger.jsonl"
        store.save_card_direct(_card("mem-A", "original"))
        gate = CardAdmissionGate(
            store=store, evictor=_AlwaysHarmful(), ledger=WriteLedger(ledger_path)
        )
        gate.merge("mem-A", _card("", "union prose"))
        (row,) = self._rows(ledger_path)
        assert row["outcome"] == "rejected_harm"
        assert row["final_id"] == ""

    def test_merge_row_reports_submitted_incoming_id_and_target(self, tmp_path) -> None:
        store = _FakeStore()
        ledger_path = tmp_path / "write_ledger.jsonl"
        store.save_card_direct(_card("mem-A", "original"))
        gate = CardAdmissionGate(
            store=store, evictor=_NeverHarmful(), ledger=WriteLedger(ledger_path)
        )
        gate.merge("mem-A", _card("", "union prose"))
        (row,) = self._rows(ledger_path)
        assert row["outcome"] == "merged"
        assert row["incoming_id"] == ""
        assert row["final_id"] == "mem-A"
        assert row["merge_targets"] == ["mem-A"]
