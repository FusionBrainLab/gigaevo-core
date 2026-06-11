"""Write-path transparency: every ingest decision leaves one ledger row.

The append-only ``write_ledger.jsonl`` is the audit trail for the card bank —
added / updated / merged / discarded / rejected_harm, each with the gate's
reason. Recording failures must never block the write path.
"""

from __future__ import annotations

import json

from gigaevo.memory.core.write_ledger import (
    WriteLedger,
    WriteLedgerRecord,
    WriteOutcome,
)
from gigaevo.memory.core.write_pipeline import MemoryWritePipeline
from gigaevo.memory.shared_memory.card_dedup import DedupDecision


class _FakeCardStore:
    def __init__(self):
        self.cards: dict = {}
        self.write_stats = {
            "processed": 0,
            "added": 0,
            "updated": 0,
            "rejected": 0,
            "updated_target_cards": 0,
        }

    def ensure_id(self, card) -> str:
        return str(getattr(card, "id", "") or "minted-id")


class _FakeStore:
    def __init__(self):
        self.card_store = _FakeCardStore()
        self.deleted: list[str] = []

    def delete(self, card_id: str) -> None:
        self.deleted.append(card_id)
        self.card_store.cards.pop(card_id, None)

    def save_card_direct(self, card) -> str:
        card_id = str(getattr(card, "id", "") or "new-1")
        self.card_store.cards[card_id] = card
        return card_id

    def apply_merges(self, merges) -> list[str]:
        return [card_id for card_id, _ in merges]


class _FakeEvictor:
    def __init__(self, evict: bool = False, sweep_ids: list[str] | None = None):
        self._evict = evict
        self._sweep_ids = sweep_ids or []

    def should_evict(self, card) -> bool:
        return self._evict

    def sweep(self, bank):
        return [cid for cid in self._sweep_ids if cid in bank]


class _FakeDedup:
    def __init__(self, decision: DedupDecision):
        self._decision = decision

    def reconcile(self, card, bank) -> DedupDecision:
        return self._decision


def _decision(action: str, *, reason: str = "", duplicate_of: str = "", merges=None):
    return DedupDecision(
        action=action,
        reason=reason,
        duplicate_of=duplicate_of,
        merges=merges or [],
    )


def _idea_card(card_id: str = "idea-1") -> dict:
    return {"id": card_id, "description": "An idea card", "category": "insight"}


def _program_card(program_id: str = "abc") -> dict:
    return {
        "id": f"program-{program_id}",
        "program_id": program_id,
        "category": "program",
        "description": "A program card",
        "fitness": 1.0,
        "code": "def f(): return 1",
    }


def _pipeline(
    tmp_path, *, evict=False, decision=None
) -> tuple[MemoryWritePipeline, _FakeStore, WriteLedger]:
    store = _FakeStore()
    ledger = WriteLedger(tmp_path / "write_ledger.jsonl")
    pipeline = MemoryWritePipeline(
        store=store,
        evictor=_FakeEvictor(evict),
        deduplicator=_FakeDedup(decision or _decision("add", reason="novel")),
        ledger=ledger,
    )
    return pipeline, store, ledger


def _rows(tmp_path) -> list[dict]:
    path = tmp_path / "write_ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestLedgerRows:
    def test_added_row_for_novel_card(self, tmp_path):
        pipeline, _, _ = _pipeline(
            tmp_path, decision=_decision("add", reason="novel mechanism")
        )
        final_id = pipeline.ingest(_idea_card())
        (row,) = _rows(tmp_path)
        assert row["outcome"] == "added"
        assert row["final_id"] == final_id
        assert row["incoming_id"] == "idea-1"
        assert row["reason"] == "novel mechanism"

    def test_discarded_row_carries_duplicate_of_and_reason(self, tmp_path):
        pipeline, store, _ = _pipeline(
            tmp_path,
            decision=_decision(
                "discard", reason="same mechanism as idea-0", duplicate_of="idea-0"
            ),
        )
        store.card_store.cards["idea-0"] = object()
        final_id = pipeline.ingest(_idea_card("idea-1"))
        (row,) = _rows(tmp_path)
        assert row["outcome"] == "discarded"
        assert row["duplicate_of"] == "idea-0"
        assert row["reason"] == "same mechanism as idea-0"
        assert final_id == "idea-0"
        assert row["final_id"] == "idea-0"

    def test_rejected_harm_row(self, tmp_path):
        pipeline, _, _ = _pipeline(tmp_path, evict=True)
        pipeline.ingest(_idea_card())
        (row,) = _rows(tmp_path)
        assert row["outcome"] == "rejected_harm"
        assert row["reason"]

    def test_updated_row_for_known_id(self, tmp_path):
        pipeline, store, _ = _pipeline(tmp_path)
        store.card_store.cards["idea-1"] = object()
        pipeline.ingest(_idea_card("idea-1"))
        (row,) = _rows(tmp_path)
        assert row["outcome"] == "updated"
        assert row["final_id"] == "idea-1"

    def test_merged_row_lists_targets(self, tmp_path):
        from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card

        merged_card = normalize_memory_card(_idea_card("idea-0"))
        pipeline, _, _ = _pipeline(
            tmp_path,
            decision=_decision(
                "update",
                reason="merged into stronger card",
                merges=[("idea-0", merged_card)],
            ),
        )
        final_id = pipeline.ingest(_idea_card("idea-1"))
        (row,) = _rows(tmp_path)
        assert row["outcome"] == "merged"
        assert row["merge_targets"] == ["idea-0"]
        assert final_id == "idea-0"

    def test_program_fast_path_added_row(self, tmp_path):
        pipeline, _, _ = _pipeline(tmp_path)
        pipeline.ingest(_program_card())
        (row,) = _rows(tmp_path)
        assert row["outcome"] == "added"
        assert row["incoming_id"] == "program-abc"

    def test_one_row_per_ingest(self, tmp_path):
        pipeline, _, _ = _pipeline(tmp_path)
        for i in range(3):
            pipeline.ingest(_idea_card(f"idea-{i}"))
        assert len(_rows(tmp_path)) == 3


class TestSweepRows:
    def test_sweep_records_one_evicted_row_per_card(self, tmp_path):
        store = _FakeStore()
        pipeline = MemoryWritePipeline(
            store=store,
            evictor=_FakeEvictor(sweep_ids=["idea-bad"]),
            deduplicator=_FakeDedup(_decision("add", reason="novel")),
            ledger=WriteLedger(tmp_path / "write_ledger.jsonl"),
        )
        pipeline.ingest(_idea_card("idea-bad"))
        pipeline.ingest(_idea_card("idea-good"))
        assert pipeline.sweep() == ["idea-bad"]
        (row,) = [r for r in _rows(tmp_path) if r["outcome"] == "evicted"]
        assert row["incoming_id"] == "idea-bad"
        assert row["final_id"] == "idea-bad"
        assert row["category"] == "insight"
        assert "harmful" in row["reason"]
        assert store.deleted == ["idea-bad"]

    def test_clean_sweep_records_nothing(self, tmp_path):
        pipeline, _, _ = _pipeline(tmp_path)
        pipeline.ingest(_idea_card())
        assert pipeline.sweep() == []
        assert [r["outcome"] for r in _rows(tmp_path)] == ["added"]


class TestWriteOutcomeEnum:
    def test_members_cover_ledger_vocabulary(self):
        assert [o.value for o in WriteOutcome] == [
            "added",
            "updated",
            "merged",
            "discarded",
            "rejected_harm",
            "evicted",
        ]

    def test_enum_member_serializes_as_plain_string_in_jsonl(self, tmp_path):
        ledger = WriteLedger(tmp_path / "write_ledger.jsonl")
        ledger.record(
            incoming_id="x", final_id="x", outcome=WriteOutcome.ADDED, reason="r"
        )
        raw = (tmp_path / "write_ledger.jsonl").read_text()
        assert '"outcome":"added"' in raw

    def test_plain_string_still_accepted(self, tmp_path):
        ledger = WriteLedger(tmp_path / "write_ledger.jsonl")
        ledger.record(incoming_id="x", final_id="x", outcome="merged")
        (row,) = _rows(tmp_path)
        assert row["outcome"] == "merged"


class TestLedgerRobustness:
    def test_ledger_io_failure_never_blocks_ingest(self, tmp_path):
        blocked = tmp_path / "write_ledger.jsonl"
        blocked.mkdir()
        store = _FakeStore()
        pipeline = MemoryWritePipeline(
            store=store,
            evictor=_FakeEvictor(),
            deduplicator=_FakeDedup(_decision("add")),
            ledger=WriteLedger(blocked),
        )
        final_id = pipeline.ingest(_idea_card())
        assert final_id == "idea-1"
        assert "idea-1" in store.card_store.cards

    def test_no_ledger_is_fine(self, tmp_path):
        store = _FakeStore()
        pipeline = MemoryWritePipeline(
            store=store,
            evictor=_FakeEvictor(),
            deduplicator=_FakeDedup(_decision("add")),
        )
        assert pipeline.ingest(_idea_card()) == "idea-1"

    def test_record_is_valid_jsonl_with_timestamp(self, tmp_path):
        ledger = WriteLedger(tmp_path / "write_ledger.jsonl")
        ledger.record(incoming_id="x", final_id="x", outcome="added", reason="r")
        (row,) = _rows(tmp_path)
        WriteLedgerRecord.model_validate(row)
        assert row["timestamp_utc"]

    def test_invalid_record_swallowed(self, tmp_path):
        ledger = WriteLedger(tmp_path / "write_ledger.jsonl")
        ledger.record(incoming_id="x", final_id="x", outcome="not-an-outcome")
        assert _rows(tmp_path) == []
