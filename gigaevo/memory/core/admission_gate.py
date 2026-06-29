"""Sole admission authority on the memory write path.

The Librarian owns dedup and prose authoring; this gate only decides
admit-or-reject on harm and records every verdict to the write ledger. The one
externally-consumed artifact is the ``WriteLedger`` (``write_ledger.jsonl``) —
its row schema is the contract (``write_ledger.v1``). Backend ``store.insert`` /
``store.merge`` events fire automatically through the store; the gate adds no
counters, no ``normalize_memory_card`` (input is a typed librarian-authored
``AnyCard``), and no ``admission.verdict`` event (no consumer).
"""

from __future__ import annotations

from typing import Any

from gigaevo.memory.core.write_ledger import WriteLedger, WriteOutcome
from gigaevo.memory.shared_memory.card_merge import merge_cards
from gigaevo.memory.shared_memory.models import AnyCard, MemoryCard


class CardAdmissionGate:
    """Harm gate + ledger over a card backend.

    Four methods make up the whole write-path admission surface:
    ``admit`` (new/known-id card), ``merge`` (synthesized union onto an existing
    card), ``bump_provenance`` (pre-gate near-dup provenance append, no LLM),
    and ``sweep`` (periodic harm eviction).
    """

    def __init__(
        self, *, store: Any, evictor: Any, ledger: WriteLedger | None = None
    ) -> None:
        self._store = store
        self._evictor = evictor
        self._ledger = ledger

    def admit(self, card: AnyCard) -> str:
        incoming_id = card.id.strip()
        known = bool(incoming_id) and self._store.get_card(incoming_id) is not None

        if self._evictor.should_evict(card):
            if known:
                self._store.delete(incoming_id)
            return self._ledger_record(
                card,
                "",
                WriteOutcome.REJECTED_HARM,
                "injection posterior confidently harmful",
            )

        final_id = self._store.save_card_direct(card)
        outcome = WriteOutcome.UPDATED if known else WriteOutcome.ADDED
        reason = "known id replaced" if known else "librarian-authored card"
        return self._ledger_record(card, final_id, outcome, reason)

    def merge(self, target_id: str, card: AnyCard) -> str:
        target = self._store.get_card(target_id)
        if target is None or not isinstance(target, MemoryCard):
            return ""
        # merge_cards preserves the target id on the survivor, so capture the
        # submitted card's id before the fold — the ledger row must report what
        # happened to the SUBMITTED card, not the merge target.
        submitted_id = card.id
        merged = merge_cards(target, card, replace_description=True)
        if self._evictor.should_evict(merged):
            self._store.delete(target_id)
            return self._ledger_record(
                merged,
                "",
                WriteOutcome.REJECTED_HARM,
                "merged card confidently harmful",
                incoming_id=submitted_id,
            )
        updated = self._store.apply_merges([(target_id, merged)])
        if not updated:
            return ""
        return self._ledger_record(
            merged,
            updated[0],
            WriteOutcome.MERGED,
            "librarian merge",
            incoming_id=submitted_id,
            merge_targets=[target_id],
        )

    def bump_provenance(self, target_id: str, child_id: str) -> str:
        target = self._store.get_card(target_id)
        if target is None or not isinstance(target, MemoryCard):
            return ""
        if child_id and child_id not in target.programs:
            self._store.save_card_direct(
                target.model_copy(update={"programs": [*target.programs, child_id]})
            )
        return self._ledger_record(
            target, target_id, WriteOutcome.UPDATED, "pre-gate provenance bump"
        )

    def sweep(self) -> list[str]:
        bank = self._store.all_cards_snapshot()
        evicted = list(self._evictor.sweep(bank))
        for cid in evicted:
            card = bank.get(cid)
            self._store.delete(cid)
            if card is not None:
                self._ledger_record(
                    card, "", WriteOutcome.EVICTED, "confidently harmful"
                )
        return evicted

    def _ledger_record(
        self,
        card: AnyCard,
        final_id: str,
        outcome: WriteOutcome,
        reason: str,
        *,
        incoming_id: str | None = None,
        merge_targets: list[str] | None = None,
    ) -> str:
        if self._ledger is not None:
            self._ledger.record(
                incoming_id=card.id if incoming_id is None else incoming_id,
                final_id=final_id,
                outcome=outcome,
                reason=reason,
                merge_targets=merge_targets,
                category=card.category,
            )
        return final_id
