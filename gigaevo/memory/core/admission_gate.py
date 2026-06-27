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
        cards = self._store.card_store.cards
        incoming_id = str(card.id or "").strip()

        if self._evictor.should_evict(card):
            if incoming_id and incoming_id in cards:
                self._store.delete(incoming_id)
            return self._ledger_record(
                card,
                "",
                WriteOutcome.REJECTED_HARM,
                "injection posterior confidently harmful",
            )

        known = bool(incoming_id) and incoming_id in cards
        final_id = self._store.save_card_direct(card)
        outcome = WriteOutcome.UPDATED if known else WriteOutcome.ADDED
        reason = "known id replaced" if known else "librarian-authored card"
        return self._ledger_record(card, final_id, outcome, reason)

    def merge(self, target_id: str, card: AnyCard) -> str:
        store = self._store.card_store
        if target_id not in store.cards:
            return ""
        card.id = target_id  # apply_merges keys storage off the card's own id
        updated = self._store.apply_merges([(target_id, card)])
        if not updated:
            return ""
        return self._ledger_record(
            card, updated[0], WriteOutcome.MERGED, "librarian merge"
        )

    def bump_provenance(self, target_id: str, child_id: str) -> str:
        store = self._store.card_store
        target = store.cards.get(target_id)
        if target is None:
            return ""
        if (
            isinstance(target, MemoryCard)
            and child_id
            and child_id not in target.programs
        ):
            target.programs = [*target.programs, child_id]
            self._store.save_card_direct(target)
        return self._ledger_record(
            target, target_id, WriteOutcome.UPDATED, "pre-gate provenance bump"
        )

    def sweep(self) -> list[str]:
        bank = self._store.card_store.cards
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
        self, card: AnyCard, final_id: str, outcome: WriteOutcome, reason: str
    ) -> str:
        if self._ledger is not None:
            self._ledger.record(
                incoming_id=str(card.id or ""),
                final_id=final_id,
                outcome=outcome,
                reason=reason,
                category=card.category,
            )
        return final_id
