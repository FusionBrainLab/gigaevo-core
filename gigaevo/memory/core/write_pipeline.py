from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.memory.core.protocols import Deduplicator, Evictor
from gigaevo.memory.core.write_ledger import WriteLedger, WriteOutcome
from gigaevo.memory.shared_memory.card_conversion import (
    is_program_card,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.card_dedup import DedupAction


class MemoryWritePipeline:
    """Behavioral twin of ``AmemGamMemory.save_card``: normalize → harm gate →
    known-id update → program fast-path → dedup reconcile → store. The store is
    the legacy memory during transition; persistence and write_stats stay there.
    Every verdict is appended to the write ledger when one is configured."""

    def __init__(
        self,
        *,
        store: Any,
        evictor: Evictor,
        deduplicator: Deduplicator,
        ledger: WriteLedger | None = None,
    ):
        self._store = store
        self._evictor = evictor
        self._dedup = deduplicator
        self._ledger = ledger

    def _record(self, **fields: Any) -> None:
        if self._ledger is not None:
            self._ledger.record(**fields)

    def ingest(self, card: Any) -> str:
        normalized = normalize_memory_card(card)
        store = self._store.card_store
        store.write_stats["processed"] += 1
        incoming_id = str(normalized.id or "").strip()
        category = str(getattr(normalized, "category", "") or "")

        if self._evictor.should_evict(normalized):
            store.write_stats["rejected"] += 1
            if incoming_id and incoming_id in store.cards:
                self._store.delete(incoming_id)
            logger.info(
                "[Memory][Store] Card {!r} rejected: injection posterior confidently harmful",
                incoming_id or "<new>",
            )
            final_id = incoming_id or store.ensure_id(normalized)
            self._record(
                incoming_id=incoming_id,
                final_id=final_id,
                outcome=WriteOutcome.REJECTED_HARM,
                reason="injection posterior confidently harmful",
                category=category,
            )
            return final_id

        if incoming_id and incoming_id in store.cards:
            store.write_stats["updated"] += 1
            final_id = self._store.save_card_direct(normalized)
            self._record(
                incoming_id=incoming_id,
                final_id=final_id,
                outcome=WriteOutcome.UPDATED,
                reason="known id replaced",
                category=category,
            )
            return final_id

        if is_program_card(normalized):
            store.write_stats["added"] += 1
            final_id = self._store.save_card_direct(normalized)
            self._record(
                incoming_id=incoming_id,
                final_id=final_id,
                outcome=WriteOutcome.ADDED,
                reason="program card fast-path",
                category=category,
            )
            return final_id

        decision = self._dedup.reconcile(normalized, store.cards)
        if decision.action is DedupAction.DISCARD:
            store.write_stats["rejected"] += 1
            duplicate_of = str(decision.duplicate_of or "")
            if duplicate_of and duplicate_of in store.cards:
                final_id = duplicate_of
            else:
                final_id = store.ensure_id(normalized)
            logger.info(
                "[Memory][Store] Card {!r} discarded as duplicate of {!r}: {}",
                incoming_id or "<new>",
                duplicate_of or "<unknown>",
                decision.reason,
            )
            self._record(
                incoming_id=incoming_id,
                final_id=final_id,
                outcome=WriteOutcome.DISCARDED,
                reason=decision.reason,
                duplicate_of=duplicate_of,
                category=category,
            )
            return final_id
        if decision.action is DedupAction.UPDATE and decision.merges:
            updated_ids = self._store.apply_merges(decision.merges)
            if updated_ids:
                store.write_stats["updated"] += 1
                store.write_stats["updated_target_cards"] += len(updated_ids)
                self._record(
                    incoming_id=incoming_id,
                    final_id=updated_ids[0],
                    outcome=WriteOutcome.MERGED,
                    reason=decision.reason,
                    merge_targets=list(updated_ids),
                    category=category,
                )
                return updated_ids[0]

        store.write_stats["added"] += 1
        final_id = self._store.save_card_direct(normalized)
        self._record(
            incoming_id=incoming_id,
            final_id=final_id,
            outcome=WriteOutcome.ADDED,
            reason=decision.reason,
            category=category,
        )
        return final_id

    def sweep(self) -> list[str]:
        bank = self._store.card_store.cards
        evicted = list(self._evictor.sweep(bank))
        for card_id in evicted:
            category = str(getattr(bank.get(card_id), "category", "") or "")
            self._store.delete(card_id)
            self._record(
                incoming_id=card_id,
                final_id=card_id,
                outcome=WriteOutcome.EVICTED,
                reason="sweep: injection posterior confidently harmful",
                category=category,
            )
        return evicted
