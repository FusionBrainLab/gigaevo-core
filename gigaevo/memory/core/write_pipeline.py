from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from gigaevo.memory.core.events import (
    emit_memory_event,
    memory_event_context,
    new_memory_decision_id,
)
from gigaevo.memory.core.protocols import Deduplicator, Evictor
from gigaevo.memory.core.write_ledger import WriteLedger, WriteOutcome
from gigaevo.memory.shared_memory.card_conversion import (
    is_program_card,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.card_dedup import DedupAction
from gigaevo.memory.shared_memory.card_store import WriteStatKey
from gigaevo.memory.shared_memory.models import AnyCard


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
        event_path: str | Path | None = None,
    ):
        self._store = store
        self._evictor = evictor
        self._dedup = deduplicator
        self._ledger = ledger
        self._event_path = Path(event_path) if event_path is not None else None

    def _write_stats(self) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in getattr(self._store.card_store, "write_stats", {}).items()
        }

    @staticmethod
    def _outcome_value(outcome: Any) -> str:
        return outcome.value if isinstance(outcome, WriteOutcome) else str(outcome)

    def _record(self, *, event_type: str = "write.ingest", **fields: Any) -> None:
        if self._ledger is not None:
            self._ledger.record(**fields)
        outcome = self._outcome_value(fields["outcome"])
        payload = {
            **fields,
            "outcome": outcome,
            "bank_card_count": len(self._store.card_store.cards),
            "write_stats": self._write_stats(),
        }
        high_signal_outcomes = {
            WriteOutcome.DISCARDED.value,
            WriteOutcome.REJECTED_HARM.value,
            WriteOutcome.EVICTED.value,
        }
        level = "INFO" if outcome in high_signal_outcomes else "DEBUG"
        emit_memory_event(
            component="WritePipeline",
            event_type=event_type,
            payload=payload,
            level=level,
            event_path=self._event_path,
        )
        logger.log(
            level,
            "[Memory][WritePipeline] event={} outcome={} incoming_id={!r} final_id={!r} "
            "category={} reason={} bank_cards={} stats={}",
            event_type,
            outcome,
            fields["incoming_id"],
            fields["final_id"],
            fields["category"],
            fields["reason"],
            payload["bank_card_count"],
            payload["write_stats"],
        )

    def ingest(self, card: dict[str, Any] | AnyCard) -> str:
        with memory_event_context(
            decision_id=new_memory_decision_id("memwrite"),
            event_path=self._event_path,
        ):
            try:
                return self._ingest(card)
            except Exception as exc:
                emit_memory_event(
                    component="WritePipeline",
                    event_type="write.exception",
                    payload={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "bank_card_count": len(self._store.card_store.cards),
                        "write_stats": self._write_stats(),
                    },
                    level="WARNING",
                )
                logger.opt(exception=True).warning(
                    "[Memory][WritePipeline] ingest failed before a verdict was recorded: {}",
                    exc,
                )
                raise

    def _ingest(self, card: dict[str, Any] | AnyCard) -> str:
        """Normalize ``card`` at this boundary (raw dicts arrive from JSON and
        GAM producers) and run it through the write verdict chain."""
        normalized = normalize_memory_card(card)
        store = self._store.card_store
        store.write_stats[WriteStatKey.PROCESSED] += 1
        incoming_id = str(normalized.id or "").strip()
        category = normalized.category
        keywords = normalized.keywords or []
        emit_memory_event(
            component="WritePipeline",
            event_type="write.request",
            payload={
                "incoming_id": incoming_id,
                "category": category,
                "card_model": type(normalized).__name__,
                "description_chars": len(normalized.description or ""),
                "keywords_count": len(keywords)
                if isinstance(keywords, (list, tuple, set))
                else 0,
                "has_task_description": bool(normalized.task_description),
                "known_card": bool(incoming_id and incoming_id in store.cards),
                "bank_card_count_before": len(store.cards),
                "write_stats": self._write_stats(),
            },
        )
        logger.debug(
            "[Memory][WritePipeline] Ingest start incoming_id={!r} category={} "
            "bank_cards={} stats={}",
            incoming_id or "<new>",
            category,
            len(store.cards),
            self._write_stats(),
        )

        if self._evictor.should_evict(normalized):
            store.write_stats[WriteStatKey.REJECTED] += 1
            if incoming_id and incoming_id in store.cards:
                self._store.delete(incoming_id)
            logger.info(
                "[Memory][Store] Card {!r} rejected: injection posterior confidently harmful",
                incoming_id or "<new>",
            )
            # The card never remains in the bank (any existing copy was just
            # deleted), so a final_id would reference a card that exists
            # nowhere; incoming_id keeps the row traceable.
            self._record(
                incoming_id=incoming_id,
                final_id="",
                outcome=WriteOutcome.REJECTED_HARM,
                reason="injection posterior confidently harmful",
                category=category,
            )
            return ""

        if incoming_id and incoming_id in store.cards:
            store.write_stats[WriteStatKey.UPDATED] += 1
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
            store.write_stats[WriteStatKey.ADDED] += 1
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
            store.write_stats[WriteStatKey.REJECTED] += 1
            duplicate_of = str(decision.duplicate_of or "")
            # Phantom/empty duplicate_of: nothing was stored, so an empty
            # final_id is the honest ledger value — minting one would
            # fabricate a card reference that exists nowhere in the bank.
            final_id = duplicate_of if duplicate_of in store.cards else ""
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
                store.write_stats[WriteStatKey.UPDATED] += 1
                store.write_stats[WriteStatKey.UPDATED_TARGET_CARDS] += len(updated_ids)
                self._record(
                    incoming_id=incoming_id,
                    final_id=updated_ids[0],
                    outcome=WriteOutcome.MERGED,
                    reason=decision.reason,
                    merge_targets=list(updated_ids),
                    category=category,
                )
                return updated_ids[0]

        store.write_stats[WriteStatKey.ADDED] += 1
        final_id = self._store.save_card_direct(normalized)
        reason = decision.reason
        if decision.action is DedupAction.UPDATE:
            reason = f"merge failed; added as new ({decision.reason})"
        self._record(
            incoming_id=incoming_id,
            final_id=final_id,
            outcome=WriteOutcome.ADDED,
            reason=reason,
            category=category,
        )
        return final_id

    def sweep(self) -> list[str]:
        with memory_event_context(
            decision_id=new_memory_decision_id("memsweep"),
            event_path=self._event_path,
        ):
            try:
                return self._sweep()
            except Exception as exc:
                emit_memory_event(
                    component="WritePipeline",
                    event_type="write.sweep.exception",
                    payload={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "bank_card_count": len(self._store.card_store.cards),
                        "write_stats": self._write_stats(),
                    },
                    level="WARNING",
                )
                logger.opt(exception=True).warning(
                    "[Memory][WritePipeline] harm sweep failed: {}", exc
                )
                raise

    def _sweep(self) -> list[str]:
        bank = self._store.card_store.cards
        bank_count_before = len(bank)
        emit_memory_event(
            component="WritePipeline",
            event_type="write.sweep.request",
            payload={
                "bank_card_count_before": bank_count_before,
                "write_stats": self._write_stats(),
            },
        )
        evicted = list(self._evictor.sweep(bank))
        for card_id in evicted:
            swept = bank.get(card_id)
            category = swept.category if swept is not None else ""
            self._store.delete(card_id)
            self._record(
                event_type="write.sweep",
                incoming_id=card_id,
                final_id=card_id,
                outcome=WriteOutcome.EVICTED,
                reason="sweep: injection posterior confidently harmful",
                category=category,
            )
        emit_memory_event(
            component="WritePipeline",
            event_type="write.sweep.summary",
            payload={
                "bank_card_count_before": bank_count_before,
                "bank_card_count_after": len(bank),
                "evicted_count": len(evicted),
                "evicted_ids": evicted,
                "write_stats": self._write_stats(),
            },
            level="INFO" if evicted else "DEBUG",
        )
        return evicted
