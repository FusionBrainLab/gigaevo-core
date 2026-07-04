"""Sole admission authority on the memory write path.

The Librarian owns dedup and prose authoring; this gate only decides
admit-or-reject on harm and records every verdict to the write ledger. The one
externally-consumed artifact is the ``WriteLedger`` (``write_ledger.jsonl``) —
its row schema is the contract (``write_ledger.v1``). Store save/merge events
fire automatically through the store (``MEMORY_STORE_WRITE``); the gate adds
no counters and no admission event of its own (no consumer).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.memory.write.eviction import Evictor
from gigaevo.memory.write.merge import merge_cards


class WriteOutcome(StrEnum):
    """Verdict of one card-bank ingest.

    ADDED / UPDATED / MERGED / REJECTED_HARM / EVICTED are each recorded as a
    write-ledger row. DISCARDED is the no-op verdict: the gate did nothing
    (merge/bump target absent or ineligible, or the store merge failed) and
    recorded no row — the submitted card was neither admitted nor rejected, so a
    caller may re-author it as fresh. It is deliberately distinct from
    REJECTED_HARM, which the gate judged and dropped and which must never be
    re-admitted.
    """

    ADDED = "added"
    UPDATED = "updated"
    MERGED = "merged"
    DISCARDED = "discarded"
    REJECTED_HARM = "rejected_harm"
    EVICTED = "evicted"


class WriteResult(BaseModel):
    """What one gate ingest did, and the bank id the card landed under.

    Replaces the former bare-``str`` return whose ``""`` conflated two verdicts a
    caller must tell apart: a benign no-op (``DISCARDED`` — retry as fresh) and a
    harmful rejection (``REJECTED_HARM`` — drop, never re-admit).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: WriteOutcome = Field(description="Gate verdict for this ingest.")
    card_id: str = Field(
        default="",
        description="Bank id the card landed under; '' when nothing landed.",
    )

    @property
    def landed(self) -> bool:
        return bool(self.card_id)

    @property
    def rejected_harm(self) -> bool:
        return self.outcome is WriteOutcome.REJECTED_HARM

    @property
    def benign_noop(self) -> bool:
        """The gate did nothing and judged nothing: safe to re-author as fresh.

        The only verdict a caller may launder back into the bank as new. Every
        other non-landed outcome is a harm-driven deletion (``REJECTED_HARM``
        today; ``EVICTED`` if a sweep verdict is ever routed through a re-author
        path) and must be dropped, not resurrected.
        """
        return self.outcome is WriteOutcome.DISCARDED


_DISCARDED = WriteResult(outcome=WriteOutcome.DISCARDED)


class WriteLedgerRecord(BaseModel):
    """One JSONL row of the write ledger: a single ingest or eviction verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(
        default="write_ledger.v1",
        description="Stable schema id for append-only write ledger rows.",
    )
    record_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique id for this ledger row.",
    )
    timestamp_utc: str = Field(
        description="ISO-8601 UTC time the verdict was recorded."
    )
    incoming_id: str = Field(description="Id of the card as submitted to the bank.")
    final_id: str = Field(
        description="Id the card ended up under (merge target for merges, '' if dropped)."
    )
    outcome: WriteOutcome = Field(description="Gate verdict for this card.")
    reason: str = Field(
        default="", description="Human-readable rationale emitted by the deciding gate."
    )
    duplicate_of: str = Field(
        default="", description="Id of the existing card this one duplicated, if any."
    )
    merge_targets: tuple[str, ...] = Field(
        default=(),
        description="Ids of the cards this one was merged into.",
    )
    category: str = Field(
        default="", description="Category of the incoming card at submission time."
    )


class WriteLedger:
    """Append-only JSONL audit trail answering "what happened to the card I
    submitted, and why" for every gate verdict. Recording must never block the
    write path: any failure is logged and swallowed."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        *,
        incoming_id: str,
        final_id: str,
        outcome: WriteOutcome | str,
        reason: str = "",
        duplicate_of: str = "",
        merge_targets: Sequence[str] | None = None,
        category: str = "",
    ) -> None:
        try:
            row = WriteLedgerRecord(
                timestamp_utc=datetime.now(UTC).isoformat(),
                incoming_id=incoming_id,
                final_id=final_id,
                outcome=WriteOutcome(outcome),
                reason=reason,
                duplicate_of=duplicate_of,
                merge_targets=tuple(merge_targets or ()),
                category=category,
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(row.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("[Memory][WriteLedger] failed to record: {}", exc)


class CardAdmissionGate:
    """Harm gate + ledger over the card store.

    Four methods make up the whole write-path admission surface:
    ``admit`` (new/known-id card), ``merge`` (synthesized union onto an existing
    card), ``bump_provenance`` (DUPLICATE-ruling provenance append, no LLM),
    and ``sweep`` (periodic harm eviction).
    """

    def __init__(
        self, *, store: MemoryStore, evictor: Evictor, ledger: WriteLedger | None = None
    ) -> None:
        self._store = store
        self._evictor = evictor
        self._ledger = ledger

    def admit(self, card: Card) -> WriteResult:
        incoming_id = card.id.strip()
        known = bool(incoming_id) and self._store.get(incoming_id) is not None

        if self._evictor.should_evict(card):
            if known:
                self._store.delete(incoming_id)
            return self._ledger_record(
                card,
                "",
                WriteOutcome.REJECTED_HARM,
                "injection posterior confidently harmful",
            )

        final_id = self._store.save(card)
        outcome = WriteOutcome.UPDATED if known else WriteOutcome.ADDED
        reason = "known id replaced" if known else "librarian-authored card"
        return self._ledger_record(card, final_id, outcome, reason)

    def merge(self, target_id: str, card: Card) -> WriteResult:
        target = self._store.get(target_id)
        if target is None or target.kind is not CardKind.INSIGHT:
            return _DISCARDED
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
        updated = self._store.apply_merges([merged])
        if not updated:
            return _DISCARDED
        return self._ledger_record(
            merged,
            updated[0],
            WriteOutcome.MERGED,
            "librarian merge",
            incoming_id=submitted_id,
            merge_targets=(target_id,),
        )

    def bump_provenance(self, target_id: str, child_id: str) -> WriteResult:
        target = self._store.get(target_id)
        if target is None or target.kind is not CardKind.INSIGHT:
            return _DISCARDED
        if child_id and child_id not in target.programs:
            self._store.save(
                target.model_copy(update={"programs": (*target.programs, child_id)})
            )
        return self._ledger_record(
            target, target_id, WriteOutcome.UPDATED, "duplicate provenance bump"
        )

    def sweep(self) -> list[str]:
        bank = self._store.snapshot()
        by_id = {card.id: card for card in bank}
        evicted = list(self._evictor.sweep(bank))
        for cid in evicted:
            self._store.delete(cid)
            card = by_id.get(cid)
            if card is not None:
                self._ledger_record(
                    card, "", WriteOutcome.EVICTED, "confidently harmful"
                )
        return evicted

    def _ledger_record(
        self,
        card: Card,
        final_id: str,
        outcome: WriteOutcome,
        reason: str,
        *,
        incoming_id: str | None = None,
        merge_targets: Sequence[str] | None = None,
    ) -> WriteResult:
        if self._ledger is not None:
            self._ledger.record(
                incoming_id=card.id if incoming_id is None else incoming_id,
                final_id=final_id,
                outcome=outcome,
                reason=reason,
                merge_targets=merge_targets,
                category=card.category,
            )
        return WriteResult(outcome=outcome, card_id=final_id)
