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

    ADDED / UPDATED / MERGED / REJECTED_HARM / REJECTED_NOVELTY / EVICTED are
    each recorded as a write-ledger row. DISCARDED is the no-op verdict: the
    gate did nothing (merge/bump target absent or ineligible, or the store
    merge failed) and recorded no row — the submitted card was neither admitted
    nor rejected, so a caller may re-author it as fresh. It is deliberately
    distinct from the judged rejections: REJECTED_HARM must never be
    re-admitted, and REJECTED_NOVELTY (the novelty judge ruled the lever
    prior-known) must not be re-authored either or the judge just re-rejects it.
    """

    ADDED = "added"
    UPDATED = "updated"
    MERGED = "merged"
    DISCARDED = "discarded"
    REJECTED_HARM = "rejected_harm"
    REJECTED_NOVELTY = "rejected_novelty"
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

    Six methods make up the whole write-path admission surface:
    ``admit`` (new/known-id card), ``merge`` (synthesized union onto an existing
    card), ``bump_provenance`` (DUPLICATE-ruling provenance append, no LLM),
    ``reject_novelty`` (ledger a novelty-judge kill), ``retire_exemplar`` (delete
    a non-harm superseded/pruned exemplar), and ``sweep`` (periodic harm
    eviction). Every harm deletion also tombstones the id for the rest of the
    run: the deletion destroys the very gain events the evictor's verdict
    rested on, so a re-authored twin arrives evidence-free and would otherwise
    churn evict → re-author → re-admit every sweep.
    """

    def __init__(
        self, *, store: MemoryStore, evictor: Evictor, ledger: WriteLedger | None = None
    ) -> None:
        self._store = store
        self._evictor = evictor
        self._ledger = ledger
        self._tombstoned: set[str] = set()

    def is_tombstoned(self, card_id: str) -> bool:
        """True iff this id was harm-deleted earlier in the run. Lets callers
        skip work the gate would reject anyway — the writer checks before
        paying the exemplar-author LLM call. In-memory only: a restart clears
        the set (accepted — the churn this kills is intra-run)."""
        return card_id in self._tombstoned

    def admit(self, card: Card) -> WriteResult:
        incoming_id = card.id.strip()
        if incoming_id in self._tombstoned:
            return self._ledger_record(
                card,
                "",
                WriteOutcome.REJECTED_HARM,
                "tombstoned: harm-evicted earlier this run",
            )
        known = bool(incoming_id) and self._store.get(incoming_id) is not None

        if self._evictor.should_evict(card):
            reason = _eviction_reason(
                self._evictor, card, "injection posterior confidently harmful"
            )
            if known:
                self._store.delete(incoming_id)
            if incoming_id:
                self._tombstoned.add(incoming_id)
            return self._ledger_record(
                card,
                "",
                WriteOutcome.REJECTED_HARM,
                reason,
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
            reason = _eviction_reason(
                self._evictor, merged, "merged card confidently harmful"
            )
            self._store.delete(target_id)
            self._tombstoned.add(target_id)
            if (
                submitted_id
                and submitted_id != target_id
                and self._store.get(submitted_id) is not None
            ):
                self._store.delete(submitted_id)
                self._tombstoned.add(submitted_id)
            # Two cards die here, so two rows: the banked target's deletion
            # (EVICTED, same convention as sweep) and the submitted partner's
            # rejection — one row would leave the target's fate unrecorded.
            self._ledger_record(target, "", WriteOutcome.EVICTED, reason)
            return self._ledger_record(
                merged,
                "",
                WriteOutcome.REJECTED_HARM,
                reason,
                incoming_id=submitted_id,
                merge_targets=(target_id,),
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

    def reject_novelty(self, card: Card, reason: str) -> WriteResult:
        """Record a novelty-judge rejection. The librarian holds the judge; the
        gate only ledgers the verdict — when the judge is on it kills a large
        share of idea authorship, and an unledgered kill of that size makes the
        ledger unable to answer "where did my cards go"."""
        return self._ledger_record(card, "", WriteOutcome.REJECTED_NOVELTY, reason)

    def retire_exemplar(
        self, card: Card, *, successor_id: str = "", reason: str
    ) -> WriteResult:
        """Delete a program exemplar for a non-harm reason and ledger it.

        Supersession/pruning is not a harm verdict — no tombstone — but the
        deletion must still be answerable from write_ledger.jsonl.
        """
        if card.kind is not CardKind.PROGRAM:
            return _DISCARDED
        self._store.delete(card.id)
        return self._ledger_record(card, successor_id, WriteOutcome.UPDATED, reason)

    def retire_twin(self, twin: Card, *, successor_id: str) -> None:
        """Delete an exemplar superseded by a strictly-better twin."""
        self.retire_exemplar(
            twin,
            successor_id=successor_id,
            reason="exemplar superseded by strictly-better twin",
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
            self._tombstoned.add(cid)
            card = by_id.get(cid)
            if card is not None:
                reason = _eviction_reason(self._evictor, card, "confidently harmful")
                self._ledger_record(card, "", WriteOutcome.EVICTED, reason)
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


def _eviction_reason(evictor: Evictor, card: Card, default: str) -> str:
    text = evictor.eviction_reason(card).strip()
    if text:
        return text
    return default
