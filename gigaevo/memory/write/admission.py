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
from contextlib import nullcontext
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.exceptions import MergeAborted
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.prior_evidence import EvictedEvidenceSink, _JsonlFileLock
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.memory.write.eviction import Evictor, foreign_retention_veto
from gigaevo.memory.write.merge import merge_cards, union_events, union_strings


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
            with _JsonlFileLock(self._path.with_suffix(self._path.suffix + ".lock")):
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
        self,
        *,
        store: MemoryStore,
        evictor: Evictor,
        ledger: WriteLedger | None = None,
        evicted_evidence_sink: EvictedEvidenceSink | None = None,
        selection_leases: InFlightSelectionRegistry | None = None,
        task_key: str = "",
        min_effective_events: float = 0.0,
    ) -> None:
        self._store = store
        self._evictor = evictor
        self._ledger = ledger
        self._evicted_evidence_sink = evicted_evidence_sink
        self._selection_leases = selection_leases
        self._task_key = task_key
        self._min_effective_events = min_effective_events
        self._tombstoned: set[str] = set()

    def is_tombstoned(self, card_id: str) -> bool:
        """True iff this id was harm-deleted earlier in the run. Lets callers
        skip work the gate would reject anyway — the writer checks before
        paying the exemplar-author LLM call. In-memory only: a restart clears
        the set (accepted — the churn this kills is intra-run)."""
        return card_id in self._tombstoned

    def admit(self, card: Card) -> WriteResult:
        with self._eviction_guard():
            incoming_id = card.id.strip()
            if incoming_id in self._tombstoned:
                return self._ledger_record(
                    card,
                    "",
                    WriteOutcome.REJECTED_HARM,
                    "tombstoned: harm-evicted earlier this run",
                )
            known = bool(incoming_id) and self._store.get(incoming_id) is not None

            if not known:
                if self._evictor.should_evict(card):
                    reason = _eviction_reason(
                        self._evictor,
                        card,
                        "injection posterior confidently harmful",
                    )
                    if incoming_id:
                        self._tombstoned.add(incoming_id)
                    return self._ledger_record(
                        card,
                        "",
                        WriteOutcome.REJECTED_HARM,
                        reason,
                    )
                final_id = self._store.save(card)
                return self._ledger_record(
                    card,
                    final_id,
                    WriteOutcome.ADDED,
                    "librarian-authored card",
                )

            harmful = False
            leased = False
            reason = ""

            def replace(fresh: Card) -> Card | None:
                nonlocal harmful, leased, reason
                merged = card.model_copy(
                    update={
                        "gain_events": union_events(
                            fresh.gain_events, card.gain_events
                        ),
                        "absorbed_ids": union_strings(
                            fresh.absorbed_ids, card.absorbed_ids
                        ),
                    }
                )
                if not self._evictor.should_evict(merged):
                    return merged
                harmful = True
                reason = _eviction_reason(
                    self._evictor,
                    merged,
                    "injection posterior confidently harmful",
                )
                if self._is_leased(fresh.id):
                    leased = True
                    return fresh
                return None

            saved = self._store.update(incoming_id, replace)
            if saved is None:
                return _DISCARDED
            if harmful:
                if leased:
                    self._log_leased_skip(incoming_id, reason)
                else:
                    self._tombstoned.add(incoming_id)
                return self._ledger_record(
                    card,
                    "",
                    WriteOutcome.REJECTED_HARM,
                    reason,
                )
            return self._ledger_record(
                saved,
                incoming_id,
                WriteOutcome.UPDATED,
                "known id replaced",
                incoming_id=incoming_id,
            )

    def merge(self, target_id: str, card: Card) -> WriteResult:
        with self._eviction_guard():
            submitted_id = card.id
            fresh_target: Card | None = None
            merged: Card | None = None
            submitted_banked = False
            reason = ""

            def fold(target: Card, partner: Card | None) -> Card | None:
                nonlocal fresh_target, merged, submitted_banked, reason
                fresh_target = target
                submitted_banked = partner is not None
                if target.kind is not CardKind.INSIGHT:
                    raise MergeAborted
                if partner is not None and self._is_leased(partner.id):
                    self._log_leased_skip(
                        partner.id, "librarian merge would retire absorbed partner"
                    )
                    raise MergeAborted
                incoming = card
                if partner is not None:
                    incoming = card.model_copy(
                        update={
                            "gain_events": union_events(
                                partner.gain_events, card.gain_events
                            ),
                            "absorbed_ids": union_strings(
                                partner.absorbed_ids, card.absorbed_ids
                            ),
                            "programs": union_strings(partner.programs, card.programs),
                        }
                    )
                merged = merge_cards(target, incoming, replace_description=True)
                if not self._evictor.should_evict(merged):
                    return merged
                reason = _eviction_reason(
                    self._evictor, merged, "merged card confidently harmful"
                )
                if self._is_leased(target.id):
                    self._log_leased_skip(target.id, reason)
                    raise MergeAborted
                return None

            result = self._store.merge_retire(target_id, submitted_id, fold)
            if result.outcome in {"target_missing", "aborted"}:
                return _DISCARDED
            if fresh_target is None or merged is None:
                raise RuntimeError("merge transaction completed without fold state")
            if result.outcome == "retired":
                self._tombstoned.add(target_id)
                if submitted_banked:
                    self._tombstoned.add(submitted_id)
                self._ledger_record(fresh_target, "", WriteOutcome.EVICTED, reason)
                return self._ledger_record(
                    merged,
                    "",
                    WriteOutcome.REJECTED_HARM,
                    reason,
                    incoming_id=submitted_id,
                    merge_targets=(target_id,),
                )
            return self._ledger_record(
                merged,
                target_id,
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
        with self._eviction_guard():
            fresh_card: Card | None = None
            blocker = ""

            def revalidate(fresh: Card) -> Card | None:
                nonlocal fresh_card, blocker
                fresh_card = fresh
                if fresh.kind is not CardKind.PROGRAM:
                    blocker = "fresh card is not a program exemplar"
                    return fresh
                if self._is_leased(fresh.id):
                    blocker = "card is leased by an in-flight mutation"
                    return fresh
                blocker = self._foreign_retention_veto(fresh) or ""
                return fresh if blocker else None

            if self._store.update(card.id, revalidate) is None:
                return _DISCARDED
            if blocker:
                logger.info(
                    "[Memory][Admission] skipped retirement of card {}: {}; {}",
                    card.id,
                    reason,
                    blocker,
                )
                return _DISCARDED
            if fresh_card is None:
                return _DISCARDED
            return self._ledger_record(
                fresh_card, successor_id, WriteOutcome.UPDATED, reason
            )

    def retire_twin(self, twin: Card, *, successor_id: str) -> WriteResult:
        """Fold a fresh superseded twin into its successor and retire it."""
        reason = "exemplar superseded by strictly-better twin"
        fresh_twin: Card | None = None
        blocker = ""

        def fold(successor: Card, partner: Card | None) -> Card:
            nonlocal fresh_twin, blocker
            if partner is None:
                raise MergeAborted
            fresh_twin = partner
            if partner.kind is not CardKind.PROGRAM:
                blocker = "fresh card is not a program exemplar"
                raise MergeAborted
            if self._is_leased(partner.id):
                blocker = "card is leased by an in-flight mutation"
                raise MergeAborted
            blocker = self._foreign_retention_veto(partner) or ""
            if blocker:
                raise MergeAborted
            return merge_cards(successor, partner, replace_description=False)

        with self._eviction_guard():
            result = self._store.merge_retire(successor_id, twin.id, fold)
        if result.outcome != "merged" or fresh_twin is None:
            if blocker:
                logger.info(
                    "[Memory][Admission] skipped retirement of card {}: {}; {}",
                    twin.id,
                    reason,
                    blocker,
                )
            return _DISCARDED
        return self._ledger_record(
            fresh_twin, successor_id, WriteOutcome.UPDATED, reason
        )

    def bump_provenance(self, target_id: str, child_id: str) -> WriteResult:
        eligible = False

        def fold(target: Card) -> Card:
            nonlocal eligible
            if target.kind is not CardKind.INSIGHT:
                return target
            eligible = True
            if not child_id or child_id in target.programs:
                return target
            return target.model_copy(update={"programs": (*target.programs, child_id)})

        target = self._store.update(target_id, fold)
        if target is None or not eligible:
            return _DISCARDED
        return self._ledger_record(
            target, target_id, WriteOutcome.UPDATED, "duplicate provenance bump"
        )

    def sweep(self) -> list[str]:
        bank = self._store.snapshot()
        candidates = list(self._evictor.sweep(bank))
        evicted: list[str] = []
        for cid in candidates:
            fresh_card: Card | None = None
            reason = ""
            rescued = False
            leased = False

            def revalidate(card: Card) -> Card | None:
                nonlocal fresh_card, reason, rescued, leased
                fresh_card = card
                if not self._evictor.should_evict(card):
                    rescued = True
                    reason = "fresh verdict no longer says evict"
                    return card
                reason = _eviction_reason(self._evictor, card, "confidently harmful")
                if self._is_leased(card.id):
                    leased = True
                    return card
                # Persist evicted evidence BEFORE the card leaves the live bank so
                # the empirical-Bayes cold prior never sees a survivorship-biased
                # cohort during the delete window (evict only, not rescued/leased).
                if self._evicted_evidence_sink is not None:
                    try:
                        self._evicted_evidence_sink.record(card)
                    except Exception as exc:
                        logger.warning(
                            "[Memory][Admission] failed to record evicted evidence: {}",
                            exc,
                        )
                return None

            with self._eviction_guard():
                if self._store.update(cid, revalidate) is None:
                    continue
            if rescued:
                logger.info(
                    "[Memory][Admission] rescued card {} from stale eviction: {}",
                    cid,
                    reason,
                )
                continue
            if leased:
                self._log_leased_skip(cid, reason)
                continue
            evicted.append(cid)
            self._tombstoned.add(cid)
            if fresh_card is not None:
                self._ledger_record(fresh_card, "", WriteOutcome.EVICTED, reason)
        return evicted

    def _eviction_guard(self):
        if self._selection_leases is None:
            return nullcontext()
        return self._selection_leases.eviction_guard()

    def _is_leased(self, card_id: str) -> bool:
        return bool(
            self._selection_leases is not None
            and self._selection_leases.is_leased(card_id)
        )

    def _foreign_retention_veto(self, card: Card) -> str | None:
        return foreign_retention_veto(
            card,
            task_key=self._task_key,
            min_effective_events=self._min_effective_events,
        )

    @staticmethod
    def _log_leased_skip(card_id: str, reason: str) -> None:
        logger.info(
            "[Memory][Admission] skipped eviction of leased card {}: {}",
            card_id,
            reason,
        )

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
