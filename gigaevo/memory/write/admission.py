"""Atomic card admission, equivalence updates, and periodic retirement."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import (
    Card,
    CardKind,
    ContextualGain,
    union_use_trials,
)
from gigaevo.memory.prior_evidence import EvictedEvidenceSink, _JsonlFileLock
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.memory.write.eviction import Evictor, foreign_retention_veto


class WriteOutcome(StrEnum):
    """Verdict of one card-bank ingest.

    Every state-changing verdict is recorded in the write ledger. DISCARDED is
    the no-op verdict: the target was absent or ineligible and no row is written,
    so the caller may route the candidate through ordinary admission.
    """

    ADDED = "added"
    UPDATED = "updated"
    DISCARDED = "discarded"
    REJECTED_RETIRED = "rejected_retired"
    REJECTED_NOVELTY = "rejected_novelty"
    REJECTED_CAPACITY = "rejected_capacity"
    RETIRED = "retired"
    EVICTED = "evicted"


class WriteResult(BaseModel):
    """What one gate ingest did, and the bank id the card landed under."""

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
    def rejected_retired(self) -> bool:
        return self.outcome is WriteOutcome.REJECTED_RETIRED

    @property
    def benign_noop(self) -> bool:
        """The gate did nothing and judged nothing: safe to re-author as fresh.

        The only verdict a caller may route back into the bank as new. Every
        explicit rejection must be dropped rather than retried under a new id.
        """
        return self.outcome is WriteOutcome.DISCARDED


_DISCARDED = WriteResult(outcome=WriteOutcome.DISCARDED)


class WriteLedgerRecord(BaseModel):
    """One JSONL row of the write ledger: a single ingest or eviction verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_utc: str = Field(
        description="ISO-8601 UTC time the verdict was recorded."
    )
    incoming_id: str = Field(description="Id of the card as submitted to the bank.")
    final_id: str = Field(
        description="Id the card ended up under; empty if it did not land."
    )
    outcome: WriteOutcome = Field(description="Gate verdict for this card.")
    reason: str = Field(
        default="", description="Human-readable rationale emitted by the deciding gate."
    )
    duplicate_of: str = Field(
        default="", description="Id of the existing card this one duplicated, if any."
    )
    incoming_description: str = Field(
        default="",
        description="Authored treatment text, retained for equivalence auditing.",
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
        incoming_description: str = "",
    ) -> None:
        try:
            row = WriteLedgerRecord(
                timestamp_utc=datetime.now(UTC).isoformat(),
                incoming_id=incoming_id,
                final_id=final_id,
                outcome=WriteOutcome(outcome),
                reason=reason,
                duplicate_of=duplicate_of,
                incoming_description=incoming_description,
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with _JsonlFileLock(self._path.with_suffix(self._path.suffix + ".lock")):
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(row.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("[Memory][WriteLedger] failed to record: {}", exc)


class CardAdmissionGate:
    """Write ledger over the card store; causal retirement lives in ``sweep``.

    Admission never consults the evictor. Periodic maintenance is the only place
    where a verdict can remove a card, and it is atomically revalidated against
    the fresh store revision before deletion. Retired ids and exact authored
    hypothesis payloads are tombstoned for the rest of the run, preventing
    deterministic re-author churn without adding a durable legacy registry.
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
        max_task_cards: int | None = None,
    ) -> None:
        if max_task_cards is not None and max_task_cards < 1:
            raise ValueError("max_task_cards must be positive when configured")
        self._store = store
        self._evictor = evictor
        self._ledger = ledger
        self._evicted_evidence_sink = evicted_evidence_sink
        self._selection_leases = selection_leases
        self._task_key = task_key
        self._min_effective_events = min_effective_events
        self._max_task_cards = max_task_cards
        self._tombstoned: set[str] = set()
        self._retired_hypotheses: set[tuple[str, CardKind, str, str]] = set()

    def is_tombstoned(self, card_id: str) -> bool:
        """True iff this id was retired earlier in the run. Lets callers
        skip work the gate would reject anyway — the writer checks before
        paying the exemplar-author LLM call. In-memory only: a restart clears
        the set (accepted — the churn this kills is intra-run)."""
        return card_id in self._tombstoned

    def admit(self, card: Card) -> WriteResult:
        with self._eviction_guard():
            incoming_id = card.id.strip()
            if (
                incoming_id in self._tombstoned
                or _hypothesis_key(card) in self._retired_hypotheses
            ):
                return self._ledger_record(
                    card,
                    "",
                    WriteOutcome.REJECTED_RETIRED,
                    "tombstoned: causally retired earlier this run",
                )
            known = bool(incoming_id) and self._store.get(incoming_id) is not None

            if not known:
                if self._task_capacity_reached(card):
                    return self._ledger_record(
                        card,
                        "",
                        WriteOutcome.REJECTED_CAPACITY,
                        f"active task card cap reached ({self._max_task_cards})",
                    )
                final_id = self._store.save(card)
                return self._ledger_record(
                    card,
                    final_id,
                    WriteOutcome.ADDED,
                    "librarian-authored card",
                )

            def replace(fresh: Card) -> Card | None:
                return fresh.model_copy(
                    update={
                        "gain_events": union_events(
                            fresh.gain_events, card.gain_events
                        ),
                        "use_trials": union_use_trials(
                            fresh.use_trials, card.use_trials
                        ),
                        "absorbed_ids": union_strings(
                            fresh.absorbed_ids, card.absorbed_ids
                        ),
                        "programs": union_strings(fresh.programs, card.programs),
                    }
                )

            saved = self._store.update(incoming_id, replace)
            if saved is None:
                return _DISCARDED
            return self._ledger_record(
                saved,
                incoming_id,
                WriteOutcome.UPDATED,
                "known id evidence updated",
                incoming_id=incoming_id,
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
                if self._is_card_leased(fresh):
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
                fresh_card, successor_id, WriteOutcome.RETIRED, reason
            )

    def update_equivalent(
        self,
        target_id: str,
        incoming: Card,
        *,
        higher_is_better: bool = True,
        min_fitness_gap: float = 0.0,
    ) -> WriteResult:
        """Pool exact-equivalent evidence without changing the banked action.

        Task-labelled events and trials remain distinct rows on one canonical
        semantic card. Raw program fitness chooses a representative only within
        one task, because scales across tasks are not comparable.
        """

        eligible = False
        representative_improved = False

        def fold(target: Card) -> Card:
            nonlocal eligible, representative_improved
            if target.kind is not incoming.kind:
                return target
            eligible = True
            same_task = target.task_key == incoming.task_key
            absorbed_ids = tuple(
                card_id
                for card_id in union_strings(target.absorbed_ids, incoming.absorbed_ids)
                if card_id != target.id
            )
            updates: dict = {
                "programs": union_strings(target.programs, incoming.programs),
                "gain_events": union_events(target.gain_events, incoming.gain_events),
                "use_trials": union_use_trials(target.use_trials, incoming.use_trials),
                "absorbed_ids": absorbed_ids,
            }
            if (
                target.kind is CardKind.PROGRAM
                and same_task
                and _fitness_improves(
                    incoming.fitness,
                    target.fitness,
                    higher_is_better=higher_is_better,
                    min_delta=min_fitness_gap,
                )
            ):
                representative_improved = True
                updates.update(
                    {
                        "program_id": incoming.program_id,
                        "fitness": incoming.fitness,
                        "code": incoming.code,
                    }
                )
            return target.model_copy(update=updates)

        target = self._store.update(target_id, fold)
        if target is None or not eligible:
            return _DISCARDED
        reason = "equivalent action; evidence pooled"
        if representative_improved:
            reason = "equivalent program strategy; best representative updated"
        return self._ledger_record(
            incoming,
            target_id,
            WriteOutcome.UPDATED,
            reason,
            incoming_id=incoming.id,
            duplicate_of=target_id,
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
            retained_elsewhere = False

            def revalidate(card: Card) -> Card | None:
                nonlocal fresh_card, reason, rescued, leased, retained_elsewhere
                fresh_card = card
                if not self._evictor.should_evict(card):
                    rescued = True
                    reason = "fresh verdict no longer says evict"
                    return card
                reason = _eviction_reason(self._evictor, card, "retirement verdict")
                if self._is_card_leased(card):
                    leased = True
                    return card
                if blocker := self._foreign_retention_veto(card):
                    retained_elsewhere = True
                    reason = blocker
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
            if retained_elsewhere:
                logger.info(
                    "[Memory][Admission] skipped retirement of card {}: {}",
                    cid,
                    reason,
                )
                continue
            evicted.append(cid)
            self._tombstoned.add(cid)
            if fresh_card is not None:
                self._retired_hypotheses.add(_hypothesis_key(fresh_card))
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

    def _is_card_leased(self, card: Card) -> bool:
        return any(
            self._is_leased(card_id) for card_id in (card.id, *card.absorbed_ids)
        )

    def _foreign_retention_veto(self, card: Card) -> str | None:
        return foreign_retention_veto(
            card,
            task_key=self._task_key,
            min_effective_events=self._min_effective_events,
        )

    def _task_capacity_reached(self, card: Card) -> bool:
        if self._max_task_cards is None:
            return False
        task_key = card.task_key or self._task_key
        return (
            sum(existing.task_key == task_key for existing in self._store.snapshot())
            >= self._max_task_cards
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
        duplicate_of: str = "",
    ) -> WriteResult:
        if self._ledger is not None:
            self._ledger.record(
                incoming_id=card.id if incoming_id is None else incoming_id,
                final_id=final_id,
                outcome=outcome,
                reason=reason,
                duplicate_of=duplicate_of,
                incoming_description=card.description,
            )
        return WriteResult(outcome=outcome, card_id=final_id)


def union_strings(a: Sequence[str], b: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*a, *b)))


def union_events(
    a: Sequence[ContextualGain],
    b: Sequence[ContextualGain],
) -> tuple[ContextualGain, ...]:
    out = list(a)
    out.extend(event for event in b if event not in out)
    return tuple(out)


def _fitness_improves(
    incoming: float | None,
    current: float | None,
    *,
    higher_is_better: bool,
    min_delta: float,
) -> bool:
    if incoming is None:
        return False
    if current is None:
        return True
    if higher_is_better:
        return incoming > current + min_delta
    return incoming < current - min_delta


def _hypothesis_key(card: Card) -> tuple[str, CardKind, str, str]:
    return (
        card.task_key,
        card.kind,
        _normalized_hypothesis_text(card.description),
        _normalized_hypothesis_text(card.explanation_summary),
    )


def _normalized_hypothesis_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _eviction_reason(evictor: Evictor, card: Card, default: str) -> str:
    text = evictor.eviction_reason(card).strip()
    if text:
        return text
    return default
