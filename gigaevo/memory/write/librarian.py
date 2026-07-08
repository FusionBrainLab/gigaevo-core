"""Librarian: the LLM-first idea-write path from a mutation diff into the bank.

One ``ingest_idea`` per valid child at birth. The librarian asks a
``NeighborSource`` for the nearest existing cards, then runs the
``ReconcileAgent`` over the parent->child diff with those neighbors as context
to author 0..N clean cards, each routed to the gate by decision: NEW (admit),
DUPLICATE (bump the target's provenance), or MERGE (union onto the target).
The agent is the only write-time dedup arbiter on the idea path — there is no
embedding-distance threshold anywhere in it (raw notes and authored prose sit
too far apart in embedding space for a cosine gate to fire on real twins). An
LLM failure lands the note verbatim — deduped to its exact banked twin when
one exists — never a silent drop.

An optional ``admission_judge`` gates freshly-authored cards on novelty against
the mutator's prior: a lever a strong model would already reach for unprompted
is rejected before it enters the bank (the bank's binding constraint is an
excess of prior-known cards, not plumbing). Off by default; a judge error fails
open and admits. The reconcile-failed verbatim path is never gated — it is the
never-silent-drop degrade path, and a second LLM hop would likely fail too.

``author_program`` fills exemplar card prose, cached by program id so a
re-seeded exemplar never re-pays the LLM.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from loguru import logger

from gigaevo.llm.agents.admission_novelty import NoveltyAdmissionAgent
from gigaevo.llm.agents.program_author import ProgramAuthorAgent, ProgramAuthorResponse
from gigaevo.llm.agents.reconcile import ReconcileAgent
from gigaevo.memory.cards import Card, CardKind, ContextualGain
from gigaevo.memory.storage.base import MemoryStore, ScoredCard
from gigaevo.memory.write.admission import CardAdmissionGate, WriteOutcome, WriteResult


class NeighborSource(Protocol):
    def nearest(
        self, text: str, k: int, kind: CardKind | None = None
    ) -> list[ScoredCard]: ...


class Librarian:
    def __init__(
        self,
        *,
        agent: ReconcileAgent,
        program_author: ProgramAuthorAgent,
        gate: CardAdmissionGate,
        store: MemoryStore,
        neighbors: NeighborSource,
        top_k: int = 5,
        max_cards: int = 3,
        task_description: str = "",
        task_description_summary: str = "",
        admission_judge: NoveltyAdmissionAgent | None = None,
    ) -> None:
        self._agent = agent
        self._program_author = program_author
        self._gate = gate
        self._store = store
        self._neighbors = neighbors
        self._top_k = top_k
        self._max_cards = max_cards
        self._task_description = task_description
        self._task_description_summary = task_description_summary
        self._admission_judge = admission_judge

    async def ingest_idea(
        self,
        *,
        base_parent_code: str,
        child_id: str,
        child_code: str,
        note: str,
        founding_gain: ContextualGain | None = None,
        sink: list[WriteResult] | None = None,
    ) -> list[WriteResult]:
        # One WriteResult per routed card: the writer needs the outcome, not
        # just an id — landed ids feed the consolidation cadence, freshly
        # ADDED ids the inline intra-batch consolidation subset.
        # ``sink`` receives each result the moment it is produced: when the
        # writer's wait_for cancels a hung ingest, the return value dies with
        # the coroutine but cards already routed through the gate are banked —
        # the sink is how they still reach the writer's accounting.
        # founding_gain seeds the child's verified parent->child delta so a fresh
        # card bids on its own evidence; it rides a NEW admit only. A MERGE or
        # DUPLICATE target keeps exactly its own evidence — the delta was measured
        # for THIS child against ITS parent, foreign context for a pre-existing
        # lever (and the founding flag would defeat merge event dedup).
        events = (founding_gain,) if founding_gain is not None else ()

        def record_result(result: WriteResult) -> WriteResult:
            if sink is not None:
                sink.append(result)
            return result

        try:
            # Idea dedup is insight-only: an idea is never a near-duplicate of
            # a program exemplar, and the reconcile agent must not be offered
            # program ids as MERGE/DUPLICATE targets.
            hits = self._neighbors.nearest(note, self._top_k, CardKind.INSIGHT)
        except Exception as exc:
            logger.warning(
                "[Memory][Librarian] neighbor retrieval failed for child {}: {}",
                child_id,
                exc,
            )
            hits = []
        allowed_targets = {h.card.id for h in hits if h.card.id}

        try:
            resp = await self._agent.arun(
                base_parent_code=base_parent_code,
                child_code=child_code,
                note=note,
                neighbors=[h.card for h in hits],
            )
        except Exception as exc:
            logger.warning(
                "[Memory][Librarian] reconcile LLM failed ({}); admitting note "
                "verbatim (NOT a drop).",
                exc,
            )
            card = Card(
                id="",
                description=note,
                programs=(child_id,),
                task_description=self._task_description,
                task_description_summary=self._task_description_summary,
                gain_events=events,
            )
            # Never gated by the novelty judge (degrade path), but the zero-LLM
            # exact-twin check still applies: a repeated note must bump its
            # banked twin's provenance, not mint a second id for the same prose.
            twin = self._desc_twin(card)
            if twin is not None:
                return [record_result(self._gate.bump_provenance(twin.id, child_id))]
            return [record_result(self._gate.admit(card))]

        out: list[WriteResult] = []
        for item in resp.items[: self._max_cards]:
            card = Card(
                id="",
                description=item.card.description,
                explanation_summary=item.card.explanation_summary,
                keywords=tuple(item.card.keywords),
                programs=(child_id,),
                task_description=self._task_description,
                task_description_summary=self._task_description_summary,
                gain_events=events,
            )
            if item.decision == "NEW":
                result = await self._admit_new(card)
            elif item.decision == "DUPLICATE":
                target_id = _allowed_target_id(item.target_id, allowed_targets)
                result = await self._land_dedup(
                    self._gate.bump_provenance(target_id, child_id)
                    if target_id
                    else None,
                    card,
                    item.target_id,
                )
            else:  # MERGE
                target_id = _allowed_target_id(item.target_id, allowed_targets)
                result = await self._land_dedup(
                    self._gate.merge(
                        target_id, card.model_copy(update={"gain_events": ()})
                    )
                    if target_id
                    else None,
                    card,
                    item.target_id,
                )
            out.append(record_result(result))
        return out

    async def _land_dedup(
        self, result: WriteResult | None, card: Card, target_id: str
    ) -> WriteResult:
        """Resolve a DUPLICATE/MERGE gate verdict to its final write result.

        A landed verdict stands. Re-authoring as NEW is whitelisted to the
        two genuinely benign cases — an empty target id (``None`` here) or a
        gate no-op (``DISCARDED``: target absent/ineligible, or the store merge
        failed). Every other non-landed verdict is a harm-driven deletion (the
        gate judged the union confidently harmful and deleted the merge target)
        and is passed through as-is — never laundered back in as a fresh card.
        A target-absent no-op is NOT benign when the target was harm-tombstoned
        earlier this run (typically by a prior item of the same ingest): the
        harm verdict must stick, so that card drops too.
        """
        if result is not None and result.landed:
            return result
        if (
            (result is None or result.benign_noop)
            and target_id
            and self._gate.is_tombstoned(target_id)
        ):
            logger.info(
                "[Memory][Librarian] dedup target {} was harm-tombstoned this "
                "run; dropping the card rather than re-authoring it as new.",
                target_id,
            )
            return result or WriteResult(outcome=WriteOutcome.REJECTED_HARM)
        if result is None or result.benign_noop:
            return await self._admit_new(card)
        logger.info(
            "[Memory][Librarian] dedup target folded to a confidently harmful "
            "union; dropping the card rather than re-authoring it as new."
        )
        return result

    async def _admit_new(self, card: Card) -> WriteResult:
        """Admit a fresh idea card, deduping exact text twins and gating it
        through the novelty judge.

        The reconcile agent only sees top-k neighbors, so a NEW ruling can bank
        a description the bank already holds verbatim when the true twin missed
        that window; an exact normalized-description match is the zero-LLM last
        line of defense and resolves as a DUPLICATE provenance bump. With no
        judge wired (the default) the rest is a plain gate admit. With a judge,
        a card the mutator would already reach for unprompted is rejected
        before it enters the bank (ledgered as REJECTED_NOVELTY); a judge error
        fails open and admits, so the gate can never silently drop the write
        path.
        """
        twin = self._desc_twin(card)
        if twin is not None:
            child_id = card.programs[0] if card.programs else ""
            return self._gate.bump_provenance(twin.id, child_id)
        if self._admission_judge is not None:
            try:
                verdict = await self._admission_judge.arun(
                    description=card.description,
                    explanation_summary=card.explanation_summary,
                )
            except Exception as exc:
                logger.warning(
                    "[Memory][Librarian] novelty gate failed ({}); admitting card "
                    "(fail-open).",
                    exc,
                )
            else:
                if not verdict.keep:
                    logger.info(
                        "[Memory][Librarian] novelty gate rejected prior-known "
                        "card: {}",
                        verdict.reason,
                    )
                    return self._gate.reject_novelty(card, verdict.reason)
        return self._gate.admit(card)

    async def author_program(
        self, *, program_id: str, code: str, fitness: float | None
    ) -> ProgramAuthorResponse:
        existing = self._store.get(f"program-{program_id}")
        if existing is not None and (existing.description or "").strip():
            return ProgramAuthorResponse(
                description=existing.description,
                explanation_summary=existing.explanation_summary,
                keywords=list(existing.keywords),
            )
        return await self._program_author.arun(code=code, fitness=fitness)

    def admit_program(
        self,
        card: Card,
        *,
        higher_is_better: bool,
        min_fitness_gap: float = 0.0,
    ) -> str:
        """Admit an exemplar card, deduping code-identical twins by fitness.

        Exemplar cards are program-id-keyed, so two distinct ids can represent
        the same source. Identity is exact via the stored source hash (or
        legacy source body). We keep the best-fitness representative, replacing
        existing twins only when the incoming card clears ``min_fitness_gap``.
        Twins are retired only AFTER the incoming card lands, so a harm-rejected
        incoming never deletes the banked representative.
        """
        twins = self._program_twins(card)
        best_twin = _best_by_fitness(twins, higher_is_better)
        if best_twin is not None and not _strictly_better(
            card.fitness,
            best_twin.fitness,
            higher_is_better,
            min_delta=min_fitness_gap,
        ):
            return ""
        result = self._gate.admit(card)
        if result.landed:
            for twin in twins:
                self._gate.retire_twin(twin, successor_id=result.card_id)
        return result.card_id

    def _program_twins(self, card: Card) -> list[Card]:
        code_identity = program_code_sha256(card)
        if not code_identity:
            return []
        out: list[Card] = []
        for other in self._store.snapshot():
            if other.kind is not CardKind.PROGRAM or other.id == card.id:
                continue
            if program_code_sha256(other) == code_identity:
                out.append(other)
        return out

    def _desc_twin(self, card: Card) -> Card | None:
        key = _desc_key(card.description)
        if not key:
            return None
        for other in self._store.snapshot():
            if (
                other.kind is CardKind.INSIGHT
                and other.id != card.id
                and _desc_key(other.description) == key
            ):
                return other
        return None


def _allowed_target_id(target_id: str, allowed_targets: set[str]) -> str:
    target_id = target_id.strip()
    if not target_id:
        return ""
    if target_id not in allowed_targets:
        logger.info(
            "[Memory][Librarian] ignoring reconcile target {} because it was not "
            "one of the offered neighbor ids",
            target_id,
        )
        return ""
    return target_id


def _strictly_better(
    incoming: float | None,
    banked: float | None,
    higher_is_better: bool,
    *,
    min_delta: float = 0.0,
) -> bool:
    if incoming is None:
        return False
    if banked is None:
        return True
    return (
        incoming > banked + min_delta
        if higher_is_better
        else incoming < banked - min_delta
    )


def _best_by_fitness(cards: list[Card], higher_is_better: bool) -> Card | None:
    best: Card | None = None
    for card in cards:
        if best is None or _strictly_better(
            card.fitness, best.fitness, higher_is_better
        ):
            best = card
    return best


def _code_key(code: str) -> str:
    """Normalize program source for exact-twin comparison: strip surrounding
    blank lines and per-line trailing whitespace so trailing-whitespace noise
    can't split a genuine code twin. Deliberately does not reformat — two
    differently-structured programs are different exemplars, not twins."""
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def code_sha256(code: str) -> str:
    key = _code_key(code)
    if not key:
        return ""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def program_code_sha256(card: Card) -> str:
    return card.code_sha256 or code_sha256(card.code)


def _desc_key(text: str) -> str:
    """Normalize idea prose for exact-twin comparison: collapse whitespace and
    casefold so formatting noise can't split a genuine twin. Deliberately not a
    fuzzy match — near-duplicates stay the reconcile/consolidation agents' call."""
    return " ".join(text.split()).casefold()
