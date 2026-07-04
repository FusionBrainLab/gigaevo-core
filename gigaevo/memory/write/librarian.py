"""Librarian: the LLM-first idea-write path from a mutation diff into the bank.

One ``ingest_idea`` per valid child at birth. The librarian asks a
``NeighborSource`` for the nearest existing cards, then runs the
``ReconcileAgent`` over the parent->child diff with those neighbors as context
to author 0..N clean cards, each routed to the gate by decision: NEW (admit),
DUPLICATE (bump the target's provenance), or MERGE (union onto the target).
The agent is the only write-time dedup arbiter on the idea path — there is no
embedding-distance threshold anywhere in it (raw notes and authored prose sit
too far apart in embedding space for a cosine gate to fire on real twins). An
LLM failure admits the note verbatim — never a silent drop.

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

from typing import Protocol

from loguru import logger

from gigaevo.llm.agents.admission_novelty import NoveltyAdmissionAgent
from gigaevo.llm.agents.program_author import ProgramAuthorAgent, ProgramAuthorResponse
from gigaevo.llm.agents.reconcile import ReconcileAgent
from gigaevo.memory.cards import Card, CardKind, ContextualGain
from gigaevo.memory.storage.base import MemoryStore, ScoredCard
from gigaevo.memory.write.admission import CardAdmissionGate


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
    ) -> list[str]:
        # founding_gain seeds the child's verified parent->child delta so a fresh
        # card bids on its own evidence; it rides a NEW admit and a MERGE union,
        # and is dropped on a DUPLICATE (the target keeps its own founding event).
        events = (founding_gain,) if founding_gain is not None else ()
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
            fid = self._gate.admit(
                Card(
                    id="",
                    description=note,
                    programs=(child_id,),
                    task_description=self._task_description,
                    task_description_summary=self._task_description_summary,
                    gain_events=events,
                )
            )
            return [fid] if fid else []

        out: list[str] = []
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
                fid = await self._admit_new(card)
            elif item.decision == "DUPLICATE":
                fid = (
                    self._gate.bump_provenance(item.target_id, child_id)
                    if item.target_id
                    else ""
                )
            else:  # MERGE
                fid = self._gate.merge(item.target_id, card) if item.target_id else ""
            # A DUPLICATE/MERGE whose target is empty or already gone from the
            # bank makes the gate a no-op (returns ""): author the idea as NEW
            # rather than silently dropping it — through the same novelty gate.
            if not fid and item.decision != "NEW":
                fid = await self._admit_new(card)
            if fid:
                out.append(fid)
        return out

    async def _admit_new(self, card: Card) -> str:
        """Admit a fresh idea card, gating it through the novelty judge first.

        With no judge wired (the default) this is a plain gate admit. With a
        judge, a card the mutator would already reach for unprompted is rejected
        before it enters the bank (returns ""); a judge error fails open and
        admits, so the gate can never silently drop the write path.
        """
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
                    return ""
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

    def admit_program(self, card: Card, *, higher_is_better: bool) -> str:
        """Admit an exemplar card, deduping code-identical twins by fitness.

        Exemplar cards are program-id-keyed, so two distinct programs that
        evolution produced with identical code (a no-op crossover that returns a
        parent verbatim is common) would otherwise bank two identical cards. A
        twin is an existing program card with the same normalized code — exact
        identity, not a prose-embedding cosine (independently-authored
        descriptions of the same code sit too far apart for any cosine gate to
        fire). We keep the higher-fitness exemplar — replacing the banked twin
        only when the incoming card is strictly better, otherwise dropping the
        redundant card — so the bank holds one best representative per program.
        The bank is updated synchronously, so a co-batch twin admitted earlier in
        the same sweep is already visible here (intra-batch safe). Re-admitting an
        exemplar already in the bank flows straight to the gate as an UPDATE (its
        own id is never its twin).
        """
        twin = self._code_twin(card)
        if twin is not None:
            if not _strictly_better(card.fitness, twin.fitness, higher_is_better):
                return ""
            self._store.delete(twin.id)
        return self._gate.admit(card)

    def _code_twin(self, card: Card) -> Card | None:
        key = _code_key(card.code)
        if not key:
            return None
        for other in self._store.snapshot():
            if (
                other.kind is CardKind.PROGRAM
                and other.id != card.id
                and _code_key(other.code) == key
            ):
                return other
        return None


def _strictly_better(
    incoming: float | None, banked: float | None, higher_is_better: bool
) -> bool:
    if incoming is None:
        return False
    if banked is None:
        return True
    return incoming > banked if higher_is_better else incoming < banked


def _code_key(code: str) -> str:
    """Normalize program source for exact-twin comparison: strip surrounding
    blank lines and per-line trailing whitespace so trailing-whitespace noise
    can't split a genuine code twin. Deliberately does not reformat — two
    differently-structured programs are different exemplars, not twins."""
    return "\n".join(line.rstrip() for line in code.strip().splitlines())
