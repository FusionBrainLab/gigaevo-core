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

``author_program`` fills exemplar card prose, cached by program id so a
re-seeded exemplar never re-pays the LLM.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from gigaevo.llm.agents.program_author import ProgramAuthorAgent, ProgramAuthorResponse
from gigaevo.llm.agents.reconcile import ReconcileAgent
from gigaevo.memory.cards import Card, CardKind
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
        program_twin_eps: float = 0.05,
        top_k: int = 5,
        max_cards: int = 3,
        task_description: str = "",
        task_description_summary: str = "",
    ) -> None:
        self._agent = agent
        self._program_author = program_author
        self._gate = gate
        self._store = store
        self._neighbors = neighbors
        self._program_twin_eps = program_twin_eps
        self._top_k = top_k
        self._max_cards = max_cards
        self._task_description = task_description
        self._task_description_summary = task_description_summary

    async def ingest_idea(
        self,
        *,
        base_parent_code: str,
        child_id: str,
        child_code: str,
        note: str,
    ) -> list[str]:
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
            )
            if item.decision == "NEW":
                fid = self._gate.admit(card)
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
            # rather than silently dropping it.
            if not fid and item.decision != "NEW":
                fid = self._gate.admit(card)
            if fid:
                out.append(fid)
        return out

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
        """Admit an exemplar card, deduping same-strategy twins by fitness.

        Exemplar cards are program-id-keyed, so two distinct programs that
        converge on the same strategy would otherwise bank two near-identical
        cards. The NeighborSource surfaces such a twin (a different-id program
        card within ``program_twin_eps``); we keep the higher-fitness exemplar — replacing the
        banked twin only when the incoming card is strictly better, otherwise
        dropping the redundant card — so the bank holds one best representative
        per strategy. Re-admitting an exemplar already in the bank flows straight
        to the gate as an UPDATE (its own id is never its twin).
        """
        twin = self._nearest_program_twin(card)
        if twin is not None:
            if not _strictly_better(card.fitness, twin.fitness, higher_is_better):
                return ""
            self._store.delete(twin.id)
        return self._gate.admit(card)

    def _nearest_program_twin(self, card: Card) -> Card | None:
        try:
            hits = self._neighbors.nearest(
                card.description, self._top_k, CardKind.PROGRAM
            )
        except Exception as exc:
            logger.warning(
                "[Memory][Librarian] neighbor retrieval failed for card {}: {}",
                card.id,
                exc,
            )
            return None
        for hit in hits:
            if hit.distance > self._program_twin_eps:
                break
            if hit.card.id != card.id:
                return hit.card
        return None


def _strictly_better(
    incoming: float | None, banked: float | None, higher_is_better: bool
) -> bool:
    if incoming is None:
        return False
    if banked is None:
        return True
    return incoming > banked if higher_is_better else incoming < banked
