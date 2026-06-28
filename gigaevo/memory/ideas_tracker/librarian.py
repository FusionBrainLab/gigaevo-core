"""Librarian: the LLM-first idea-write path from a mutation diff into the bank.

One ``ingest_idea`` per valid child at birth. The librarian:
1. asks a ``NeighborSource`` for the nearest existing cards;
2. if the closest is within ``eps`` (a near-duplicate), short-circuits the LLM
   and just bumps that card's provenance through the gate (pre-gate path);
3. otherwise runs the ``ReconcileAgent`` over the parent->child diff to author
   0..N clean cards, each routed to the gate by decision: NEW (admit),
   DUPLICATE (bump the target's provenance), or MERGE (union onto the target).
   An LLM failure admits the note verbatim — never a silent drop.

``author_program`` fills exemplar ``ProgramCard`` prose, cached by program id so
a re-seeded exemplar never re-pays the LLM.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.memory.core.events import emit_memory_event
from gigaevo.memory.shared_memory.models import CardT, MemoryCard, ProgramCard


class NeighborSource(Protocol):
    def nearest(
        self, note: str, k: int, card_type: type[CardT]
    ) -> list[tuple[CardT, float]]: ...


class Librarian:
    def __init__(
        self,
        *,
        agent: Any,
        program_author: Any,
        gate: Any,
        store: Any,
        neighbors: NeighborSource,
        eps: float = 0.05,
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
        self._eps = eps
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
            # Idea dedup is MemoryCard-only: an idea is never a near-duplicate of
            # a program exemplar, and the reconcile agent must not be offered
            # program ids as MERGE/DUPLICATE targets.
            hits = self._neighbors.nearest(note, self._top_k, MemoryCard)
        except Exception as exc:
            emit_memory_event(
                component="librarian",
                event_type="neighbor.retrieval_failed",
                payload={"error": str(exc), "child_id": child_id},
                level="WARNING",
            )
            hits = []
        if hits and hits[0][1] <= self._eps:
            bumped = self._gate.bump_provenance(hits[0][0].id, child_id)
            return [bumped] if bumped else []

        try:
            resp = await self._agent.arun(
                base_parent_code=base_parent_code,
                child_code=child_code,
                note=note,
                neighbors=[c for c, _ in hits],
            )
        except Exception as exc:
            logger.warning(
                "[Memory][Librarian] reconcile LLM failed ({}); admitting note "
                "verbatim (NOT a drop).",
                exc,
            )
            fid = self._gate.admit(
                MemoryCard(
                    id="",
                    description=note,
                    keywords=[],
                    programs=[child_id],
                    task_description=self._task_description,
                    task_description_summary=self._task_description_summary,
                )
            )
            return [fid] if fid else []

        out: list[str] = []
        for item in resp.items[: self._max_cards]:
            card = MemoryCard(
                id="",
                description=item.card.description,
                explanation_summary=item.card.explanation_summary,
                keywords=item.card.keywords,
                programs=[child_id],
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
        existing = self._store.get_card(f"program-{program_id}")
        if existing is not None and (existing.description or "").strip():
            return ProgramAuthorResponse(
                description=existing.description,
                explanation_summary=existing.explanation_summary,
                keywords=list(existing.keywords),
            )
        return await self._program_author.arun(code=code, fitness=fitness)

    def admit_program(self, card: ProgramCard, *, higher_is_better: bool) -> str:
        """Admit an exemplar card, deduping same-strategy twins by fitness.

        Exemplar cards are program-id-keyed, so two distinct programs that
        converge on the same strategy would otherwise bank two near-identical
        cards. The NeighborSource surfaces such a twin (a different-id program
        card within ``eps``); we keep the higher-fitness exemplar — replacing the
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

    def _nearest_program_twin(self, card: ProgramCard) -> ProgramCard | None:
        try:
            hits = self._neighbors.nearest(card.description, self._top_k, ProgramCard)
        except Exception as exc:
            emit_memory_event(
                component="librarian",
                event_type="neighbor.retrieval_failed",
                payload={"error": str(exc), "card_id": card.id},
                level="WARNING",
            )
            return None
        for neighbor, distance in hits:
            if distance > self._eps:
                break
            if isinstance(neighbor, ProgramCard) and neighbor.id != card.id:
                return neighbor
        return None


def _strictly_better(
    incoming: float | None, banked: float | None, higher_is_better: bool
) -> bool:
    if incoming is None:
        return False
    if banked is None:
        return True
    return incoming > banked if higher_is_better else incoming < banked
