"""Librarian: the LLM-first idea-write path from a mutation diff into the bank.

One ``ingest_idea`` per valid child at birth. The librarian:
1. asks a ``NeighborSource`` for the nearest existing cards;
2. if the closest is within ``eps`` (a near-duplicate), short-circuits the LLM
   and just bumps that card's provenance through the gate (pre-gate path);
3. otherwise runs the ``ReconcileAgent`` over the parent->child diff to author
   0..N clean cards, each routed to the gate as NEW (admit) or MERGE/DUPLICATE
   (merge onto the target). An LLM failure admits the note verbatim — never a
   silent drop.

``author_program`` fills exemplar ``ProgramCard`` prose, cached by program id so
a re-seeded exemplar never re-pays the LLM.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.memory.shared_memory.card_conversion import AnyCard
from gigaevo.memory.shared_memory.models import MemoryCard


class NeighborSource(Protocol):
    def nearest(self, note: str, k: int) -> list[tuple[AnyCard, float]]: ...


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
        base_parent_id: str,
        base_parent_code: str,
        child_id: str,
        child_code: str,
        note: str,
    ) -> list[str]:
        try:
            hits = self._neighbors.nearest(note, self._top_k)
        except Exception:
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
                keywords=item.card.keywords,
                programs=[child_id],
                task_description=self._task_description,
                task_description_summary=self._task_description_summary,
            )
            if item.decision == "NEW":
                fid = self._gate.admit(card)
            else:  # DUPLICATE | MERGE
                fid = self._gate.merge(item.target_id, card) if item.target_id else ""
            if fid:
                out.append(fid)
        return out

    async def author_program(
        self, *, program_id: str, code: str, fitness: float | None
    ) -> ProgramAuthorResponse:
        existing = self._store.card_store.cards.get(f"program-{program_id}")
        if existing is not None and (existing.description or "").strip():
            return ProgramAuthorResponse(
                description=existing.description, keywords=list(existing.keywords)
            )
        return await self._program_author.arun(code=code, fitness=fitness)
