"""Minimal author → retrieve → equivalence memory write pipeline."""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from gigaevo.llm.agents.admission_novelty import NoveltyAdmissionAgent
from gigaevo.llm.agents.card_author import CardAuthorAgent
from gigaevo.llm.agents.equivalence import EquivalenceAgent
from gigaevo.llm.agents.program_author import ProgramAuthorAgent
from gigaevo.memory.cards import Card, CardKind, ContextualGain, card_brief
from gigaevo.memory.storage.base import MemoryStore, ScoredCard
from gigaevo.memory.write.admission import CardAdmissionGate, WriteOutcome, WriteResult
from gigaevo.memory.write.decisions import (
    ArchiveStatus,
    ValidityStatus,
    WriteDecision,
)


class NeighborSource(Protocol):
    def nearest(
        self, text: str, k: int, kind: CardKind | None = None
    ) -> list[ScoredCard]: ...


class Librarian:
    """Owns one clean write protocol for both insights and program exemplars.

    An author first proposes at most one actionable card without seeing the
    bank. Only then do we retrieve same-kind neighbors from that authored
    action and ask a strict interventional-equivalence judge. ``EQUIVALENT``
    preserves the existing treatment payload and appends provenance; ``NEW``
    admits the authored candidate. There is no union prose or merge decision.
    """

    def __init__(
        self,
        *,
        author: CardAuthorAgent,
        equivalence: EquivalenceAgent,
        program_author: ProgramAuthorAgent,
        gate: CardAdmissionGate,
        store: MemoryStore,
        neighbors: NeighborSource,
        top_k: int = 5,
        task_key: str = "",
        task_description: str = "",
        task_description_summary: str = "",
        admission_judge: NoveltyAdmissionAgent | None = None,
    ) -> None:
        self._author = author
        self._equivalence = equivalence
        self._program_author = program_author
        self._gate = gate
        self._store = store
        self._neighbors = neighbors
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        self._top_k = top_k
        self._task_key = task_key
        self._task_description = task_description
        self._task_description_summary = task_description_summary
        self._admission_judge = admission_judge
        self._reviewed_programs: set[str] = set()

    async def ingest_idea(
        self,
        *,
        base_parent_code: str,
        child_id: str,
        child_code: str,
        mutation_report: str,
        parent_fitness: float | None,
        child_fitness: float,
        signed_gain: float | None,
        validity_status: ValidityStatus,
        archive_status: ArchiveStatus,
        founding_gain: ContextualGain | None = None,
    ) -> list[WriteResult]:
        """Author and route zero or one insight card."""
        response = await self._author.arun(
            base_parent_code=base_parent_code,
            child_code=child_code,
            mutation_report=mutation_report,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            signed_gain=signed_gain,
            validity_status=validity_status,
            archive_status=archive_status,
        )
        if response.decision is WriteDecision.DROP or response.card is None:
            logger.info(
                "[Memory][Librarian] idea source={} author={}",
                child_id,
                WriteDecision.DROP.value,
            )
            return []

        events = (founding_gain,) if founding_gain is not None else ()
        card = Card(
            id="",
            task_key=self._task_key,
            description=response.card.description,
            explanation_summary=response.card.explanation_summary,
            programs=(child_id,),
            task_description=self._task_description,
            task_description_summary=self._task_description_summary,
            gain_events=events,
        )
        result = await self._route(card)
        logger.info(
            "[Memory][Librarian] idea source={} author={} write={}",
            child_id,
            WriteDecision.NEW.value,
            result.outcome.value,
        )
        return [result]

    async def ingest_program(
        self,
        *,
        program_id: str,
        code: str,
        fitness: float | None,
        archive_rank: int,
        higher_is_better: bool,
        min_fitness_gap: float = 0.0,
        store_code: bool = False,
    ) -> WriteResult:
        """Author a strategy family and retain its best concrete representative."""
        if self._program_was_reviewed(program_id):
            return WriteResult(outcome=WriteOutcome.DISCARDED)
        try:
            response = await self._program_author.arun(
                code=code,
                fitness=fitness,
                archive_rank=archive_rank,
            )
        except Exception:
            # Do not cache failures: a later refresh may retry a transient LLM error.
            raise
        self._reviewed_programs.add(program_id)
        if response.decision is WriteDecision.DROP or response.card is None:
            logger.info(
                "[Memory][Librarian] program source={} author={}",
                program_id,
                WriteDecision.DROP.value,
            )
            return WriteResult(outcome=WriteOutcome.DISCARDED)

        card = Card(
            kind=CardKind.PROGRAM,
            id=f"program-{program_id}",
            task_key=self._task_key,
            program_id=program_id,
            programs=(program_id,),
            task_description=self._task_description,
            task_description_summary=self._task_description_summary,
            description=response.card.description,
            explanation_summary=response.card.explanation_summary,
            fitness=fitness,
            code=code if store_code else "",
        )
        result = await self._route(
            card,
            higher_is_better=higher_is_better,
            min_fitness_gap=min_fitness_gap,
        )
        logger.info(
            "[Memory][Librarian] program source={} author={} write={}",
            program_id,
            WriteDecision.NEW.value,
            result.outcome.value,
        )
        return result

    async def _route(
        self,
        card: Card,
        *,
        higher_is_better: bool = True,
        min_fitness_gap: float = 0.0,
    ) -> WriteResult:
        hits = self._same_task_neighbors(card)
        if not hits:
            return await self._admit_new(card)

        offered = {hit.card.id: hit.card for hit in hits if hit.card.id}
        try:
            decision = await self._equivalence.arun(
                candidate=card,
                neighbors=list(offered.values()),
            )
        except Exception as exc:
            logger.warning(
                "[Memory][Librarian] equivalence check failed ({}); treating "
                "authored candidate as NEW.",
                exc,
            )
            return await self._admit_new(card)

        if decision.decision is WriteDecision.NEW:
            return await self._admit_new(card)
        target = offered.get(decision.target_id.strip())
        if target is None:
            logger.warning(
                "[Memory][Librarian] equivalence target {!r} was not offered; "
                "treating candidate as NEW.",
                decision.target_id,
            )
            return await self._admit_new(card)

        result = self._gate.update_equivalent(
            target.id,
            card,
            higher_is_better=higher_is_better,
            min_fitness_gap=min_fitness_gap,
        )
        return result if result.landed else await self._admit_new(card)

    def _same_task_neighbors(self, card: Card) -> list[ScoredCard]:
        """Retrieve by authored semantics, then keep same-kind/same-task cards."""
        if self._top_k == 0:
            return []
        foreign = sum(
            1
            for existing in self._store.snapshot()
            if existing.kind is card.kind and existing.task_key != card.task_key
        )
        try:
            hits = self._neighbors.nearest(
                card_brief(card), self._top_k + foreign, card.kind
            )
        except Exception as exc:
            logger.warning(
                "[Memory][Librarian] authored-card retrieval failed: {}", exc
            )
            return []
        return [
            hit
            for hit in hits
            if hit.card.kind is card.kind
            and hit.card.task_key == card.task_key
            and hit.card.id != card.id
        ][: self._top_k]

    async def _admit_new(self, card: Card) -> WriteResult:
        if card.kind is CardKind.INSIGHT and self._admission_judge is not None:
            try:
                verdict = await self._admission_judge.arun(
                    description=card.description,
                    explanation_summary=card.explanation_summary,
                )
            except Exception as exc:
                logger.warning(
                    "[Memory][Librarian] novelty gate failed ({}); admitting "
                    "candidate.",
                    exc,
                )
            else:
                if not verdict.keep:
                    return self._gate.reject_novelty(card, verdict.reason)
        return self._gate.admit(card)

    def _program_was_reviewed(self, program_id: str) -> bool:
        if program_id in self._reviewed_programs:
            return True
        for card in self._store.snapshot():
            if card.kind is not CardKind.PROGRAM:
                continue
            if card.program_id == program_id or program_id in card.programs:
                self._reviewed_programs.add(program_id)
                return True
        return False
