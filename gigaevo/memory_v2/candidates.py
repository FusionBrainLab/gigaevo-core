"""Candidate retrieval for the memory-v2 posterior policy."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Literal, Protocol, runtime_checkable

from loguru import logger

from gigaevo.memory.cards import Card
from gigaevo.memory.events import MemoryResearch, emit_memory_event
from gigaevo.memory.read.exclusion import (
    CardExcluder,
    NullExcluder,
    expand_exclude_ids,
    is_card_excluded,
)
from gigaevo.memory.read.interfaces import Shortlister
from gigaevo.memory.storage.base import MemoryStore, ResearchResult
from gigaevo.memory_v2.models import (
    RetrievalRecord,
    RetrievalSpecification,
    RetrievalStatus,
)
from gigaevo.memory_v2.rng import EventRNG
from gigaevo.programs.program import Program


@dataclass(frozen=True)
class CandidateSlate:
    """One store snapshot and its context-specific retrieved subset."""

    lineage_registry: tuple[Card, ...]
    candidates: tuple[Card, ...]
    retrieval: RetrievalRecord


@runtime_checkable
class CandidateSource(Protocol):
    @property
    def specification(self) -> RetrievalSpecification: ...

    async def prepare(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        pending_by_bank_card: Mapping[str, int],
        max_pending_per_card: int,
    ) -> ResearchResult: ...

    async def candidate_snapshot(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        pending_by_bank_card: Mapping[str, int],
        max_pending_per_card: int,
        rng_key: str,
        research: ResearchResult | None = None,
    ) -> CandidateSlate: ...


class _BankCandidateSource:
    def __init__(
        self,
        *,
        store: MemoryStore,
        excluder: CardExcluder | None,
        allow_cross_task: bool,
        allowed_kinds: Sequence[str],
    ) -> None:
        self.store = store
        self.excluder = excluder if excluder is not None else NullExcluder()
        self.allow_cross_task = allow_cross_task
        self.allowed_kinds = frozenset(str(kind) for kind in allowed_kinds)

    def _snapshot(
        self,
        program: Program,
        *,
        task_key: str,
        pending_by_bank_card: Mapping[str, int],
        max_pending_per_card: int,
    ) -> tuple[tuple[Card, ...], tuple[Card, ...], frozenset[str]]:
        snapshot = self.store.snapshot()
        registry = self._registry_from_snapshot(snapshot, task_key=task_key)
        registry_ids = {card.id for card in registry}
        excluded = self.excluder.exclude_for(program).union(
            card_id
            for card in snapshot
            if card.id not in registry_ids
            for card_id in (card.id, *card.absorbed_ids)
        )
        excluded = expand_exclude_ids(snapshot, excluded)
        eligible: list[Card] = []
        for card in registry:
            if is_card_excluded(card, excluded):
                continue
            lineage_ids = (card.id, *card.absorbed_ids)
            if (
                sum(pending_by_bank_card.get(card_id, 0) for card_id in lineage_ids)
                >= max_pending_per_card
            ):
                excluded = excluded.union(lineage_ids)
                continue
            eligible.append(card)
        return registry, tuple(eligible), excluded

    def _registry_from_snapshot(
        self, snapshot: Sequence[Card], *, task_key: str
    ) -> tuple[Card, ...]:
        cards = [
            card
            for card in snapshot
            if card.description.strip()
            and str(card.kind) in self.allowed_kinds
            and (
                self.allow_cross_task
                or not card.task_key
                or not task_key
                or card.task_key == task_key
            )
        ]
        return tuple(sorted(cards, key=lambda card: card.id))


class WholeBankCandidateSource(_BankCandidateSource):
    """Explicit control source that sends the complete eligible bank downstream."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        shortlister: Shortlister | None = None,
        excluder: CardExcluder | None = None,
        allow_cross_task: bool = True,
        allowed_kinds: Sequence[str] = ("insight", "program"),
        max_candidates: int | None = None,
        exploration_candidates: int | None = None,
        mutation_mode: str | None = None,
        research_timeout_seconds: float | None = None,
        selection_logic: str | None = None,
    ) -> None:
        # Keep the target swappable in Hydra without a second near-duplicate
        # memory preset. These agentic-only dependencies are intentionally unused.
        del (
            shortlister,
            max_candidates,
            exploration_candidates,
            mutation_mode,
            research_timeout_seconds,
            selection_logic,
        )
        super().__init__(
            store=store,
            excluder=excluder,
            allow_cross_task=allow_cross_task,
            allowed_kinds=allowed_kinds,
        )
        self._specification = RetrievalSpecification(
            name="whole_bank",
            max_candidates=0,
            exploration_candidates=0,
        )

    @property
    def specification(self) -> RetrievalSpecification:
        return self._specification

    async def prepare(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        pending_by_bank_card: Mapping[str, int],
        max_pending_per_card: int,
    ) -> ResearchResult:
        del (
            program,
            task_key,
            task_description,
            metrics_description,
            parent_context,
            pending_by_bank_card,
            max_pending_per_card,
        )
        return ResearchResult()

    async def candidate_snapshot(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        pending_by_bank_card: Mapping[str, int],
        max_pending_per_card: int,
        rng_key: str,
        research: ResearchResult | None = None,
    ) -> CandidateSlate:
        del task_description, metrics_description, parent_context, research
        registry, eligible, _ = self._snapshot(
            program,
            task_key=task_key,
            pending_by_bank_card=pending_by_bank_card,
            max_pending_per_card=max_pending_per_card,
        )
        ids = tuple(card.id for card in eligible)
        status: RetrievalStatus = "whole_bank" if ids else "empty"
        return CandidateSlate(
            lineage_registry=registry,
            candidates=eligible,
            retrieval=RetrievalRecord(
                specification=self.specification,
                status=status,
                rng_key=rng_key,
                eligible_bank_card_ids=ids,
                core_bank_card_ids=ids,
                exploration_bank_card_ids=(),
                candidate_bank_card_ids=ids,
                conditional_tail_inclusion_probability=0.0,
                random_slate_probability=1.0,
            ),
        )


class AgenticCandidateSource(_BankCandidateSource):
    """Research a relevant core, then retain uniform discovery support."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        shortlister: Shortlister,
        excluder: CardExcluder | None = None,
        allow_cross_task: bool = False,
        allowed_kinds: Sequence[str] = ("insight", "program"),
        max_candidates: int = 12,
        exploration_candidates: int = 4,
        mutation_mode: str = "rewrite",
        research_timeout_seconds: float = 240.0,
        selection_logic: Literal["legacy_fill", "core_priority"] = "legacy_fill",
    ) -> None:
        if not math.isfinite(research_timeout_seconds) or research_timeout_seconds <= 0:
            raise ValueError("research_timeout_seconds must be finite and positive")
        super().__init__(
            store=store,
            excluder=excluder,
            allow_cross_task=allow_cross_task,
            allowed_kinds=allowed_kinds,
        )
        self.shortlister = shortlister
        self.research_timeout_seconds = research_timeout_seconds
        self._specification = RetrievalSpecification(
            name=(
                "agentic_research_core_priority"
                if selection_logic == "core_priority"
                else "agentic_research"
            ),
            max_candidates=max_candidates,
            exploration_candidates=exploration_candidates,
            mutation_mode=mutation_mode,
        )

    @property
    def specification(self) -> RetrievalSpecification:
        return self._specification

    async def prepare(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        pending_by_bank_card: Mapping[str, int],
        max_pending_per_card: int,
    ) -> ResearchResult:
        _, eligible, excluded = self._snapshot(
            program,
            task_key=task_key,
            pending_by_bank_card=pending_by_bank_card,
            max_pending_per_card=max_pending_per_card,
        )
        if not eligible:
            return ResearchResult()
        return await self._research(
            program,
            task_description=task_description,
            metrics_description=metrics_description,
            parent_context=parent_context,
            exclude_ids=excluded,
        )

    async def candidate_snapshot(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        pending_by_bank_card: Mapping[str, int],
        max_pending_per_card: int,
        rng_key: str,
        research: ResearchResult | None = None,
    ) -> CandidateSlate:
        if research is None:
            research = await self.prepare(
                program,
                task_key=task_key,
                task_description=task_description,
                metrics_description=metrics_description,
                parent_context=parent_context,
                pending_by_bank_card=pending_by_bank_card,
                max_pending_per_card=max_pending_per_card,
            )
        # The writer may merge or retire cards while research awaits the LLM.
        # Freeze the actionable slate from a fresh bank view; research hits that
        # disappeared are ignored and newly eligible cards enter uniform discovery.
        registry, eligible, _ = self._snapshot(
            program,
            task_key=task_key,
            pending_by_bank_card=pending_by_bank_card,
            max_pending_per_card=max_pending_per_card,
        )
        if not eligible:
            return CandidateSlate(
                lineage_registry=registry,
                candidates=(),
                retrieval=RetrievalRecord(
                    specification=self.specification,
                    status="empty",
                    rng_key=rng_key,
                    eligible_bank_card_ids=(),
                    core_bank_card_ids=(),
                    exploration_bank_card_ids=(),
                    candidate_bank_card_ids=(),
                    conditional_tail_inclusion_probability=0.0,
                    random_slate_probability=1.0,
                    research_iterations=research.iterations,
                ),
            )
        eligible_by_id = {card.id: card for card in eligible}
        core_limit = self.specification.max_candidates - (
            self.specification.exploration_candidates
        )
        core_ids = tuple(
            list(
                dict.fromkeys(
                    card.id for card in research.cards if card.id in eligible_by_id
                )
            )[:core_limit]
        )
        remaining_ids = tuple(
            card.id for card in eligible if card.id not in set(core_ids)
        )
        draw_budget = (
            self.specification.exploration_candidates
            if self.specification.name == "agentic_research_core_priority"
            else self.specification.max_candidates - len(core_ids)
        )
        draw_count = min(len(remaining_ids), draw_budget)
        if draw_count:
            permutation = (
                EventRNG(rng_key)
                .generator("retrieval-exploration")
                .permutation(len(remaining_ids))
            )
            exploration_ids = tuple(
                remaining_ids[int(index)] for index in permutation[:draw_count]
            )
        else:
            exploration_ids = ()
        candidate_ids = core_ids + exploration_ids
        tail_probability = draw_count / len(remaining_ids) if remaining_ids else 0.0
        random_slate_probability = (
            1.0 / math.comb(len(remaining_ids), draw_count) if draw_count else 1.0
        )
        status: RetrievalStatus = "agentic" if core_ids else "uniform_fallback"
        return CandidateSlate(
            lineage_registry=registry,
            candidates=tuple(eligible_by_id[card_id] for card_id in candidate_ids),
            retrieval=RetrievalRecord(
                specification=self.specification,
                status=status,
                rng_key=rng_key,
                eligible_bank_card_ids=tuple(card.id for card in eligible),
                core_bank_card_ids=core_ids,
                exploration_bank_card_ids=exploration_ids,
                candidate_bank_card_ids=candidate_ids,
                conditional_tail_inclusion_probability=tail_probability,
                random_slate_probability=random_slate_probability,
                research_iterations=research.iterations,
            ),
        )

    async def _research(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        exclude_ids: frozenset[str],
    ) -> ResearchResult:
        started = perf_counter()
        try:
            async with asyncio.timeout(self.research_timeout_seconds):
                return await self.shortlister.shortlist(
                    parents=[program],
                    mutation_mode=self.specification.mutation_mode,
                    task_description=task_description,
                    metrics_description=metrics_description,
                    exclude_ids=exclude_ids,
                    parent_contexts=[parent_context or ""],
                )
        except TimeoutError:
            emit_memory_event(
                MemoryResearch(
                    outcome="failed",
                    exclude_count=len(exclude_ids),
                    duration_ms=(perf_counter() - started) * 1000.0,
                    error=(
                        "agentic retrieval exceeded "
                        f"{self.research_timeout_seconds:.1f}s"
                    ),
                )
            )
            logger.warning(
                "[MemoryV2][Retrieval] research exceeded {:.1f}s; using a uniform "
                "slate",
                self.research_timeout_seconds,
            )
            return ResearchResult()
        except Exception:
            logger.opt(exception=True).warning(
                "[MemoryV2][Retrieval] research failed; using a uniform slate"
            )
            return ResearchResult()
