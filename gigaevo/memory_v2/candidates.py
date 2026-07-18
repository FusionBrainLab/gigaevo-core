"""Full-bank candidate eligibility plus optional pre-treatment RAG assessment."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Protocol, runtime_checkable

from loguru import logger

from gigaevo.memory.cards import Card
from gigaevo.memory.events import MemoryResearch, emit_memory_event
from gigaevo.memory.read.exclusion import (
    CardExcluder,
    NullExcluder,
    expand_exclude_ids,
    is_card_excluded,
)
from gigaevo.memory.read.interfaces import Shortlister, policy_digest
from gigaevo.memory.storage.base import MemoryStore, ResearchFailure, ResearchResult
from gigaevo.memory_v2.models import (
    ApplicabilityRecord,
    ApplicabilitySpecification,
    ApplicabilityStatus,
    CandidateUniverseRecord,
    CandidateUniverseSpecification,
    CandidateUniverseStatus,
    canonical_digest,
)
from gigaevo.programs.program import Program


@dataclass(frozen=True)
class CandidateSlate:
    """One eligible-bank snapshot and its optional contextual RAG annotation."""

    lineage_registry: tuple[Card, ...]
    candidates: tuple[Card, ...]
    candidate_universe: CandidateUniverseRecord
    applicability: ApplicabilityRecord


@dataclass(frozen=True)
class PreparedApplicability:
    """Research result plus the exact pre-research candidate universe."""

    result: ResearchResult = field(default_factory=ResearchResult)
    assessed_bank_card_ids: tuple[str, ...] = ()


@runtime_checkable
class ApplicabilityProvider(Protocol):
    """Produces semantic applicability before the posterior action draw."""

    @property
    def specification(self) -> ApplicabilitySpecification: ...

    async def assess(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        exclude_ids: frozenset[str],
    ) -> ResearchResult: ...


class NullApplicabilityProvider:
    """Whole-bank baseline: no RAG signal and no LLM call."""

    def __init__(
        self,
        *,
        mutation_mode: str = "rewrite",
        retrieval_applicability_contrast: bool = False,
    ) -> None:
        if retrieval_applicability_contrast:
            raise ValueError(
                "null applicability cannot enable retrieval-applicability contrast"
            )
        self._specification = ApplicabilitySpecification(
            name="none",
            mutation_mode=mutation_mode,
            retrieval_applicability_contrast=retrieval_applicability_contrast,
            policy_digest=canonical_digest(
                {
                    "class": f"{type(self).__module__}.{type(self).__qualname__}",
                    "mutation_mode": mutation_mode,
                    "retrieval_applicability_contrast": (
                        retrieval_applicability_contrast
                    ),
                }
            ),
        )

    @property
    def specification(self) -> ApplicabilitySpecification:
        return self._specification

    async def assess(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
        exclude_ids: frozenset[str],
    ) -> ResearchResult:
        del (
            program,
            task_description,
            metrics_description,
            parent_context,
            exclude_ids,
        )
        return ResearchResult()


class AgenticApplicabilityProvider:
    """Runs semantic research but never controls the Bayesian candidate universe."""

    def __init__(
        self,
        *,
        shortlister: Shortlister,
        mutation_mode: str = "rewrite",
        research_timeout_seconds: float = 240.0,
        retrieval_applicability_contrast: bool = True,
    ) -> None:
        if not math.isfinite(research_timeout_seconds) or research_timeout_seconds <= 0:
            raise ValueError("research_timeout_seconds must be finite and positive")
        if not retrieval_applicability_contrast:
            raise ValueError(
                "agentic applicability requires retrieval-applicability contrast"
            )
        self.shortlister = shortlister
        self.mutation_mode = mutation_mode
        self.research_timeout_seconds = research_timeout_seconds
        self._specification = ApplicabilitySpecification(
            name="agentic_research",
            mutation_mode=mutation_mode,
            retrieval_applicability_contrast=retrieval_applicability_contrast,
            policy_digest=canonical_digest(
                {
                    "provider": f"{type(self).__module__}.{type(self).__qualname__}",
                    "shortlister": policy_digest(shortlister),
                    "mutation_mode": mutation_mode,
                    "research_timeout_seconds": research_timeout_seconds,
                    "retrieval_applicability_contrast": (
                        retrieval_applicability_contrast
                    ),
                }
            ),
        )

    @property
    def specification(self) -> ApplicabilitySpecification:
        return self._specification

    async def assess(
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
                    mutation_mode=self.mutation_mode,
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
                        "agentic applicability exceeded "
                        f"{self.research_timeout_seconds:.1f}s"
                    ),
                )
            )
            logger.warning(
                "[MemoryV2][Applicability] research exceeded {:.1f}s; "
                "continuing with a neutral RAG signal",
                self.research_timeout_seconds,
            )
            return ResearchResult(
                failure=ResearchFailure.TIMEOUT,
            )
        except Exception as exc:
            emit_memory_event(
                MemoryResearch(
                    outcome="failed",
                    exclude_count=len(exclude_ids),
                    duration_ms=(perf_counter() - started) * 1000.0,
                    error=str(exc),
                )
            )
            logger.opt(exception=True).warning(
                "[MemoryV2][Applicability] research failed; continuing with a "
                "neutral RAG signal"
            )
            return ResearchResult(
                failure=ResearchFailure.SHORTLISTER_EXCEPTION,
            )


@runtime_checkable
class CandidateSource(Protocol):
    @property
    def specification(self) -> CandidateUniverseSpecification: ...

    @property
    def applicability_specification(self) -> ApplicabilitySpecification: ...

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
    ) -> PreparedApplicability: ...

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
        research: PreparedApplicability | None = None,
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
    """Always send the eligible bank to the posterior, with optional RAG labels."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        applicability: ApplicabilityProvider | None = None,
        excluder: CardExcluder | None = None,
        allow_cross_task: bool = True,
        allowed_kinds: Sequence[str] = ("insight", "program"),
    ) -> None:
        super().__init__(
            store=store,
            excluder=excluder,
            allow_cross_task=allow_cross_task,
            allowed_kinds=allowed_kinds,
        )
        self.applicability = (
            applicability if applicability is not None else NullApplicabilityProvider()
        )
        self._specification = CandidateUniverseSpecification(
            policy_digest=canonical_digest(
                {
                    "source": f"{type(self).__module__}.{type(self).__qualname__}",
                    "excluder": policy_digest(self.excluder),
                    "allow_cross_task": allow_cross_task,
                    "allowed_kinds": tuple(sorted(self.allowed_kinds)),
                }
            )
        )

    @property
    def specification(self) -> CandidateUniverseSpecification:
        return self._specification

    @property
    def applicability_specification(self) -> ApplicabilitySpecification:
        return self.applicability.specification

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
    ) -> PreparedApplicability:
        _, eligible, excluded = self._snapshot(
            program,
            task_key=task_key,
            pending_by_bank_card=pending_by_bank_card,
            max_pending_per_card=max_pending_per_card,
        )
        if not eligible:
            return PreparedApplicability()
        return PreparedApplicability(
            result=await self.applicability.assess(
                program,
                task_description=task_description,
                metrics_description=metrics_description,
                parent_context=parent_context,
                exclude_ids=excluded,
            ),
            assessed_bank_card_ids=tuple(card.id for card in eligible),
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
        research: PreparedApplicability | None = None,
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
        # The writer may merge or retire cards while the LLM assesses
        # applicability. Refresh eligibility, retain every current candidate,
        # and mark newly arrived cards unassessed rather than negative.
        registry, eligible, _ = self._snapshot(
            program,
            task_key=task_key,
            pending_by_bank_card=pending_by_bank_card,
            max_pending_per_card=max_pending_per_card,
        )
        eligible_ids = tuple(card.id for card in eligible)
        eligible_id_set = frozenset(eligible_ids)
        assessed_ids = tuple(
            sorted(eligible_id_set.intersection(research.assessed_bank_card_ids))
        )
        applicable_ids = tuple(
            sorted(
                {
                    card.id
                    for card in research.result.cards
                    if card.id in set(assessed_ids)
                }
            )
        )
        if self.applicability_specification.name == "none":
            applicability = ApplicabilityRecord(
                specification=self.applicability_specification,
                status=ApplicabilityStatus.DISABLED,
            )
        elif research.result.failure is not None:
            applicability = ApplicabilityRecord(
                specification=self.applicability_specification,
                status=ApplicabilityStatus.FAILED,
                research_iterations=research.result.iterations,
                failure=research.result.failure,
            )
        else:
            applicability = ApplicabilityRecord(
                specification=self.applicability_specification,
                status=(
                    ApplicabilityStatus.ASSESSED
                    if applicable_ids
                    else ApplicabilityStatus.EMPTY
                ),
                assessed_bank_card_ids=assessed_ids,
                applicable_bank_card_ids=applicable_ids,
                research_iterations=research.result.iterations,
                summary=research.result.summary,
            )
        status = (
            CandidateUniverseStatus.ELIGIBLE_BANK
            if eligible_ids
            else CandidateUniverseStatus.EMPTY
        )
        return CandidateSlate(
            lineage_registry=registry,
            candidates=eligible,
            candidate_universe=CandidateUniverseRecord(
                specification=self.specification,
                status=status,
                eligible_bank_card_ids=eligible_ids,
            ),
            applicability=applicability,
        )
