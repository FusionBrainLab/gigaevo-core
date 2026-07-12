"""MemoryReader — the read-system facade.

Shortlist (agentic research over the store) → reputation → auction → budget →
render, every stage swappable behind a small Protocol. Fails to an empty
selection on every error path so a memory outage can never sink a mutation.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from loguru import logger
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.context import GlobalMemoryContext, MemoryContextModel
from gigaevo.memory.events import (
    MemoryReadSelection,
    emit_memory_event,
    memory_event_context,
    new_decision_id,
)
from gigaevo.memory.read.auction import AuctionBid
from gigaevo.memory.read.exclusion import is_card_excluded
from gigaevo.memory.read.interfaces import (
    Auctioneer,
    Budgeter,
    CandidateProjector,
    CardRenderer,
    ProbePolicy,
    ReputationModel,
    Shortlister,
)
from gigaevo.memory.read.probe import NoColdProbePolicy
from gigaevo.memory.read.projection import AuctionCandidateProjector

_MILLISECONDS_PER_SECOND = 1000.0
_TIMING_DECIMALS = 3


class MemorySelection(BaseModel):
    """Result of memory card selection for mutation guidance."""

    model_config = ConfigDict(frozen=True)

    cards: tuple[str, ...] = Field(
        default=(),
        description="Rendered mutator-facing text blocks, one per selected card.",
    )
    card_ids: tuple[str, ...] = Field(
        default=(),
        description="Bank ids of the selected cards, aligned with ``cards``.",
    )
    slate: tuple[AuctionBid, ...] = Field(
        default=(),
        description="Per-candidate auction audit records (winners and losers).",
    )


def _elapsed_ms(started: float) -> float:
    return round(
        (perf_counter() - started) * _MILLISECONDS_PER_SECOND, _TIMING_DECIMALS
    )


def _ids(items: list[Any]) -> tuple[str, ...]:
    return tuple(str(item_id) for item in items if (item_id := getattr(item, "id", "")))


class MemoryReader:
    """Retrieve → auction → budget → render over a researched shortlist.

    The shortlister's recall width lives in the store's ``ResearchConfig``;
    ``max_cards`` here is the injection budget the budgeter caps to.
    """

    def __init__(
        self,
        *,
        shortlister: Shortlister,
        reputation: ReputationModel,
        auctioneer: Auctioneer,
        budgeter: Budgeter,
        renderer: CardRenderer,
        context_model: MemoryContextModel | None = None,
        candidate_projector: CandidateProjector | None = None,
        probe_policy: ProbePolicy | None = None,
        max_cards: int = 1,
        rng: Any = None,
    ) -> None:
        self._shortlister = shortlister
        self._reputation = reputation
        self._auctioneer = auctioneer
        self._budgeter = budgeter
        self._renderer = renderer
        self._context_model = (
            context_model if context_model is not None else GlobalMemoryContext()
        )
        self._projector = (
            candidate_projector
            if candidate_projector is not None
            else AuctionCandidateProjector(context_model=self._context_model)
        )
        self._probe_policy = (
            probe_policy if probe_policy is not None else NoColdProbePolicy()
        )
        self._max_cards = max_cards
        self._rng = rng if rng is not None else np.random.default_rng()
        self._lock = asyncio.Lock()

    async def select(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str] = frozenset(),
        parent_contexts: list[str] | None = None,
    ) -> MemorySelection:
        parent_ids = _ids(parents)
        with memory_event_context(
            decision_id=new_decision_id(),
            program_id=parent_ids[0] if parent_ids else "",
            parent_ids=parent_ids,
        ):
            base = MemoryReadSelection(
                mutation_mode=mutation_mode,
                max_cards=self._max_cards,
                exclude_ids=tuple(sorted(exclude_ids)),
            )
            if self._max_cards <= 0:
                emit_memory_event(
                    base.model_copy(update={"empty_reason": "max_cards_nonpositive"})
                )
                return MemorySelection()
            try:
                return await self._select(
                    parents=parents,
                    mutation_mode=mutation_mode,
                    task_description=task_description,
                    metrics_description=metrics_description,
                    exclude_ids=exclude_ids,
                    parent_contexts=parent_contexts,
                    base=base,
                )
            except Exception as exc:
                logger.opt(exception=True).warning(
                    "[Memory][Reader] selection failed; returning empty: {}", exc
                )
                emit_memory_event(
                    base.model_copy(
                        update={
                            "empty_reason": "exception",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                )
                return MemorySelection()

    async def _select(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str],
        parent_contexts: list[str] | None,
        base: MemoryReadSelection,
    ) -> MemorySelection:
        started_total = perf_counter()
        started_research = perf_counter()
        async with self._lock:
            result = await self._shortlister.shortlist(
                parents=parents,
                mutation_mode=mutation_mode,
                task_description=task_description,
                metrics_description=metrics_description,
                exclude_ids=exclude_ids,
                parent_contexts=parent_contexts,
            )
        research_ms = _elapsed_ms(started_research)
        candidates = {
            card.id: card
            for card in result.cards
            if not is_card_excluded(card, exclude_ids)
        }

        started_reputation = perf_counter()
        decision_context = self._context_model.read_context(parents)
        baseline = self._projector.decision_baseline(decision_context)
        blocks = {
            card.id: self._reputation.card_stats(card, decision_context)
            for card in candidates.values()
        }
        auction_input = []
        for card_id, block in blocks.items():
            card = candidates[card_id]
            auction_input.append(
                self._projector.project(
                    card=card,
                    block=block,
                    reputation=self._reputation,
                    context=decision_context,
                )
            )
        reputation_ms = _elapsed_ms(started_reputation)

        started_auction = perf_counter()
        auction_winner_ids, slate = self._auctioneer.run(
            auction_input, self._rng, baseline=baseline
        )
        auction_ms = _elapsed_ms(started_auction)
        started_budget = perf_counter()
        budgeted_ids = self._budgeter.cap(auction_winner_ids, slate, self._max_cards)
        budgeted_ids, slate = self._probe_policy.apply(
            budgeted_ids=budgeted_ids,
            slate=list(slate),
            max_cards=self._max_cards,
            rng=self._rng,
        )
        budget_ms = _elapsed_ms(started_budget)
        started_render = perf_counter()
        rendered = [
            (cid, text)
            for cid in budgeted_ids
            if (text := self._renderer.render(candidates[cid], blocks.get(cid)))
        ]
        render_ms = _elapsed_ms(started_render)
        card_ids = tuple(cid for cid, _ in rendered)
        render_dropped_ids = tuple(cid for cid in budgeted_ids if cid not in card_ids)

        empty_reason = ""
        if not card_ids:
            if not candidates:
                empty_reason = "research_empty"
            elif not auction_winner_ids:
                empty_reason = "auction_rejected"
            elif not budgeted_ids:
                empty_reason = "budget_empty"
            else:
                empty_reason = "render_empty"
        # Terminal telemetry is isolated: the selection is already computed, so a
        # failure emitting/serializing it must not discard a valid selection.
        try:
            emit_memory_event(
                base.model_copy(
                    update={
                        "research_iterations": result.iterations,
                        "candidate_ids": tuple(candidates),
                        "auction_winner_ids": tuple(auction_winner_ids),
                        "budgeted_ids": tuple(budgeted_ids),
                        "render_dropped_ids": render_dropped_ids,
                        "selected_ids": card_ids,
                        "slate": tuple(bid.model_dump(mode="json") for bid in slate),
                        "empty_reason": empty_reason,
                        "timing_ms": {
                            "research": research_ms,
                            "reputation": reputation_ms,
                            "auction": auction_ms,
                            "budget": budget_ms,
                            "render": render_ms,
                            "total": _elapsed_ms(started_total),
                        },
                    }
                )
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Reader] selection telemetry emit failed; keeping selection"
            )
        logger.debug(
            "[Memory][Reader] Selected {}/{} card(s) after auction+budget (ids={})",
            len(card_ids),
            len(auction_input),
            list(card_ids),
        )
        return MemorySelection(
            cards=tuple(text for _, text in rendered),
            card_ids=card_ids,
            slate=tuple(slate),
        )
