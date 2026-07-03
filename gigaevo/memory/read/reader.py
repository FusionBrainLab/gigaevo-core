"""MemoryReader — the read-system facade.

Shortlist (agentic research over the store) → reputation → auction → budget →
render, every stage swappable behind a small Protocol. Fails to an empty
selection on every error path so a memory outage can never sink a mutation.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from loguru import logger
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.events import (
    MemoryReadSelection,
    emit_memory_event,
    memory_event_context,
    new_decision_id,
)
from gigaevo.memory.read.auction import AuctionBid, AuctionCandidate
from gigaevo.memory.storage.base import ResearchResult

_MILLISECONDS_PER_SECOND = 1000.0
_TIMING_DECIMALS = 3


@runtime_checkable
class Shortlister(Protocol):
    """Turns the mutation context into researched candidate cards."""

    async def shortlist(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str] = frozenset(),
        parent_contexts: list[str] | None = None,
    ) -> ResearchResult: ...


@runtime_checkable
class ReputationModel(Protocol):
    """Owns all per-card efficacy statistics derived from injection outcomes."""

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None: ...

    def card_posterior(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, float]: ...

    def card_magnitude(
        self, card: Card, context: DecisionContext | None = None
    ) -> float | None: ...


@runtime_checkable
class Auctioneer(Protocol):
    """Decides which candidate cards are injected into a mutation prompt."""

    def run(
        self, candidates: list[AuctionCandidate], rng: Any
    ) -> tuple[list[str], list[AuctionBid]]: ...


@runtime_checkable
class Budgeter(Protocol):
    """Caps the auction's emergent winner set to the mutator-facing budget."""

    def cap(
        self, card_ids: list[str], slate: list[AuctionBid], max_cards: int
    ) -> list[str]: ...


@runtime_checkable
class CardRenderer(Protocol):
    """Renders one card into its mutator-facing text block from its resolved
    ``card_stats`` block (the same authority the auction bid on)."""

    def render(self, card: Card | None, block: CardStatsBlock | None = None) -> str: ...


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
        max_cards: int = 1,
        rng: Any = None,
    ) -> None:
        self._shortlister = shortlister
        self._reputation = reputation
        self._auctioneer = auctioneer
        self._budgeter = budgeter
        self._renderer = renderer
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
        candidates = {card.id: card for card in result.cards}

        started_reputation = perf_counter()
        # Query cell = parents[0]: no base parent is chosen until the mutator
        # runs, so the read anchors on the primary parent's live cell. Writes
        # later anchor each gain_event on the mutator-named base; both map
        # through the same behavior_space tessellation, so in-cell matching
        # stays consistent.
        decision_context = (
            DecisionContext(
                parent_metrics=dict(getattr(parents[0], "metrics", None) or {})
            )
            if parents
            else None
        )
        auction_input = [
            AuctionCandidate(
                card_id=card.id,
                posterior_a=posterior_a,
                posterior_b=posterior_b,
                magnitude=self._reputation.card_magnitude(card, decision_context),
            )
            for card in candidates.values()
            for posterior_a, posterior_b in (
                self._reputation.card_posterior(card, decision_context),
            )
        ]
        reputation_ms = _elapsed_ms(started_reputation)

        started_auction = perf_counter()
        auction_winner_ids, slate = self._auctioneer.run(auction_input, self._rng)
        auction_ms = _elapsed_ms(started_auction)
        started_budget = perf_counter()
        budgeted_ids = self._budgeter.cap(auction_winner_ids, slate, self._max_cards)
        budget_ms = _elapsed_ms(started_budget)
        started_render = perf_counter()
        rendered = [
            (cid, text)
            for cid in budgeted_ids
            if (
                text := self._renderer.render(
                    candidates[cid],
                    self._reputation.card_stats(candidates[cid], decision_context),
                )
            )
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
