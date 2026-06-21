from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger
import numpy as np

from gigaevo.memory.core.auctioneer import AuctionCandidate
from gigaevo.memory.core.events import (
    emit_memory_event,
    memory_event_context,
    new_memory_decision_id,
)
from gigaevo.memory.core.protocols import (
    Auctioneer,
    Budgeter,
    CardRenderer,
    CardRetriever,
    CardShortlister,
    ReputationModel,
)
from gigaevo.memory.core.selection import MemorySelection

_MILLISECONDS_PER_SECOND = 1000.0
_TIMING_DECIMALS = 3


def _empty() -> MemorySelection:
    return MemorySelection(cards=[], card_ids=[])


def _elapsed_ms(started: float) -> float:
    return round(
        (perf_counter() - started) * _MILLISECONDS_PER_SECOND, _TIMING_DECIMALS
    )


def _ids(items: list[Any]) -> list[str]:
    return [str(item_id) for item in items if (item_id := getattr(item, "id", ""))]


def _raw_memory_summary(raw_memory: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(raw_memory).__name__}
    if isinstance(raw_memory, dict):
        final_decision = (
            raw_memory["final_decision"] if "final_decision" in raw_memory else None
        )
        summary["keys"] = sorted(str(k) for k in raw_memory.keys())
        summary["has_final_decision"] = final_decision is not None
        if isinstance(final_decision, dict):
            top_ideas = (
                final_decision["top_ideas"] if "top_ideas" in final_decision else None
            )
            summary["final_mode"] = (
                final_decision["mode"] if "mode" in final_decision else None
            )
            summary["top_ideas_count"] = (
                len(top_ideas) if isinstance(top_ideas, list) else None
            )
    return summary


class MemoryReadPipeline:
    """Retrieve → shortlist → auction → budget → render.

    Replaces the deleted legacy ``MemorySelectorAgent.select()`` (goldens frozen
    in tests/memory/test_core_read_pipeline.py; auction draws seed-exact, but
    the mechanism render line and the post-auction ``max_cards`` cap are
    intentional deltas), with every stage swappable. Fails to an empty selection
    on every error path so a memory outage can never sink a mutation.
    """

    def __init__(
        self,
        *,
        retriever: CardRetriever | None,
        selector: CardShortlister,
        auctioneer: Auctioneer,
        budgeter: Budgeter,
        renderer: CardRenderer,
        reputation: ReputationModel,
        rng: Any = None,
        event_path: str | Path | None = None,
    ) -> None:
        self._retriever = retriever
        self._selector = selector
        self._auctioneer = auctioneer
        self._budgeter = budgeter
        self._renderer = renderer
        self._reputation = reputation
        self._rng = rng if rng is not None else np.random.default_rng()
        self._event_path = Path(event_path) if event_path is not None else None
        self._lock = asyncio.Lock()

    async def select(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int = 1,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
    ) -> MemorySelection:
        parent_ids = _ids(parents)
        decision_id = new_memory_decision_id()
        event_payload_base = {
            "mutation_mode": mutation_mode,
            "max_cards": max_cards,
            "exclude_count": len(exclude_ids),
            "exclude_ids": sorted(exclude_ids),
            "random_drop_dose": random_drop_dose,
            "task_description_chars": len(task_description),
            "metrics_description_chars": len(metrics_description),
        }
        program_id = parent_ids[0] if parent_ids else ""
        with memory_event_context(
            decision_id=decision_id,
            program_id=program_id,
            parent_ids=parent_ids,
            event_path=self._event_path,
        ):
            emit_memory_event(
                component="ReadPipeline",
                event_type="read.request",
                payload=event_payload_base,
            )
            return await self._select_with_events(
                parents=parents,
                mutation_mode=mutation_mode,
                task_description=task_description,
                metrics_description=metrics_description,
                max_cards=max_cards,
                exclude_ids=exclude_ids,
                random_drop_dose=random_drop_dose,
                event_payload_base=event_payload_base,
            )

    async def _select_with_events(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int = 1,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
        event_payload_base: dict[str, Any],
    ) -> MemorySelection:
        if max_cards <= 0:
            emit_memory_event(
                component="ReadPipeline",
                event_type="read.selection",
                payload={
                    **event_payload_base,
                    "selected_ids": [],
                    "slate": [],
                    "empty_reason": "max_cards_nonpositive",
                },
            )
            return _empty()
        if self._retriever is None:
            logger.warning("[Memory][ReadPipeline] no retriever; empty selection")
            emit_memory_event(
                component="ReadPipeline",
                event_type="read.selection",
                payload={
                    **event_payload_base,
                    "selected_ids": [],
                    "slate": [],
                    "empty_reason": "missing_retriever",
                },
            )
            return _empty()

        try:
            return await self._select(
                parents=parents,
                mutation_mode=mutation_mode,
                task_description=task_description,
                metrics_description=metrics_description,
                max_cards=max_cards,
                exclude_ids=exclude_ids,
                random_drop_dose=random_drop_dose,
                event_payload_base=event_payload_base,
            )
        except Exception as exc:
            logger.opt(exception=True).warning(
                "[Memory][ReadPipeline] selection failed; returning empty: {}", exc
            )
            emit_memory_event(
                component="ReadPipeline",
                event_type="read.selection",
                payload={
                    **event_payload_base,
                    "selected_ids": [],
                    "slate": [],
                    "empty_reason": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                level="WARNING",
            )
            return _empty()

    async def _select(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
        event_payload_base: dict[str, Any] | None = None,
    ) -> MemorySelection:
        started_total = perf_counter()
        event_payload_base = event_payload_base or {}
        retriever = self._retriever
        if retriever is None:
            return _empty()
        core_request = self._selector.build_core_request(
            parents=parents,
            mutation_mode=mutation_mode,
            task_description=task_description,
            metrics_description=metrics_description,
            max_cards=max_cards,
        )
        query = self._selector.build_query(
            parents=parents,
            mutation_mode=mutation_mode,
            task_description=task_description,
            metrics_description=metrics_description,
            max_cards=max_cards,
        )

        started_research = perf_counter()
        async with self._lock:
            result = await asyncio.to_thread(
                retriever.research,
                query,
                planning_request=core_request,
                exclude_ids=exclude_ids,
                random_drop_dose=random_drop_dose,
            )
        research_ms = _elapsed_ms(started_research)
        raw_memory_summary = _raw_memory_summary(result.raw_memory)
        emit_memory_event(
            component="ReadPipeline",
            event_type="read.retrieval",
            payload={
                **event_payload_base,
                "query_chars": len(query),
                "core_request_chars": len(core_request),
                "integrated_memory_chars": len(result.integrated_memory or ""),
                "raw_memory": raw_memory_summary,
                "duration_ms": research_ms,
            },
        )

        started_shortlist = perf_counter()
        candidate_ids = self._selector.shortlist(result.raw_memory)
        shortlist_ms = _elapsed_ms(started_shortlist)
        started_fetch = perf_counter()
        fetched = {
            cid: card
            for cid in candidate_ids
            if (card := retriever.get_card(cid)) is not None
        }
        fetch_ms = _elapsed_ms(started_fetch)
        fetched_ids = list(fetched.keys())
        missing_ids = [cid for cid in candidate_ids if cid not in fetched]
        started_reputation = perf_counter()
        auction_input: list[AuctionCandidate] = []
        for cid, card in fetched.items():
            posterior_a, posterior_b = self._reputation.card_posterior(card)
            auction_input.append(
                AuctionCandidate(
                    card_id=cid, posterior_a=posterior_a, posterior_b=posterior_b
                )
            )
        reputation_ms = _elapsed_ms(started_reputation)

        started_auction = perf_counter()
        auction_winner_ids, slate = self._auctioneer.run(auction_input, self._rng)
        auction_ms = _elapsed_ms(started_auction)
        started_budget = perf_counter()
        budgeted_ids = self._budgeter.cap(auction_winner_ids, slate, max_cards)
        budget_ms = _elapsed_ms(started_budget)
        started_render = perf_counter()
        rendered = [
            (cid, text)
            for cid in budgeted_ids
            if (text := self._renderer.render(fetched[cid]))
        ]
        render_ms = _elapsed_ms(started_render)
        rendered_id_set = {cid for cid, _ in rendered}
        render_dropped_ids = [cid for cid in budgeted_ids if cid not in rendered_id_set]
        card_ids = [cid for cid, _ in rendered]
        cards = [text for _, text in rendered]
        empty_reason = ""
        if not card_ids:
            if not candidate_ids:
                empty_reason = "shortlist_empty"
            elif not fetched:
                empty_reason = "fetch_empty"
            elif not auction_winner_ids:
                empty_reason = "auction_rejected"
            elif not rendered:
                empty_reason = "render_empty"
            else:
                empty_reason = "budget_or_render_empty"
        emit_memory_event(
            component="ReadPipeline",
            event_type="read.selection",
            payload={
                **event_payload_base,
                "query_chars": len(query),
                "core_request_chars": len(core_request),
                "raw_memory": raw_memory_summary,
                "timing_ms": {
                    "research": research_ms,
                    "shortlist": shortlist_ms,
                    "fetch": fetch_ms,
                    "reputation": reputation_ms,
                    "auction": auction_ms,
                    "budget": budget_ms,
                    "render": render_ms,
                    "total": _elapsed_ms(started_total),
                },
                "candidate_ids": candidate_ids,
                "candidate_count": len(candidate_ids),
                "fetched_ids": fetched_ids,
                "missing_ids": missing_ids,
                "auction_input_count": len(auction_input),
                "auction_winner_ids": auction_winner_ids,
                "budgeted_ids": budgeted_ids,
                "render_dropped_ids": render_dropped_ids,
                "selected_ids": card_ids,
                "selected_count": len(card_ids),
                "slate": [bid.model_dump(mode="json") for bid in slate],
                "empty_reason": empty_reason,
            },
        )

        if card_ids:
            logger.debug(
                "[Memory][ReadPipeline] Selected {}/{} card(s) after auction+budget (ids={})",
                len(card_ids),
                len(auction_input),
                card_ids,
            )
        else:
            logger.debug(
                "[Memory][ReadPipeline] No cards selected from {} candidate(s) "
                "after auction+budget",
                len(auction_input),
            )
        return MemorySelection(cards=cards, card_ids=card_ids, slate=slate)
