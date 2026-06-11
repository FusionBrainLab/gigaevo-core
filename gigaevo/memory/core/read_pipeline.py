from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
import numpy as np

from gigaevo.memory.core.protocols import (
    Auctioneer,
    Budgeter,
    CardRenderer,
    CardRetriever,
    CardShortlister,
    ReputationModel,
)
from gigaevo.memory.core.selection import MemorySelection


def _empty() -> MemorySelection:
    return MemorySelection(cards=[], card_ids=[])


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
    ) -> None:
        self._retriever = retriever
        self._selector = selector
        self._auctioneer = auctioneer
        self._budgeter = budgeter
        self._renderer = renderer
        self._reputation = reputation
        self._rng = rng if rng is not None else np.random.default_rng()
        self._lock = asyncio.Lock()

    async def select(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int = 1,
    ) -> MemorySelection:
        if max_cards <= 0:
            return _empty()
        if self._retriever is None:
            logger.warning("[Memory][ReadPipeline] no retriever; empty selection")
            return _empty()

        try:
            return await self._select(
                parents=parents,
                mutation_mode=mutation_mode,
                task_description=task_description,
                metrics_description=metrics_description,
                max_cards=max_cards,
            )
        except Exception as exc:
            logger.opt(exception=True).warning(
                "[Memory][ReadPipeline] selection failed; returning empty: {}", exc
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
    ) -> MemorySelection:
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

        async with self._lock:
            result = await asyncio.to_thread(
                retriever.research, query, planning_request=core_request
            )

        candidate_ids = self._selector.shortlist(result.raw_memory)
        fetched = {cid: retriever.get_card(cid) for cid in candidate_ids}
        auction_input = [
            (cid, *self._reputation.card_posterior(fetched[cid]))
            for cid in candidate_ids
            if fetched[cid] is not None
        ]

        card_ids, slate = self._auctioneer.run(auction_input, self._rng)
        card_ids = self._budgeter.cap(card_ids, slate, max_cards)
        rendered = [
            (cid, text)
            for cid in card_ids
            if (text := self._renderer.render(fetched[cid]))
        ]
        card_ids = [cid for cid, _ in rendered]
        cards = [text for _, text in rendered]

        if card_ids:
            logger.debug(
                "[Memory][ReadPipeline] Auction kept {}/{} idea(s) (ids={})",
                len(card_ids),
                len(auction_input),
                card_ids,
            )
        else:
            logger.debug(
                "[Memory][ReadPipeline] Auction kept no cards from {} candidate(s)",
                len(auction_input),
            )
        return MemorySelection(cards=cards, card_ids=card_ids, slate=slate)
