"""Memory provider abstraction for Hydra-injected memory selection.

The provider is a strategy object injected into the DAG pipeline via Hydra.
- ``NullMemoryProvider`` — no-op, returns empty selection (default: ``memory=none``)
- ``SelectorMemoryProvider`` — assembles a ``MemoryReadPipeline`` over the shared
  card bank (``memory=reader`` or ``memory=full``)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio

from loguru import logger

from gigaevo.memory.backend_factory import MemoryBackendFactory
from gigaevo.memory.core import (
    Auctioneer,
    BetaBinomialReputation,
    Budgeter,
    CardRenderer,
    CardShortlister,
    EfficacyCardRenderer,
    GamRetriever,
    LLMCardSelector,
    MemoryReadPipeline,
    MemorySelection,
    ReputationModel,
    ThompsonAuctioneer,
    TopThetaBudgeter,
)
from gigaevo.memory.shared_memory.memory_config import GamConfig
from gigaevo.programs.program import Program


class MemoryProvider(ABC):
    """Abstract memory provider injected via Hydra."""

    @abstractmethod
    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        """Select memory cards relevant to this program."""


class NullMemoryProvider(MemoryProvider):
    """No-op provider. Returns empty selection. Default when ``memory=none``."""

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        return MemorySelection(cards=[], card_ids=[])


class SelectorMemoryProvider(MemoryProvider):
    """Assembles the modular ``MemoryReadPipeline`` lazily on first use.

    The backend factory is required and Hydra-composed (``memory/backend``
    group; ``config/memory/local.yaml`` wires it). Every other stage except
    the renderer is Hydra-injectable (config/memory/<group>/; the renderer is
    constructor-injectable only) and defaults to the production stack: GAM
    retriever, LLM shortlist, Thompson auction, top-theta budget, efficacy
    renderer. Backend construction is deferred to first use to avoid heavy
    initialization at Hydra config resolution time.

    Optional ``checkpoint_dir`` overrides the backend factory's configured
    checkpoint dir at runtime (the engine pins per-run artefacts under the
    Hydra output dir).
    """

    def __init__(
        self,
        *,
        backend: MemoryBackendFactory,
        # matches config/memory/local.yaml — one card per mutation is the
        # experimental protocol the shipped configs run
        max_cards: int = 1,
        checkpoint_dir: str | None = None,
        retriever: GamRetriever | None = None,
        selector: CardShortlister | None = None,
        auctioneer: Auctioneer | None = None,
        budgeter: Budgeter | None = None,
        renderer: CardRenderer | None = None,
        reputation: ReputationModel | None = None,
    ) -> None:
        self._max_cards = max_cards
        self._checkpoint_dir = checkpoint_dir
        self._backend_factory = backend
        self._retriever = retriever
        self._selector = selector if selector is not None else LLMCardSelector()
        self._auctioneer = (
            auctioneer if auctioneer is not None else ThompsonAuctioneer()
        )
        self._budgeter = budgeter if budgeter is not None else TopThetaBudgeter()
        self._renderer = renderer if renderer is not None else EfficacyCardRenderer()
        self._reputation = (
            reputation if reputation is not None else BetaBinomialReputation()
        )
        self._pipeline: MemoryReadPipeline | None = None
        self._build_lock = asyncio.Lock()

    def _build_retriever(self) -> GamRetriever:
        retriever = self._retriever if self._retriever is not None else GamRetriever()
        if retriever.backend is not None:
            return retriever
        gam = GamConfig(
            enable_bm25=retriever.enable_bm25,
            allowed_tools=list(retriever.allowed_tools),
            top_k_by_tool=dict(retriever.top_k_by_tool),
            # A falsy pipeline_mode degrades to the working "experimental" mode,
            # not the dead "default" under which the selector returns no cards.
            pipeline_mode=retriever.pipeline_mode or "experimental",
            max_cards=self._max_cards,
            **(
                {"max_iters": retriever.max_iters}
                if retriever.max_iters is not None
                else {}
            ),
        )
        # Read-side backend never ingests; evictor/dedup are write-path
        # components plumbed through IdeaTracker into the write pipeline.
        backend = self._backend_factory.build(
            checkpoint_dir=self._checkpoint_dir,
            gam=gam,
        )
        return retriever.bind(backend)

    def _get_pipeline(self) -> MemoryReadPipeline:
        if self._pipeline is None:
            retriever = self._build_retriever()
            logger.info(
                "[Memory][Provider] Assembled MemoryReadPipeline "
                "(checkpoint_dir={}, backend={})",
                self._checkpoint_dir,
                type(retriever.backend).__module__,
            )
            self._pipeline = MemoryReadPipeline(
                retriever=retriever,
                selector=self._selector,
                auctioneer=self._auctioneer,
                budgeter=self._budgeter,
                renderer=self._renderer,
                reputation=self._reputation,
            )
        return self._pipeline

    async def _ensure_pipeline(self) -> MemoryReadPipeline:
        if self._pipeline is not None:
            return self._pipeline
        # Build off the event loop — loading the embedding model is seconds of
        # blocking work that would otherwise stall every other program-stage
        # sharing the loop; the lock collapses a concurrent first-selection
        # race down to a single build.
        async with self._build_lock:
            if self._pipeline is None:
                return await asyncio.to_thread(self._get_pipeline)
            return self._pipeline

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        pipeline = await self._ensure_pipeline()
        return await pipeline.select(
            parents=[program],
            mutation_mode="rewrite",
            task_description=task_description,
            metrics_description=metrics_description,
            max_cards=self._max_cards,
        )
