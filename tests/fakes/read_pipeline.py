"""Factory for a production-shaped MemoryReadPipeline over a stub or real backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from gigaevo.memory.core import (
    BetaBinomialReputation,
    EfficacyCardRenderer,
    GamRetriever,
    LLMCardSelector,
    MemoryReadPipeline,
    ThompsonAuctioneer,
    TopThetaBudgeter,
)


def make_read_pipeline(
    backend: Any | None, *, seed: int = 20260604
) -> MemoryReadPipeline:
    return MemoryReadPipeline(
        retriever=GamRetriever(backend) if backend is not None else None,
        selector=LLMCardSelector(),
        auctioneer=ThompsonAuctioneer(),
        budgeter=TopThetaBudgeter(),
        renderer=EfficacyCardRenderer(),
        reputation=BetaBinomialReputation(),
        rng=np.random.default_rng(seed),
    )
