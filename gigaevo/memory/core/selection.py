from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.core.auctioneer import AuctionBid


class MemorySelection(BaseModel):
    """Result of memory card selection for mutation guidance."""

    model_config = ConfigDict(frozen=True)

    cards: list[str] = Field(
        description="Rendered mutator-facing text blocks, one per selected card."
    )
    card_ids: list[str] = Field(
        description="Bank ids of the selected cards, aligned with ``cards``."
    )
    slate: list[AuctionBid] = Field(
        default_factory=list,
        description="Per-candidate auction audit records (winners and losers).",
    )
