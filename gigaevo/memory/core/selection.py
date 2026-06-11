from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MemorySelection(BaseModel):
    """Result of memory card selection for mutation guidance."""

    model_config = ConfigDict(frozen=True)

    cards: list[str]
    card_ids: list[str]
    slate: list[dict] = Field(default_factory=list)
