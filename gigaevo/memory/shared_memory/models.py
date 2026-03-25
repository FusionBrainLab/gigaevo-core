from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Strategy = Literal["exploration", "exploitation", "hybrid"]


class MemoryCardExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanations: list[str] = Field(default_factory=list)
    summary: str = ""


class MemoryCard(BaseModel):
    """Canonical memory card used by gigaevo.memory."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str = "general"
    description: str
    task_description: str = ""
    task_description_summary: str = ""
    strategy: Strategy | None = None
    last_generation: int = 0
    programs: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    evolution_statistics: dict[str, Any] = Field(default_factory=dict)
    explanation: MemoryCardExplanation = Field(default_factory=MemoryCardExplanation)
    works_with: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class LocalMemorySnapshot(BaseModel):
    """Persisted local memory state."""

    model_config = ConfigDict(extra="forbid")

    memory_cards: dict[str, MemoryCard] = Field(default_factory=dict)


__all__ = [
    "LocalMemorySnapshot",
    "MemoryCard",
    "MemoryCardExplanation",
    "Strategy",
]
