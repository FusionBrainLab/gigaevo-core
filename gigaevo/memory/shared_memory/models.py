from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Strategy = Literal["exploration", "exploitation", "hybrid"]


class MemoryCardExplanation(BaseModel):
    """Explanation field with history and summary."""

    model_config = ConfigDict(extra="forbid")

    explanations: list[str] = Field(default_factory=list)
    summary: str = ""


class MemoryCard(BaseModel):
    """Canonical general memory card (ideas, insights)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    category: str = "general"
    description: str = ""
    task_description: str = ""
    task_description_summary: str = ""
    strategy: str = ""
    last_generation: int = 0
    programs: list[str] = Field(default_factory=list)
    aliases: list[Any] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    evolution_statistics: dict[str, Any] = Field(default_factory=dict)
    explanation: MemoryCardExplanation = Field(default_factory=MemoryCardExplanation)
    works_with: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class ConnectedIdea(BaseModel):
    """Reference to an idea card linked to a program."""

    model_config = ConfigDict(extra="allow")

    idea_id: str = ""
    description: str = ""


class ProgramCard(BaseModel):
    """Memory card representing a top-performing program."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    category: str = "program"
    program_id: str = ""
    task_description: str = ""
    task_description_summary: str = ""
    description: str = ""
    fitness: float | None = None
    code: str = ""
    connected_ideas: list[ConnectedIdea | dict[str, Any]] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    strategy: str = ""
    links: list[str] = Field(default_factory=list)


AnyCard = MemoryCard | ProgramCard


class LocalMemorySnapshot(BaseModel):
    """Persisted local memory state."""

    model_config = ConfigDict(extra="forbid")

    memory_cards: dict[str, MemoryCard] = Field(default_factory=dict)


__all__ = [
    "AnyCard",
    "ConnectedIdea",
    "LocalMemorySnapshot",
    "MemoryCard",
    "MemoryCardExplanation",
    "ProgramCard",
    "Strategy",
]
