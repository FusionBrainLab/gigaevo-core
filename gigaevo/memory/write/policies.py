"""Small immutable policies for memory authoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DedupPolicy(BaseModel):
    """Neighbor recall for authored-action equivalence checking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    online_top_k: int = Field(
        default=5,
        ge=0,
        description="Same-kind neighbors retrieved from the authored candidate "
        "before strict equivalence checking.",
    )
    across_tasks: bool = Field(
        default=False,
        description="Search the whole bank for semantic duplicates instead of "
        "only the authoring task. Evidence remains task-labelled.",
    )


class ProgramExemplarPolicy(BaseModel):
    """Bounded write policy for program exemplar cards.

    Program cards are stable semantic strategy families backed by concrete
    high-fitness representatives. The policy caps authoring and residency.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(
        default=True,
        description="Whether the writer authors and maintains program exemplars.",
    )
    top_k_per_refresh: int | None = Field(
        default=4,
        ge=0,
        description="Maximum number of top programs authored per write refresh; "
        "None keeps all programs selected by best_programs_percent.",
    )
    max_cards: int = Field(
        default=12,
        ge=0,
        description="Per-task hard cap on program exemplar cards kept in the bank.",
    )
    min_fitness_gap: float = Field(
        default=0.0,
        ge=0.0,
        description="Absolute improvement required to replace an existing "
        "equivalent family's representative. Default is strict improvement with no "
        "task-scale "
        "epsilon.",
    )
    store_code: bool = Field(
        default=False,
        description="Whether program cards retain the full source body.",
    )
