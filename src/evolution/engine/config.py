from __future__ import annotations

from typing import Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from src.evolution.mutation.parent_selector import (
    ParentSelector,
    RandomParentSelector,
)


class EngineConfig(BaseModel):
    """Configuration options controlling EvolutionEngine behaviour."""

    loop_interval: float = Field(default=1.0, gt=0)
    max_elites_per_generation: int = Field(default=20, gt=0)
    max_mutations_per_generation: int = Field(default=50, gt=0)
    max_consecutive_errors: int = Field(default=5, gt=0)
    generation_timeout: float = Field(default=600.0, gt=0)
    log_interval: int = Field(default=1, gt=0)
    cleanup_interval: int = Field(default=100, gt=0)
    max_generations: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of generations to run (None = unlimited)",
    )
    required_behavior_keys: Set[str] = Field(default_factory=set)
    parent_selector: ParentSelector = Field(
        default_factory=lambda: RandomParentSelector(num_parents=1)
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
