from __future__ import annotations

from typing import Set, Optional
from pydantic import BaseModel, Field

class EngineConfig(BaseModel):
    """Configuration options controlling EvolutionEngine behaviour."""

    loop_interval: float = Field(default=1.0, gt=0)
    max_elites_per_generation: int = Field(default=20, gt=0)
    max_mutations_per_generation: int = Field(default=50, gt=0)
    max_consecutive_errors: int = Field(default=5, gt=0)
    generation_timeout: float = Field(default=300.0, gt=0)
    log_interval: int = Field(default=1, gt=0)
    cleanup_interval: int = Field(default=100, gt=0)
    max_generations: Optional[int] = Field(default=None, gt=0, description="Maximum number of generations to run (None = unlimited)")

    required_behavior_keys: Set[str] = Field(default_factory=set)

    # Whether to log detailed validation failures (may be noisy)
    log_validation_failures: bool = Field(default=True)

    model_config = {"extra": "forbid"} 