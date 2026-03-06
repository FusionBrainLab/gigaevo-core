from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.evolution.engine.acceptor import (
    DefaultProgramEvolutionAcceptor,
    ProgramEvolutionAcceptor,
)
from gigaevo.evolution.mutation.parent_selector import (
    ParentSelector,
    RandomParentSelector,
)


class EngineConfig(BaseModel):
    """Configuration options controlling EvolutionEngine behaviour."""

    loop_interval: float = Field(default=1.0, gt=0)
    max_elites_per_generation: int = Field(default=20, gt=0)
    max_mutations_per_generation: int = Field(default=50, gt=0)
    generation_timeout: float = Field(default=4800.0, gt=0)
    metrics_collection_interval: float = Field(
        default=1.0, gt=0, description="Interval in seconds for metrics collection"
    )
    max_generations: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of generations to run (None = unlimited)",
    )
    parent_selector: ParentSelector = Field(
        default_factory=lambda: RandomParentSelector(num_parents=1)
    )
    program_acceptor: ProgramEvolutionAcceptor = Field(
        default_factory=lambda: DefaultProgramEvolutionAcceptor(),
        description="Acceptor for determining if programs should be accepted for evolution",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
