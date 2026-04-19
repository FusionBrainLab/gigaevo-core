from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.evolution.engine.acceptor import (
    DefaultProgramEvolutionAcceptor,
    ProgramEvolutionAcceptor,
)
from gigaevo.evolution.engine.stopper import EvolutionStopper
from gigaevo.evolution.mutation.parent_selector import (
    ParentSelector,
    RandomParentSelector,
)


class EngineConfig(BaseModel):
    """Configuration options controlling EvolutionEngine behaviour."""

    loop_interval: float = Field(default=1.0, gt=0)
    max_elites_per_generation: int = Field(default=20, gt=0)
    max_mutations_per_generation: int = Field(default=50, gt=0)
    generation_timeout: float | None = Field(
        default=None,
        description="Deprecated — no longer used. Individual program timeouts are "
        "handled by dag_timeout / stage_timeout.",
    )
    metrics_collection_interval: float = Field(
        default=1.0, gt=0, description="Interval in seconds for metrics collection"
    )
    parent_selector: ParentSelector = Field(
        default_factory=lambda: RandomParentSelector(num_parents=1)
    )
    program_acceptor: ProgramEvolutionAcceptor = Field(
        default_factory=lambda: DefaultProgramEvolutionAcceptor(),
        description="Acceptor for determining if programs should be accepted for evolution",
    )
    stopper: EvolutionStopper = Field(
        default_factory=EvolutionStopper,
        description="Pluggable stopping criterion. Authoritative termination signal for the "
        "engine loop. Configured via the ``stopper`` Hydra group "
        "(``config/stopper/``). Default is a no-op stopper that never stops.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SteadyStateEngineConfig(EngineConfig):
    """Extra knobs for :class:`SteadyStateEvolutionEngine`.

    Inherits all ``EngineConfig`` fields.  Their meanings in steady-state:

    * ``max_mutations_per_generation`` — **epoch size**.  An epoch refresh is
      triggered after this many programs have been processed (ingested or
      discarded).  This is the closest analog to "generation size" — one epoch
      ≈ one generation's worth of work, but without the idle barrier.
    * ``max_elites_per_generation`` — passed to ``select_elites()`` each call.

    Termination is controlled by ``stopper`` (see ``config/stopper/``), e.g.
    ``MaxGenerationsStopper`` caps total epochs.
    """

    max_in_flight: int = Field(
        default=5,
        gt=0,
        description=(
            "Max mutant programs in the pipeline (produced but not yet "
            "ingested/discarded).  Backpressure: the mutation loop blocks "
            "when this many programs are awaiting DAG evaluation.  "
            "Optimal value depends on server count and concurrent runs: "
            "~4 concurrent per GPU server is the sweet spot (measured on "
            "Qwen3-235B).  Default 5 is tuned for 3-4 servers with 4 runs."
        ),
    )

    @property
    def epoch_trigger_count(self) -> int:
        """Epoch size = ``max_mutations_per_generation`` (reused, not a new knob)."""
        return self.max_mutations_per_generation
