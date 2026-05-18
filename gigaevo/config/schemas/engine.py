from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from gigaevo.config.schemas._base import FrozenStrictModel
from gigaevo.config.schemas.migration_bus import MigrationBusConfig

if TYPE_CHECKING:
    from gigaevo.evolution.engine.acceptor import ProgramEvolutionAcceptor
    from gigaevo.evolution.engine.config import EngineConfig as RuntimeEngineConfig
    from gigaevo.evolution.mutation.parent_selector import ParentSelector


class RandomParentSelectorConfig(FrozenStrictModel):
    """Picks ``num_parents`` programs uniformly at random for mutation."""

    kind: Literal["random"] = "random"
    num_parents: int = Field(default=1, ge=1)

    def build(self) -> ParentSelector:
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        return RandomParentSelector(num_parents=self.num_parents)


class AllCombinationsParentSelectorConfig(FrozenStrictModel):
    """Yields every ``num_parents``-sized combination of the selected
    elites; used for crossover-shaped mutation operators."""

    kind: Literal["all_combinations"] = "all_combinations"
    num_parents: int = Field(default=1, ge=1)

    def build(self) -> ParentSelector:
        from gigaevo.evolution.mutation.parent_selector import (
            AllCombinationsParentSelector,
        )

        return AllCombinationsParentSelector(num_parents=self.num_parents)


ParentSelectorConfig = Annotated[
    RandomParentSelectorConfig | AllCombinationsParentSelectorConfig,
    Field(discriminator="kind"),
]


class StandardAcceptorConfig(FrozenStrictModel):
    """The shipped composite acceptor: state + metrics existence +
    validity + required behavior keys + mutation context.

    The behavior keys arrive from the algorithm subtree at build time —
    the schema declares ``required_behavior_keys`` as None to defer to
    the cross-field resolution in the experiment root. The runtime
    acceptor stores the keys as a ``set`` and performs set subtraction
    in its accept hot path; the schema accepts ``list[str]`` for
    serialisation friendliness and converts at build time.
    ``validity_key`` defaults to the canonical ``VALIDITY_KEY``
    constant in the acceptor module."""

    kind: Literal["standard"] = "standard"
    required_behavior_keys: list[str] | None = None
    validity_key: str | None = Field(default=None, min_length=1)

    def build(
        self, *, required_behavior_keys: list[str]
    ) -> ProgramEvolutionAcceptor:
        from gigaevo.evolution.engine.acceptor import (
            VALIDITY_KEY,
            StandardEvolutionAcceptor,
        )

        keys_list = (
            self.required_behavior_keys
            if self.required_behavior_keys is not None
            else required_behavior_keys
        )
        return StandardEvolutionAcceptor(
            required_behavior_keys=set(keys_list),
            validity_key=self.validity_key or VALIDITY_KEY,
        )


AcceptorConfig = Annotated[
    StandardAcceptorConfig,
    Field(discriminator="kind"),
]


class _EngineConfigBase(FrozenStrictModel):
    """Common engine knobs shared by the generational and steady-state
    variants. Mirrors :class:`gigaevo.evolution.engine.config.EngineConfig`.

    The ``max_elites_per_generation`` and ``max_mutations_per_generation``
    defaults agree with the preset constants
    :data:`gigaevo.config.defaults.DEFAULT_MAX_ELITES_PER_GENERATION` and
    :data:`gigaevo.config.defaults.DEFAULT_MAX_MUTATIONS_PER_GENERATION`
    so direct schema construction yields the same shape as
    :func:`gigaevo.config.engine_presets.build_generational`."""

    loop_interval: float = Field(default=1.0, gt=0.0)
    max_elites_per_generation: int = Field(default=5, gt=0)
    max_mutations_per_generation: int = Field(default=8, gt=0)
    metrics_collection_interval: float = Field(default=1.0, gt=0.0)
    max_generations: int | None = Field(default=None, ge=1)
    parent_selector: ParentSelectorConfig = Field(
        default_factory=lambda: RandomParentSelectorConfig(num_parents=1)
    )
    program_acceptor: AcceptorConfig = Field(
        default_factory=lambda: StandardAcceptorConfig()
    )


class GenerationalEngineConfig(_EngineConfigBase):
    """Step-wise generational engine. One mutation/evaluation barrier
    per generation; the engine waits for all programs in a generation
    to complete before advancing. Use steady-state for ~8-9x throughput."""

    kind: Literal["generational"] = "generational"

    def build_runtime_config(
        self, *, required_behavior_keys: list[str]
    ) -> RuntimeEngineConfig:
        from gigaevo.evolution.engine.config import EngineConfig as RuntimeEngineConfig

        return RuntimeEngineConfig(
            loop_interval=self.loop_interval,
            max_elites_per_generation=self.max_elites_per_generation,
            max_mutations_per_generation=self.max_mutations_per_generation,
            metrics_collection_interval=self.metrics_collection_interval,
            max_generations=self.max_generations,
            parent_selector=self.parent_selector.build(),
            program_acceptor=self.program_acceptor.build(
                required_behavior_keys=required_behavior_keys
            ),
        )


class SteadyStateEngineConfig(_EngineConfigBase):
    """Continuous mutation/evaluation interleaving — no generational
    barrier. Programs are evaluated and ingested immediately as they
    complete."""

    kind: Literal["steady_state"] = "steady_state"
    # ``max_in_flight=8`` mirrors the
    # ``gigaevo.config.engine_presets._STEADY_STATE_MAX_IN_FLIGHT``
    # preset constant tuned for ~3-4 GPU servers with 4 concurrent runs.
    max_in_flight: int = Field(default=8, gt=0)

    def build_runtime_config(
        self, *, required_behavior_keys: list[str]
    ) -> RuntimeEngineConfig:
        from gigaevo.evolution.engine.config import SteadyStateEngineConfig as Runtime

        return Runtime(
            loop_interval=self.loop_interval,
            max_elites_per_generation=self.max_elites_per_generation,
            max_mutations_per_generation=self.max_mutations_per_generation,
            metrics_collection_interval=self.metrics_collection_interval,
            max_generations=self.max_generations,
            parent_selector=self.parent_selector.build(),
            program_acceptor=self.program_acceptor.build(
                required_behavior_keys=required_behavior_keys
            ),
            max_in_flight=self.max_in_flight,
        )


class BusedEngineConfig(_EngineConfigBase):
    """Generational engine plus cross-run migration bus.

    ``BusedEvolutionEngine`` wraps the base engine with a
    :class:`MigrationNode` that publishes rejected-but-valid programs
    to a Redis Stream and drains migrants from peer runs before each
    generation step. The schema composes a :class:`MigrationBusConfig`
    with the base engine knobs plus ``max_imports_per_generation``
    which caps the per-step migrant intake.

    The cross-field validator on :class:`ExperimentConfig`
    (_bus_engine_requires_bandit_router) requires the experiment's
    llm to be a :class:`BanditRouterConfig` when this engine variant
    is selected — the bus's reward signal needs the bandit's
    bookkeeping to propagate."""

    kind: Literal["bus"] = "bus"
    migration_bus: MigrationBusConfig
    max_imports_per_generation: int = Field(default=10, ge=1)

    def build_runtime_config(
        self, *, required_behavior_keys: list[str]
    ) -> RuntimeEngineConfig:
        from gigaevo.evolution.engine.config import EngineConfig as RuntimeEngineConfig

        return RuntimeEngineConfig(
            loop_interval=self.loop_interval,
            max_elites_per_generation=self.max_elites_per_generation,
            max_mutations_per_generation=self.max_mutations_per_generation,
            metrics_collection_interval=self.metrics_collection_interval,
            max_generations=self.max_generations,
            parent_selector=self.parent_selector.build(),
            program_acceptor=self.program_acceptor.build(
                required_behavior_keys=required_behavior_keys
            ),
        )


EngineConfig = Annotated[
    GenerationalEngineConfig | SteadyStateEngineConfig | BusedEngineConfig,
    Field(discriminator="kind"),
]
