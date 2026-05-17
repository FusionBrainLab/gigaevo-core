"""Preset builders for the shipped evolution-engine YAMLs.

Two builders cover ``config/evolution/{default,steady_state}.yaml``,
returning fully-validated :class:`GenerationalEngineConfig` or
:class:`SteadyStateEngineConfig` instances with parent-selector and
acceptor wiring resolved.

Both YAMLs use the same parent selector / acceptor pair:
``AllCombinationsParentSelector`` for crossover-shaped mutation and
``StandardEvolutionAcceptor`` for the canonical accept policy. The
preset reuses those defaults across both engine variants so a future
engine YAML changing one constructor parameter only adjusts the
specific preset, not the shared scaffolding.
"""

from __future__ import annotations

from gigaevo.config.defaults import (
    DEFAULT_LOOP_INTERVAL_S,
    DEFAULT_MAX_ELITES_PER_GENERATION,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    DEFAULT_NUM_PARENTS,
)
from gigaevo.config.schemas import (
    AllCombinationsParentSelectorConfig,
    EngineConfig,
    GenerationalEngineConfig,
    StandardAcceptorConfig,
    SteadyStateEngineConfig,
)


# steady_state.yaml hardcodes max_in_flight=8 directly rather than
# routing through a ${max_in_flight} constant; pin that value here so
# build_steady_state's default matches the YAML byte-equal.
_STEADY_STATE_YAML_MAX_IN_FLIGHT: int = 8


def _default_parent_selector(
    num_parents: int = DEFAULT_NUM_PARENTS,
) -> AllCombinationsParentSelectorConfig:
    return AllCombinationsParentSelectorConfig(num_parents=num_parents)


def _default_program_acceptor() -> StandardAcceptorConfig:
    """The acceptor's ``required_behavior_keys`` are resolved at
    build_object_graph time from the algorithm subtree's union of
    behavior keys, so the schema leaves the field as None to defer
    that resolution."""
    return StandardAcceptorConfig()


def build_generational(
    *,
    loop_interval: float = DEFAULT_LOOP_INTERVAL_S,
    max_elites_per_generation: int = DEFAULT_MAX_ELITES_PER_GENERATION,
    max_mutations_per_generation: int = DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    max_generations: int | None = DEFAULT_MAX_GENERATIONS,
    num_parents: int = DEFAULT_NUM_PARENTS,
) -> GenerationalEngineConfig:
    """Step-wise generational engine matching
    ``config/evolution/default.yaml``. One mutation/evaluation
    barrier per generation; the engine waits for every program in a
    generation to complete before advancing. Use the steady-state
    variant for ~8-9x throughput when the workload tolerates
    interleaved completion."""
    return GenerationalEngineConfig(
        loop_interval=loop_interval,
        max_elites_per_generation=max_elites_per_generation,
        max_mutations_per_generation=max_mutations_per_generation,
        max_generations=max_generations,
        parent_selector=_default_parent_selector(num_parents=num_parents),
        program_acceptor=_default_program_acceptor(),
    )


def build_steady_state(
    *,
    loop_interval: float = DEFAULT_LOOP_INTERVAL_S,
    max_elites_per_generation: int = DEFAULT_MAX_ELITES_PER_GENERATION,
    max_mutations_per_generation: int = DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    max_generations: int | None = DEFAULT_MAX_GENERATIONS,
    max_in_flight: int = _STEADY_STATE_YAML_MAX_IN_FLIGHT,
    num_parents: int = DEFAULT_NUM_PARENTS,
) -> SteadyStateEngineConfig:
    """Continuous mutation/evaluation interleaving matching
    ``config/evolution/steady_state.yaml``. Programs are evaluated and
    ingested immediately as they complete — no generational barrier.
    ``max_in_flight`` bounds the in-flight DAG queue (backpressure on
    the mutation loop); the YAML pins 8, tuned for ~3-4 GPU servers
    with 4 concurrent runs."""
    return SteadyStateEngineConfig(
        loop_interval=loop_interval,
        max_elites_per_generation=max_elites_per_generation,
        max_mutations_per_generation=max_mutations_per_generation,
        max_generations=max_generations,
        max_in_flight=max_in_flight,
        parent_selector=_default_parent_selector(num_parents=num_parents),
        program_acceptor=_default_program_acceptor(),
    )


__all__: list[str] = [
    "EngineConfig",
    "build_generational",
    "build_steady_state",
]
