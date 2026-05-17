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
    BusedEngineConfig,
    BusTopologyConfig,
    EngineConfig,
    GenerationalEngineConfig,
    MigrationBusConfig,
    RedisStreamTransportConfig,
    RingTopologyConfig,
    StandardAcceptorConfig,
    SteadyStateEngineConfig,
    TopologyConfig,
)


# steady_state.yaml hardcodes max_in_flight=8 directly rather than
# routing through a ${max_in_flight} constant; pin that value here so
# build_steady_state's default matches the YAML byte-equal.
_STEADY_STATE_YAML_MAX_IN_FLIGHT: int = 8

# migration_bus/{bus,ring}.yaml literals — the YAML hardcodes these
# rather than reading from constants.
_BUS_YAML_MAX_IMPORTS_PER_GENERATION: int = 10
_BUS_YAML_MIGRATION_BUS_DB: int = 15
_BUS_YAML_MAX_BUFFER_SIZE: int = 30
_BUS_YAML_CONSUME_INTERVAL_S: float = 3.0
_BUS_YAML_MAX_CONSUME_PER_POLL: int = 20
_BUS_YAML_MAX_STREAM_LEN: int = 1000
_BUS_YAML_CLAIM_TTL_S: int = 120


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


def _bus_migration_config(
    *,
    run_id: str,
    stream_key: str,
    host: str,
    port: int,
    bus_db: int,
    topology: TopologyConfig,
    max_buffer_size: int,
    consume_interval: float,
    max_consume_per_poll: int,
    max_stream_len: int,
    claim_ttl: int,
) -> MigrationBusConfig:
    """Build the MigrationBusConfig that backs the bus engine. The
    transport's run_id must match the bus run_id (enforced by the
    cross-field validator on MigrationBusConfig); both default to the
    same caller-supplied value."""
    return MigrationBusConfig(
        run_id=run_id,
        transport=RedisStreamTransportConfig(
            run_id=run_id,
            stream_key=stream_key,
            host=host,
            port=port,
            db=bus_db,
            max_stream_len=max_stream_len,
            claim_ttl=claim_ttl,
        ),
        topology=topology,
        max_buffer_size=max_buffer_size,
        consume_interval=consume_interval,
        max_consume_per_poll=max_consume_per_poll,
    )


def build_bus_engine(
    *,
    run_id: str,
    problem_name: str,
    host: str = "localhost",
    port: int = 6379,
    bus_db: int = _BUS_YAML_MIGRATION_BUS_DB,
    max_imports_per_generation: int = _BUS_YAML_MAX_IMPORTS_PER_GENERATION,
    max_buffer_size: int = _BUS_YAML_MAX_BUFFER_SIZE,
    consume_interval: float = _BUS_YAML_CONSUME_INTERVAL_S,
    max_consume_per_poll: int = _BUS_YAML_MAX_CONSUME_PER_POLL,
    max_stream_len: int = _BUS_YAML_MAX_STREAM_LEN,
    claim_ttl: int = _BUS_YAML_CLAIM_TTL_S,
    loop_interval: float = DEFAULT_LOOP_INTERVAL_S,
    max_elites_per_generation: int = DEFAULT_MAX_ELITES_PER_GENERATION,
    max_mutations_per_generation: int = DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    max_generations: int | None = DEFAULT_MAX_GENERATIONS,
    num_parents: int = DEFAULT_NUM_PARENTS,
) -> BusedEngineConfig:
    """Fully-connected migration bus matching
    ``config/migration_bus/bus.yaml``.

    Each run publishes rejected-but-valid programs to a shared Redis
    Stream; any other run can claim them exclusively via SETNX. The
    ``BusTopology`` accepts envelopes from any run except the local
    one. ``run_id`` is the local identity (canonical form
    ``f"{problem_name}@db{redis_db}"``); ``problem_name`` shapes the
    stream key as ``f"gigaevo:{problem_name}:migration_bus"`` matching
    the YAML.

    All YAML-hardcoded literals (max_stream_len=1000, claim_ttl=120,
    max_buffer_size=30, etc.) are pinned as preset defaults; the
    cross-field validator on MigrationBusConfig enforces the run_id /
    transport.run_id agreement that the YAML expresses through shared
    ``${problem.name}@db${redis.db}`` interpolation."""
    return BusedEngineConfig(
        migration_bus=_bus_migration_config(
            run_id=run_id,
            stream_key=f"gigaevo:{problem_name}:migration_bus",
            host=host,
            port=port,
            bus_db=bus_db,
            topology=BusTopologyConfig(),
            max_buffer_size=max_buffer_size,
            consume_interval=consume_interval,
            max_consume_per_poll=max_consume_per_poll,
            max_stream_len=max_stream_len,
            claim_ttl=claim_ttl,
        ),
        max_imports_per_generation=max_imports_per_generation,
        loop_interval=loop_interval,
        max_elites_per_generation=max_elites_per_generation,
        max_mutations_per_generation=max_mutations_per_generation,
        max_generations=max_generations,
        parent_selector=_default_parent_selector(num_parents=num_parents),
        program_acceptor=_default_program_acceptor(),
    )


def build_ring_engine(
    *,
    run_id: str,
    problem_name: str,
    ring_run_ids: list[str],
    host: str = "localhost",
    port: int = 6379,
    bus_db: int = _BUS_YAML_MIGRATION_BUS_DB,
    max_imports_per_generation: int = _BUS_YAML_MAX_IMPORTS_PER_GENERATION,
    max_buffer_size: int = _BUS_YAML_MAX_BUFFER_SIZE,
    consume_interval: float = _BUS_YAML_CONSUME_INTERVAL_S,
    max_consume_per_poll: int = _BUS_YAML_MAX_CONSUME_PER_POLL,
    max_stream_len: int = _BUS_YAML_MAX_STREAM_LEN,
    claim_ttl: int = _BUS_YAML_CLAIM_TTL_S,
    loop_interval: float = DEFAULT_LOOP_INTERVAL_S,
    max_elites_per_generation: int = DEFAULT_MAX_ELITES_PER_GENERATION,
    max_mutations_per_generation: int = DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    max_generations: int | None = DEFAULT_MAX_GENERATIONS,
    num_parents: int = DEFAULT_NUM_PARENTS,
) -> BusedEngineConfig:
    """Directed-ring migration matching
    ``config/migration_bus/ring.yaml``. ``ring_run_ids`` defines the
    ring order; each run accepts migrants only from its predecessor.

    The local ``run_id`` must appear in ``ring_run_ids`` — the
    MigrationBusConfig cross-field validator catches missing-from-ring
    misconfigurations at load time rather than letting the bus
    silently reject every envelope."""
    return BusedEngineConfig(
        migration_bus=_bus_migration_config(
            run_id=run_id,
            stream_key=f"gigaevo:{problem_name}:migration_bus",
            host=host,
            port=port,
            bus_db=bus_db,
            topology=RingTopologyConfig(run_ids=list(ring_run_ids)),
            max_buffer_size=max_buffer_size,
            consume_interval=consume_interval,
            max_consume_per_poll=max_consume_per_poll,
            max_stream_len=max_stream_len,
            claim_ttl=claim_ttl,
        ),
        max_imports_per_generation=max_imports_per_generation,
        loop_interval=loop_interval,
        max_elites_per_generation=max_elites_per_generation,
        max_mutations_per_generation=max_mutations_per_generation,
        max_generations=max_generations,
        parent_selector=_default_parent_selector(num_parents=num_parents),
        program_acceptor=_default_program_acceptor(),
    )


__all__: list[str] = [
    "EngineConfig",
    "build_bus_engine",
    "build_generational",
    "build_ring_engine",
    "build_steady_state",
]
