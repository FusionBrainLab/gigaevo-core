"""Preset builders for the evolution-engine schemas.

Two builders cover the generational and steady-state variants,
returning fully-validated :class:`GenerationalEngineConfig` or
:class:`SteadyStateEngineConfig`. Two more wrap the
:class:`BusedEngineConfig` variant for fully-connected and ring
migration topologies.

The default parent selector (``AllCombinationsParentSelectorConfig``)
and acceptor (``StandardAcceptorConfig``) are shared across the
non-bus variants; a private helper keeps that wiring in one place.
"""

from __future__ import annotations

from typing import Final

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


# Steady-state engine: in-flight DAG queue cap, tuned for ~3-4 GPU
# servers with 4 concurrent runs.
_STEADY_STATE_MAX_IN_FLIGHT: Final[int] = 8

# Migration-bus literals shared across the bus and ring topology
# variants.
_BUS_MAX_IMPORTS_PER_GENERATION: Final[int] = 10
_BUS_MIGRATION_BUS_DB: Final[int] = 15
_BUS_MAX_BUFFER_SIZE: Final[int] = 30
_BUS_CONSUME_INTERVAL_S: Final[float] = 3.0
_BUS_MAX_CONSUME_PER_POLL: Final[int] = 20
_BUS_MAX_STREAM_LEN: Final[int] = 1000
_BUS_CLAIM_TTL_S: Final[int] = 120


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
    """Step-wise generational engine. One mutation/evaluation barrier
    per generation; the engine waits for every program in a
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
    max_in_flight: int = _STEADY_STATE_MAX_IN_FLIGHT,
    num_parents: int = DEFAULT_NUM_PARENTS,
) -> SteadyStateEngineConfig:
    """Continuous mutation/evaluation interleaving. Programs are
    evaluated and ingested immediately as they complete -- no
    generational barrier. ``max_in_flight`` bounds the in-flight DAG
    queue (backpressure on the mutation loop) and defaults to 8,
    tuned for ~3-4 GPU servers with 4 concurrent runs."""
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
    bus_db: int = _BUS_MIGRATION_BUS_DB,
    max_imports_per_generation: int = _BUS_MAX_IMPORTS_PER_GENERATION,
    max_buffer_size: int = _BUS_MAX_BUFFER_SIZE,
    consume_interval: float = _BUS_CONSUME_INTERVAL_S,
    max_consume_per_poll: int = _BUS_MAX_CONSUME_PER_POLL,
    max_stream_len: int = _BUS_MAX_STREAM_LEN,
    claim_ttl: int = _BUS_CLAIM_TTL_S,
    loop_interval: float = DEFAULT_LOOP_INTERVAL_S,
    max_elites_per_generation: int = DEFAULT_MAX_ELITES_PER_GENERATION,
    max_mutations_per_generation: int = DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    max_generations: int | None = DEFAULT_MAX_GENERATIONS,
    num_parents: int = DEFAULT_NUM_PARENTS,
) -> BusedEngineConfig:
    """Fully-connected migration bus.

    Each run publishes rejected-but-valid programs to a shared Redis
    Stream; any other run can claim them exclusively via SETNX. The
    ``BusTopology`` accepts envelopes from any run except the local
    one. ``run_id`` is the local identity (canonical form
    ``f"{problem_name}@db{redis_db}"``); ``problem_name`` shapes the
    stream key as ``f"gigaevo:{problem_name}:migration_bus"``.

    The cross-field validator on ``MigrationBusConfig`` enforces that
    the local ``run_id`` agrees with ``transport.run_id``, so the
    builder threads the same value through both fields."""
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
    bus_db: int = _BUS_MIGRATION_BUS_DB,
    max_imports_per_generation: int = _BUS_MAX_IMPORTS_PER_GENERATION,
    max_buffer_size: int = _BUS_MAX_BUFFER_SIZE,
    consume_interval: float = _BUS_CONSUME_INTERVAL_S,
    max_consume_per_poll: int = _BUS_MAX_CONSUME_PER_POLL,
    max_stream_len: int = _BUS_MAX_STREAM_LEN,
    claim_ttl: int = _BUS_CLAIM_TTL_S,
    loop_interval: float = DEFAULT_LOOP_INTERVAL_S,
    max_elites_per_generation: int = DEFAULT_MAX_ELITES_PER_GENERATION,
    max_mutations_per_generation: int = DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    max_generations: int | None = DEFAULT_MAX_GENERATIONS,
    num_parents: int = DEFAULT_NUM_PARENTS,
) -> BusedEngineConfig:
    """Directed-ring migration. ``ring_run_ids`` defines the ring
    order; each run accepts migrants only from its predecessor.

    The local ``run_id`` must appear in ``ring_run_ids`` -- the
    ``MigrationBusConfig`` cross-field validator catches missing-from-
    ring misconfigurations at load time rather than letting the bus
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
