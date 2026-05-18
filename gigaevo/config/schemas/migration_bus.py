from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from gigaevo.config.schemas._base import FrozenStrictModel

if TYPE_CHECKING:
    from gigaevo.evolution.bus.node import MigrationNode
    from gigaevo.evolution.bus.topology import Topology
    from gigaevo.evolution.bus.transport import Transport


class BusTopologyConfig(FrozenStrictModel):
    """Fully-connected bus: every run accepts migrants from every
    other run (and rejects its own). The runtime ``BusTopology`` is
    parameterless — this schema is a marker for the discriminated
    union."""

    kind: Literal["bus"] = "bus"

    def build(self) -> Topology:
        from gigaevo.evolution.bus.topology import BusTopology

        return BusTopology()


class RingTopologyConfig(FrozenStrictModel):
    """Ring topology: each run accepts migrants only from its
    predecessor in ``run_ids``. The list defines the ring order and
    wraps around so the first run accepts from the last."""

    kind: Literal["ring"] = "ring"
    run_ids: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def _run_ids_unique(self) -> RingTopologyConfig:
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError(
                f"RingTopologyConfig.run_ids must be unique; got {self.run_ids}"
            )
        return self

    def build(self) -> Topology:
        from gigaevo.evolution.bus.topology import RingTopology

        return RingTopology(run_ids=list(self.run_ids))


TopologyConfig = Annotated[
    BusTopologyConfig | RingTopologyConfig,
    Field(discriminator="kind"),
]


class RedisStreamTransportConfig(FrozenStrictModel):
    """Redis Streams transport with SETNX exclusive claiming. The
    schema mirrors the runtime ``RedisStreamTransport`` constructor
    surface; ``run_id`` and ``stream_key`` are required because they
    encode the experiment identity and the cross-run namespace."""

    run_id: str = Field(min_length=1)
    stream_key: str = Field(min_length=1)
    host: str = Field(default="localhost", min_length=1)
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=15, ge=0)
    max_stream_len: int = Field(default=1000, ge=1)
    claim_ttl: int = Field(default=120, ge=1)
    block_ms: int = Field(default=5000, ge=0)

    def build(self) -> Transport:
        from gigaevo.evolution.bus.transport import RedisStreamTransport

        return RedisStreamTransport(
            run_id=self.run_id,
            stream_key=self.stream_key,
            host=self.host,
            port=self.port,
            db=self.db,
            max_stream_len=self.max_stream_len,
            claim_ttl=self.claim_ttl,
            block_ms=self.block_ms,
        )


class MigrationBusConfig(FrozenStrictModel):
    """Cross-run migration coordinator. Composes the transport, the
    topology filter, and the per-node tuning knobs.

    ``run_id`` is the local run's identity — by convention
    f"{problem.name}@db{redis.db}" so concurrent runs of the same
    experiment land in distinct namespaces. The ring topology's
    ``run_ids`` list must contain the local ``run_id`` (the runtime
    ``RingTopology`` silently rejects everything otherwise); the
    cross-field validator below catches that misconfiguration at
    load time.

    ``max_imports_per_generation`` is NOT carried here — it is a
    parameter of ``BusedEvolutionEngine`` (not ``MigrationNode``) and
    lives on :class:`BusedEngineConfig`."""

    run_id: str = Field(min_length=1)
    transport: RedisStreamTransportConfig
    topology: TopologyConfig
    max_buffer_size: int = Field(default=50, ge=1)
    consume_interval: float = Field(default=5.0, gt=0.0)
    max_consume_per_poll: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def _run_id_matches_transport(self) -> MigrationBusConfig:
        if self.transport.run_id != self.run_id:
            raise ValueError(
                f"MigrationBusConfig.run_id ({self.run_id!r}) must equal "
                f"transport.run_id ({self.transport.run_id!r}) — the bus "
                "and the transport must agree on the local identity"
            )
        return self

    @model_validator(mode="after")
    def _local_run_in_ring(self) -> MigrationBusConfig:
        topology = self.topology
        if isinstance(topology, RingTopologyConfig):
            if self.run_id not in topology.run_ids:
                raise ValueError(
                    f"MigrationBusConfig.run_id {self.run_id!r} must appear "
                    f"in the ring topology run_ids {topology.run_ids}"
                )
        return self

    def build(self) -> MigrationNode:
        from gigaevo.evolution.bus.node import MigrationNode

        return MigrationNode(
            run_id=self.run_id,
            transport=self.transport.build(),
            topology=self.topology.build(),
            max_buffer_size=self.max_buffer_size,
            consume_interval=self.consume_interval,
            max_consume_per_poll=self.max_consume_per_poll,
        )
