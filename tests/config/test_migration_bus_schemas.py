"""Unit tests for the migration-bus typed schemas."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    BusTopologyConfig,
    MigrationBusConfig,
    RedisStreamTransportConfig,
    RingTopologyConfig,
    TopologyConfig,
)


def _transport(run_id: str = "exp@db0") -> RedisStreamTransportConfig:
    return RedisStreamTransportConfig(
        run_id=run_id,
        stream_key="gigaevo:exp:migration_bus",
    )


class TestBusTopologyConfig:
    def test_construct(self) -> None:
        cfg = BusTopologyConfig()
        assert cfg.kind == "bus"

    def test_builds_runtime(self) -> None:
        from gigaevo.evolution.bus.topology import BusTopology

        assert isinstance(BusTopologyConfig().build(), BusTopology)


class TestRingTopologyConfig:
    def test_construct_with_two_ids(self) -> None:
        cfg = RingTopologyConfig(run_ids=["a@db0", "b@db1"])
        assert cfg.kind == "ring"
        assert cfg.run_ids == ["a@db0", "b@db1"]

    def test_single_run_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RingTopologyConfig(run_ids=["only_one"])

    def test_duplicate_run_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            RingTopologyConfig(run_ids=["same", "same"])

    def test_builds_runtime(self) -> None:
        from gigaevo.evolution.bus.topology import RingTopology

        ring = RingTopologyConfig(run_ids=["a", "b", "c"]).build()
        assert isinstance(ring, RingTopology)


class TestTopologyUnion:
    def test_round_trip_each_variant(self) -> None:
        ta: TypeAdapter[TopologyConfig] = TypeAdapter(TopologyConfig)
        for cfg in (
            BusTopologyConfig(),
            RingTopologyConfig(run_ids=["a", "b"]),
        ):
            parsed = ta.validate_python(cfg.model_dump())
            assert type(parsed) is type(cfg)

    def test_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[TopologyConfig] = TypeAdapter(TopologyConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "no_such_topology"})

    def test_each_variant_has_unique_kind(self) -> None:
        kinds = [
            BusTopologyConfig.model_fields["kind"].default,
            RingTopologyConfig.model_fields["kind"].default,
        ]
        assert set(kinds) == {"bus", "ring"}


class TestRedisStreamTransportConfig:
    def test_minimal(self) -> None:
        cfg = _transport()
        assert cfg.run_id == "exp@db0"
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.db == 15
        assert cfg.max_stream_len == 1000

    def test_run_id_required(self) -> None:
        with pytest.raises(ValidationError):
            RedisStreamTransportConfig(stream_key="x")  # type: ignore[call-arg]

    def test_stream_key_required(self) -> None:
        with pytest.raises(ValidationError):
            RedisStreamTransportConfig(run_id="x")  # type: ignore[call-arg]

    def test_port_range_enforced(self) -> None:
        with pytest.raises(ValidationError):
            RedisStreamTransportConfig(
                run_id="x", stream_key="s", port=70_000
            )

    def test_empty_run_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedisStreamTransportConfig(run_id="", stream_key="s")

    def test_max_stream_len_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RedisStreamTransportConfig(
                run_id="x", stream_key="s", max_stream_len=0
            )


class TestMigrationBusConfig:
    def test_bus_topology_construct(self) -> None:
        cfg = MigrationBusConfig(
            run_id="exp@db0",
            transport=_transport("exp@db0"),
            topology=BusTopologyConfig(),
        )
        assert cfg.run_id == "exp@db0"
        assert isinstance(cfg.topology, BusTopologyConfig)

    def test_ring_topology_construct(self) -> None:
        cfg = MigrationBusConfig(
            run_id="exp@db0",
            transport=_transport("exp@db0"),
            topology=RingTopologyConfig(run_ids=["exp@db0", "exp@db1"]),
        )
        assert isinstance(cfg.topology, RingTopologyConfig)

    def test_run_id_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must equal"):
            MigrationBusConfig(
                run_id="exp@db0",
                transport=_transport("exp@db1"),
                topology=BusTopologyConfig(),
            )

    def test_local_run_must_be_in_ring(self) -> None:
        with pytest.raises(ValidationError, match="must appear in the ring"):
            MigrationBusConfig(
                run_id="exp@db0",
                transport=_transport("exp@db0"),
                topology=RingTopologyConfig(run_ids=["other@db1", "third@db2"]),
            )

    def test_consume_interval_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            MigrationBusConfig(
                run_id="exp@db0",
                transport=_transport("exp@db0"),
                topology=BusTopologyConfig(),
                consume_interval=0.0,
            )

    def test_json_round_trip_preserves_ring_config(self) -> None:
        cfg = MigrationBusConfig(
            run_id="exp@db0",
            transport=_transport("exp@db0"),
            topology=RingTopologyConfig(run_ids=["exp@db0", "exp@db1", "exp@db2"]),
            max_buffer_size=100,
            consume_interval=2.5,
            max_consume_per_poll=15,
            max_imports_per_generation=20,
        )
        parsed = MigrationBusConfig.model_validate_json(cfg.model_dump_json())
        assert parsed.run_id == "exp@db0"
        assert isinstance(parsed.topology, RingTopologyConfig)
        assert parsed.topology.run_ids == ["exp@db0", "exp@db1", "exp@db2"]
        assert parsed.max_buffer_size == 100
        assert parsed.consume_interval == 2.5
        assert parsed.max_consume_per_poll == 15
        assert parsed.max_imports_per_generation == 20

    def test_build_constructs_runtime_migration_node(self) -> None:
        from gigaevo.evolution.bus.node import MigrationNode

        cfg = MigrationBusConfig(
            run_id="exp@db0",
            transport=_transport("exp@db0"),
            topology=BusTopologyConfig(),
        )
        node = cfg.build()
        assert isinstance(node, MigrationNode)
        assert node.run_id == "exp@db0"
