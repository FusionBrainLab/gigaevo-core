"""Unit tests for the engine-side typed schemas."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    AcceptorConfig,
    AllCombinationsParentSelectorConfig,
    BusTopologyConfig,
    BusedEngineConfig,
    EngineConfig,
    GenerationalEngineConfig,
    MigrationBusConfig,
    ParentSelectorConfig,
    RandomParentSelectorConfig,
    RedisStreamTransportConfig,
    StandardAcceptorConfig,
    SteadyStateEngineConfig,
)


class TestParentSelectorConfig:
    def test_random_selector_default(self) -> None:
        cfg = RandomParentSelectorConfig()
        assert cfg.kind == "random"
        assert cfg.num_parents == 1

    def test_num_parents_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RandomParentSelectorConfig(num_parents=0)

    def test_all_combinations_selector(self) -> None:
        cfg = AllCombinationsParentSelectorConfig(num_parents=3)
        assert cfg.kind == "all_combinations"
        assert cfg.num_parents == 3

    def test_union_round_trip(self) -> None:
        ta: TypeAdapter[ParentSelectorConfig] = TypeAdapter(ParentSelectorConfig)
        for cfg in (
            RandomParentSelectorConfig(num_parents=2),
            AllCombinationsParentSelectorConfig(num_parents=4),
        ):
            parsed = ta.validate_python(cfg.model_dump())
            assert type(parsed) is type(cfg)
            assert parsed.num_parents == cfg.num_parents

    def test_random_builds_runtime(self) -> None:
        from gigaevo.evolution.mutation.parent_selector import RandomParentSelector

        sel = RandomParentSelectorConfig(num_parents=2).build()
        assert isinstance(sel, RandomParentSelector)
        assert sel.num_parents == 2

    def test_all_combinations_builds_runtime(self) -> None:
        from gigaevo.evolution.mutation.parent_selector import (
            AllCombinationsParentSelector,
        )

        sel = AllCombinationsParentSelectorConfig(num_parents=3).build()
        assert isinstance(sel, AllCombinationsParentSelector)
        assert sel.num_parents == 3


class TestStandardAcceptorConfig:
    def test_default_construct(self) -> None:
        cfg = StandardAcceptorConfig()
        assert cfg.kind == "standard"
        assert cfg.required_behavior_keys is None

    def _find_required_keys_subacceptor(self, composite):  # type: ignore[no-untyped-def]
        """Locate the RequiredBehaviorKeysAcceptor inside the composite
        and return its required_behavior_keys set."""
        from gigaevo.evolution.engine.acceptor import RequiredBehaviorKeysAcceptor

        for sub in composite.acceptors:
            if isinstance(sub, RequiredBehaviorKeysAcceptor):
                return sub.required_behavior_keys
        raise AssertionError("no RequiredBehaviorKeysAcceptor in composite")

    def test_explicit_keys_override(self) -> None:
        cfg = StandardAcceptorConfig(required_behavior_keys=["fitness", "validity"])
        acceptor = cfg.build(required_behavior_keys=["ignored"])
        keys = self._find_required_keys_subacceptor(acceptor)
        assert keys == {"fitness", "validity"}

    def test_keys_resolved_from_caller_when_unset(self) -> None:
        cfg = StandardAcceptorConfig()
        acceptor = cfg.build(required_behavior_keys=["fitness", "complexity"])
        keys = self._find_required_keys_subacceptor(acceptor)
        assert keys == {"fitness", "complexity"}

    def test_keys_stored_as_set_for_subtraction_hot_path(self) -> None:
        """RequiredBehaviorKeysAcceptor does ``self.required_behavior_keys
        - present_keys`` on every program. Storing a list would crash
        with TypeError on the first accept call."""
        cfg = StandardAcceptorConfig()
        acceptor = cfg.build(required_behavior_keys=["fitness"])
        keys = self._find_required_keys_subacceptor(acceptor)
        assert isinstance(keys, set)
        assert keys - {"fitness", "extra"} == set()

    def test_custom_validity_key(self) -> None:
        from gigaevo.evolution.engine.acceptor import ValidityMetricAcceptor

        cfg = StandardAcceptorConfig(validity_key="custom_valid")
        acceptor = cfg.build(required_behavior_keys=["fitness"])
        validity_subacceptors = [
            s for s in acceptor.acceptors if isinstance(s, ValidityMetricAcceptor)
        ]
        assert len(validity_subacceptors) == 1
        assert validity_subacceptors[0].validity_key == "custom_valid"

    def test_empty_validity_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StandardAcceptorConfig(validity_key="")

    def test_union_round_trip(self) -> None:
        ta: TypeAdapter[AcceptorConfig] = TypeAdapter(AcceptorConfig)
        parsed = ta.validate_python({"kind": "standard"})
        assert isinstance(parsed, StandardAcceptorConfig)


class TestEngineConfig:
    def test_generational_defaults(self) -> None:
        cfg = GenerationalEngineConfig()
        assert cfg.kind == "generational"
        assert cfg.loop_interval == 1.0
        assert cfg.max_generations is None

    def test_steady_state_carries_max_in_flight(self) -> None:
        cfg = SteadyStateEngineConfig(max_in_flight=8)
        assert cfg.kind == "steady_state"
        assert cfg.max_in_flight == 8

    def test_loop_interval_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SteadyStateEngineConfig(loop_interval=0.0)

    def test_max_in_flight_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SteadyStateEngineConfig(max_in_flight=0)

    def test_max_generations_optional(self) -> None:
        cfg = GenerationalEngineConfig(max_generations=200)
        assert cfg.max_generations == 200
        with pytest.raises(ValidationError):
            GenerationalEngineConfig(max_generations=0)

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SteadyStateEngineConfig(max_in_flit=8)  # type: ignore[call-arg]

    def test_build_runtime_generational(self) -> None:
        from gigaevo.evolution.engine.config import (
            EngineConfig as RuntimeEngineConfig,
        )

        runtime = GenerationalEngineConfig().build_runtime_config(
            required_behavior_keys=["fitness"]
        )
        assert isinstance(runtime, RuntimeEngineConfig)
        assert runtime.loop_interval == 1.0

    def test_build_runtime_steady_state(self) -> None:
        from gigaevo.evolution.engine.config import (
            SteadyStateEngineConfig as RuntimeSS,
        )

        runtime = SteadyStateEngineConfig(max_in_flight=4).build_runtime_config(
            required_behavior_keys=["fitness"]
        )
        assert isinstance(runtime, RuntimeSS)
        assert runtime.max_in_flight == 4

    def test_engine_union_round_trip(self) -> None:
        ta: TypeAdapter[EngineConfig] = TypeAdapter(EngineConfig)
        for cfg in (
            GenerationalEngineConfig(max_elites_per_generation=10),
            SteadyStateEngineConfig(max_in_flight=12),
        ):
            parsed = ta.validate_python(cfg.model_dump())
            assert type(parsed) is type(cfg)

    def test_engine_union_json_round_trip(self) -> None:
        ta: TypeAdapter[EngineConfig] = TypeAdapter(EngineConfig)
        cfg = SteadyStateEngineConfig(
            max_in_flight=8,
            parent_selector=AllCombinationsParentSelectorConfig(num_parents=2),
        )
        as_json = ta.dump_json(cfg)
        parsed = ta.validate_json(as_json)
        assert isinstance(parsed, SteadyStateEngineConfig)
        assert parsed.max_in_flight == 8
        assert isinstance(parsed.parent_selector, AllCombinationsParentSelectorConfig)

    def test_engine_union_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[EngineConfig] = TypeAdapter(EngineConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "no_such_engine"})


def _bus_config(run_id: str = "exp@db0") -> MigrationBusConfig:
    return MigrationBusConfig(
        run_id=run_id,
        transport=RedisStreamTransportConfig(
            run_id=run_id,
            stream_key=f"gigaevo:{run_id}:bus",
        ),
        topology=BusTopologyConfig(),
    )


class TestBusedEngineConfig:
    def test_construct_default(self) -> None:
        cfg = BusedEngineConfig(migration_bus=_bus_config())
        assert cfg.kind == "bus"
        assert cfg.max_imports_per_generation == 10

    def test_max_imports_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            BusedEngineConfig(
                migration_bus=_bus_config(), max_imports_per_generation=0
            )

    def test_migration_bus_required(self) -> None:
        with pytest.raises(ValidationError):
            BusedEngineConfig()  # type: ignore[call-arg]

    def test_round_trip_through_engine_union(self) -> None:
        ta: TypeAdapter[EngineConfig] = TypeAdapter(EngineConfig)
        cfg = BusedEngineConfig(
            migration_bus=_bus_config("exp@db0"),
            max_imports_per_generation=25,
        )
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, BusedEngineConfig)
        assert parsed.max_imports_per_generation == 25

    def test_build_runtime_config_returns_engine_config(self) -> None:
        from gigaevo.evolution.engine.config import (
            EngineConfig as RuntimeEngineConfig,
        )

        cfg = BusedEngineConfig(migration_bus=_bus_config())
        runtime = cfg.build_runtime_config(required_behavior_keys=["fitness"])
        # BusedEngineConfig.build_runtime_config returns the base
        # EngineConfig; the BusedEvolutionEngine itself is constructed
        # by build_object_graph wrapping this config + the migration_node.
        assert isinstance(runtime, RuntimeEngineConfig)

    def test_each_variant_has_unique_kind(self) -> None:
        kinds = [
            GenerationalEngineConfig.model_fields["kind"].default,
            SteadyStateEngineConfig.model_fields["kind"].default,
            BusedEngineConfig.model_fields["kind"].default,
        ]
        assert len(set(kinds)) == len(kinds)
        assert set(kinds) == {"generational", "steady_state", "bus"}
