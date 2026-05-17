"""Unit tests for the evolution-engine preset builders."""

from __future__ import annotations

from pydantic import TypeAdapter

from gigaevo.config.defaults import (
    DEFAULT_LOOP_INTERVAL_S,
    DEFAULT_MAX_ELITES_PER_GENERATION,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    DEFAULT_NUM_PARENTS,
)
from gigaevo.config.engine_presets import build_generational, build_steady_state
from gigaevo.config.schemas import (
    AllCombinationsParentSelectorConfig,
    EngineConfig,
    GenerationalEngineConfig,
    StandardAcceptorConfig,
    SteadyStateEngineConfig,
)


class TestBuildGenerational:
    def test_defaults_match_constants(self) -> None:
        cfg = build_generational()
        assert isinstance(cfg, GenerationalEngineConfig)
        assert cfg.kind == "generational"
        assert cfg.loop_interval == DEFAULT_LOOP_INTERVAL_S
        assert cfg.max_elites_per_generation == DEFAULT_MAX_ELITES_PER_GENERATION
        assert cfg.max_mutations_per_generation == DEFAULT_MAX_MUTATIONS_PER_GENERATION
        assert cfg.max_generations == DEFAULT_MAX_GENERATIONS

    def test_uses_all_combinations_parent_selector(self) -> None:
        cfg = build_generational()
        assert isinstance(
            cfg.parent_selector, AllCombinationsParentSelectorConfig
        )
        assert cfg.parent_selector.num_parents == DEFAULT_NUM_PARENTS

    def test_uses_standard_acceptor_with_deferred_keys(self) -> None:
        cfg = build_generational()
        assert isinstance(cfg.program_acceptor, StandardAcceptorConfig)
        assert cfg.program_acceptor.required_behavior_keys is None

    def test_override_num_parents(self) -> None:
        cfg = build_generational(num_parents=4)
        assert cfg.parent_selector.num_parents == 4  # type: ignore[union-attr]

    def test_override_max_generations(self) -> None:
        cfg = build_generational(max_generations=500)
        assert cfg.max_generations == 500


class TestBuildSteadyState:
    def test_defaults_match_yaml(self) -> None:
        cfg = build_steady_state()
        assert isinstance(cfg, SteadyStateEngineConfig)
        assert cfg.kind == "steady_state"
        # steady_state.yaml hardcodes max_in_flight=8
        assert cfg.max_in_flight == 8

    def test_uses_all_combinations_parent_selector(self) -> None:
        cfg = build_steady_state()
        assert isinstance(
            cfg.parent_selector, AllCombinationsParentSelectorConfig
        )

    def test_uses_standard_acceptor_with_deferred_keys(self) -> None:
        cfg = build_steady_state()
        assert isinstance(cfg.program_acceptor, StandardAcceptorConfig)
        assert cfg.program_acceptor.required_behavior_keys is None

    def test_override_max_in_flight(self) -> None:
        cfg = build_steady_state(max_in_flight=4)
        assert cfg.max_in_flight == 4


class TestRunsThroughEngineUnion:
    def test_generational_round_trips(self) -> None:
        ta: TypeAdapter[EngineConfig] = TypeAdapter(EngineConfig)
        cfg = build_generational()
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, GenerationalEngineConfig)

    def test_steady_state_round_trips(self) -> None:
        ta: TypeAdapter[EngineConfig] = TypeAdapter(EngineConfig)
        cfg = build_steady_state()
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, SteadyStateEngineConfig)
        assert parsed.max_in_flight == 8


class TestBuildRuntimeConfig:
    def test_generational_build_runtime(self) -> None:
        from gigaevo.evolution.engine.config import (
            EngineConfig as RuntimeEngineConfig,
        )

        cfg = build_generational()
        runtime = cfg.build_runtime_config(required_behavior_keys=["fitness"])
        assert isinstance(runtime, RuntimeEngineConfig)
        assert runtime.loop_interval == DEFAULT_LOOP_INTERVAL_S

    def test_steady_state_build_runtime_carries_max_in_flight(self) -> None:
        from gigaevo.evolution.engine.config import (
            SteadyStateEngineConfig as RuntimeSteadyState,
        )

        cfg = build_steady_state(max_in_flight=12)
        runtime = cfg.build_runtime_config(required_behavior_keys=["fitness"])
        assert isinstance(runtime, RuntimeSteadyState)
        assert runtime.max_in_flight == 12


class TestYAMLParity:
    """The two presets exist to match config/evolution/{default,
    steady_state}.yaml byte-equal on every observable field. Tests
    pin each YAML default to the corresponding preset default."""

    def test_generational_matches_default_yaml(self) -> None:
        cfg = build_generational()
        # default.yaml uses interpolations resolving to constants/llm.yaml
        # values which are picked up via DEFAULT_*.
        assert cfg.loop_interval == DEFAULT_LOOP_INTERVAL_S
        assert cfg.max_elites_per_generation == DEFAULT_MAX_ELITES_PER_GENERATION
        assert cfg.max_mutations_per_generation == DEFAULT_MAX_MUTATIONS_PER_GENERATION
        assert cfg.max_generations == DEFAULT_MAX_GENERATIONS
        # parent_selector is AllCombinationsParentSelector(num_parents=2)
        assert cfg.parent_selector.kind == "all_combinations"  # type: ignore[union-attr]
        assert cfg.parent_selector.num_parents == 2  # type: ignore[union-attr]

    def test_steady_state_matches_yaml_max_in_flight(self) -> None:
        cfg = build_steady_state()
        # steady_state.yaml hardcodes max_in_flight: 8 (not a constant)
        assert cfg.max_in_flight == 8

    def test_steady_state_inherits_default_yaml_overrides(self) -> None:
        """steady_state.yaml has `defaults: - default - _self_` so it
        inherits the parent_selector + program_acceptor wiring from
        default.yaml. The preset must therefore produce the same
        selectors as build_generational does."""
        gen = build_generational()
        ss = build_steady_state()
        assert type(ss.parent_selector) is type(gen.parent_selector)
        assert ss.parent_selector.num_parents == gen.parent_selector.num_parents  # type: ignore[union-attr]
        assert type(ss.program_acceptor) is type(gen.program_acceptor)
