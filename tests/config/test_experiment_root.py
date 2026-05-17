"""Unit tests for the ExperimentConfig root and its cross-field validators."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gigaevo.config.schemas import (
    BanditRouterConfig,
    ChatOpenAIConfig,
    DataPlaneSettings,
    DefaultPipelineBuilderConfig,
    EnsembleRouterConfig,
    ExperimentConfig,
    FitnessArchiveRemoverConfig,
    FitnessProportionalEliteSelectorConfig,
    GenerationalEngineConfig,
    IslandConfig,
    MultiIslandConfig,
    PipelineConfig,
    ProblemConfig,
    RedisConfig,
    SingleIslandConfig,
    SteadyStateEngineConfig,
    SumArchiveSelectorConfig,
    TopFitnessMigrantSelectorConfig,
    BehaviorSpaceConfig,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _bspace() -> BehaviorSpaceConfig:
    return BehaviorSpaceConfig(
        keys=["fitness"],
        bounds=[(0.0, 1.0)],
        resolutions=[100],
        binning_types=["linear"],
    )


def _island(island_id: str = "main", max_size: int | None = None) -> IslandConfig:
    return IslandConfig(
        island_id=island_id,
        max_size=max_size,
        behavior_space=_bspace(),
        archive_selector=SumArchiveSelectorConfig(
            fitness_keys=["fitness"],
            fitness_key_higher_is_better=[True],
        ),
        elite_selector=FitnessProportionalEliteSelectorConfig(fitness_key="fitness"),
        archive_remover=(
            FitnessArchiveRemoverConfig(fitness_key="fitness")
            if max_size is not None
            else None
        ),
        migrant_selector=TopFitnessMigrantSelectorConfig(fitness_key="fitness"),
    )


def _kwargs(name: str = "hotpot_test", **overrides) -> dict:  # type: ignore[no-untyped-def]
    redis = RedisConfig()
    defaults = {
        "name": name,
        "redis": redis,
        "dataplane": DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{name}"),
        "problem": ProblemConfig(name=name, problem_dir=Path("/srv/gigaevo/problems/x")),
        "algorithm": SingleIslandConfig(island=_island()),
        "engine": SteadyStateEngineConfig(
            max_in_flight=5, max_mutations_per_generation=50
        ),
        "pipeline": PipelineConfig(builder=DefaultPipelineBuilderConfig()),
        "llm": EnsembleRouterConfig(models=[ChatOpenAIConfig(model="gpt-4o-mini")]),
    }
    defaults.update(overrides)
    return defaults


class TestExperimentRoot:
    def test_minimal_root_constructs(self) -> None:
        cfg = ExperimentConfig(**_kwargs())
        assert cfg.name == "hotpot_test"
        assert cfg.seed == 42

    def test_name_pattern_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(**_kwargs(name="bad name!"))

    def test_seed_override(self) -> None:
        cfg = ExperimentConfig(**_kwargs(seed=7))
        assert cfg.seed == 7

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(**_kwargs(extra_field="oops"))

    def test_frozen(self) -> None:
        cfg = ExperimentConfig(**_kwargs())
        with pytest.raises(ValidationError):
            cfg.seed = 99  # type: ignore[misc]


class TestKeyPrefixConvention:
    def test_correct_prefix_accepted(self) -> None:
        ExperimentConfig(**_kwargs())  # default key_prefix matches

    def test_wrong_prefix_rejected(self) -> None:
        redis = RedisConfig()
        bad = DataPlaneSettings(redis=redis, key_prefix="wrong:prefix")
        with pytest.raises(ValidationError, match="key_prefix must equal"):
            ExperimentConfig(**_kwargs(dataplane=bad))

    def test_expected_prefix_property(self) -> None:
        cfg = ExperimentConfig(**_kwargs(name="foo_bar"))
        assert cfg.expected_key_prefix == "gigaevo:foo_bar"


class TestMultiIslandMaxSize:
    def test_cap_with_remover_passes(self) -> None:
        algo = MultiIslandConfig(
            islands=[_island("a", max_size=100), _island("b", max_size=100)]
        )
        ExperimentConfig(**_kwargs(algorithm=algo))

    def test_cap_without_remover_caught_at_leaf(self) -> None:
        """The IslandConfig validator fires before the experiment root
        validator. The experiment-level check is defense-in-depth."""
        with pytest.raises(ValidationError, match="archive_remover"):
            MultiIslandConfig(
                islands=[
                    IslandConfig(
                        island_id="a",
                        max_size=100,
                        behavior_space=_bspace(),
                        archive_selector=SumArchiveSelectorConfig(
                            fitness_keys=["fitness"],
                            fitness_key_higher_is_better=[True],
                        ),
                        elite_selector=FitnessProportionalEliteSelectorConfig(
                            fitness_key="fitness"
                        ),
                        archive_remover=None,
                        migrant_selector=TopFitnessMigrantSelectorConfig(
                            fitness_key="fitness"
                        ),
                    ),
                    _island("b"),
                ]
            )


class TestSteadyStateInFlight:
    def test_in_flight_within_budget_passes(self) -> None:
        engine = SteadyStateEngineConfig(
            max_in_flight=5, max_mutations_per_generation=50
        )
        ExperimentConfig(**_kwargs(engine=engine))

    def test_in_flight_exceeds_budget_rejected(self) -> None:
        engine = SteadyStateEngineConfig(
            max_in_flight=100, max_mutations_per_generation=10
        )
        with pytest.raises(ValidationError, match="max_in_flight"):
            ExperimentConfig(**_kwargs(engine=engine))

    def test_generational_engine_skips_in_flight_check(self) -> None:
        engine = GenerationalEngineConfig(max_elites_per_generation=10)
        ExperimentConfig(**_kwargs(engine=engine))


class TestExperimentId:
    def test_id_is_deterministic_for_same_config(self) -> None:
        cfg1 = ExperimentConfig(**_kwargs(name="repro_test"))
        cfg2 = ExperimentConfig(**_kwargs(name="repro_test"))
        assert cfg1.experiment_id == cfg2.experiment_id
        assert len(cfg1.experiment_id) == 12

    def test_id_differs_for_different_seeds(self) -> None:
        cfg1 = ExperimentConfig(**_kwargs(seed=1))
        cfg2 = ExperimentConfig(**_kwargs(seed=2))
        assert cfg1.experiment_id != cfg2.experiment_id

    def test_id_differs_for_different_names(self) -> None:
        cfg1 = ExperimentConfig(**_kwargs(name="alpha"))
        cfg2 = ExperimentConfig(**_kwargs(name="beta"))
        assert cfg1.experiment_id != cfg2.experiment_id


class TestBusInvariant:
    """Bus engine variants ship in hydra-2.5; the validator runs as a
    no-op against the currently-shipped engine kinds. Documenting the
    behavior here so the next variant landing trips the test on
    forgotten test updates."""

    def test_no_bus_kind_today(self) -> None:
        cfg = ExperimentConfig(**_kwargs())
        assert getattr(cfg.engine, "kind", None) in ("steady_state", "generational")

    def test_validator_dormant_against_current_kinds(self) -> None:
        bandit_llm = BanditRouterConfig(models=[ChatOpenAIConfig(model="m")])
        cfg = ExperimentConfig(**_kwargs(llm=bandit_llm))
        # The validator only fires when engine.kind == "bus" which is not
        # in the current EngineConfig union — passes by no-op.
        assert cfg.llm.kind == "bandit"


class TestJSONRoundTrip:
    def test_full_root_json_round_trip(self) -> None:
        cfg = ExperimentConfig(**_kwargs(name="round_trip"))
        as_json = cfg.model_dump_json()
        parsed = ExperimentConfig.model_validate_json(as_json)
        assert parsed.name == "round_trip"
        assert parsed.experiment_id == cfg.experiment_id

    def test_cross_field_validators_fire_on_json_load(self) -> None:
        """A maliciously-edited config.json with a broken key_prefix
        must still trip the cross-field validator at load time, not
        only at Python construction."""
        cfg = ExperimentConfig(**_kwargs(name="json_validator"))
        as_dict = cfg.model_dump()
        as_dict["dataplane"]["key_prefix"] = "tampered:prefix"
        with pytest.raises(ValidationError, match="key_prefix must equal"):
            ExperimentConfig.model_validate(as_dict)


class TestOutputDir:
    def test_default_is_outputs(self) -> None:
        cfg = ExperimentConfig(**_kwargs())
        assert cfg.output_dir == Path("outputs")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="output_dir"):
            ExperimentConfig(**_kwargs(output_dir=Path("")))

    def test_cwd_dot_rejected(self) -> None:
        with pytest.raises(ValidationError, match="output_dir"):
            ExperimentConfig(**_kwargs(output_dir=Path(".")))

    def test_custom_path_accepted(self) -> None:
        cfg = ExperimentConfig(**_kwargs(output_dir=Path("/srv/gigaevo/out")))
        assert cfg.output_dir == Path("/srv/gigaevo/out")
