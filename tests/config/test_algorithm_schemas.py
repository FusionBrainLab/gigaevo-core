"""Unit tests for the algorithm-side typed schemas."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    AlgorithmConfig,
    BehaviorSpaceConfig,
    FitnessArchiveRemoverConfig,
    FitnessProportionalEliteSelectorConfig,
    IslandConfig,
    MultiIslandConfig,
    SingleIslandConfig,
    SumArchiveSelectorConfig,
    TopFitnessMigrantSelectorConfig,
    WeightedEliteSelectorConfig,
)


def _bspace_1d() -> BehaviorSpaceConfig:
    return BehaviorSpaceConfig(
        keys=["fitness"],
        bounds=[(0.0, 1.0)],
        resolutions=[150],
        binning_types=["linear"],
    )


def _island(island_id: str = "main", max_size: int | None = None) -> IslandConfig:
    return IslandConfig(
        island_id=island_id,
        max_size=max_size,
        behavior_space=_bspace_1d(),
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


class TestBehaviorSpaceConfig:
    def test_single_dim(self) -> None:
        bs = _bspace_1d()
        assert bs.primary_key == "fitness"
        assert bs.dynamic

    def test_three_dim_topology(self) -> None:
        bs = BehaviorSpaceConfig(
            keys=["dag_depth", "max_fan_in", "n_retrieval"],
            bounds=[(1, 11), (1, 9), (0, 5)],
            resolutions=[5, 5, 6],
            binning_types=["linear", "linear", "linear"],
        )
        assert bs.primary_key == "dag_depth"

    def test_lengths_must_align(self) -> None:
        with pytest.raises(ValidationError, match="length"):
            BehaviorSpaceConfig(
                keys=["a", "b"],
                bounds=[(0.0, 1.0)],
                resolutions=[10, 10],
                binning_types=["linear", "linear"],
            )

    def test_inverted_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError, match="min"):
            BehaviorSpaceConfig(
                keys=["fitness"],
                bounds=[(1.0, 0.0)],
                resolutions=[10],
                binning_types=["linear"],
            )

    def test_zero_resolution_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorSpaceConfig(
                keys=["fitness"],
                bounds=[(0.0, 1.0)],
                resolutions=[0],
                binning_types=["linear"],
            )

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorSpaceConfig(
                keys=["f"],
                bounds=[(0.0, 1.0)],
                resolutions=[10],
                binning_types=["linear"],
                ynamic=True,  # type: ignore[call-arg]
            )

    def test_build_constructs_dynamic_space(self) -> None:
        bs = _bspace_1d().build()
        from gigaevo.evolution.strategies.models import DynamicBehaviorSpace

        assert isinstance(bs, DynamicBehaviorSpace)
        assert bs.behavior_keys == ["fitness"]

    def test_build_static_when_not_dynamic(self) -> None:
        cfg = BehaviorSpaceConfig(
            keys=["f"],
            bounds=[(0.0, 1.0)],
            resolutions=[10],
            binning_types=["linear"],
            dynamic=False,
        )
        bs = cfg.build()
        from gigaevo.evolution.strategies.models import (
            BehaviorSpace,
            DynamicBehaviorSpace,
        )

        assert isinstance(bs, BehaviorSpace)
        assert not isinstance(bs, DynamicBehaviorSpace)


class TestSelectorConfigs:
    def test_sum_archive_selector_alignment(self) -> None:
        with pytest.raises(ValidationError, match="lengths must match"):
            SumArchiveSelectorConfig(
                fitness_keys=["a", "b"],
                fitness_key_higher_is_better=[True],
            )

    def test_fitness_proportional_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            FitnessProportionalEliteSelectorConfig(
                fitness_key="fitness", temperature=0.0
            )

    def test_weighted_lambda_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            WeightedEliteSelectorConfig(fitness_key="fitness", lambda_=0.0)

    def test_selector_builds_runtime_instance(self) -> None:
        sel = FitnessProportionalEliteSelectorConfig(fitness_key="f").build()
        from gigaevo.evolution.strategies.elite_selectors import (
            FitnessProportionalEliteSelector,
        )

        assert isinstance(sel, FitnessProportionalEliteSelector)
        assert sel.fitness_key == "f"


class TestIslandConfig:
    def test_minimal_island(self) -> None:
        isl = _island()
        assert isl.island_id == "main"
        assert isl.archive_remover is None

    def test_island_id_pattern_enforced(self) -> None:
        with pytest.raises(ValidationError):
            IslandConfig(
                island_id="bad island name!",
                behavior_space=_bspace_1d(),
                archive_selector=SumArchiveSelectorConfig(
                    fitness_keys=["f"], fitness_key_higher_is_better=[True]
                ),
                elite_selector=FitnessProportionalEliteSelectorConfig(fitness_key="f"),
                migrant_selector=TopFitnessMigrantSelectorConfig(fitness_key="f"),
            )

    def test_remover_required_when_max_size_set(self) -> None:
        with pytest.raises(ValidationError, match="archive_remover"):
            IslandConfig(
                island_id="main",
                max_size=100,
                behavior_space=_bspace_1d(),
                archive_selector=SumArchiveSelectorConfig(
                    fitness_keys=["f"], fitness_key_higher_is_better=[True]
                ),
                elite_selector=FitnessProportionalEliteSelectorConfig(fitness_key="f"),
                archive_remover=None,
                migrant_selector=TopFitnessMigrantSelectorConfig(fitness_key="f"),
            )

    def test_build_returns_runtime_island_config(self) -> None:
        from gigaevo.evolution.strategies.island import (
            IslandConfig as RuntimeIslandConfig,
        )

        runtime = _island(max_size=50).build()
        assert isinstance(runtime, RuntimeIslandConfig)
        assert runtime.island_id == "main"
        assert runtime.max_size == 50


class TestSingleAndMultiIsland:
    def test_single_island_construct(self) -> None:
        cfg = SingleIslandConfig(island=_island())
        assert cfg.kind == "single_island"

    def test_multi_island_requires_at_least_two(self) -> None:
        with pytest.raises(ValidationError):
            MultiIslandConfig(islands=[_island("solo")])

    def test_multi_island_unique_ids(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            MultiIslandConfig(
                islands=[_island("dup"), _island("dup")],
            )

    def test_multi_island_construct(self) -> None:
        cfg = MultiIslandConfig(islands=[_island("a"), _island("b")])
        assert cfg.kind == "multi_island"
        assert cfg.migration_interval == 50

    def test_algorithm_union_round_trip(self) -> None:
        cfg = SingleIslandConfig(island=_island())
        ta: TypeAdapter[AlgorithmConfig] = TypeAdapter(AlgorithmConfig)
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, SingleIslandConfig)
        assert parsed.island.island_id == "main"

    def test_algorithm_union_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[AlgorithmConfig] = TypeAdapter(AlgorithmConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "no_such_algo", "island": {}})
