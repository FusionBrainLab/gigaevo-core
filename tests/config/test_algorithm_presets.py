"""Unit tests for the algorithm preset builders."""

from __future__ import annotations

import pytest

from gigaevo.config.algorithm_presets import (
    build_multi_island_fitness_complexity,
    build_single_island,
    build_single_island_2d,
    build_single_island_fitness_prop_fixed_temp,
    build_single_island_weighted,
    build_topology_3d,
    build_topology_3d_7step,
    build_topology_3d_ret,
)
from gigaevo.config.schemas import (
    FitnessProportionalEliteSelectorConfig,
    MultiIslandConfig,
    SingleIslandConfig,
    WeightedEliteSelectorConfig,
)


class TestSingleIsland:
    def test_default_shape(self) -> None:
        cfg = build_single_island()
        assert isinstance(cfg, SingleIslandConfig)
        island = cfg.island
        assert island.island_id == "fitness_island"
        assert island.max_size == 75
        # Behavior space is 1-D on "fitness"
        assert island.behavior_space.keys == ["fitness"]
        assert island.behavior_space.resolutions == [150]
        # Elite selector is FitnessProp with auto-temperature
        assert isinstance(
            island.elite_selector, FitnessProportionalEliteSelectorConfig
        )
        assert island.elite_selector.temperature is None

    def test_custom_fitness_key_propagates(self) -> None:
        cfg = build_single_island(
            fitness_key="custom_score", higher_is_better=False
        )
        island = cfg.island
        assert island.archive_selector.fitness_keys == ["custom_score"]  # type: ignore[union-attr]
        assert island.archive_selector.fitness_key_higher_is_better == [False]  # type: ignore[union-attr]
        assert island.elite_selector.fitness_key == "custom_score"  # type: ignore[union-attr]
        assert island.elite_selector.fitness_key_higher_is_better is False  # type: ignore[union-attr]

    def test_uncapped_island_drops_remover(self) -> None:
        cfg = build_single_island(max_size=None)
        assert cfg.island.archive_remover is None


class TestSingleIslandFixedTemp:
    def test_default_temperature(self) -> None:
        cfg = build_single_island_fitness_prop_fixed_temp()
        assert isinstance(
            cfg.island.elite_selector, FitnessProportionalEliteSelectorConfig
        )
        assert cfg.island.elite_selector.temperature == 0.14

    def test_temperature_override(self) -> None:
        cfg = build_single_island_fitness_prop_fixed_temp(temperature=0.5)
        assert cfg.island.elite_selector.temperature == 0.5  # type: ignore[union-attr]


class TestSingleIslandWeighted:
    def test_uses_weighted_selector(self) -> None:
        cfg = build_single_island_weighted()
        assert isinstance(
            cfg.island.elite_selector, WeightedEliteSelectorConfig
        )
        assert cfg.island.elite_selector.lambda_ == 10.0
        assert cfg.island.elite_selector.epsilon == 1e-8

    def test_custom_lambda(self) -> None:
        cfg = build_single_island_weighted(lambda_=5.0, epsilon=1e-6)
        assert cfg.island.elite_selector.lambda_ == 5.0  # type: ignore[union-attr]
        assert cfg.island.elite_selector.epsilon == 1e-6  # type: ignore[union-attr]


class TestSingleIsland2D:
    def test_default_shape(self) -> None:
        cfg = build_single_island_2d()
        bs = cfg.island.behavior_space
        assert bs.keys == ["fitness", "runtime"]
        assert bs.resolutions == [30, 5]
        # 30 * 5 = 150 cells, matching the 1-D budget
        assert bs.resolutions[0] * bs.resolutions[1] == 150

    def test_custom_runtime_bounds(self) -> None:
        cfg = build_single_island_2d(runtime_bounds=(0.0, 1800.0))
        assert cfg.island.behavior_space.bounds[1] == (0.0, 1800.0)


class TestTopology3D:
    def test_default_topology_3d_shape(self) -> None:
        cfg = build_topology_3d()
        bs = cfg.island.behavior_space
        assert bs.keys == ["dag_depth", "max_dependency_fan_in", "n_deep_retrieval"]
        assert bs.bounds == [(1.0, 11.0), (1.0, 9.0), (0.0, 5.0)]
        assert bs.resolutions == [5, 5, 6]

    def test_topology_3d_ret_uses_n_retrievals(self) -> None:
        cfg = build_topology_3d_ret()
        bs = cfg.island.behavior_space
        assert bs.keys[2] == "n_retrievals"
        assert bs.bounds[2] == (0.0, 6.0)

    def test_topology_3d_7step_tightened_bounds(self) -> None:
        cfg = build_topology_3d_7step()
        bs = cfg.island.behavior_space
        assert bs.bounds == [(1.0, 7.0), (1.0, 6.0), (0.0, 4.0)]
        assert bs.resolutions == [4, 3, 5]
        # 4 * 3 * 5 = 60 cells
        assert bs.resolutions[0] * bs.resolutions[1] * bs.resolutions[2] == 60


class TestMultiIslandFitnessComplexity:
    def test_two_islands_fitness_and_simplicity(self) -> None:
        cfg = build_multi_island_fitness_complexity()
        assert isinstance(cfg, MultiIslandConfig)
        assert len(cfg.islands) == 2
        assert cfg.islands[0].island_id == "fitness_island"
        assert cfg.islands[1].island_id == "simplicity_island"

    def test_first_island_carries_validity_axis_at_primary_resolution(
        self,
    ) -> None:
        cfg = build_multi_island_fitness_complexity()
        bs = cfg.islands[0].behavior_space
        assert bs.keys == ["fitness", "is_valid"]
        # Fitness island uses the [150, 2] grid (primary x validity);
        # not the [20, 2] grid the simplicity island carries.
        assert bs.resolutions == [150, 2]

    def test_second_island_carries_complexity_axis(self) -> None:
        cfg = build_multi_island_fitness_complexity()
        bs = cfg.islands[1].behavior_space
        assert bs.keys == ["fitness", "complexity_score"]
        assert bs.resolutions == [20, 10]

    def test_simplicity_island_archive_subtracts_complexity(self) -> None:
        """The simplicity island archive selector pins
        ``fitness_keys=[primary, complexity_score]`` with
        ``higher_is_better=[True, False]`` so the sum subtracts
        complexity from fitness. Without the second key the archive
        ignores complexity entirely and the island degenerates to a
        duplicate fitness island."""
        cfg = build_multi_island_fitness_complexity()
        sel = cfg.islands[1].archive_selector
        assert sel.fitness_keys == ["fitness", "complexity_score"]  # type: ignore[union-attr]
        assert sel.fitness_key_higher_is_better == [True, False]  # type: ignore[union-attr]

    def test_simplicity_island_remover_evicts_most_complex(self) -> None:
        """The simplicity island archive remover pins
        ``fitness_key="complexity_score"`` with
        ``higher_is_better=False`` so the island sheds its
        most-complex programs first (not its lowest-fitness)."""
        cfg = build_multi_island_fitness_complexity()
        remover = cfg.islands[1].archive_remover
        assert remover is not None
        assert remover.fitness_key == "complexity_score"  # type: ignore[union-attr]
        assert remover.fitness_key_higher_is_better is False  # type: ignore[union-attr]

    def test_migration_knobs_from_defaults(self) -> None:
        cfg = build_multi_island_fitness_complexity()
        assert cfg.migration_interval == 25
        assert cfg.max_migrants_per_island == 5
        assert cfg.enable_migration is True

    def test_custom_validity_key(self) -> None:
        cfg = build_multi_island_fitness_complexity(validity_key="my_validity")
        assert cfg.islands[0].behavior_space.keys[1] == "my_validity"


class TestRunsThroughAlgorithmUnion:
    """Every preset must produce a config that round-trips through the
    AlgorithmConfig discriminated union — defense-in-depth that the
    schema discriminator agrees with the concrete builder output."""

    @pytest.mark.parametrize(
        "builder",
        [
            build_single_island,
            build_single_island_fitness_prop_fixed_temp,
            build_single_island_weighted,
            build_single_island_2d,
            build_topology_3d,
            build_topology_3d_ret,
            build_topology_3d_7step,
        ],
    )
    def test_single_island_presets_round_trip(self, builder) -> None:
        from pydantic import TypeAdapter

        from gigaevo.config.schemas import AlgorithmConfig

        cfg = builder()
        ta: TypeAdapter[AlgorithmConfig] = TypeAdapter(AlgorithmConfig)
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, SingleIslandConfig)

    def test_multi_island_preset_round_trips(self) -> None:
        from pydantic import TypeAdapter

        from gigaevo.config.schemas import AlgorithmConfig

        cfg = build_multi_island_fitness_complexity()
        ta: TypeAdapter[AlgorithmConfig] = TypeAdapter(AlgorithmConfig)
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, MultiIslandConfig)


class TestPresetsInvariantCompliance:
    """Each preset must satisfy IslandConfig's max_size + remover
    invariant: if max_size is set then archive_remover must be set."""

    @pytest.mark.parametrize(
        "builder",
        [
            build_single_island,
            build_single_island_weighted,
            build_single_island_2d,
            build_topology_3d,
        ],
    )
    def test_default_max_size_paired_with_remover(self, builder) -> None:
        cfg = builder()
        assert cfg.island.max_size is not None
        assert cfg.island.archive_remover is not None

    def test_max_size_uncapped_works(self) -> None:
        cfg = build_single_island(max_size=None)
        assert cfg.island.archive_remover is None
