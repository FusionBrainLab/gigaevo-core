"""Preset builders for the algorithm schemas.

Each builder returns a fully-validated :class:`SingleIslandConfig`
or :class:`MultiIslandConfig`.

The fitness key and direction default to ``"fitness"`` / ``True`` --
the canonical shape for most experiments -- but every builder
accepts ``fitness_key`` and ``higher_is_better`` overrides so a
problem whose primary metric differs gets the right wiring.

Defaults for the island id, per-island archive cap, binning
resolutions, and migration knobs come from
:mod:`gigaevo.config.defaults`.
"""

from __future__ import annotations

from gigaevo.config.defaults import (
    DEFAULT_BINNING_TYPE,
    DEFAULT_ENABLE_MIGRATION,
    DEFAULT_ISLAND_ID,
    DEFAULT_ISLAND_MAX_SIZE,
    DEFAULT_MAX_MIGRANTS_PER_ISLAND,
    DEFAULT_MIGRATION_INTERVAL,
    DEFAULT_PRIMARY_RESOLUTION,
    DEFAULT_VALIDITY_RESOLUTION,
)
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


def _fitness_archive_selector(
    fitness_key: str, higher_is_better: bool
) -> SumArchiveSelectorConfig:
    return SumArchiveSelectorConfig(
        fitness_keys=[fitness_key],
        fitness_key_higher_is_better=[higher_is_better],
    )


def _fitness_archive_remover(
    fitness_key: str, higher_is_better: bool
) -> FitnessArchiveRemoverConfig:
    return FitnessArchiveRemoverConfig(
        fitness_key=fitness_key,
        fitness_key_higher_is_better=higher_is_better,
    )


def _top_fitness_migrant_selector(
    fitness_key: str, higher_is_better: bool
) -> TopFitnessMigrantSelectorConfig:
    return TopFitnessMigrantSelectorConfig(
        fitness_key=fitness_key,
        fitness_key_higher_is_better=higher_is_better,
    )


def _fitness_behavior_space(
    *,
    fitness_bounds: tuple[float, float] = (0.0, 1.0),
    primary_resolution: int = DEFAULT_PRIMARY_RESOLUTION,
) -> BehaviorSpaceConfig:
    """1-D behavior space on the canonical ``fitness`` axis."""
    return BehaviorSpaceConfig(
        keys=["fitness"],
        bounds=[fitness_bounds],
        resolutions=[primary_resolution],
        binning_types=[DEFAULT_BINNING_TYPE],
    )


def _island_with_default_selectors(
    *,
    fitness_key: str,
    higher_is_better: bool,
    behavior_space: BehaviorSpaceConfig,
    elite_selector: FitnessProportionalEliteSelectorConfig
    | WeightedEliteSelectorConfig,
    island_id: str = DEFAULT_ISLAND_ID,
    max_size: int | None = DEFAULT_ISLAND_MAX_SIZE,
) -> IslandConfig:
    return IslandConfig(
        island_id=island_id,
        max_size=max_size,
        behavior_space=behavior_space,
        archive_selector=_fitness_archive_selector(fitness_key, higher_is_better),
        elite_selector=elite_selector,
        archive_remover=(
            _fitness_archive_remover(fitness_key, higher_is_better)
            if max_size is not None
            else None
        ),
        migrant_selector=_top_fitness_migrant_selector(
            fitness_key, higher_is_better
        ),
    )


def build_single_island(
    *,
    fitness_key: str = "fitness",
    higher_is_better: bool = True,
    island_id: str = DEFAULT_ISLAND_ID,
    max_size: int | None = DEFAULT_ISLAND_MAX_SIZE,
    fitness_bounds: tuple[float, float] = (0.0, 1.0),
    primary_resolution: int = DEFAULT_PRIMARY_RESOLUTION,
    temperature: float | None = None,
) -> SingleIslandConfig:
    """1-D MAP-Elites with ``FitnessProportionalEliteSelector``. The
    default ``temperature=None`` enables the runtime selector's
    auto-temperature heuristic; pass a float (e.g. ``0.14``) for the
    fixed-temperature variant."""
    return SingleIslandConfig(
        island=_island_with_default_selectors(
            fitness_key=fitness_key,
            higher_is_better=higher_is_better,
            behavior_space=_fitness_behavior_space(
                fitness_bounds=fitness_bounds,
                primary_resolution=primary_resolution,
            ),
            elite_selector=FitnessProportionalEliteSelectorConfig(
                fitness_key=fitness_key,
                fitness_key_higher_is_better=higher_is_better,
                temperature=temperature,
            ),
            island_id=island_id,
            max_size=max_size,
        )
    )


def build_single_island_fitness_prop_fixed_temp(
    *,
    fitness_key: str = "fitness",
    higher_is_better: bool = True,
    temperature: float = 0.14,
    **kwargs,  # type: ignore[no-untyped-def]
) -> SingleIslandConfig:
    """Fixed-temperature softmax variant. ``temperature`` operates in
    normalised [0, 1] fitness space — 0.14 gives a ~7x weight ratio
    per unit of normalised fitness (moderate differentiation)."""
    return build_single_island(
        fitness_key=fitness_key,
        higher_is_better=higher_is_better,
        temperature=temperature,
        **kwargs,
    )


def build_single_island_weighted(
    *,
    fitness_key: str = "fitness",
    higher_is_better: bool = True,
    lambda_: float = 10.0,
    epsilon: float = 1e-8,
    island_id: str = DEFAULT_ISLAND_ID,
    max_size: int | None = DEFAULT_ISLAND_MAX_SIZE,
    fitness_bounds: tuple[float, float] = (0.0, 1.0),
    primary_resolution: int = DEFAULT_PRIMARY_RESOLUTION,
) -> SingleIslandConfig:
    """1-D MAP-Elites with ``WeightedEliteSelector`` — sigmoid x
    child-count penalty (ShinkaEvolve-style)."""
    return SingleIslandConfig(
        island=_island_with_default_selectors(
            fitness_key=fitness_key,
            higher_is_better=higher_is_better,
            behavior_space=_fitness_behavior_space(
                fitness_bounds=fitness_bounds,
                primary_resolution=primary_resolution,
            ),
            elite_selector=WeightedEliteSelectorConfig(
                fitness_key=fitness_key,
                fitness_key_higher_is_better=higher_is_better,
                lambda_=lambda_,
                epsilon=epsilon,
            ),
            island_id=island_id,
            max_size=max_size,
        )
    )


def build_single_island_2d(
    *,
    fitness_key: str = "fitness",
    higher_is_better: bool = True,
    runtime_bounds: tuple[float, float] = (0.0, 3600.0),
    runtime_resolution: int = 5,
    fitness_resolution: int = 30,
    fitness_bounds: tuple[float, float] = (0.0, 1.0),
    island_id: str = DEFAULT_ISLAND_ID,
    max_size: int | None = DEFAULT_ISLAND_MAX_SIZE,
) -> SingleIslandConfig:
    """2-D MAP-Elites (fitness x runtime). 30 x 5 = 150 cells by
    default, matching the 1-D budget but distributing across a
    runtime axis."""
    behavior_space = BehaviorSpaceConfig(
        keys=["fitness", "runtime"],
        bounds=[fitness_bounds, runtime_bounds],
        resolutions=[fitness_resolution, runtime_resolution],
        binning_types=[DEFAULT_BINNING_TYPE, "linear"],
    )
    return SingleIslandConfig(
        island=_island_with_default_selectors(
            fitness_key=fitness_key,
            higher_is_better=higher_is_better,
            behavior_space=behavior_space,
            elite_selector=FitnessProportionalEliteSelectorConfig(
                fitness_key=fitness_key,
                fitness_key_higher_is_better=higher_is_better,
            ),
            island_id=island_id,
            max_size=max_size,
        )
    )


def build_topology_3d(
    *,
    fitness_key: str = "fitness",
    higher_is_better: bool = True,
    third_axis: str = "n_deep_retrieval",
    bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] = ((1.0, 11.0), (1.0, 9.0), (0.0, 5.0)),
    resolutions: tuple[int, int, int] = (5, 5, 6),
    island_id: str = DEFAULT_ISLAND_ID,
    max_size: int | None = DEFAULT_ISLAND_MAX_SIZE,
) -> SingleIslandConfig:
    """3-D MAP-Elites on (dag_depth x max_dependency_fan_in x
    third_axis). Default shape is ``third_axis="n_deep_retrieval"``,
    5x5x6=150 cells; the ``_ret`` and ``_7step`` siblings reshape the
    third axis or tighten the bounds."""
    behavior_space = BehaviorSpaceConfig(
        keys=["dag_depth", "max_dependency_fan_in", third_axis],
        bounds=list(bounds),
        resolutions=list(resolutions),
        binning_types=[DEFAULT_BINNING_TYPE] * 3,
    )
    return SingleIslandConfig(
        island=_island_with_default_selectors(
            fitness_key=fitness_key,
            higher_is_better=higher_is_better,
            behavior_space=behavior_space,
            elite_selector=FitnessProportionalEliteSelectorConfig(
                fitness_key=fitness_key,
                fitness_key_higher_is_better=higher_is_better,
            ),
            island_id=island_id,
            max_size=max_size,
        )
    )


def build_topology_3d_ret(
    *, fitness_key: str = "fitness", higher_is_better: bool = True
) -> SingleIslandConfig:
    """Topology-3D variant: third axis is the combined
    ``n_retrievals`` (retrieve + retrieve_deep), bounds (0, 6)."""
    return build_topology_3d(
        fitness_key=fitness_key,
        higher_is_better=higher_is_better,
        third_axis="n_retrievals",
        bounds=((1.0, 11.0), (1.0, 9.0), (0.0, 6.0)),
        resolutions=(5, 5, 6),
    )


def build_topology_3d_7step(
    *, fitness_key: str = "fitness", higher_is_better: bool = True
) -> SingleIslandConfig:
    """Topology-3D variant for 7-step max chains: 4 x 3 x 5 = 60
    cells with tightened bounds."""
    return build_topology_3d(
        fitness_key=fitness_key,
        higher_is_better=higher_is_better,
        third_axis="n_deep_retrieval",
        bounds=((1.0, 7.0), (1.0, 6.0), (0.0, 4.0)),
        resolutions=(4, 3, 5),
    )


def build_multi_island_fitness_complexity(
    *,
    fitness_key: str = "fitness",
    higher_is_better: bool = True,
    validity_key: str = "is_valid",
    fitness_bounds: tuple[float, float] = (0.0, 1.0),
    validity_bounds: tuple[float, float] = (0.0, 1.0),
    complexity_bounds: tuple[float, float] = (0.0, 100.0),
    fitness_island_primary_resolution: int = DEFAULT_PRIMARY_RESOLUTION,
    fitness_island_validity_resolution: int = DEFAULT_VALIDITY_RESOLUTION,
    simplicity_island_fitness_resolution: int = 20,
    simplicity_island_complexity_resolution: int = 10,
    island_max_size: int | None = DEFAULT_ISLAND_MAX_SIZE,
    migration_interval: int = DEFAULT_MIGRATION_INTERVAL,
    max_migrants_per_island: int = DEFAULT_MAX_MIGRANTS_PER_ISLAND,
    enable_migration: bool = DEFAULT_ENABLE_MIGRATION,
) -> MultiIslandConfig:
    """Two-island fitness/complexity shape.

    Fitness island (``primary_resolution x validity_resolution``
    cells, 150 x 2 by default): archive scores by primary fitness
    only; archive remover drops lowest-fitness elites.

    Simplicity island (20 x 10 cells, a coarser fitness grid plus a
    complexity axis): archive scores by the sum of primary fitness
    MINUS complexity (``fitness_keys=[primary, complexity_score]``
    with ``higher_is_better=[True, False]``), biasing the archive
    toward simpler high-fitness programs. Archive remover drops the
    most-complex programs first — the goal on this island is code
    simplicity, so the eviction policy mirrors that.

    Migration knobs flow from defaults; the bandit selects between
    islands by recent improvement rate every
    ``migration_interval`` generations."""
    fitness_island = IslandConfig(
        island_id="fitness_island",
        max_size=island_max_size,
        behavior_space=BehaviorSpaceConfig(
            keys=[fitness_key, validity_key],
            bounds=[fitness_bounds, validity_bounds],
            resolutions=[
                fitness_island_primary_resolution,
                fitness_island_validity_resolution,
            ],
            binning_types=[DEFAULT_BINNING_TYPE, DEFAULT_BINNING_TYPE],
        ),
        archive_selector=_fitness_archive_selector(
            fitness_key, higher_is_better
        ),
        elite_selector=FitnessProportionalEliteSelectorConfig(
            fitness_key=fitness_key,
            fitness_key_higher_is_better=higher_is_better,
        ),
        archive_remover=(
            _fitness_archive_remover(fitness_key, higher_is_better)
            if island_max_size is not None
            else None
        ),
        migrant_selector=_top_fitness_migrant_selector(
            fitness_key, higher_is_better
        ),
    )
    simplicity_island = IslandConfig(
        island_id="simplicity_island",
        max_size=island_max_size,
        behavior_space=BehaviorSpaceConfig(
            keys=[fitness_key, "complexity_score"],
            bounds=[fitness_bounds, complexity_bounds],
            resolutions=[
                simplicity_island_fitness_resolution,
                simplicity_island_complexity_resolution,
            ],
            binning_types=[DEFAULT_BINNING_TYPE, DEFAULT_BINNING_TYPE],
        ),
        archive_selector=SumArchiveSelectorConfig(
            fitness_keys=[fitness_key, "complexity_score"],
            fitness_key_higher_is_better=[higher_is_better, False],
        ),
        elite_selector=FitnessProportionalEliteSelectorConfig(
            fitness_key=fitness_key,
            fitness_key_higher_is_better=higher_is_better,
        ),
        archive_remover=(
            FitnessArchiveRemoverConfig(
                fitness_key="complexity_score",
                fitness_key_higher_is_better=False,
            )
            if island_max_size is not None
            else None
        ),
        migrant_selector=_top_fitness_migrant_selector(
            fitness_key, higher_is_better
        ),
    )
    return MultiIslandConfig(
        islands=[fitness_island, simplicity_island],
        migration_interval=migration_interval,
        max_migrants_per_island=max_migrants_per_island,
        enable_migration=enable_migration,
    )


__all__: list[str] = [
    "AlgorithmConfig",
    "build_multi_island_fitness_complexity",
    "build_single_island",
    "build_single_island_2d",
    "build_single_island_fitness_prop_fixed_temp",
    "build_single_island_weighted",
    "build_topology_3d",
    "build_topology_3d_7step",
    "build_topology_3d_ret",
]
