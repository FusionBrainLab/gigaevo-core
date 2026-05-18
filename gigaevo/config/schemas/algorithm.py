from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from gigaevo.config.schemas._base import FrozenStrictModel

if TYPE_CHECKING:
    from gigaevo.database.redis_program_storage import RedisProgramStorage
    from gigaevo.evolution.strategies.elite_selectors import EliteSelector
    from gigaevo.evolution.strategies.migrant_selectors import MigrantSelector
    from gigaevo.evolution.strategies.models import BehaviorSpace
    from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
    from gigaevo.evolution.strategies.removers import ArchiveRemover
    from gigaevo.evolution.strategies.selectors import ArchiveSelector


BinningType = Literal["linear"]


class BehaviorSpaceConfig(FrozenStrictModel):
    """Declarative shape of a MAP-Elites behavior space.

    Each list is parallel: ``keys[i]`` has bounds ``bounds[i]``,
    resolution ``resolutions[i]``, and binning ``binning_types[i]``.
    ``dynamic`` chooses ``DynamicBehaviorSpace`` over the static
    counterpart; the initial bounds act as hard limits in either case.
    """

    keys: list[str] = Field(min_length=1)
    bounds: list[tuple[float, float]] = Field(min_length=1)
    resolutions: list[int] = Field(min_length=1)
    binning_types: list[BinningType] = Field(min_length=1)
    dynamic: bool = True
    expansion_buffer_ratio: float = Field(default=0.1, ge=0.0)

    @model_validator(mode="after")
    def lists_aligned(self) -> BehaviorSpaceConfig:
        n = len(self.keys)
        for name, value in (
            ("bounds", self.bounds),
            ("resolutions", self.resolutions),
            ("binning_types", self.binning_types),
        ):
            if len(value) != n:
                raise ValueError(
                    f"{name} length ({len(value)}) must equal keys length ({n})"
                )
        for i, (lo, hi) in enumerate(self.bounds):
            if lo > hi:
                raise ValueError(
                    f"bounds[{i}] invalid: min ({lo}) > max ({hi}) for key {self.keys[i]!r}"
                )
        for i, r in enumerate(self.resolutions):
            if r <= 0:
                raise ValueError(
                    f"resolutions[{i}] must be > 0 for key {self.keys[i]!r}"
                )
        return self

    @property
    def primary_key(self) -> str:
        """The first behavior key is treated as the primary metric for
        fitness-aware selectors that need a scalar to optimise."""
        return self.keys[0]

    @property
    def behavior_keys(self) -> list[str]:
        """Mirror of the runtime ``BehaviorSpace.behavior_keys`` for
        consumers that need the ordered key list without materialising
        the runtime object."""
        return list(self.keys)

    def build(self) -> BehaviorSpace:
        from gigaevo.evolution.strategies.models import (
            BehaviorSpace,
            DynamicBehaviorSpace,
            LinearBinning,
        )

        bins: dict[str, LinearBinning] = {}
        for key, (lo, hi), num_bins, binning in zip(
            self.keys, self.bounds, self.resolutions, self.binning_types
        ):
            if binning != "linear":
                raise ValueError(
                    f"binning_type {binning!r} not yet supported in the schema"
                )
            bins[key] = LinearBinning(
                min_val=float(lo), max_val=float(hi), num_bins=num_bins
            )
        if self.dynamic:
            return DynamicBehaviorSpace(
                bins=bins, expansion_buffer_ratio=self.expansion_buffer_ratio
            )
        return BehaviorSpace(bins=bins)


class SumArchiveSelectorConfig(FrozenStrictModel):
    """Sum-of-fitnesses archive selector. Used by every algorithm YAML
    currently shipped under config/algorithm/."""

    kind: Literal["sum"] = "sum"
    fitness_keys: list[str] = Field(min_length=1)
    fitness_key_higher_is_better: list[bool] = Field(min_length=1)

    @model_validator(mode="after")
    def keys_aligned(self) -> SumArchiveSelectorConfig:
        if len(self.fitness_keys) != len(self.fitness_key_higher_is_better):
            raise ValueError(
                "fitness_keys and fitness_key_higher_is_better lengths must match"
            )
        return self

    def build(self) -> ArchiveSelector:
        from gigaevo.evolution.strategies.selectors import SumArchiveSelector

        return SumArchiveSelector(
            fitness_keys=self.fitness_keys,
            fitness_key_higher_is_better=self.fitness_key_higher_is_better,
        )


ArchiveSelectorConfig = Annotated[
    SumArchiveSelectorConfig,
    Field(discriminator="kind"),
]


class FitnessProportionalEliteSelectorConfig(FrozenStrictModel):
    """Boltzmann softmax sampling. ``temperature=None`` enables the
    auto-temperature heuristic from the runtime selector."""

    kind: Literal["fitness_proportional"] = "fitness_proportional"
    fitness_key: str = Field(min_length=1)
    fitness_key_higher_is_better: bool = True
    temperature: float | None = Field(default=None, gt=0.0)

    def build(self) -> EliteSelector:
        from gigaevo.evolution.strategies.elite_selectors import (
            FitnessProportionalEliteSelector,
        )

        return FitnessProportionalEliteSelector(
            fitness_key=self.fitness_key,
            fitness_key_higher_is_better=self.fitness_key_higher_is_better,
            temperature=self.temperature,
        )


class WeightedEliteSelectorConfig(FrozenStrictModel):
    """ShinkaEvolve-style sigmoid + child-count weighted sampling."""

    kind: Literal["weighted"] = "weighted"
    fitness_key: str = Field(min_length=1)
    fitness_key_higher_is_better: bool = True
    lambda_: float = Field(default=10.0, gt=0.0)
    epsilon: float = Field(default=1e-8, gt=0.0)

    def build(self) -> EliteSelector:
        from gigaevo.evolution.strategies.elite_selectors import (
            WeightedEliteSelector,
        )

        return WeightedEliteSelector(
            fitness_key=self.fitness_key,
            fitness_key_higher_is_better=self.fitness_key_higher_is_better,
            lambda_=self.lambda_,
            epsilon=self.epsilon,
        )


EliteSelectorConfig = Annotated[
    FitnessProportionalEliteSelectorConfig | WeightedEliteSelectorConfig,
    Field(discriminator="kind"),
]


class FitnessArchiveRemoverConfig(FrozenStrictModel):
    """Removes the lowest-fitness program when an island archive exceeds
    its ``max_size``."""

    kind: Literal["fitness"] = "fitness"
    fitness_key: str = Field(min_length=1)
    fitness_key_higher_is_better: bool = True

    def build(self) -> ArchiveRemover:
        from gigaevo.evolution.strategies.removers import FitnessArchiveRemover

        return FitnessArchiveRemover(
            fitness_key=self.fitness_key,
            fitness_key_higher_is_better=self.fitness_key_higher_is_better,
        )


ArchiveRemoverConfig = Annotated[
    FitnessArchiveRemoverConfig,
    Field(discriminator="kind"),
]


class TopFitnessMigrantSelectorConfig(FrozenStrictModel):
    """Picks the top-fitness programs as migrants."""

    kind: Literal["top_fitness"] = "top_fitness"
    fitness_key: str = Field(min_length=1)
    fitness_key_higher_is_better: bool = True

    def build(self) -> MigrantSelector:
        from gigaevo.evolution.strategies.migrant_selectors import (
            TopFitnessMigrantSelector,
        )

        return TopFitnessMigrantSelector(
            fitness_key=self.fitness_key,
            fitness_key_higher_is_better=self.fitness_key_higher_is_better,
        )


MigrantSelectorConfig = Annotated[
    TopFitnessMigrantSelectorConfig,
    Field(discriminator="kind"),
]


class IslandConfig(FrozenStrictModel):
    """Schema-side island descriptor. Builds the runtime
    ``gigaevo.evolution.strategies.island.IslandConfig`` via
    :meth:`build`. Selectors are typed by their schema discriminator
    rather than by ``_target_`` strings, so a typo in a fitness key or
    a wrong selector class name is caught at load time."""

    island_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    max_size: int | None = Field(default=None, ge=1)
    behavior_space: BehaviorSpaceConfig
    archive_selector: ArchiveSelectorConfig
    elite_selector: EliteSelectorConfig
    archive_remover: ArchiveRemoverConfig | None = None
    migrant_selector: MigrantSelectorConfig

    @model_validator(mode="after")
    def remover_required_when_capped(self) -> IslandConfig:
        if self.max_size is not None and self.archive_remover is None:
            raise ValueError(
                f"island {self.island_id!r}: max_size is set but archive_remover is None"
            )
        return self

    def build(self):  # type: ignore[no-untyped-def]
        from gigaevo.evolution.strategies.island import (
            IslandConfig as RuntimeIslandConfig,
        )

        return RuntimeIslandConfig(
            island_id=self.island_id,
            max_size=self.max_size,
            behavior_space=self.behavior_space.build(),
            archive_selector=self.archive_selector.build(),
            elite_selector=self.elite_selector.build(),
            archive_remover=(
                self.archive_remover.build()
                if self.archive_remover is not None
                else None
            ),
            migrant_selector=self.migrant_selector.build(),
        )


class SingleIslandConfig(FrozenStrictModel):
    """One island, no migration. Covers single_island*.yaml and the
    topology_3d*.yaml variants which are single-island with a
    three-dimensional behavior space. The runtime backing is
    ``MapElitesMultiIsland`` with a one-element islands list and
    ``enable_migration=False`` — a single class serves both shapes."""

    kind: Literal["single_island"] = "single_island"
    island: IslandConfig

    def build(
        self, *, program_storage: RedisProgramStorage
    ) -> MapElitesMultiIsland:
        from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland

        return MapElitesMultiIsland(
            island_configs=[self.island.build()],
            program_storage=program_storage,
            enable_migration=False,
        )


class MultiIslandConfig(FrozenStrictModel):
    """Multiple islands with periodic migrant exchange."""

    kind: Literal["multi_island"] = "multi_island"
    # ``default_factory=list`` lets tyro render help text for the
    # inactive branch of ``AlgorithmConfig`` when ``SingleIslandConfig``
    # is selected. ``validate_default=True`` makes ``min_length=2`` fire
    # when a caller constructs ``MultiIslandConfig()`` directly without
    # an ``islands`` argument; the constructor exits with a typed error
    # instead of accepting an empty list silently.
    islands: list[IslandConfig] = Field(
        default_factory=list, min_length=2, validate_default=True
    )
    # ``migration_interval=25`` matches the shipped preset value in
    # ``gigaevo.config.defaults.DEFAULT_MIGRATION_INTERVAL`` so callers
    # that instantiate :class:`MultiIslandConfig` directly observe the
    # same cadence the preset builder would have produced.
    migration_interval: int = Field(default=25, ge=1)
    max_migrants_per_island: int = Field(default=5, ge=1)
    enable_migration: bool = True

    @model_validator(mode="after")
    def unique_island_ids(self) -> MultiIslandConfig:
        ids = [i.island_id for i in self.islands]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate island_id in islands list: {ids}")
        return self

    def build(
        self, *, program_storage: RedisProgramStorage
    ) -> MapElitesMultiIsland:
        from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland

        return MapElitesMultiIsland(
            island_configs=[i.build() for i in self.islands],
            program_storage=program_storage,
            migration_interval=self.migration_interval,
            enable_migration=self.enable_migration,
            max_migrants_per_island=self.max_migrants_per_island,
        )


AlgorithmConfig = Annotated[
    SingleIslandConfig | MultiIslandConfig,
    Field(discriminator="kind"),
]
