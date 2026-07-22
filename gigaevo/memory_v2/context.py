"""Typed adapters from the evolution engine to replayable memory-v2 context."""

from __future__ import annotations

from collections.abc import Mapping
import itertools
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.evolution.strategies.island import (
    METADATA_KEY_CURRENT_ISLAND,
    METADATA_KEY_HOME_ISLAND,
    MapElitesIsland,
)
from gigaevo.evolution.strategies.models import BehaviorSpace
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.memory_v2.models import (
    BehaviorCoordinate,
    EnvironmentFingerprint,
    EvolutionContext,
    LineageCreditConfig,
    MapElitesContext,
    RewardDefinition,
    canonical_digest,
)
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program


class MapContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    neighbor_radius: int = Field(default=1, ge=1, le=3)


class ArchiveMembership(BaseModel):
    """Validated projection of the storage-opaque Program metadata boundary."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    current_island: str | None = Field(  # type: ignore[literal-required]
        default=None, alias=METADATA_KEY_CURRENT_ISLAND
    )
    home_island: str | None = Field(  # type: ignore[literal-required]
        default=None, alias=METADATA_KEY_HOME_ISLAND
    )


@runtime_checkable
class DecisionContextSource(Protocol):
    async def snapshot(self, program: Program) -> EvolutionContext: ...


@runtime_checkable
class TrajectoryIdSource(Protocol):
    @property
    def trajectory_id(self) -> str: ...


class MapElitesContextSource:
    """Freeze one coherent pre-treatment view of a MAP-Elites parent."""

    def __init__(
        self,
        *,
        strategy: MapElitesMultiIsland,
        metrics_context: MetricsContext,
        environment: EnvironmentFingerprint,
        config: MapContextConfig,
        credit: LineageCreditConfig,
        trajectory_id_source: TrajectoryIdSource,
    ) -> None:
        self.strategy = strategy
        self.metrics_context = metrics_context
        self.environment = environment
        self.config = config
        self.credit = credit
        self.trajectory_id_source = trajectory_id_source

    @property
    def behavior_keys(self) -> tuple[str, ...]:
        """Return the common ordered model axes from the live behavior spaces."""

        spaces = [
            island.config.behavior_space for island in self.strategy.islands.values()
        ]
        schemas = {tuple(space.behavior_keys) for space in spaces}
        if not schemas:
            raise ValueError("memory v2 requires at least one MAP-Elites island")
        if len(schemas) != 1:
            raise ValueError(
                "memory v2 requires one shared ordered behavior schema across "
                f"islands, got {sorted(schemas)!r}"
            )
        behavior_keys = next(iter(schemas))
        semantic_schemas = {
            tuple(
                (
                    key,
                    canonical_digest(space.bins[key].model_transform_payload()),
                )
                for key in behavior_keys
            )
            for space in spaces
        }
        if len(semantic_schemas) != 1:
            raise ValueError(
                "memory v2 requires one shared stable behavior transform across islands"
            )
        return behavior_keys

    async def snapshot(self, program: Program) -> EvolutionContext:
        # Keep direct callers as strict as the Hydra path that resolves this
        # property while constructing the feature map.
        _ = self.behavior_keys
        island = self._island_for(program)
        elites, behavior_space = await island.snapshot_archive_state()
        required = tuple(behavior_space.behavior_keys)
        missing_metrics = set(required) - program.metrics.keys()
        if missing_metrics:
            raise ValueError(
                f"parent lacks behavior metrics: {sorted(missing_metrics)}"
            )

        parent_cell = behavior_space.get_cell(program.metrics)
        occupied_cells = {
            behavior_space.get_cell(elite.metrics): elite for elite in elites
        }
        coordinates = tuple(
            self._coordinate(key, program.metrics[key], behavior_space)
            for key in required
        )
        archive_fingerprint = canonical_digest(
            [
                {
                    "id": elite.id,
                    "revision": elite.atomic_counter,
                    "cell": behavior_space.get_cell(elite.metrics),
                }
                for elite in sorted(elites, key=lambda row: row.id)
            ]
        )
        schema_payload = [
            {
                "key": row.key,
                "dynamic_lower_bound": row.dynamic_lower_bound,
                "dynamic_upper_bound": row.dynamic_upper_bound,
                "num_bins": row.num_bins,
                "model_transform": behavior_space.bins[
                    row.key
                ].model_transform_payload(),
            }
            for row in coordinates
        ]
        total_cells = behavior_space.total_cells
        primary_key = self.metrics_context.get_primary_key()
        bounds = self.metrics_context.get_bounds(primary_key)
        if bounds is None:
            raise ValueError(
                f"memory v2 requires finite bounds for primary metric {primary_key!r}"
            )
        reward = RewardDefinition(
            primary_metric=primary_key,
            higher_is_better=self.metrics_context.is_higher_better(primary_key),
            metric_lower_bound=bounds[0],
            metric_upper_bound=bounds[1],
            endpoint=(
                "bounded_proximal_utility"
                if self.credit.lineage_depth == 1
                else "bounded_lineage_utility"
            ),
            lineage_depth=self.credit.lineage_depth,
            lineage_opportunity_budget=(
                1 if self.credit.lineage_depth == 1 else self.credit.opportunity_budget
            ),
        )
        return EvolutionContext(
            run_id=self.trajectory_id_source.trajectory_id,
            environment=self.environment,
            parent_id=program.id,
            parent_iteration=program.iteration,
            parent_generation=program.generation,
            parent_metrics=dict(program.metrics),
            reward=reward,
            map_elites=MapElitesContext(
                island_id=island.config.island_id,
                strategy_generation=self.strategy.generation,
                archive_size=len(occupied_cells),
                total_cells=total_cells,
                coverage=len(occupied_cells) / total_cells,
                parent_quality_quantile=self._quality_quantile(
                    program,
                    elites,
                    primary_key=primary_key,
                    higher_is_better=reward.higher_is_better,
                ),
                parent_cell=parent_cell,
                parent_cell_occupied=parent_cell in occupied_cells,
                neighbor_occupancy=self._neighbor_occupancy(
                    parent_cell,
                    occupied_cells,
                    coordinates,
                ),
                coordinates=coordinates,
                semantic_schema_hash=canonical_digest(
                    [
                        behavior_space.bins[key].model_transform_payload()
                        for key in required
                    ]
                ),
                behavior_schema_hash=canonical_digest(schema_payload),
                archive_fingerprint=archive_fingerprint,
            ),
        )

    def _island_for(self, program: Program) -> MapElitesIsland:
        membership = ArchiveMembership.model_validate(program.metadata)
        island_id = membership.current_island or membership.home_island
        if island_id is not None:
            island = self.strategy.islands.get(island_id)
            if island is not None:
                return island
        if len(self.strategy.islands) == 1:
            return next(iter(self.strategy.islands.values()))
        raise ValueError(
            f"parent {program.id!r} has no valid island in a multi-island strategy"
        )

    def _coordinate(
        self,
        key: str,
        value: float,
        behavior_space: BehaviorSpace,
    ) -> BehaviorCoordinate:
        binning = behavior_space.bins[key]
        return BehaviorCoordinate(
            key=key,
            raw_value=float(value),
            semantic_normalized=binning.normalize_for_model(value),
            dynamic_normalized=binning.normalize_for_archive(value),
            cell_index=binning.get_index(float(value)),
            num_bins=binning.num_bins,
            dynamic_lower_bound=binning.min_val,
            dynamic_upper_bound=binning.max_val,
        )

    def _neighbor_occupancy(
        self,
        parent_cell: tuple[int, ...],
        occupied_cells: Mapping[tuple[int, ...], Program],
        coordinates: tuple[BehaviorCoordinate, ...],
    ) -> float:
        radius = self.config.neighbor_radius
        axes = [
            range(max(0, index - radius), min(row.num_bins, index + radius + 1))
            for index, row in zip(parent_cell, coordinates)
        ]
        neighbors = [cell for cell in itertools.product(*axes) if cell != parent_cell]
        if not neighbors:
            return 0.0
        return sum(cell in occupied_cells for cell in neighbors) / len(neighbors)

    def _quality_quantile(
        self,
        parent: Program,
        elites: list[Program],
        *,
        primary_key: str,
        higher_is_better: bool,
    ) -> float | None:
        parent_fitness = self.metrics_context.strict_fitness(
            parent.metrics, primary_key
        )
        if parent_fitness is None:
            return None
        values = [
            fitness
            for elite in elites
            for fitness in (
                self.metrics_context.strict_fitness(elite.metrics, primary_key),
            )
            if fitness is not None
        ]
        if not values:
            return None
        oriented_parent = parent_fitness if higher_is_better else -parent_fitness
        oriented = [value if higher_is_better else -value for value in values]
        below = sum(value < oriented_parent for value in oriented)
        tied = sum(value == oriented_parent for value in oriented)
        return (below + 0.5 * tied) / len(oriented)
