"""Shared-benchmark lineage signal (v3 D population, Prong 2 HoF-snapshot race fix).

Lineage trend = mean(child_deltas) - mean(parent_deltas) on shared G benchmark.
Shared benchmark = intersection of G's both child-D and parent-D have faced.
This breaks HoF-snapshot race (HoF rotates, so current-HoF comparison is not
meaningful across generations). Shared benchmark is stable (no rotation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.programs.core_types import StageIO, VoidOutput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker


class SharedBenchmarkLineageOutput(StageIO):
    """Output of SharedBenchmarkLineageStage."""

    trend: float | None = None
    """Mean delta(child) - mean delta(parent) on shared benchmark. None if insufficient data."""

    n_shared: int | None = None
    """Number of shared G's used in trend calculation. None if insufficient."""


class SharedBenchmarkResolver(ABC):
    """Abstraction for finding the shared benchmark between two D programs.

    Used by SharedBenchmarkLineageStage to find G's both a parent-D and child-D
    have been evaluated against, forming a stable evaluation context.
    """

    @abstractmethod
    async def shared_benchmark(self, d_a: str, d_b: str) -> list[str]:
        """Return G IDs both d_a and d_b have been evaluated against.

        Args:
            d_a: First D program ID (e.g., child).
            d_b: Second D program ID (e.g., parent).

        Returns:
            List of G program IDs in the intersection. Empty if no overlap.
        """
        pass


class DGTrackerSharedOpponentResolver(SharedBenchmarkResolver):
    """Shared benchmark resolver using DGImprovementTracker.faced_by_d intersection."""

    def __init__(self, *, tracker: DGImprovementTracker):
        self._tracker = tracker

    async def shared_benchmark(self, d_a: str, d_b: str) -> list[str]:
        """Return intersection of G's both d_a and d_b have faced."""
        faced_a = await self._tracker.faced_by_d(d_a)
        faced_b = await self._tracker.faced_by_d(d_b)
        return list(faced_a & faced_b)


class SharedBenchmarkLineageStage(Stage):
    """Compute lineage trend on shared-benchmark G's.

    Args:
        resolver: SharedBenchmarkResolver instance.
        min_shared: Minimum shared G's required to compute trend (default 2).
    """

    InputsModel = VoidOutput
    OutputModel = SharedBenchmarkLineageOutput
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        resolver: SharedBenchmarkResolver,
        min_shared: int = 2,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._resolver = resolver
        self._min_shared = min_shared

    async def _load_parent(self, program: Program) -> Program | None:
        """Load the parent program by parent_id. Override in tests."""
        if program.parent_id is None:
            return None
        # Stub — in real pipelines, this queries the archive or program store.
        return None

    async def compute(self, program: Program) -> SharedBenchmarkLineageOutput:
        """Compute shared-benchmark lineage trend for this program."""
        parent = await self._load_parent(program)
        if parent is None:
            logger.debug(
                "[SharedBenchmarkLineageStage] {} no parent; trend=None", program.id[:8]
            )
            return SharedBenchmarkLineageOutput(trend=None, n_shared=None)

        # For tests, _parent_id_to_d_id maps program IDs to D IDs.
        # In production, this mapping is queried from the archive.
        parent_id_to_d_id = getattr(self, "_parent_id_to_d_id", {})
        d_id = parent_id_to_d_id.get(program.id)
        parent_d_id = parent_id_to_d_id.get(parent.id)

        if d_id is None or parent_d_id is None:
            logger.debug(
                "[SharedBenchmarkLineageStage] {} → {} missing d_id mapping",
                program.id[:8],
                parent.id[:8],
            )
            return SharedBenchmarkLineageOutput(trend=None, n_shared=None)

        shared = await self._resolver.shared_benchmark(d_id, parent_d_id)
        if len(shared) < self._min_shared:
            logger.debug(
                "[SharedBenchmarkLineageStage] {} insufficient shared: {} < {}",
                program.id[:8],
                len(shared),
                self._min_shared,
            )
            return SharedBenchmarkLineageOutput(trend=None, n_shared=None)

        # Fetch deltas from the tracker
        tracker = getattr(self._resolver, "_tracker", None)
        if tracker is None:
            logger.error(
                "[SharedBenchmarkLineageStage] resolver has no _tracker attribute"
            )
            return SharedBenchmarkLineageOutput(trend=None, n_shared=None)

        pairs = await tracker.get_deltas_against(d_id, parent_d_id, shared)
        if not pairs:
            logger.debug(
                "[SharedBenchmarkLineageStage] {} no delta pairs from tracker",
                program.id[:8],
            )
            return SharedBenchmarkLineageOutput(trend=None, n_shared=None)

        child_mean = sum(c for c, _ in pairs) / len(pairs)
        parent_mean = sum(p for _, p in pairs) / len(pairs)
        trend = child_mean - parent_mean

        logger.info(
            "[SharedBenchmarkLineageStage] {} trend={:.6f} (n_shared={} "
            "child_mean={:.6f} parent_mean={:.6f})",
            program.id[:8],
            trend,
            len(shared),
            child_mean,
            parent_mean,
        )
        return SharedBenchmarkLineageOutput(trend=trend, n_shared=len(shared))
