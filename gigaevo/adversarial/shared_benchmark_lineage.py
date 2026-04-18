"""Shared-benchmark lineage signal (v3 D population, Prong 2 HoF-snapshot race fix).

Lineage trend = mean(child_deltas) - mean(parent_deltas) on shared G benchmark.
Shared benchmark = intersection of G's both child-D and parent-D have faced.
This breaks HoF-snapshot race (HoF rotates, so current-HoF comparison is not
meaningful across generations). Shared benchmark is stable (no rotation).

Caching contract: the stage uses ``CacheOnlyInput`` with ``cache_on`` plumbed
to the current opponent-id list from ``FetchOpponentIdsStage``. The tracker
pairs backing the shared benchmark only grow meaningfully when the G HoF
rotates (new G's enter the eval set), so opponent-id change is a defensible
invalidation signal — within a single HoF window the trend estimate is stable
because appended pairs are marginal and tracker deltas are immutable once
written. This matches the invalidation pattern used by ``LineageStage`` /
``InsightsStage`` in ``asymmetric_pipeline.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.adversarial.structured_logging import emit_lineage_trend
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.core_types import ProgramStageResult, StageIO
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import CacheOnlyInput

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker


class SharedBenchmarkLineageOutput(StageIO):
    """Output of SharedBenchmarkLineageStage."""

    trend: float | None = None
    """Mean delta(child) - mean delta(parent) on shared benchmark. None if insufficient data."""

    n_shared: int | None = None
    """Number of shared G's used in trend calculation. None if insufficient."""


class SharedBenchmarkResolver(ABC):
    """Abstraction for finding the shared benchmark between two D programs."""

    @abstractmethod
    async def shared_benchmark(self, d_a: str, d_b: str) -> list[str]:
        """Return G IDs both d_a and d_b have been evaluated against."""
        pass


class DGTrackerSharedOpponentResolver(SharedBenchmarkResolver):
    """Shared benchmark resolver using DGImprovementTracker.faced_by_d intersection."""

    def __init__(self, *, tracker: DGImprovementTracker):
        self._tracker = tracker

    async def shared_benchmark(self, d_a: str, d_b: str) -> list[str]:
        faced_a = await self._tracker.faced_by_d(d_a)
        faced_b = await self._tracker.faced_by_d(d_b)
        return list(faced_a & faced_b)


class SharedBenchmarkLineageStage(Stage):
    """Compute lineage trend on shared-benchmark G's (D-side, v3 §3.5 Prong 2).

    Args:
        resolver: SharedBenchmarkResolver instance.
        storage: ProgramStorage for loading parent by ``program.parent_id``.
            Optional for tests that override ``_load_parent``.
        min_shared: Minimum shared G's required to compute trend (default 2).

    Caching: ``InputsModel = CacheOnlyInput`` — wire ``cache_on`` from
    ``FetchOpponentIdsStage`` to invalidate on HoF rotation.
    """

    InputsModel = CacheOnlyInput
    OutputModel = SharedBenchmarkLineageOutput

    def __init__(
        self,
        *,
        resolver: SharedBenchmarkResolver,
        storage: ProgramStorage | None = None,
        min_shared: int = 2,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._resolver = resolver
        self._storage = storage
        self._min_shared = min_shared

    async def _load_parent(self, program: Program) -> Program | None:
        """Load the parent program. Override in tests if no storage is supplied."""
        if program.parent_id is None:
            return None
        if self._storage is None:
            return None
        parents = await self._storage.mget([program.parent_id])
        if not parents:
            return None
        parent = parents[0]
        return parent

    def _d_id(self, program: Program) -> str:
        """D program id == d_id in tracker. Test hook via ``_parent_id_to_d_id``."""
        mapping = getattr(self, "_parent_id_to_d_id", None)
        if mapping is not None:
            mapped = mapping.get(program.id)
            if mapped is not None:
                return mapped
        return program.id

    def _emit_lineage_event(
        self,
        program_id: str,
        d_id: str,
        parent_d_id: str,
        trend: float | None,
        n_shared: int,
    ) -> None:
        payload = emit_lineage_trend(
            program_id=program_id,
            d_id=d_id,
            parent_d_id=parent_d_id,
            trend=trend,
            n_shared=n_shared,
        )
        logger.info("[LINEAGE_TREND] {}", json.dumps(payload))

    def _write_metrics(
        self, program: Program, trend: float | None, n_shared: int | None
    ) -> None:
        """Stash trend + n_shared on program.metrics for BD / telemetry use."""
        if program.metrics is None:
            return
        program.metrics["lineage_trend"] = trend if trend is not None else 0.0
        program.metrics["n_shared"] = n_shared if n_shared is not None else 0

    async def compute(self, program: Program) -> SharedBenchmarkLineageOutput:
        """Compute shared-benchmark lineage trend for this program."""
        d_id = self._d_id(program)

        parent = await self._load_parent(program)
        if parent is None:
            logger.debug(
                "[SharedBenchmarkLineageStage] {} no parent; trend=None",
                program.id[:8],
            )
            self._emit_lineage_event(program.id, d_id, "", None, 0)
            self._write_metrics(program, None, None)
            return SharedBenchmarkLineageOutput(trend=None, n_shared=None)

        parent_d_id = self._d_id(parent)

        shared = await self._resolver.shared_benchmark(d_id, parent_d_id)
        if len(shared) < self._min_shared:
            logger.debug(
                "[SharedBenchmarkLineageStage] {} insufficient shared: {} < {}",
                program.id[:8],
                len(shared),
                self._min_shared,
            )
            self._emit_lineage_event(program.id, d_id, parent_d_id, None, len(shared))
            self._write_metrics(program, None, None)
            return SharedBenchmarkLineageOutput(trend=None, n_shared=None)

        tracker = getattr(self._resolver, "_tracker", None)
        if tracker is None:
            logger.error(
                "[SharedBenchmarkLineageStage] resolver has no _tracker attribute"
            )
            self._emit_lineage_event(program.id, d_id, parent_d_id, None, len(shared))
            self._write_metrics(program, None, None)
            return SharedBenchmarkLineageOutput(trend=None, n_shared=None)

        pairs = await tracker.get_deltas_against(d_id, parent_d_id, shared)
        if not pairs:
            logger.debug(
                "[SharedBenchmarkLineageStage] {} no delta pairs from tracker",
                program.id[:8],
            )
            self._emit_lineage_event(program.id, d_id, parent_d_id, None, len(shared))
            self._write_metrics(program, None, None)
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
        self._emit_lineage_event(program.id, d_id, parent_d_id, trend, len(shared))
        self._write_metrics(program, trend, len(shared))
        return SharedBenchmarkLineageOutput(trend=trend, n_shared=len(shared))
