"""Tracker-coverage BD axis stages for v3 asymmetric adversarial evolution.

Writes tracker_coverage_count metrics (inverted-index cardinality) into D and G
metrics dicts for use as BD axes. These stages run after DGTrackerStage (pairs
recorded) and before EnsureMetricsStage (metrics available for binning).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gigaevo.programs.core_types import VoidOutput
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker
    from gigaevo.programs.program import Program


class ComputeDWinsCountStage(Stage):
    """Write tracker_coverage_count (D career wins) for D's BD axis y.

    Queries DGImprovementTracker for count_g_beaten_by_d(program_id),
    which is the cardinality of the dg_d_wins:{program_id} Redis SET.
    This is the number of distinct G programs this D has beaten (positive delta).
    """

    InputsModel = VoidOutput
    OutputModel = VoidOutput
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        dg_tracker: DGImprovementTracker,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tracker = dg_tracker

    async def compute(self, program: Program) -> None:
        """Fetch and store tracker_coverage_count in program.metrics."""
        count = await self._tracker.count_g_beaten_by_d(program.id)
        program.metrics["tracker_coverage_count"] = count


class ComputeGResistedCountStage(Stage):
    """Write g_tracker_coverage_count (G career resisted) for G's BD fallback axis.

    Queries DGImprovementTracker for count_d_resisted_by_g(program_id),
    which is the cardinality of the dg_g_resisted:{program_id} Redis SET.
    This is the number of distinct D programs this G has resisted (non-positive delta).
    """

    InputsModel = VoidOutput
    OutputModel = VoidOutput
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        dg_tracker: DGImprovementTracker,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tracker = dg_tracker

    async def compute(self, program: Program) -> None:
        """Fetch and store g_tracker_coverage_count in program.metrics."""
        count = await self._tracker.count_d_resisted_by_g(program.id)
        program.metrics["g_tracker_coverage_count"] = count
