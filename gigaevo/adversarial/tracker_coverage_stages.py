"""Tracker-coverage BD axis stages for v3 asymmetric adversarial evolution.

Writes tracker_coverage_count metrics (inverted-index cardinality) into D and G
metrics dicts for use as BD axes. These stages run after DGTrackerStage (pairs
recorded) and contribute to the candidate dict consumed by EnsureMetricsStage
via a downstream ``MergeDictStage`` (``MergeCoverageMetricsStage``).

Output model: :class:`FloatDictContainer` so the emitted dict flows through a
data_flow_edge into the merge stage. The stage still writes to
``program.metrics`` as a convenience for downstream consumers that inspect the
program object directly, but the authoritative path for ``wins`` to reach
``EnsureMetricsStage`` is the merge-chain output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import FloatDictContainer

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker
    from gigaevo.programs.program import Program


class ComputeDWinsCountStage(Stage):
    """Write wins (D career wins) for D's BD axis y (§3.2).

    Queries DGImprovementTracker for count_g_beaten_by_d(program_id),
    which is the cardinality of the dg_d_wins:{program_id} Redis SET.
    This is the number of distinct G programs this D has beaten (positive delta).
    """

    InputsModel = VoidInput
    OutputModel = FloatDictContainer
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        dg_tracker: DGImprovementTracker,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tracker = dg_tracker

    async def compute(self, program: Program) -> FloatDictContainer:
        count = await self._tracker.count_g_beaten_by_d(program.id)
        program.metrics["wins"] = count
        logger.info(
            "[METRIC_EMIT] {}",
            json.dumps(
                {
                    "event": "METRIC_EMIT",
                    "program_id": program.id,
                    "metric_name": "wins",
                    "metric_value": int(count),
                    "source": "ComputeDWinsCountStage",
                }
            ),
        )
        return FloatDictContainer(data={"wins": float(count)})


class ComputeGResistedCountStage(Stage):
    """Write wins (G career resisted) for G's BD axis y (§3.1).

    Queries DGImprovementTracker for count_d_resisted_by_g(program_id),
    which is the cardinality of the dg_g_resisted:{program_id} Redis SET.
    This is the number of distinct D programs this G has resisted (non-positive delta).
    """

    InputsModel = VoidInput
    OutputModel = FloatDictContainer
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        dg_tracker: DGImprovementTracker,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tracker = dg_tracker

    async def compute(self, program: Program) -> FloatDictContainer:
        count = await self._tracker.count_d_resisted_by_g(program.id)
        program.metrics["wins"] = count
        logger.info(
            "[METRIC_EMIT] {}",
            json.dumps(
                {
                    "event": "METRIC_EMIT",
                    "program_id": program.id,
                    "metric_name": "wins",
                    "metric_value": int(count),
                    "source": "ComputeGResistedCountStage",
                }
            ),
        )
        return FloatDictContainer(data={"wins": float(count)})
