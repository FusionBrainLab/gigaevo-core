"""Chain structural metrics stage.

Extracts DAG topology features from chain programs and stores them
directly in ``program.metrics``.  Used as MAP-Elites behavioral
characterization dimensions in the ``topology_3d`` algorithm config.

Runs on ALL pipeline arms (control + treatment) to avoid pipeline confounds.
"""

from __future__ import annotations

from loguru import logger

from gigaevo.evolution.scheduling.feature_extractor import ChainFeatureExtractor
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.stage_registry import StageRegistry

_EXTRACTOR = ChainFeatureExtractor()

# Keys surfaced as program.metrics for MAP-Elites behavioral characterization.
STRUCTURAL_METRIC_KEYS = (
    "dag_depth",
    "max_dependency_fan_in",
    "n_deep_retrieval",
    "n_retrievals",
)


@StageRegistry.register(
    description="Extract chain structural metrics (dag_depth, max_fan_in, n_deep_retrieval, n_retrievals)"
)
class ChainStructuralMetricsStage(Stage):
    """Extract chain topology features and store them in ``program.metrics``.

    Uses :class:`ChainFeatureExtractor` (pure regex, <1ms) to compute:

    - ``dag_depth``: longest path from root to leaf in the dependency DAG
    - ``max_dependency_fan_in``: maximum in-degree across all steps
    - ``n_deep_retrieval``: count of ``retrieve_deep`` calls (k=10)
    - ``n_retrievals``: total count of all retrieval calls (retrieve + retrieve_deep)

    These are stored directly on the program so the MAP-Elites behavior
    space can key on them.  The stage also returns a :class:`FloatDictContainer`
    for pipeline consistency.
    """

    InputsModel = VoidInput
    OutputModel = FloatDictContainer

    async def compute(self, program: Program) -> FloatDictContainer:
        features = _EXTRACTOR.extract(program)
        structural = {k: features[k] for k in STRUCTURAL_METRIC_KEYS}
        program.add_metrics(structural)
        logger.debug(
            "[{}] structural metrics: dag_depth={}, max_fan_in={}, n_deep_ret={}, n_ret={}",
            self.stage_name,
            structural["dag_depth"],
            structural["max_dependency_fan_in"],
            structural["n_deep_retrieval"],
            structural["n_retrievals"],
        )
        return FloatDictContainer(data=structural)
