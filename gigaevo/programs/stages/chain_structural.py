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
# The semantic keys parse json_document chain specs (zeros on any other
# program format) — the chains_bd3d behavior space keys on them.
SEMANTIC_CHAIN_METRIC_KEYS = (
    "hop_depth",
    "passages_fetched",
    "instr_chars",
)
STRUCTURAL_METRIC_KEYS = (
    "dag_depth",
    "max_dependency_fan_in",
    "n_deep_retrieval",
    "n_retrievals",
    *SEMANTIC_CHAIN_METRIC_KEYS,
)


@StageRegistry.register(
    description="Extract chain structural metrics (dag_depth, max_fan_in, n_deep_retrieval, n_retrievals, hop_depth, passages_fetched, instr_chars)"
)
class ChainStructuralMetricsStage(Stage):
    """Extract chain topology features and store them in ``program.metrics``.

    Uses :class:`ChainFeatureExtractor` (<1ms) to compute:

    - ``dag_depth``: longest path from root to leaf in the dependency DAG
    - ``max_dependency_fan_in``: maximum in-degree across all steps
    - ``n_deep_retrieval``: count of ``retrieve_deep`` calls (k=10)
    - ``n_retrievals``: total count of all retrieval calls (retrieve + retrieve_deep)
    - ``hop_depth``: retrieve→reason→retrieve chain length (json_document specs)
    - ``passages_fetched``: k-weighted evidence budget (json_document specs)
    - ``instr_chars``: guidance-field + system_prompt characters (json_document specs)

    These are stored directly on the program so the MAP-Elites behavior
    space can key on them (``topology_3d_ret``, ``chains_bd3d``).  The stage
    also returns a :class:`FloatDictContainer` for pipeline consistency.
    """

    InputsModel = VoidInput
    OutputModel = FloatDictContainer

    async def compute(self, program: Program) -> FloatDictContainer:
        features = _EXTRACTOR.extract(program)
        structural = {k: features[k] for k in STRUCTURAL_METRIC_KEYS}
        program.add_metrics(structural)
        logger.debug("[{}] structural metrics: {}", self.stage_name, structural)
        return FloatDictContainer(data=structural)
