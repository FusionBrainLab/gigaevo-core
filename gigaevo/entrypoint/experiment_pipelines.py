"""Experiment-specific pipeline builders.

Each builder extends :class:`DefaultPipelineBuilder` with additional stages
needed for a particular experiment.  Referenced via Hydra pipeline YAML configs
(e.g. ``config/pipeline/structural_metrics.yaml``).
"""

from __future__ import annotations

from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.dag.automata import ExecutionOrderDependency
from gigaevo.programs.stages.chain_structural import ChainStructuralMetricsStage


class StructuralMetricsPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline + chain structural metrics for MAP-Elites topology BC.

    Adds :class:`ChainStructuralMetricsStage` which extracts ``dag_depth``,
    ``max_dependency_fan_in``, and ``n_deep_retrieval`` from program code and
    stores them in ``program.metrics``.  These are used as behavioral
    characterization dimensions in the ``topology_3d`` algorithm config.

    The stage runs on ALL arms (control + treatment) to avoid pipeline
    confounds — control's behavior space simply ignores the extra keys.
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float | None = None,
    ):
        # stage_timeout default handled by parent
        kwargs: dict = {"dag_timeout": dag_timeout}
        if stage_timeout is not None:
            kwargs["stage_timeout"] = stage_timeout
        super().__init__(ctx, **kwargs)
        self._add_structural_metrics_stage()

    def _add_structural_metrics_stage(self) -> None:
        stage_timeout = self._stage_timeout

        self.add_stage(
            "ChainStructuralMetricsStage",
            lambda: ChainStructuralMetricsStage(timeout=stage_timeout),
        )

        # Run after ValidateCodeStage succeeds (needs valid code to parse).
        self.add_exec_dep(
            "ChainStructuralMetricsStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )
