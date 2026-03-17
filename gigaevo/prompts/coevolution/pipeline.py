"""Pipeline builder for the prompt evolution problem.

Uses PromptExecutionStage + PromptFitnessStage instead of the standard
validate.py / CallValidatorFunction path.
"""

from __future__ import annotations

from gigaevo.entrypoint.constants import (
    DEFAULT_DAG_CONCURRENCY,
    DEFAULT_MAX_INSIGHTS,
    DEFAULT_SIMPLE_STAGE_TIMEOUT,
    MAX_CODE_LENGTH,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.collector import (
    AncestorProgramIds,
    DescendantProgramIds,
    EvolutionaryStatisticsCollector,
)
from gigaevo.programs.stages.complexity import ComputeComplexityStage
from gigaevo.programs.stages.insights import InsightsStage
from gigaevo.programs.stages.insights_lineage import (
    LineagesFromAncestors,
    LineageStage,
    LineagesToDescendants,
)
from gigaevo.programs.stages.json_processing import MergeDictStage
from gigaevo.programs.stages.metrics import EnsureMetricsStage
from gigaevo.programs.stages.mutation_context import MutationContextStage
from gigaevo.programs.stages.validation import ValidateCodeStage
from gigaevo.prompts.coevolution.stages import PromptExecutionStage, PromptFitnessStage
from gigaevo.prompts.coevolution.stats import PromptStatsProvider
from gigaevo.runner.dag_blueprint import DAGBlueprint


class PromptEvolutionPipelineBuilder:
    """Builds the DAG pipeline for the prompt_evolution problem.

    Uses PromptExecutionStage + PromptFitnessStage instead of:
      - ValidateCodeStage (syntax check is still done)
      - CallProgramFunction
      - CallValidatorFunction / FetchMetrics
      - FormatterStage

    The metrics (fitness, is_valid, prompt_length) come directly from
    PromptFitnessStage which queries the main run's Redis for outcome stats.

    Args:
        ctx: EvolutionContext (llm, storage, problem, prompts_dir)
        stats_provider: PromptStatsProvider injected for fitness computation
        stage_timeout: Per-stage timeout in seconds
        dag_timeout: Overall DAG timeout in seconds
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        stats_provider: PromptStatsProvider,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        dag_timeout: float = 3600.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 3.0,
        required_prefix: str | None = None,
    ):
        self._ctx = ctx
        self._stats_provider = stats_provider
        self._stage_timeout = stage_timeout
        self._dag_timeout = dag_timeout
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._required_prefix = required_prefix

    def build_blueprint(self) -> DAGBlueprint:
        ctx = self._ctx
        st = self._stage_timeout
        metrics_context = ctx.problem_ctx.metrics_context
        llm_wrapper = ctx.llm_wrapper
        storage = ctx.storage
        task_description = ctx.problem_ctx.task_description
        prompts_dir = ctx.prompts_dir
        stats_provider = self._stats_provider

        nodes = {
            "ValidateCodeStage": lambda: ValidateCodeStage(
                max_code_length=MAX_CODE_LENGTH,
                timeout=st,
                safe_mode=True,
            ),
            "PromptExecutionStage": lambda: PromptExecutionStage(
                required_prefix=self._required_prefix, timeout=st
            ),
            "PromptFitnessStage": lambda: PromptFitnessStage(
                stats_provider=stats_provider,
                prior_alpha=self._prior_alpha,
                prior_beta=self._prior_beta,
                timeout=st,
            ),
            "ComputeComplexityStage": lambda: ComputeComplexityStage(timeout=st),
            "MergeMetricsStage": lambda: MergeDictStage[str, float](timeout=st),
            "EnsureMetricsStage": lambda: EnsureMetricsStage(
                metrics_factory=metrics_context.get_sentinels,
                metrics_context=metrics_context,
                timeout=st,
            ),
            "InsightsStage": lambda: InsightsStage(
                llm=llm_wrapper,
                task_description=task_description,
                metrics_context=metrics_context,
                max_insights=DEFAULT_MAX_INSIGHTS,
                timeout=st,
                prompts_dir=prompts_dir,
            ),
            "DescendantProgramIds": lambda: DescendantProgramIds(
                storage=storage,
                selector=AncestrySelector(
                    metrics_context=metrics_context,
                    strategy="best_fitness",
                    max_selected=1,
                ),
                timeout=st,
            ),
            "AncestorProgramIds": lambda: AncestorProgramIds(
                storage=storage,
                selector=AncestrySelector(
                    metrics_context=metrics_context,
                    strategy="best_fitness",
                    max_selected=2,
                ),
                timeout=st,
            ),
            "LineageStage": lambda: LineageStage(
                llm=llm_wrapper,
                task_description=task_description,
                metrics_context=metrics_context,
                storage=storage,
                timeout=st,
                prompts_dir=prompts_dir,
            ),
            "LineagesToDescendants": lambda: LineagesToDescendants(
                storage=storage,
                source_stage_name="LineageStage",
                timeout=st,
            ),
            "LineagesFromAncestors": lambda: LineagesFromAncestors(
                storage=storage,
                source_stage_name="LineageStage",
                timeout=st,
            ),
            "MutationContextStage": lambda: MutationContextStage(
                metrics_context=metrics_context,
                timeout=st,
            ),
            "EvolutionaryStatisticsCollector": lambda: EvolutionaryStatisticsCollector(
                storage=storage,
                metrics_context=metrics_context,
                timeout=st,
            ),
        }

        data_flow_edges = [
            DataFlowEdge.create(
                source="PromptExecutionStage",
                destination="PromptFitnessStage",
                input_name="execution_output",
            ),
            DataFlowEdge.create(
                source="PromptFitnessStage",
                destination="MergeMetricsStage",
                input_name="first",
            ),
            DataFlowEdge.create(
                source="ComputeComplexityStage",
                destination="MergeMetricsStage",
                input_name="second",
            ),
            DataFlowEdge.create(
                source="MergeMetricsStage",
                destination="EnsureMetricsStage",
                input_name="candidate",
            ),
            DataFlowEdge.create(
                source="EnsureMetricsStage",
                destination="MutationContextStage",
                input_name="metrics",
            ),
            DataFlowEdge.create(
                source="InsightsStage",
                destination="MutationContextStage",
                input_name="insights",
            ),
            DataFlowEdge.create(
                source="DescendantProgramIds",
                destination="LineagesToDescendants",
                input_name="descendant_ids",
            ),
            DataFlowEdge.create(
                source="AncestorProgramIds",
                destination="LineagesFromAncestors",
                input_name="ancestor_ids",
            ),
            DataFlowEdge.create(
                source="LineagesToDescendants",
                destination="MutationContextStage",
                input_name="lineage_descendants",
            ),
            DataFlowEdge.create(
                source="LineagesFromAncestors",
                destination="MutationContextStage",
                input_name="lineage_ancestors",
            ),
            DataFlowEdge.create(
                source="EvolutionaryStatisticsCollector",
                destination="MutationContextStage",
                input_name="evolutionary_statistics",
            ),
        ]

        exec_deps = {
            "PromptExecutionStage": [
                ExecutionOrderDependency.on_success("ValidateCodeStage"),
            ],
            "PromptFitnessStage": [
                ExecutionOrderDependency.always_after("PromptExecutionStage"),
            ],
            "InsightsStage": [
                ExecutionOrderDependency.always_after("EnsureMetricsStage"),
            ],
            "LineageStage": [
                ExecutionOrderDependency.always_after("EnsureMetricsStage"),
            ],
            "LineagesToDescendants": [
                ExecutionOrderDependency.always_after("LineageStage"),
            ],
            "LineagesFromAncestors": [
                ExecutionOrderDependency.always_after("LineageStage"),
            ],
            "EvolutionaryStatisticsCollector": [
                ExecutionOrderDependency.always_after("EnsureMetricsStage"),
            ],
        }

        return DAGBlueprint(
            nodes=nodes,
            data_flow_edges=data_flow_edges,
            exec_order_deps=exec_deps,
            dag_timeout=self._dag_timeout,
            max_parallel_stages=DEFAULT_DAG_CONCURRENCY,
        )
