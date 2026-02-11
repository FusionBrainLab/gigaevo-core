from __future__ import annotations

from typing import Callable

from gigaevo.entrypoint.constants import (
    DEFAULT_DAG_CONCURRENCY,
    DEFAULT_DAG_TIMEOUT,
    DEFAULT_MAX_INSIGHTS,
    DEFAULT_STAGE_TIMEOUT,
    MAX_CODE_LENGTH,
    MAX_MEMORY_MB,
    MAX_OUTPUT_SIZE,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.problems.layout import ProblemLayout
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cma_optimization import CMANumericalOptimizationStage
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
from gigaevo.programs.stages.python_executors.execution import (
    CallFileFunction,
    CallProgramFunction,
    CallValidatorFunction,
)
from gigaevo.programs.stages.validation import ValidateCodeStage
from gigaevo.runner.dag_blueprint import DAGBlueprint

StageFactory = Callable[[], Stage]


class PipelineBuilder:
    """Mutable builder for pipeline nodes/edges/deps producing a DAGBlueprint."""

    def __init__(self, ctx: EvolutionContext):
        self.ctx = ctx
        self._nodes: dict[str, StageFactory] = {}
        self._data_flow_edges: list[DataFlowEdge] = []
        self._deps: dict[str, list[ExecutionOrderDependency]] = {}
        self._dag_timeout: float = DEFAULT_DAG_TIMEOUT
        self._max_parallel: int = DEFAULT_DAG_CONCURRENCY

    # Stage operations - add, replace, remove
    def add_stage(self, name: str, factory: StageFactory) -> "PipelineBuilder":
        self._nodes[name] = factory
        return self

    def replace_stage(self, name: str, factory: StageFactory) -> "PipelineBuilder":
        self._nodes[name] = factory
        return self

    def remove_stage(self, name: str) -> "PipelineBuilder":
        self._nodes.pop(name, None)
        self._data_flow_edges = [
            edge
            for edge in self._data_flow_edges
            if edge.source_stage != name and edge.destination_stage != name
        ]
        self._deps.pop(name, None)
        for stage, deps in list(self._deps.items()):
            self._deps[stage] = [d for d in deps if d.stage_name != name]
        return self

    # Data flow operations - add, remove
    def add_data_flow_edge(
        self, src: str, dst: str, input_name: str
    ) -> "PipelineBuilder":
        """Add a data flow edge with semantic input naming."""
        self._data_flow_edges.append(
            DataFlowEdge.create(source=src, destination=dst, input_name=input_name)
        )
        return self

    def remove_data_flow_edge(self, src: str, dst: str) -> "PipelineBuilder":
        """Remove a data flow edge."""
        self._data_flow_edges = [
            e
            for e in self._data_flow_edges
            if not (e.source_stage == src and e.destination_stage == dst)
        ]
        return self

    # Execution order dependency operations - add, remove
    def add_exec_dep(
        self, stage: str, dep: ExecutionOrderDependency
    ) -> "PipelineBuilder":
        self._deps.setdefault(stage, []).append(dep)
        return self

    def remove_exec_dep(
        self, stage: str, dep: ExecutionOrderDependency
    ) -> "PipelineBuilder":
        if stage in self._deps:
            self._deps[stage] = [d for d in self._deps[stage] if d != dep]
        return self

    # Set limits for the pipeline
    def set_limits(
        self, *, dag_timeout: float | None, max_parallel: int | None
    ) -> "PipelineBuilder":
        if dag_timeout is not None:
            self._dag_timeout = dag_timeout
        if max_parallel is not None:
            self._max_parallel = max_parallel
        return self

    # Build the pipeline blueprint
    def build_blueprint(self) -> DAGBlueprint:
        return DAGBlueprint(
            nodes=self._nodes,
            data_flow_edges=self._data_flow_edges,
            exec_order_deps=self._deps or None,
            dag_timeout=self._dag_timeout,
            max_parallel_stages=self._max_parallel,
        )


class DefaultPipelineBuilder(PipelineBuilder):
    """Recreates the current default pipeline (no context added)."""

    def __init__(self, ctx: EvolutionContext):
        super().__init__(ctx)
        self._contribute_default_nodes()
        self._contribute_default_edges()
        self._contribute_default_deps()

    def _contribute_default_nodes(self) -> None:
        # Context is available for future wiring
        metrics_context = self.ctx.problem_ctx.metrics_context
        problem_ctx = self.ctx.problem_ctx
        llm_wrapper = self.ctx.llm_wrapper
        storage = self.ctx.storage
        task_description = self.ctx.problem_ctx.task_description

        # ValidateCompiles
        self.add_stage(
            "ValidateCodeStage",
            lambda: ValidateCodeStage(
                max_code_length=MAX_CODE_LENGTH,
                timeout=DEFAULT_STAGE_TIMEOUT,
                safe_mode=True,
            ),
        )

        # ExecuteCode: run program.code with optional data from DAG
        self.add_stage(
            "CallProgramFunction",
            lambda: CallProgramFunction(
                function_name="entrypoint",
                python_path=[problem_ctx.problem_dir.resolve()],
                timeout=DEFAULT_STAGE_TIMEOUT,
                max_memory_mb=MAX_MEMORY_MB,
                max_output_size=MAX_OUTPUT_SIZE,
            ),
        )

        # RunValidation
        validator_path = problem_ctx.problem_dir / "validate.py"
        self.add_stage(
            "CallValidatorFunction",
            lambda: CallValidatorFunction(
                path=validator_path,
                function_name="validate",
                timeout=DEFAULT_STAGE_TIMEOUT,
                max_memory_mb=MAX_MEMORY_MB,
                max_output_size=MAX_OUTPUT_SIZE,
            ),
        )

        # Insights stages
        self.add_stage(
            "InsightsStage",
            lambda: InsightsStage(
                llm=llm_wrapper,
                task_description=task_description,
                metrics_context=metrics_context,
                max_insights=DEFAULT_MAX_INSIGHTS,
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_stage(
            "DescendantProgramIds",
            lambda: DescendantProgramIds(
                storage=storage,
                selector=AncestrySelector(
                    metrics_context=metrics_context,
                    strategy="best_fitness",
                    max_selected=1,
                ),
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )
        self.add_stage(
            "AncestorProgramIds",
            lambda: AncestorProgramIds(
                storage=storage,
                selector=AncestrySelector(
                    metrics_context=metrics_context,
                    strategy="best_fitness",
                    max_selected=2,
                ),
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_stage(
            "LineageStage",
            lambda: LineageStage(
                llm=llm_wrapper,
                task_description=task_description,
                metrics_context=metrics_context,
                storage=storage,
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_stage(
            "LineagesToDescendants",
            lambda: LineagesToDescendants(
                storage=storage,
                source_stage_name="LineageStage",
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_stage(
            "LineagesFromAncestors",
            lambda: LineagesFromAncestors(
                storage=storage,
                source_stage_name="LineageStage",
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_stage(
            "MutationContextStage",
            lambda: MutationContextStage(
                metrics_context=metrics_context,
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_stage(
            "ComputeComplexityStage",
            lambda: ComputeComplexityStage(
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_stage(
            "MergeMetricsStage",
            lambda: MergeDictStage[str, float](
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_stage(
            "EnsureMetricsStage",
            lambda: EnsureMetricsStage(
                metrics_factory=metrics_context.get_sentinels,
                metrics_context=metrics_context,
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )
        self.add_stage(
            "EvolutionaryStatisticsCollector",
            lambda: EvolutionaryStatisticsCollector(
                storage=storage,
                metrics_context=metrics_context,
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

    def _contribute_default_edges(self) -> None:
        self.add_data_flow_edge(
            "CallProgramFunction", "CallValidatorFunction", "payload"
        )
        self.add_data_flow_edge("CallValidatorFunction", "MergeMetricsStage", "first")
        self.add_data_flow_edge("ComputeComplexityStage", "MergeMetricsStage", "second")
        self.add_data_flow_edge("MergeMetricsStage", "EnsureMetricsStage", "candidate")
        self.add_data_flow_edge("EnsureMetricsStage", "MutationContextStage", "metrics")
        self.add_data_flow_edge("InsightsStage", "MutationContextStage", "insights")
        self.add_data_flow_edge(
            "DescendantProgramIds", "LineagesToDescendants", "descendant_ids"
        )
        self.add_data_flow_edge(
            "AncestorProgramIds", "LineagesFromAncestors", "ancestor_ids"
        )
        self.add_data_flow_edge(
            "LineagesToDescendants", "MutationContextStage", "lineage_descendants"
        )
        self.add_data_flow_edge(
            "LineagesFromAncestors", "MutationContextStage", "lineage_ancestors"
        )
        self.add_data_flow_edge(
            "EvolutionaryStatisticsCollector",
            "MutationContextStage",
            "evolutionary_statistics",
        )

    def _contribute_default_deps(self) -> None:
        self._deps = {
            "CallProgramFunction": [
                ExecutionOrderDependency.on_success("ValidateCodeStage")
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


class ContextPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline with AddContext stage and wiring enabled."""

    def __init__(self, ctx: EvolutionContext):
        super().__init__(ctx)
        self._add_context_stage_and_edges()

    def _add_context_stage_and_edges(self) -> None:
        problem_ctx = self.ctx.problem_ctx

        # AddContext stage: runs build_context from context.py to produce a dict
        self.add_stage(
            "AddContext",
            lambda: CallFileFunction(
                path=problem_ctx.problem_dir / ProblemLayout.CONTEXT_FILE,
                function_name="build_context",
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )

        self.add_data_flow_edge("AddContext", "CallProgramFunction", "context")
        self.add_data_flow_edge("AddContext", "CallValidatorFunction", "context")


class CMAOptPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline + CMA-ES numerical constant optimisation.

    Inherits :class:`DefaultPipelineBuilder` and inserts a
    :class:`CMANumericalOptimizationStage` between ``ValidateCodeStage``
    and ``CallProgramFunction``.  If the problem provides a ``context.py``
    the ``AddContext`` stage is wired automatically (same as
    :class:`ContextPipelineBuilder`).

    Execution order::

        ValidateCodeStage ─(success)─► CMAOptStage ─(always)─► CallProgramFunction
        AddContext* ───────(always)──►              ─(data)──►
        (* only when context.py exists)

    If CMA fails, the program still runs with the original code.

    Override ``_cma_stage_kwargs`` in a subclass to tweak hyper-parameters.
    """

    # Sensible defaults for quick experiments – override in subclasses.
    CMA_SCORE_KEY: str = "fitness"
    CMA_SIGMA0: float = 0.2
    CMA_MAX_GENERATIONS: int = 20
    CMA_POPULATION_SIZE: int = 10
    CMA_MAX_PARALLEL: int = 10
    CMA_EVAL_TIMEOUT: int = 30
    # Current experiment policy: CMA tunes float literals only.
    # Integer literals are left to mutation/structural evolution.
    CMA_TUNE_FLOATS_ONLY: bool = True

    def __init__(self, ctx: EvolutionContext):
        super().__init__(ctx)
        has_context = ctx.problem_ctx.is_contextual
        if has_context:
            self._add_context_stage_and_edges()
        self._add_cma_optimization(has_context=has_context)

    def _add_context_stage_and_edges(self) -> None:
        """Add the AddContext stage (same as ContextPipelineBuilder)."""
        problem_ctx = self.ctx.problem_ctx
        self.add_stage(
            "AddContext",
            lambda: CallFileFunction(
                path=problem_ctx.problem_dir / ProblemLayout.CONTEXT_FILE,
                function_name="build_context",
                timeout=DEFAULT_STAGE_TIMEOUT,
            ),
        )
        self.add_data_flow_edge("AddContext", "CallProgramFunction", "context")
        self.add_data_flow_edge("AddContext", "CallValidatorFunction", "context")

    def _cma_stage_kwargs(self) -> dict:
        """Return extra kwargs forwarded to :class:`CMANumericalOptimizationStage`.

        Override in a subclass to customise CMA hyper-parameters without
        rewriting the whole pipeline.
        """
        return {}

    def _add_cma_optimization(self, *, has_context: bool) -> None:
        problem_ctx = self.ctx.problem_ctx
        validator_path = problem_ctx.problem_dir / "validate.py"

        extra = self._cma_stage_kwargs()

        self.add_stage(
            "CMAOptStage",
            lambda: CMANumericalOptimizationStage(
                validator_path=validator_path,
                score_key=extra.pop("score_key", self.CMA_SCORE_KEY),
                function_name="entrypoint",
                validator_fn="validate",
                python_path=[problem_ctx.problem_dir.resolve()],
                minimize=False,
                sigma0=extra.pop("sigma0", self.CMA_SIGMA0),
                max_generations=extra.pop("max_generations", self.CMA_MAX_GENERATIONS),
                population_size=extra.pop("population_size", self.CMA_POPULATION_SIZE),
                max_parallel=extra.pop("max_parallel", self.CMA_MAX_PARALLEL),
                eval_timeout=extra.pop("eval_timeout", self.CMA_EVAL_TIMEOUT),
                skip_integers=extra.pop("skip_integers", self.CMA_TUNE_FLOATS_ONLY),
                update_program_code=True,
                timeout=4 * DEFAULT_STAGE_TIMEOUT,
                max_memory_mb=MAX_MEMORY_MB,
                **extra,
            ),
        )

        # CMA runs after validation succeeds
        self.add_exec_dep(
            "CMAOptStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )

        # If context exists, wire it into CMA and wait for it
        if has_context:
            self.add_data_flow_edge("AddContext", "CMAOptStage", "context")
            self.add_exec_dep(
                "CMAOptStage",
                ExecutionOrderDependency.always_after("AddContext"),
            )

        # Program execution waits for CMA (but runs even if CMA fails)
        self.add_exec_dep(
            "CallProgramFunction",
            ExecutionOrderDependency.always_after("CMAOptStage"),
        )


class CustomPipelineBuilder(PipelineBuilder):
    """Starts with an empty pipeline. Users compose everything explicitly."""
