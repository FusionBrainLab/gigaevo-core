from __future__ import annotations

from collections.abc import Callable

from gigaevo.entrypoint.constants import (
    DEFAULT_DAG_CONCURRENCY,
    DEFAULT_MAX_INSIGHTS,
    DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION,
    DEFAULT_SIMPLE_STAGE_TIMEOUT,
    MAX_CODE_LENGTH,
    MAX_MEMORY_MB,
    MAX_OUTPUT_SIZE,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.problems.layout import ProblemLayout
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.collector import (
    AncestorProgramIds,
    DescendantProgramIds,
    EvolutionaryStatisticsCollector,
)
from gigaevo.programs.stages.complexity import ComputeComplexityStage
from gigaevo.programs.stages.formatter import FormatterStage
from gigaevo.programs.stages.insights import InsightsStage
from gigaevo.programs.stages.insights_lineage import (
    LineagesFromAncestors,
    LineageStage,
    LineagesToDescendants,
)
from gigaevo.programs.stages.json_processing import MergeDictStage
from gigaevo.programs.stages.metrics import EnsureMetricsStage
from gigaevo.programs.stages.mutation_context import MutationContextStage
from gigaevo.programs.stages.optimization.cma import CMANumericalOptimizationStage
from gigaevo.programs.stages.optimization.optuna import (
    OptunaOptimizationStage,
    OptunaPayloadBridge,
    PayloadResolver,
)
from gigaevo.programs.stages.python_executors.execution import (
    CallFileFunction,
    CallProgramFunction,
    CallValidatorFunction,
    FetchArtifact,
    FetchMetrics,
)
from gigaevo.programs.stages.validation import ValidateCodeStage
from gigaevo.runner.dag_blueprint import DAGBlueprint

StageFactory = Callable[[], Stage]


class PipelineBuilder:
    """Mutable builder for pipeline nodes/edges/deps producing a DAGBlueprint."""

    def __init__(self, ctx: EvolutionContext, *, dag_timeout: float = 3600.0):
        self.ctx = ctx
        self._nodes: dict[str, StageFactory] = {}
        self._data_flow_edges: list[DataFlowEdge] = []
        self._deps: dict[str, list[ExecutionOrderDependency]] = {}
        self._dag_timeout: float = dag_timeout
        self._max_parallel: int = DEFAULT_DAG_CONCURRENCY

    # Stage operations - add, replace, remove
    def add_stage(self, name: str, factory: StageFactory) -> PipelineBuilder:
        self._nodes[name] = factory
        return self

    def replace_stage(self, name: str, factory: StageFactory) -> PipelineBuilder:
        self._nodes[name] = factory
        return self

    def remove_stage(self, name: str) -> PipelineBuilder:
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
    ) -> PipelineBuilder:
        """Add a data flow edge with semantic input naming."""
        self._data_flow_edges.append(
            DataFlowEdge.create(source=src, destination=dst, input_name=input_name)
        )
        return self

    def remove_data_flow_edge(self, src: str, dst: str) -> PipelineBuilder:
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
    ) -> PipelineBuilder:
        self._deps.setdefault(stage, []).append(dep)
        return self

    def remove_exec_dep(
        self, stage: str, dep: ExecutionOrderDependency
    ) -> PipelineBuilder:
        if stage in self._deps:
            self._deps[stage] = [d for d in self._deps[stage] if d != dep]
        return self

    # Set limits for the pipeline
    def set_limits(
        self, *, dag_timeout: float | None, max_parallel: int | None
    ) -> PipelineBuilder:
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

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
    ):
        super().__init__(ctx, dag_timeout=dag_timeout)
        self._stage_timeout = stage_timeout
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
        prompts_dir = self.ctx.prompts_dir
        stage_timeout = self._stage_timeout

        # ValidateCompiles
        self.add_stage(
            "ValidateCodeStage",
            lambda: ValidateCodeStage(
                max_code_length=MAX_CODE_LENGTH,
                timeout=stage_timeout,
                safe_mode=True,
            ),
        )

        # ExecuteCode: run program.code with optional data from DAG
        self.add_stage(
            "CallProgramFunction",
            lambda: CallProgramFunction(
                function_name="entrypoint",
                python_path=[problem_ctx.problem_dir.resolve()],
                timeout=stage_timeout,
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
                timeout=stage_timeout,
                max_memory_mb=MAX_MEMORY_MB,
                max_output_size=MAX_OUTPUT_SIZE,
            ),
        )

        # Extract metrics and artifact from validation result (artifact output unused for now)
        self.add_stage(
            "FetchMetrics",
            lambda: FetchMetrics(timeout=stage_timeout),
        )
        self.add_stage(
            "FetchArtifact",
            lambda: FetchArtifact(timeout=stage_timeout),
        )
        self.add_stage(
            "FormatterStage",
            lambda: FormatterStage(timeout=stage_timeout),
        )

        # Insights stages
        self.add_stage(
            "InsightsStage",
            lambda: InsightsStage(
                llm=llm_wrapper,
                task_description=task_description,
                metrics_context=metrics_context,
                max_insights=DEFAULT_MAX_INSIGHTS,
                timeout=stage_timeout,
                prompts_dir=prompts_dir,
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
                timeout=stage_timeout,
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
                timeout=stage_timeout,
            ),
        )

        self.add_stage(
            "LineageStage",
            lambda: LineageStage(
                llm=llm_wrapper,
                task_description=task_description,
                metrics_context=metrics_context,
                storage=storage,
                timeout=stage_timeout,
                prompts_dir=prompts_dir,
            ),
        )

        self.add_stage(
            "LineagesToDescendants",
            lambda: LineagesToDescendants(
                storage=storage,
                source_stage_name="LineageStage",
                timeout=stage_timeout,
            ),
        )

        self.add_stage(
            "LineagesFromAncestors",
            lambda: LineagesFromAncestors(
                storage=storage,
                source_stage_name="LineageStage",
                timeout=stage_timeout,
            ),
        )

        self.add_stage(
            "MutationContextStage",
            lambda: MutationContextStage(
                metrics_context=metrics_context,
                timeout=stage_timeout,
            ),
        )

        self.add_stage(
            "ComputeComplexityStage",
            lambda: ComputeComplexityStage(
                timeout=stage_timeout,
            ),
        )

        self.add_stage(
            "MergeMetricsStage",
            lambda: MergeDictStage[str, float](
                timeout=stage_timeout,
            ),
        )

        self.add_stage(
            "EnsureMetricsStage",
            lambda: EnsureMetricsStage(
                metrics_factory=metrics_context.get_sentinels,
                metrics_context=metrics_context,
                timeout=stage_timeout,
            ),
        )
        self.add_stage(
            "EvolutionaryStatisticsCollector",
            lambda: EvolutionaryStatisticsCollector(
                storage=storage,
                metrics_context=metrics_context,
                timeout=stage_timeout,
            ),
        )

    def _contribute_default_edges(self) -> None:
        self.add_data_flow_edge(
            "CallProgramFunction", "CallValidatorFunction", "payload"
        )
        self.add_data_flow_edge(
            "CallValidatorFunction", "FetchMetrics", "validation_result"
        )
        self.add_data_flow_edge(
            "CallValidatorFunction", "FetchArtifact", "validation_result"
        )
        self.add_data_flow_edge("FetchMetrics", "MergeMetricsStage", "first")
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
        self.add_data_flow_edge("FetchArtifact", "FormatterStage", "data")
        self.add_data_flow_edge("FormatterStage", "MutationContextStage", "formatted")

    def _contribute_default_deps(self) -> None:
        self._deps = {
            "CallProgramFunction": [
                ExecutionOrderDependency.on_success("ValidateCodeStage")
            ],
            "FetchMetrics": [
                ExecutionOrderDependency.always_after("CallValidatorFunction"),
            ],
            "FetchArtifact": [
                ExecutionOrderDependency.always_after("CallValidatorFunction"),
            ],
            "FormatterStage": [
                ExecutionOrderDependency.always_after("FetchArtifact"),
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

    def __init__(self, ctx: EvolutionContext, *, dag_timeout: float = 3600.0):
        super().__init__(ctx, dag_timeout=dag_timeout)
        self._add_context_stage_and_edges()

    def _add_context_stage_and_edges(self) -> None:
        problem_ctx = self.ctx.problem_ctx

        # AddContext stage: runs build_context from context.py to produce a dict
        self.add_stage(
            "AddContext",
            lambda: CallFileFunction(
                path=problem_ctx.problem_dir / ProblemLayout.CONTEXT_FILE,
                function_name="build_context",
                timeout=DEFAULT_SIMPLE_STAGE_TIMEOUT,
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

    # Sensible defaults – override in subclasses.
    CMA_SCORE_KEY: str = "fitness"
    CMA_SIGMA0: float = 0.2
    CMA_MAX_GENERATIONS: int = 20
    CMA_POPULATION_SIZE: int = 10
    CMA_MAX_PARALLEL: int = 10
    # Current experiment policy: CMA tunes float literals only.
    # Integer literals are left to mutation/structural evolution.
    CMA_TUNE_FLOATS_ONLY: bool = True

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        optimization_time_budget: float | None = None,
    ):
        super().__init__(ctx, dag_timeout=dag_timeout)
        self._optimization_time_budget = (
            optimization_time_budget
            if optimization_time_budget is not None
            else dag_timeout * DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION
        )
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
                timeout=DEFAULT_SIMPLE_STAGE_TIMEOUT,
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

        max_gen = extra.pop("max_generations", self.CMA_MAX_GENERATIONS)
        pop_size = extra.pop("population_size", self.CMA_POPULATION_SIZE)
        max_par = extra.pop("max_parallel", self.CMA_MAX_PARALLEL)

        budget = self._optimization_time_budget

        # Derive eval_timeout from budget if not explicitly overridden.
        n_rounds = -(-max_gen * pop_size // max_par)  # ceil division
        default_eval_to = max(30, min(300, budget * 0.9 / max(n_rounds, 1)))
        eval_to = extra.pop("eval_timeout", int(default_eval_to))

        # Stage timeout: capped to the optimization budget.
        stage_timeout = min((n_rounds + 1) * eval_to, int(budget))

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
                max_generations=max_gen,
                population_size=pop_size,
                max_parallel=max_par,
                eval_timeout=eval_to,
                skip_integers=extra.pop("skip_integers", self.CMA_TUNE_FLOATS_ONLY),
                update_program_code=True,
                timeout=stage_timeout,
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


class OptunaOptPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline + LLM-guided Optuna hyperparameter optimisation.

    Inherits :class:`DefaultPipelineBuilder` and inserts an
    :class:`OptunaOptimizationStage` between ``ValidateCodeStage``
    and ``CallProgramFunction``.  If the problem provides a ``context.py``
    the ``AddContext`` stage is wired automatically.

    Execution order::

        ValidateCodeStage ─(success)─► OptunaOptStage ─(always)─► CallProgramFunction
        AddContext* ───────(always)──►                 ─(data)──►
        (* only when context.py exists)

    If Optuna fails, the program still runs with the original code.

    Override ``_optuna_stage_kwargs`` in a subclass to tweak hyper-parameters.
    """

    # Sensible defaults – override in subclasses.
    OPTUNA_SCORE_KEY: str | None = None  # None -> auto-detect from problem_ctx
    OPTUNA_MAX_PARALLEL: int = 10

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 7200.0,
        optimization_time_budget: float | None = None,
    ):
        super().__init__(ctx, dag_timeout=dag_timeout)
        self._optimization_time_budget = (
            optimization_time_budget
            if optimization_time_budget is not None
            else dag_timeout * DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION
        )
        has_context = ctx.problem_ctx.is_contextual
        if has_context:
            self._add_context_stage_and_edges()
        self._add_optuna_optimization(has_context=has_context)

    def _add_context_stage_and_edges(self) -> None:
        """Add the AddContext stage (same as ContextPipelineBuilder)."""
        problem_ctx = self.ctx.problem_ctx
        self.add_stage(
            "AddContext",
            lambda: CallFileFunction(
                path=problem_ctx.problem_dir / ProblemLayout.CONTEXT_FILE,
                function_name="build_context",
                timeout=DEFAULT_SIMPLE_STAGE_TIMEOUT,
            ),
        )
        self.add_data_flow_edge("AddContext", "CallProgramFunction", "context")
        self.add_data_flow_edge("AddContext", "CallValidatorFunction", "context")

    def _optuna_stage_kwargs(self) -> dict:
        """Return extra kwargs forwarded to :class:`OptunaOptimizationStage`.

        Override in a subclass to customise Optuna hyper-parameters without
        rewriting the whole pipeline.
        """
        return {}

    def _add_optuna_optimization(self, *, has_context: bool) -> None:
        problem_ctx = self.ctx.problem_ctx
        llm_wrapper = self.ctx.llm_wrapper
        metrics_ctx = problem_ctx.metrics_context
        primary_spec = metrics_ctx.get_primary_spec()

        validator_path = problem_ctx.problem_dir / "validate2.py"
        task_description = problem_ctx.task_description

        extra = self._optuna_stage_kwargs()

        max_par = extra.pop("max_parallel", self.OPTUNA_MAX_PARALLEL)
        score_key = extra.pop(
            "score_key", self.OPTUNA_SCORE_KEY or metrics_ctx.get_primary_key()
        )
        minimize = extra.pop("minimize", not primary_spec.higher_is_better)

        budget = self._optimization_time_budget

        # Pass None for eval_timeout and n_trials so the stage auto-computes
        # them from the optimization budget + baseline runtime.
        # Explicit overrides from _optuna_stage_kwargs() still work.
        n_trials = extra.pop("n_trials", None)
        eval_to = extra.pop("eval_timeout", None)

        # Stage timeout = the full optimization budget
        stage_timeout = int(budget)

        self.add_stage(
            "OptunaOptStage",
            lambda: OptunaOptimizationStage(
                llm=llm_wrapper,
                validator_path=validator_path,
                score_key=score_key,
                function_name="entrypoint",
                validator_fn="validate",
                python_path=[problem_ctx.problem_dir.resolve()],
                minimize=minimize,
                n_trials=n_trials,
                max_parallel=max_par,
                eval_timeout=eval_to,
                update_program_code=True,
                task_description=task_description,
                optimization_time_budget=budget,
                timeout=stage_timeout,
                max_memory_mb=MAX_MEMORY_MB,
                **extra,
            ),
        )

        # Optuna runs after validation succeeds
        self.add_exec_dep(
            "OptunaOptStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )

        # If context exists, wire it into Optuna and wait for it
        if has_context:
            self.add_data_flow_edge("AddContext", "OptunaOptStage", "context")
            self.add_exec_dep(
                "OptunaOptStage",
                ExecutionOrderDependency.always_after("AddContext"),
            )

        # -- Bypass: skip CallProgramFunction when Optuna succeeds --------
        #
        # OptunaPayloadBridge extracts best_program_output from Optuna.
        # PayloadResolver picks whichever payload source completed.
        # CallValidatorFunction always runs (single source of truth).
        self.add_stage(
            "OptunaPayloadBridge",
            lambda: OptunaPayloadBridge(timeout=DEFAULT_SIMPLE_STAGE_TIMEOUT),
        )
        self.add_stage(
            "PayloadResolver",
            lambda: PayloadResolver(timeout=DEFAULT_SIMPLE_STAGE_TIMEOUT),
        )

        # Data flow: Optuna → bridge → resolver → validator
        self.add_data_flow_edge(
            "OptunaOptStage", "OptunaPayloadBridge", "optuna_output"
        )
        self.add_data_flow_edge(
            "OptunaPayloadBridge", "PayloadResolver", "optuna_payload"
        )
        self.add_data_flow_edge(
            "CallProgramFunction", "PayloadResolver", "program_payload"
        )

        # Replace the default CallProgramFunction → CallValidatorFunction edge
        # with PayloadResolver → CallValidatorFunction.
        self.remove_data_flow_edge("CallProgramFunction", "CallValidatorFunction")
        self.add_data_flow_edge("PayloadResolver", "CallValidatorFunction", "payload")

        # CallProgramFunction only runs when Optuna fails (fallback path).
        self.add_exec_dep(
            "CallProgramFunction",
            ExecutionOrderDependency.on_failure("OptunaOptStage"),
        )


class CustomPipelineBuilder(PipelineBuilder):
    """Starts with an empty pipeline. Users compose everything explicitly."""
