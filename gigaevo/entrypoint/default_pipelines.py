from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import yaml

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
from gigaevo.programs.stages.archive_gate import ArchivePotentialGateStage
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.chain_structural import ChainStructuralMetricsStage
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
from gigaevo.programs.stages.python_executors.execution import (
    CallFileFunction,
    CallProgramFunction,
    CallValidatorFunction,
    FetchArtifact,
    FetchMetrics,
)
from gigaevo.programs.stages.runtime_metrics import RuntimeFitnessStage
from gigaevo.programs.stages.validation import ValidateCodeStage
from gigaevo.runner.dag_blueprint import DAGBlueprint

StageFactory = Callable[[], Stage]


class PipelineFeature:
    """Self-contained contributor to a mutable pipeline blueprint.

    A feature owns one coherent slice of the DAG: the stages it adds, the
    data-flow edges between those stages and the surrounding graph, and any
    execution-order dependencies required for correctness. Builders compose
    features instead of hand-editing one monolithic stage graph.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def apply(self, builder: PipelineBuilder) -> None:
        raise NotImplementedError


class PipelineBuilder:
    """Mutable builder for pipeline nodes/edges/deps producing a DAGBlueprint."""

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        max_parallel: int | None = None,
    ):
        self.ctx = ctx
        self._nodes: dict[str, StageFactory] = {}
        self._data_flow_edges: list[DataFlowEdge] = []
        self._deps: dict[str, list[ExecutionOrderDependency]] = {}
        self._dag_timeout: float = dag_timeout
        self._max_parallel: int = (
            max_parallel if max_parallel is not None else DEFAULT_DAG_CONCURRENCY
        )
        self._stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT
        self._max_insights: int = DEFAULT_MAX_INSIGHTS
        self._max_code_length: int = MAX_CODE_LENGTH
        self._archive_gate_enabled: bool = False
        self._optimization_time_budget: float | None = None

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

    def apply_feature(self, feature: PipelineFeature) -> PipelineBuilder:
        """Apply a named pipeline feature to this mutable builder."""
        feature.apply(self)
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
    """Default Python-source pipeline with automatic contextual wiring."""

    OPTUNA_SCORE_KEY: str | None = None
    OPTUNA_MAX_PARALLEL: int = 10

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = DEFAULT_MAX_INSIGHTS,
        max_code_length: int = MAX_CODE_LENGTH,
        archive_gate_enabled: bool = False,
        include_legacy_feedback: bool = True,
        program_format_feature: PipelineFeature | None = None,
    ):
        super().__init__(ctx, dag_timeout=dag_timeout, max_parallel=max_parallel)
        self._stage_timeout = stage_timeout
        self._max_insights = max_insights
        self._max_code_length = max_code_length
        self._archive_gate_enabled = archive_gate_enabled
        self._program_format_feature = program_format_feature
        self._optimization_time_budget: float | None = None
        self.apply_feature(SourceProgramEvaluationFeature())
        if program_format_feature is not None:
            self.apply_feature(program_format_feature)
        if ctx.problem_ctx.is_contextual:
            self._add_context_stage_and_edges()
        self.apply_feature(MetricAssemblyFeature())
        self.apply_feature(MutationPromptContextFeature())
        if archive_gate_enabled:
            self.apply_feature(ArchivePotentialFilterFeature())
        if include_legacy_feedback:
            self.apply_feature(LegacyLineageInsightFeature())

    def _add_context_stage_and_edges(self) -> None:
        """Add the problem context stage once.

        Normal builders call this automatically when
        ``ctx.problem_ctx.is_contextual`` is true. A few specialist builders
        still force it explicitly; keeping the helper idempotent prevents
        duplicate edges/dependencies.
        """
        if "AddContext" in self._nodes:
            return
        self.apply_feature(ProblemContextBuildFeature())

    def _require_python_source_optimizer(self, optimizer_name: str) -> None:
        if self._program_format_feature is None:
            return
        raise ValueError(
            f"{optimizer_name} optimization requires program_format=python_source; "
            "non-Python program formats need a format-aware optimizer."
        )

    def _optuna_stage_kwargs(self) -> dict:
        """Extra kwargs forwarded to :class:`OptunaOptimizationStage`.

        Override in a subclass to customise Optuna hyper-parameters.
        """
        return {}

    def _wire_optuna_stage(self) -> None:
        """Insert OptunaOptimizationStage between validate and program-call.

        Requires ``self._optimization_time_budget`` set before calling.
        Detects context auto-magically via the presence of an ``AddContext``
        stage in ``self._nodes``.
        """
        self._require_python_source_optimizer("Optuna")

        from gigaevo.programs.stages.optimization.optuna import (
            OptunaOptimizationStage,
            OptunaPayloadBridge,
            PayloadResolver,
        )

        if self._optimization_time_budget is None:
            raise ValueError(
                "_optimization_time_budget must be set before _wire_optuna_stage()"
            )

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
        n_trials = extra.pop("n_trials", None)
        eval_to = extra.pop("eval_timeout", None)
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

        self.add_exec_dep(
            "OptunaOptStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )

        if "AddContext" in self._nodes:
            self.add_data_flow_edge("AddContext", "OptunaOptStage", "context")
            self.add_exec_dep(
                "OptunaOptStage",
                ExecutionOrderDependency.on_success("AddContext"),
            )

        # Bypass: when Optuna succeeds, OptunaPayloadBridge → PayloadResolver
        # feeds the validator; CallProgramFunction only runs on Optuna failure.
        bridge_timeout = self._stage_timeout
        self.add_stage(
            "OptunaPayloadBridge",
            lambda: OptunaPayloadBridge(timeout=bridge_timeout),
        )
        self.add_stage(
            "PayloadResolver",
            lambda: PayloadResolver(timeout=bridge_timeout),
        )

        self.add_data_flow_edge(
            "OptunaOptStage", "OptunaPayloadBridge", "optuna_output"
        )
        self.add_data_flow_edge(
            "OptunaPayloadBridge", "PayloadResolver", "optuna_payload"
        )
        self.add_data_flow_edge(
            "CallProgramFunction", "PayloadResolver", "program_payload"
        )

        self.remove_data_flow_edge("CallProgramFunction", "CallValidatorFunction")
        self.add_data_flow_edge("PayloadResolver", "CallValidatorFunction", "payload")

        self.add_exec_dep(
            "CallProgramFunction",
            ExecutionOrderDependency.on_failure("OptunaOptStage"),
        )

    def _contribute_default_nodes(self) -> None:
        """Deprecated compatibility wrapper for older builder subclasses."""
        self.apply_feature(SourceProgramEvaluationFeature())
        if self.ctx.problem_ctx.is_contextual:
            self._add_context_stage_and_edges()
        self.apply_feature(MetricAssemblyFeature())
        self.apply_feature(MutationPromptContextFeature())
        if self._archive_gate_enabled:
            self.apply_feature(ArchivePotentialFilterFeature())
        self.apply_feature(LegacyLineageInsightFeature())

    def _contribute_default_edges(self) -> None:
        """Deprecated: feature objects now contribute their own edges."""

    def _contribute_default_deps(self) -> None:
        """Deprecated: feature objects now contribute their own dependencies."""


class SourceProgramEvaluationFeature(PipelineFeature):
    """Python source validation, execution, validator call, and artifact formatting.

    This is the base evaluator for normal Python programs: check the candidate
    source, call its ``entrypoint()``, pass that payload to problem
    ``validate.py``, and split the validator result into metrics and artifact
    channels.
    """

    name = "source_program_evaluation"
    description = (
        "Validate Python source, run entrypoint(), call validate.py, and expose "
        "metrics plus formatted artifact text."
    )

    def apply(self, builder: PipelineBuilder) -> None:
        problem_ctx = builder.ctx.problem_ctx
        stage_timeout = builder._stage_timeout
        max_code_length = builder._max_code_length

        # ValidateCompiles
        builder.add_stage(
            "ValidateCodeStage",
            lambda: ValidateCodeStage(
                max_code_length=max_code_length,
                timeout=stage_timeout,
                safe_mode=False,
            ),
        )

        # ExecuteCode: run program.code with optional data from DAG
        builder.add_stage(
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
        builder.add_stage(
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
        builder.add_stage(
            "FetchMetrics",
            lambda: FetchMetrics(timeout=stage_timeout),
        )
        builder.add_stage(
            "FetchArtifact",
            lambda: FetchArtifact(timeout=stage_timeout),
        )
        builder.add_stage(
            "FormatterStage",
            lambda: FormatterStage(timeout=stage_timeout),
        )

        builder.add_data_flow_edge(
            "CallProgramFunction", "CallValidatorFunction", "payload"
        )
        builder.add_data_flow_edge(
            "CallValidatorFunction", "FetchMetrics", "validation_result"
        )
        builder.add_data_flow_edge(
            "CallValidatorFunction", "FetchArtifact", "validation_result"
        )
        builder.add_data_flow_edge("FetchArtifact", "FormatterStage", "data")
        builder.add_exec_dep(
            "CallProgramFunction",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )
        builder.add_exec_dep(
            "FetchMetrics",
            ExecutionOrderDependency.always_after("CallValidatorFunction"),
        )
        builder.add_exec_dep(
            "FetchArtifact",
            ExecutionOrderDependency.always_after("CallValidatorFunction"),
        )
        builder.add_exec_dep(
            "FormatterStage",
            ExecutionOrderDependency.always_after("FetchArtifact"),
        )


class ArchivePotentialFilterFeature(PipelineFeature):
    """Optional archive-admission gate for expensive downstream LLM stages.

    The stage asks the configured archive gate provider whether the candidate
    could enter any island archive. Downstream feature-specific LLM stages can
    depend on this gate to avoid spending tokens on dominated programs.
    """

    name = "archive_potential_filter"
    description = "Gate expensive LLM feedback stages on archive-admission potential."

    def apply(self, builder: PipelineBuilder) -> None:
        stage_timeout = builder._stage_timeout
        gate_provider = builder.ctx.archive_gate_provider

        builder.add_stage(
            "ArchivePotentialGateStage",
            lambda: ArchivePotentialGateStage(
                provider=gate_provider,
                timeout=stage_timeout,
            ),
        )
        builder.add_exec_dep(
            "ArchivePotentialGateStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )
        builder.add_exec_dep(
            "ArchivePotentialGateStage",
            ExecutionOrderDependency.always_after("EnsureMetricsStage"),
        )


class ChainStructuralMetricsFeature(PipelineFeature):
    """Extract chain topology metrics for structural MAP-Elites archives.

    Adds ``dag_depth``, ``max_dependency_fan_in``, ``n_deep_retrieval``, and
    ``n_retrievals`` to ``program.metrics`` after the candidate has parsed and
    validated. Enable this when an algorithm's behavior space uses chain
    structure keys, for example ``algorithm=topology_3d_ret``.
    """

    name = "chain_structural_metrics"
    description = "Compute chain topology metrics for behavior-space archives."

    def apply(self, builder: PipelineBuilder) -> None:
        stage_timeout = builder._stage_timeout

        builder.add_stage(
            "ChainStructuralMetricsStage",
            lambda: ChainStructuralMetricsStage(timeout=stage_timeout),
        )
        builder.add_exec_dep(
            "ChainStructuralMetricsStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )


class LegacyLineageInsightFeature(PipelineFeature):
    """Original insights-plus-lineage feedback path.

    This keeps the historical ``InsightsStage`` / ``LineageStage`` machinery in
    one isolated feature for pipelines that still request
    :class:`DefaultPipelineBuilder`. New guided-memory pipelines use
    ``IntraMemoryStage`` + ``MutationSuggestionStage`` instead and do not apply
    this feature.
    """

    name = "legacy_lineage_insights"
    description = (
        "Generate legacy ProgramInsights and lineage transition summaries for "
        "MutationContextStage."
    )

    def apply(self, builder: PipelineBuilder) -> None:
        metrics_context = builder.ctx.problem_ctx.metrics_context
        storage = builder.ctx.storage
        llm_wrapper = builder.ctx.llm_wrapper
        task_description = builder.ctx.problem_ctx.task_description
        prompts_dir = builder.ctx.prompts_dir
        stage_timeout = builder._stage_timeout
        max_insights = builder._max_insights

        builder.add_stage(
            "InsightsStage",
            lambda: InsightsStage(
                llm=llm_wrapper,
                task_description=task_description,
                metrics_context=metrics_context,
                max_insights=max_insights,
                timeout=stage_timeout,
                prompts_dir=prompts_dir,
            ),
        )

        # Shared with LineageStage so its preprocess can short-circuit
        # (Q → failed-child) analyses that this selector would never pick.
        descendant_selector = AncestrySelector(
            metrics_context=metrics_context,
            strategy="best_fitness",
            max_selected=1,
        )
        builder.add_stage(
            "DescendantProgramIds",
            lambda: DescendantProgramIds(
                storage=storage,
                selector=descendant_selector,
                timeout=stage_timeout,
            ),
        )
        builder.add_stage(
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

        builder.add_stage(
            "LineageStage",
            lambda: LineageStage(
                llm=llm_wrapper,
                task_description=task_description,
                metrics_context=metrics_context,
                storage=storage,
                timeout=stage_timeout,
                prompts_dir=prompts_dir,
                descendant_selector=descendant_selector,
            ),
        )

        builder.add_stage(
            "LineagesToDescendants",
            lambda: LineagesToDescendants(
                storage=storage,
                source_stage_name="LineageStage",
                timeout=stage_timeout,
            ),
        )

        builder.add_stage(
            "LineagesFromAncestors",
            lambda: LineagesFromAncestors(
                storage=storage,
                source_stage_name="LineageStage",
                timeout=stage_timeout,
            ),
        )

        builder.add_data_flow_edge("InsightsStage", "MutationContextStage", "insights")
        builder.add_data_flow_edge(
            "DescendantProgramIds", "LineagesToDescendants", "descendant_ids"
        )
        builder.add_data_flow_edge(
            "AncestorProgramIds", "LineagesFromAncestors", "ancestor_ids"
        )
        builder.add_data_flow_edge(
            "LineagesToDescendants", "MutationContextStage", "lineage_descendants"
        )
        builder.add_data_flow_edge(
            "LineagesFromAncestors", "MutationContextStage", "lineage_ancestors"
        )

        builder.add_exec_dep(
            "InsightsStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )
        builder.add_exec_dep(
            "InsightsStage",
            ExecutionOrderDependency.always_after("EnsureMetricsStage"),
        )
        builder.add_exec_dep(
            "LineageStage",
            ExecutionOrderDependency.always_after("EnsureMetricsStage"),
        )
        builder.add_exec_dep(
            "LineagesToDescendants",
            ExecutionOrderDependency.always_after("LineageStage"),
        )
        builder.add_exec_dep(
            "LineagesFromAncestors",
            ExecutionOrderDependency.always_after("LineageStage"),
        )
        if builder._archive_gate_enabled:
            builder.add_exec_dep(
                "InsightsStage",
                ExecutionOrderDependency.on_success("ArchivePotentialGateStage"),
            )


class MetricAssemblyFeature(PipelineFeature):
    """Combine validation metrics with framework metrics and collect run stats."""

    name = "metric_assembly"
    description = (
        "Merge validator metrics with code complexity, ensure sentinel values, "
        "and collect evolutionary statistics."
    )

    def apply(self, builder: PipelineBuilder) -> None:
        metrics_context = builder.ctx.problem_ctx.metrics_context
        storage = builder.ctx.storage
        stage_timeout = builder._stage_timeout

        builder.add_stage(
            "ComputeComplexityStage",
            lambda: ComputeComplexityStage(
                timeout=stage_timeout,
            ),
        )
        builder.add_stage(
            "MergeMetricsStage",
            lambda: MergeDictStage[str, float](
                timeout=stage_timeout,
            ),
        )
        builder.add_stage(
            "EnsureMetricsStage",
            lambda: EnsureMetricsStage(
                metrics_factory=metrics_context.get_sentinels,
                metrics_context=metrics_context,
                timeout=stage_timeout,
            ),
        )
        builder.add_stage(
            "EvolutionaryStatisticsCollector",
            lambda: EvolutionaryStatisticsCollector(
                storage=storage,
                metrics_context=metrics_context,
                timeout=stage_timeout,
            ),
        )

        builder.add_data_flow_edge("FetchMetrics", "MergeMetricsStage", "first")
        builder.add_data_flow_edge(
            "ComputeComplexityStage", "MergeMetricsStage", "second"
        )
        builder.add_data_flow_edge(
            "MergeMetricsStage", "EnsureMetricsStage", "candidate"
        )
        builder.add_exec_dep(
            "EvolutionaryStatisticsCollector",
            ExecutionOrderDependency.always_after("EnsureMetricsStage"),
        )


class MutationPromptContextFeature(PipelineFeature):
    """Assemble all feedback channels into the mutator-facing context object."""

    name = "mutation_prompt_context"
    description = (
        "Create MutationContextStage and wire metrics, formatted artifacts, and "
        "evolutionary statistics into the mutation prompt context."
    )

    def apply(self, builder: PipelineBuilder) -> None:
        metrics_context = builder.ctx.problem_ctx.metrics_context
        stage_timeout = builder._stage_timeout

        builder.add_stage(
            "MutationContextStage",
            lambda: MutationContextStage(
                metrics_context=metrics_context,
                timeout=stage_timeout,
            ),
        )
        builder.add_data_flow_edge(
            "EnsureMetricsStage", "MutationContextStage", "metrics"
        )
        builder.add_data_flow_edge(
            "EvolutionaryStatisticsCollector",
            "MutationContextStage",
            "evolutionary_statistics",
        )
        builder.add_data_flow_edge(
            "FormatterStage", "MutationContextStage", "formatted"
        )


class ProblemContextBuildFeature(PipelineFeature):
    """Run ``context.py`` and feed its payload into program and validator calls.

    Contextual problems provide a ``build_context`` function that returns
    per-evaluation data. This feature wires that data into both the candidate
    program execution and the problem validator so contextual pipelines do not
    need bespoke boilerplate.
    """

    name = "problem_context_build"
    description = (
        "Call problem context.py:build_context and pass the result to execution "
        "and validation stages."
    )

    def apply(self, builder: PipelineBuilder) -> None:
        problem_ctx = builder.ctx.problem_ctx
        stage_timeout = builder._stage_timeout

        builder.add_stage(
            "AddContext",
            lambda: CallFileFunction(
                path=problem_ctx.problem_dir / ProblemLayout.CONTEXT_FILE,
                function_name="build_context",
                timeout=stage_timeout,
            ),
        )
        builder.add_data_flow_edge("AddContext", "CallProgramFunction", "context")
        builder.add_data_flow_edge("AddContext", "CallValidatorFunction", "context")
        builder.add_exec_dep(
            "CallProgramFunction",
            ExecutionOrderDependency.on_success("AddContext"),
        )
        builder.add_exec_dep(
            "CallValidatorFunction",
            ExecutionOrderDependency.on_success("AddContext"),
        )


class AlgoTuneSpeedPipelineBuilder(DefaultPipelineBuilder):
    """Contextual pipeline variant using execution speed as primary fitness."""

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = DEFAULT_MAX_INSIGHTS,
        max_code_length: int = MAX_CODE_LENGTH,
        archive_gate_enabled: bool = False,
        program_format_feature: PipelineFeature | None = None,
    ):
        if not ctx.problem_ctx.is_contextual:
            raise ValueError(
                "AlgoTuneSpeedPipelineBuilder requires a contextual problem; "
                "RuntimeFitnessStage times entrypoint(context)."
            )
        super().__init__(
            ctx,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
            max_parallel=max_parallel,
            max_insights=max_insights,
            max_code_length=max_code_length,
            archive_gate_enabled=archive_gate_enabled,
            program_format_feature=program_format_feature,
        )
        self._add_runtime_fitness_stage()

    def _load_runtime_evaluation_config(self) -> tuple[int, int]:
        metrics_path = self.ctx.problem_ctx.problem_dir / "metrics.yaml"
        try:
            data = yaml.safe_load(metrics_path.read_text()) or {}
        except Exception:
            return 1, 0
        runtime_cfg = data.get("runtime_evaluation", {})
        repetitions = runtime_cfg.get("timing_repetitions", 1)
        warmups = runtime_cfg.get("warmup_repetitions", 0)
        try:
            repetitions_int = int(repetitions)
        except Exception:
            repetitions_int = 1
        try:
            warmups_int = int(warmups)
        except Exception:
            warmups_int = 0
        return max(1, repetitions_int), max(0, warmups_int)

    def _add_runtime_fitness_stage(self) -> None:
        repetitions, warmups = self._load_runtime_evaluation_config()
        problem_dir = self.ctx.problem_ctx.problem_dir
        stage_timeout = self._stage_timeout
        self.add_stage(
            "RuntimeFitnessStage",
            lambda: RuntimeFitnessStage(
                source_stage_name="CallProgramFunction",
                problem_dir=problem_dir,
                timing_repetitions=repetitions,
                warmup_repetitions=warmups,
                timeout=stage_timeout,
            ),
        )
        self.remove_data_flow_edge("FetchMetrics", "MergeMetricsStage")
        self.add_data_flow_edge("FetchMetrics", "RuntimeFitnessStage", "candidate")
        self.add_data_flow_edge("AddContext", "RuntimeFitnessStage", "context")
        self.add_data_flow_edge("RuntimeFitnessStage", "MergeMetricsStage", "first")
        self.add_exec_dep(
            "RuntimeFitnessStage",
            ExecutionOrderDependency.on_success("AddContext"),
        )


class CMAOptPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline + CMA-ES numerical constant optimisation.

    Inherits :class:`DefaultPipelineBuilder` and inserts a
    :class:`CMANumericalOptimizationStage` between ``ValidateCodeStage``
    and ``CallProgramFunction``.  If the problem provides a ``context.py``
    the ``AddContext`` stage is wired automatically by the base builder.

    Execution order::

        ValidateCodeStage ─(success)─► CMAOptStage ─(always)─► CallProgramFunction
        AddContext* ───────(success)─►              ─(data)──►
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
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = DEFAULT_MAX_INSIGHTS,
        max_code_length: int = MAX_CODE_LENGTH,
        optimization_time_budget: float | None = None,
        archive_gate_enabled: bool = False,
        program_format_feature: PipelineFeature | None = None,
    ):
        super().__init__(
            ctx,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
            max_parallel=max_parallel,
            max_insights=max_insights,
            max_code_length=max_code_length,
            archive_gate_enabled=archive_gate_enabled,
            program_format_feature=program_format_feature,
        )
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
        super()._add_context_stage_and_edges()

    def _cma_stage_kwargs(self) -> dict:
        """Return extra kwargs forwarded to :class:`CMANumericalOptimizationStage`.

        Override in a subclass to customise CMA hyper-parameters without
        rewriting the whole pipeline.
        """
        return {}

    def _add_cma_optimization(self, *, has_context: bool) -> None:
        from gigaevo.programs.stages.optimization.cma import (
            CMANumericalOptimizationStage,
        )

        self._require_python_source_optimizer("CMA")

        problem_ctx = self.ctx.problem_ctx
        validator_path = problem_ctx.problem_dir / "validate.py"

        extra = self._cma_stage_kwargs()

        max_gen = extra.pop("max_generations", self.CMA_MAX_GENERATIONS)
        pop_size = extra.pop("population_size", self.CMA_POPULATION_SIZE)
        max_par = extra.pop("max_parallel", self.CMA_MAX_PARALLEL)

        if self._optimization_time_budget is None:
            raise ValueError(
                "_optimization_time_budget must be set before _add_cma_optimization()"
            )
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
                ExecutionOrderDependency.on_success("AddContext"),
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
    the ``AddContext`` stage is wired automatically by the base builder.

    Execution order::

        ValidateCodeStage ─(success)─► OptunaOptStage ─(failure)─► CallProgramFunction
        AddContext* ───────(success)─►                 ─(data)────►
        (* only when context.py exists)

    If Optuna fails, the program still runs with the original code.

    Override ``_optuna_stage_kwargs`` in a subclass to tweak hyper-parameters.
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 7200.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = DEFAULT_MAX_INSIGHTS,
        max_code_length: int = MAX_CODE_LENGTH,
        optimization_time_budget: float | None = None,
        archive_gate_enabled: bool = False,
        program_format_feature: PipelineFeature | None = None,
    ):
        super().__init__(
            ctx,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
            max_parallel=max_parallel,
            max_insights=max_insights,
            max_code_length=max_code_length,
            archive_gate_enabled=archive_gate_enabled,
            program_format_feature=program_format_feature,
        )
        self._optimization_time_budget = (
            optimization_time_budget
            if optimization_time_budget is not None
            else dag_timeout * DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION
        )
        if ctx.problem_ctx.is_contextual:
            self._add_context_stage_and_edges()
        self._wire_optuna_stage()

    def _add_context_stage_and_edges(self) -> None:
        super()._add_context_stage_and_edges()


class CustomPipelineBuilder(PipelineBuilder):
    """Starts with an empty pipeline. Users compose everything explicitly."""
