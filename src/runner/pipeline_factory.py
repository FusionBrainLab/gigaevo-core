from __future__ import annotations

# Pipeline builders that produce DAGSpec using small, composable classes.
#
# DefaultPipelineBuilder reproduces the current default pipeline from run.py.
# ContextPipelineBuilder extends the default with an AddContext stage and wiring.
# CustomPipelineBuilder starts empty for fully user-defined pipelines.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.database.redis_program_storage import RedisProgramStorage
from src.programs.automata import ExecutionOrderDependency
from src.programs.metrics.context import MetricsContext
from src.programs.metrics.formatter import MetricsFormatter
from src.programs.stages.base import Stage
from src.programs.stages.execution import (
    RunProgramCodeWithOptionalProducedData,
    ValidatorCodeExecutor,
)
from src.programs.stages.insights import (
    GenerateLLMInsightsStage,
    InsightsConfig,
)
from src.programs.stages.insights_lineage import (
    GenerateLineageInsightsStage,
    LineageInsightsConfig,
)
from src.programs.stages.metrics import EnsureMetricsStage
from src.programs.stages.validation import ValidateCodeStage
from src.problems.context import ProblemContext
from src.llm.wrapper import LLMWrapper
from src.runner.dag_spec import DAGSpec


StageFactory = Callable[[], Stage]


@dataclass(slots=True)
class PipelineContext:
    problem_ctx: ProblemContext
    metrics_context: MetricsContext
    metrics_formatter: MetricsFormatter | None
    llm_wrapper: dict[str, LLMWrapper]
    storage: RedisProgramStorage
    task_description: str
    add_context: bool = False


class PipelineBuilder:
    """Mutable builder for pipeline nodes/edges/deps producing a DAGSpec."""

    def __init__(self, ctx: PipelineContext):
        self.ctx = ctx
        self._nodes: Dict[str, StageFactory] = {}
        self._edges: Dict[str, List[str]] = {}
        self._deps: Dict[str, List[ExecutionOrderDependency]] = {}
        self._dag_timeout: float = 1800.0
        self._max_parallel: int = 8

    # Stage operations
    def add_stage(self, name: str, factory: StageFactory) -> "PipelineBuilder":
        self._nodes[name] = factory
        return self

    def replace_stage(
        self, name: str, factory: StageFactory
    ) -> "PipelineBuilder":
        self._nodes[name] = factory
        return self

    def remove_stage(self, name: str) -> "PipelineBuilder":
        self._nodes.pop(name, None)
        # Remove edges and deps that reference this stage
        self._edges.pop(name, None)
        for src, dsts in list(self._edges.items()):
            self._edges[src] = [d for d in dsts if d != name]
        self._deps.pop(name, None)
        for stage, deps in list(self._deps.items()):
            self._deps[stage] = [d for d in deps if d.stage_name != name]
        return self

    # Edge operations
    def add_edge(self, src: str, dst: str) -> "PipelineBuilder":
        self._edges.setdefault(src, []).append(dst)
        return self

    def remove_edge(self, src: str, dst: str) -> "PipelineBuilder":
        if src in self._edges:
            self._edges[src] = [d for d in self._edges[src] if d != dst]
        return self

    # Execution order dependency operations
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

    def set_limits(
        self, *, dag_timeout: float | None, max_parallel: int | None
    ) -> "PipelineBuilder":
        if dag_timeout is not None:
            self._dag_timeout = dag_timeout
        if max_parallel is not None:
            self._max_parallel = max_parallel
        return self

    def build_spec(self) -> DAGSpec:
        return DAGSpec(
            nodes=self._nodes,
            edges=self._edges,
            exec_order_deps=self._deps or None,
            dag_timeout=self._dag_timeout,
            max_parallel_stages=self._max_parallel,
        )


class DefaultPipelineBuilder(PipelineBuilder):
    """Recreates the current default pipeline (no AddContext)."""

    def __init__(self, ctx: PipelineContext):
        super().__init__(ctx)
        self._contribute_default_nodes()
        self._contribute_default_edges()
        self._contribute_default_deps()

    def _contribute_default_nodes(self) -> None:
        # Context is available for future wiring
        metrics_context = self.ctx.metrics_context
        problem_ctx = self.ctx.problem_ctx
        llm_wrapper = self.ctx.llm_wrapper
        metrics_formatter = self.ctx.metrics_formatter
        storage = self.ctx.storage
        task_description = self.ctx.task_description

        # ValidateCompiles
        self.add_stage(
            "ValidateCompiles",
            lambda: ValidateCodeStage(
                stage_name="ValidateCompiles",
                max_code_length=24000,
                timeout=30.0,
                safe_mode=True,
            ),
        )

        # ExecuteCode: run program.code with optional data from DAG
        self.add_stage(
            "ExecuteCode",
            lambda: RunProgramCodeWithOptionalProducedData(
                stage_name="ExecuteCode",
                function_name="entrypoint",
                python_path=[problem_ctx.problem_dir.resolve()],
                timeout=600.0,
                max_memory_mb=512,
            ),
        )

        # RunValidation
        validator_path = problem_ctx.problem_dir / "validate.py"
        self.add_stage(
            "RunValidation",
            lambda: ValidatorCodeExecutor(
                stage_name="RunValidation",
                validator_path=validator_path,
                function_name="validate",
                timeout=60.0,
            ),
        )

        # Insights stages
        self.add_stage(
            "LLMInsights",
            lambda: GenerateLLMInsightsStage(
                config=InsightsConfig(
                    llm_wrapper=llm_wrapper["insights"],
                    evolutionary_task_description=task_description,
                    max_insights=8,
                    output_format="text",
                    metrics_context=metrics_context,
                    metrics_formatter=metrics_formatter,
                    metadata_key="insights",
                    excluded_error_stages=[
                        "FactoryMetricUpdate",
                    ],
                ),
                timeout=600.0,
            ),
        )

        self.add_stage(
            "LineageInsights",
            lambda: GenerateLineageInsightsStage(
                config=LineageInsightsConfig(
                    llm_wrapper=llm_wrapper["lineage"],
                    metrics_context=metrics_context,
                    metrics_formatter=metrics_formatter,
                    parent_selection_strategy="best_fitness",
                    task_description=task_description,
                ),
                storage=storage,
                timeout=600,
            ),
        )

        # ValidationMetricUpdate
        def _validation_metrics_factory() -> dict[str, Any]:
            primary_spec = metrics_context.get_primary_spec()
            return {
                primary_spec.key: (
                    primary_spec.lower_bound
                    if primary_spec.higher_is_better
                    else primary_spec.upper_bound
                ),
                "is_valid": 0,
            }

        self.add_stage(
            "ValidationMetricUpdate",
            lambda: EnsureMetricsStage(
                stage_name="ValidationMetricUpdate",
                metrics_factory=_validation_metrics_factory,
                metrics_context=metrics_context,
                timeout=15.0,
            ),
        )

    def _contribute_default_edges(self) -> None:
        # Default DAG without AddContext
        self._edges = {
            "ValidateCompiles": [],
            "ExecuteCode": ["RunValidation"],
            "RunValidation": ["ValidationMetricUpdate"],
            "LLMInsights": [],
            "LineageInsights": [],
            "ValidationMetricUpdate": [],
        }

    def _contribute_default_deps(self) -> None:
        self._deps = {
            "ExecuteCode": [
                ExecutionOrderDependency.on_success("ValidateCompiles"),
            ],
            "ValidationMetricUpdate": [
                ExecutionOrderDependency.always_after("ValidateCompiles"),
                ExecutionOrderDependency.always_after("ExecuteCode"),
                ExecutionOrderDependency.always_after("RunValidation"),
            ],
            "LLMInsights": [
                ExecutionOrderDependency.always_after("ValidateCompiles"),
                ExecutionOrderDependency.always_after("ExecuteCode"),
                ExecutionOrderDependency.always_after("RunValidation"),
                ExecutionOrderDependency.always_after("ValidationMetricUpdate"),
            ],
            "LineageInsights": [
                ExecutionOrderDependency.always_after("ValidateCompiles"),
                ExecutionOrderDependency.always_after("ExecuteCode"),
                ExecutionOrderDependency.always_after("RunValidation"),
                ExecutionOrderDependency.always_after("ValidationMetricUpdate"),
            ],
        }


class ContextPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline with AddContext stage and wiring enabled."""

    def __init__(self, ctx: PipelineContext):
        super().__init__(ctx)
        self._add_context_stage_and_edges()

    def _add_context_stage_and_edges(self) -> None:
        # Using existing context from parent; no local variables needed

        # No replacements; context is provided by edges only

        # Wire AddContext → ValidateCompiles
        self._edges.setdefault("AddContext", []).append("ValidateCompiles")
        # Ensure ExecuteCode runs after ValidateCompiles; provide data and optional context
        self._deps.setdefault("ExecuteCode", []).append(
            ExecutionOrderDependency.on_success("ValidateCompiles")
        )
        self._edges.setdefault("ExecuteCode", []).append("RunValidation")
        # Wire context and data to validation in positional order: context first, then data
        self._edges.setdefault("AddContext", []).append("ExecuteCode")
        self._edges.setdefault("AddContext", []).append("RunValidation")
        # Ensure metrics consumes validation output via regular edge
        self._edges.setdefault("RunValidation", []).append(
            "ValidationMetricUpdate"
        )


class CustomPipelineBuilder(PipelineBuilder):
    """Starts with an empty pipeline. Users compose everything explicitly."""

    # Intentionally empty; inherits all behavior from PipelineBuilder
