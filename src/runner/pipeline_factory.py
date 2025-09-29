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
    RunConstantPythonCode,
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
from src.problems.layout import ProblemLayout
from src.llm.wrapper import LLMWrapper
from src.runner.dag_spec import DAGSpec
from src.programs.automata import DataFlowEdge


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
        self._data_flow_edges: List[DataFlowEdge] = []
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
        # Remove data flow edges that reference this stage
        self._data_flow_edges = [
            edge for edge in self._data_flow_edges 
            if edge.source_stage != name and edge.destination_stage != name
        ]
        # Remove deps that reference this stage
        self._deps.pop(name, None)
        for stage, deps in list(self._deps.items()):
            self._deps[stage] = [d for d in deps if d.stage_name != name]
        return self

    # Data flow operations
    def add_data_flow_edge(self, src: str, dst: str, input_name: str) -> "PipelineBuilder":
        """Add a data flow edge with semantic input naming."""
        self._data_flow_edges.append(DataFlowEdge.create(
            source=src,
            destination=dst,
            input_name=input_name
        ))
        return self

    def remove_data_flow_edge(self, src: str, dst: str) -> "PipelineBuilder":
        """Remove a data flow edge."""
        self._data_flow_edges = [e for e in self._data_flow_edges if not (e.source_stage == src and e.destination_stage == dst)]
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
            data_flow_edges=self._data_flow_edges,
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
            "ValidateCodeStage",
            lambda: ValidateCodeStage(
                stage_name="ValidateCompiles",
                max_code_length=24000,
                timeout=30.0,
                safe_mode=True,
            ),
        )

        # ExecuteCode: run program.code with optional data from DAG
        self.add_stage(
            "RunProgramCodeWithOptionalProducedData",
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
            "ValidatorCodeExecutor",
            lambda: ValidatorCodeExecutor(
                stage_name="RunValidation",
                validator_path=validator_path,
                function_name="validate",
                timeout=240.0,
            ),
        )

        # Insights stages
        self.add_stage(
            "GenerateLLMInsightsStage",
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
            "GenerateLineageInsightsStage",
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

        self.add_stage(
            "EnsureMetricsStage",
            lambda: EnsureMetricsStage(
                stage_name="ValidationMetricUpdate",
                metrics_factory=metrics_context.get_worst_with_coalesce,
                metrics_context=metrics_context,
                timeout=15.0,
            ),
        )

    def _contribute_default_edges(self) -> None:
        # Add semantic data flow edges for better data flow clarity
        self.add_data_flow_edge("RunProgramCodeWithOptionalProducedData", "ValidatorCodeExecutor", "program_output")
        self.add_data_flow_edge("ValidatorCodeExecutor", "EnsureMetricsStage", "validation_result")

    def _contribute_default_deps(self) -> None:
        self._deps = {
            "RunProgramCodeWithOptionalProducedData": [
                ExecutionOrderDependency.on_success("ValidateCodeStage")
            ],
            "GenerateLLMInsightsStage": [
                ExecutionOrderDependency.always_after("EnsureMetricsStage"),
            ],
            "GenerateLineageInsightsStage": [
                ExecutionOrderDependency.always_after("EnsureMetricsStage"),
            ],
        }


class ContextPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline with AddContext stage and wiring enabled."""

    def __init__(self, ctx: PipelineContext):
        super().__init__(ctx)
        self._add_context_stage_and_edges()

    def _add_context_stage_and_edges(self) -> None:
        problem_ctx = self.ctx.problem_ctx

        # AddContext stage: runs build_context from context.py to produce a dict
        self.add_stage(
            "AddContext",
            lambda: RunConstantPythonCode(
                stage_name="AddContext",
                context_path=problem_ctx.problem_dir / ProblemLayout.CONTEXT_FILE,
                function_name="build_context",
                timeout=30.0,
            ),
        )

        # Provide context to ExecuteCode and RunValidation with semantic names
        self.add_data_flow_edge("AddContext", "RunProgramCodeWithOptionalProducedData", "context_data")
        self.add_data_flow_edge("AddContext", "ValidatorCodeExecutor", "context_data")



class CustomPipelineBuilder(PipelineBuilder):
    """Starts with an empty pipeline. Users compose everything explicitly."""

    # Intentionally empty; inherits all behavior from PipelineBuilder
