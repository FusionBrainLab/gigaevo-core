from __future__ import annotations

from typing import Callable

from gigaevo.entrypoint.constants import (
    DEFAULT_DAG_CONCURRENCY,
    DEFAULT_DAG_TIMEOUT,
    DEFAULT_MAX_INSIGHTS,
    DEFAULT_STAGE_TIMEOUT,
    MAX_CODE_LENGTH,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.problems.layout import ProblemLayout
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.collector import AncestorProgramIds, DescendantProgramIds
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

trait_description = """
Assess how modular the submitted code is.

Focus on whether the solution decomposes the task into small, cohesive, reusable functions with clear interfaces and minimal coupling. Reward:
- Clear separation of concerns: the top-level function orchestrates; helpers do one thing well.
- Small functions (preferably < 40 LOC) with descriptive names, docstrings, and type hints.
- Low coupling / high cohesion: helpers don’t reach into each other’s internals; parameters carry needed data.
- Reuse over repetition (little to no copy-paste).
- Controlled side effects: I/O and state changes isolated behind thin adapters; core logic mostly pure.
- Testability and composability: helpers can be unit-tested in isolation; minimal global state.

Penalize:
- Monolithic or god functions, deep nesting, and long parameter lists.
- Mixed responsibilities in a single function.
- Hidden dependencies, global/mutable shared state, and tight coupling.
- Duplicate logic instead of extracting helpers.

Scoring rubric (0–100):
- 90–100: Highly modular; clear orchestration + focused helpers; minimal coupling; excellent docs/types.
- 70–89: Generally modular; a few oversized or mixed-concern helpers; minor duplication/coupling.
- 40–69: Partially modular; main function still heavy; noticeable duplication and side-effect tangling.
- 0–39: Monolithic; few/no helpers; tightly coupled, hard to test.

Ignore performance or algorithmic optimality; evaluate modularity only.
"""


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


class CustomPipelineBuilder(PipelineBuilder):
    """Starts with an empty pipeline. Users compose everything explicitly."""
