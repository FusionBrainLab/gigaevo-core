from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Set, cast
from pydantic import BaseModel, Field

from src.programs.program import Program, ProgramStageResult, StageState

# pylint: disable=E1101  # Pydantic Field type confusion for transition_rules
from src.programs.utils import format_error_for_llm


class DataFlowEdge(BaseModel):
    """Represents a data flow connection between stages with semantic input naming."""
    source_stage: str = Field(
        ..., description="Name of the source stage that produces data"
    )
    destination_stage: str = Field(
        ..., description="Name of the destination stage that consumes data"
    )
    input_name: str = Field(
        ..., description="Semantic name for this input in the destination stage"
    )

    @classmethod
    def create(cls, source: str, destination: str, input_name: str) -> "DataFlowEdge":
        """Create a data flow edge with explicit naming."""
        return cls(
            source_stage=source,
            destination_stage=destination,
            input_name=input_name
        )


class ExecutionOrderDependency(BaseModel):
    stage_name: str = Field(
        ..., description="Name of the stage this dependency refers to"
    )
    condition: Literal["success", "failure", "always"] = Field(
        ..., description="When this dependency is considered satisfied"
    )

    def is_satisfied(self, result: Optional[ProgramStageResult]) -> bool:
        if result is None or result.status in (
            StageState.PENDING,
            StageState.RUNNING,
        ):
            return False
        if self.condition == "always":
            return True
        if self.condition == "success":
            return result.status == StageState.COMPLETED
        if self.condition == "failure":
            return result.status in (
                StageState.FAILED,
                StageState.CANCELLED,
                StageState.SKIPPED,
            )
        return False  # fallback

    @classmethod
    def on_success(cls, stage_name: str) -> "ExecutionOrderDependency":
        return cls(stage_name=stage_name, condition="success")

    @classmethod
    def on_failure(cls, stage_name: str) -> "ExecutionOrderDependency":
        return cls(stage_name=stage_name, condition="failure")

    @classmethod
    def always_after(cls, stage_name: str) -> "ExecutionOrderDependency":
        return cls(stage_name=stage_name, condition="always")


class StageTransitionRule(BaseModel):
    stage_name: str = Field(...)
    regular_dependencies: Set[str] = Field(default_factory=set)
    execution_order_dependencies: List[ExecutionOrderDependency] = Field(
        default_factory=list
    )

    def can_transition_to_ready(self, program: Program) -> bool:
        """Decide readiness based on execution-order dependencies only.

        Regular (dataflow) edges do not gate readiness. Readiness for dataflow
        is handled by the DAG runner using join policy and input bounds.
        """
        if not self.execution_order_dependencies:
            return True

        # Execution-order: each must be satisfied
        for dep in self.execution_order_dependencies:
            result = program.stage_results.get(dep.stage_name)
            if not dep.is_satisfied(result):
                return False

        return True

    def should_skip(self, program: Program) -> bool:
        """Decide auto-skip only based on execution-order deps not satisfied.

        Regular (dataflow) edges never cause auto-skip. The DAG runner will
        skip a stage if its join policy and input bounds are not met at launch.
        """
        if not self.execution_order_dependencies:
            return False

        for dep in self.execution_order_dependencies:
            result = program.stage_results.get(dep.stage_name)
            if (
                result is not None
                and not dep.is_satisfied(result)
                and dep.condition != "always"
            ):
                return True

        return False


class DAGAutomata(BaseModel):
    transition_rules: Dict[str, StageTransitionRule] = Field(
        default_factory=dict
    )

    def add_regular_dependency(self, stage_name: str, dependency: str) -> None:
        rules = cast(Dict[str, StageTransitionRule], self.transition_rules)
        rules.setdefault(
            stage_name, StageTransitionRule(stage_name=stage_name)
        ).regular_dependencies.add(dependency)

    def add_execution_order_dependency(
        self, stage_name: str, dependency: ExecutionOrderDependency
    ) -> None:
        rules = cast(Dict[str, StageTransitionRule], self.transition_rules)
        rules.setdefault(
            stage_name, StageTransitionRule(stage_name=stage_name)
        ).execution_order_dependencies.append(dependency)

    def get_ready_stages(
        self,
        program: Program,
        available_stages: Set[str],
        done: Set[str],
        running: Set[str],
        skipped: Set[str],
    ) -> Set[str]:
        rules = cast(Dict[str, StageTransitionRule], self.transition_rules)
        return {
            stage_name
            for stage_name in available_stages - done - running - skipped
            if (rule := rules.get(stage_name)) is None
            or rule.can_transition_to_ready(program)
        }

    def get_stages_to_skip(
        self,
        program: Program,
        available_stages: Set[str],
        done: Set[str],
        running: Set[str],
        skipped: Set[str],
    ) -> Set[str]:
        rules = cast(Dict[str, StageTransitionRule], self.transition_rules)
        return {
            stage_name
            for stage_name in available_stages - done - running - skipped
            if (rule := rules.get(stage_name)) and rule.should_skip(program)
        }

    def create_skip_result(
        self, stage_name: str, program: Program
    ) -> ProgramStageResult:
        rules = cast(Dict[str, StageTransitionRule], self.transition_rules)
        rule = rules.get(stage_name)
        failed_deps = [
            dep
            for dep in (rule.regular_dependencies if rule else [])
            if (res := program.stage_results.get(dep))
            and res.status
            in (StageState.FAILED, StageState.CANCELLED, StageState.SKIPPED)
        ]
        failed_exec_deps = [
            f"{dep.stage_name} (expected {dep.condition})"
            for dep in (rule.execution_order_dependencies if rule else [])
            if not dep.is_satisfied(program.stage_results.get(dep.stage_name))
            and dep.condition != "always"
        ]

        context = []
        if failed_deps:
            context.append(f"Regular deps failed: {failed_deps}")
        if failed_exec_deps:
            context.append(
                f"Execution-order deps not satisfied: {failed_exec_deps}"
            )

        return ProgramStageResult(
            status=StageState.SKIPPED,
            error=format_error_for_llm(
                error="Stage skipped due to dependency issue",
                stage_name=stage_name,
                context="; ".join(context),
            ),
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    def get_stage_dependencies(self, stage_name: str) -> Dict[str, any]:
        rules = cast(Dict[str, StageTransitionRule], self.transition_rules)
        rule = rules.get(stage_name)
        return {
            "regular": list(rule.regular_dependencies) if rule else [],
            "execution_order": [
                {"stage": dep.stage_name, "condition": dep.condition}
                for dep in (rule.execution_order_dependencies if rule else [])
            ],
        }

    def validate_dependencies(self, available_stages: Set[str]) -> List[str]:
        errors = []
        rules = cast(Dict[str, StageTransitionRule], self.transition_rules)
        for stage_name, rule in rules.items():
            for dep in rule.regular_dependencies:
                if dep not in available_stages:
                    errors.append(
                        f"Stage '{stage_name}' has regular dependency on non-existent stage '{dep}'"
                    )
            for exec_dep in rule.execution_order_dependencies:
                if exec_dep.stage_name not in available_stages:
                    errors.append(
                        f"Stage '{stage_name}' has execution order dependency on non-existent stage '{exec_dep.stage_name}'"
                    )
        return errors
