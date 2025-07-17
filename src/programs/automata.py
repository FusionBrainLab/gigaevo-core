from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.utils import format_error_for_llm


class ExecutionOrderDependency(BaseModel):
    stage_name: str = Field(..., description="Name of the stage this dependency refers to")
    condition: Literal["success", "failure", "always"] = Field(..., description="When this dependency is considered satisfied")

    def is_satisfied(self, result: Optional[ProgramStageResult]) -> bool:
        if result is None or result.status in (StageState.PENDING, StageState.RUNNING):
            return False
        if self.condition == "always":
            return True
        if self.condition == "success":
            return result.status == StageState.COMPLETED
        if self.condition == "failure":
            return result.status in (StageState.FAILED, StageState.CANCELLED, StageState.SKIPPED)
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
    execution_order_dependencies: List[ExecutionOrderDependency] = Field(default_factory=list)

    def can_transition_to_ready(self, program: Program) -> bool:
        if not self.regular_dependencies and not self.execution_order_dependencies:
            return True

        # Regular: must all be COMPLETED
        for dep in self.regular_dependencies:
            result = program.stage_results.get(dep)
            if result is None or result.status != StageState.COMPLETED:
                return False

        # Execution-order: each must be satisfied
        for dep in self.execution_order_dependencies:
            result = program.stage_results.get(dep.stage_name)
            if not dep.is_satisfied(result):
                return False

        return True
    
    def should_skip(self, program: Program) -> bool:
        if not self.regular_dependencies and not self.execution_order_dependencies:
            return False

        for dep in self.regular_dependencies:
            result = program.stage_results.get(dep)
            if result and result.status in (StageState.FAILED, StageState.CANCELLED, StageState.SKIPPED):
                return True

        for dep in self.execution_order_dependencies:
            result = program.stage_results.get(dep.stage_name)
            if result is not None and not dep.is_satisfied(result) and dep.condition != "always":
                return True

        return False


class DAGAutomata(BaseModel):
    transition_rules: Dict[str, StageTransitionRule] = Field(default_factory=dict)

    def add_regular_dependency(self, stage_name: str, dependency: str) -> None:
        self.transition_rules.setdefault(stage_name, StageTransitionRule(stage_name=stage_name)).regular_dependencies.add(dependency)

    def add_execution_order_dependency(self, stage_name: str, dependency: ExecutionOrderDependency) -> None:
        self.transition_rules.setdefault(stage_name, StageTransitionRule(stage_name=stage_name)).execution_order_dependencies.append(dependency)

    def get_ready_stages(
        self,
        program: Program,
        available_stages: Set[str],
        done: Set[str],
        running: Set[str],
        skipped: Set[str],
    ) -> Set[str]:
        return {
            stage_name
            for stage_name in available_stages - done - running - skipped
            if (rule := self.transition_rules.get(stage_name)) is None or rule.can_transition_to_ready(program)
        }

    def get_stages_to_skip(
        self,
        program: Program,
        available_stages: Set[str],
        done: Set[str],
        running: Set[str],
        skipped: Set[str],
    ) -> Set[str]:
        return {
            stage_name
            for stage_name in available_stages - done - running - skipped
            if (rule := self.transition_rules.get(stage_name)) and rule.should_skip(program)
        }

    def create_skip_result(self, stage_name: str, program: Program) -> ProgramStageResult:
        rule = self.transition_rules.get(stage_name)
        failed_deps = [
            dep for dep in (rule.regular_dependencies if rule else [])
            if (res := program.stage_results.get(dep)) and res.status in (
                StageState.FAILED, StageState.CANCELLED, StageState.SKIPPED
            )
        ]
        failed_exec_deps = [
            f"{dep.stage_name} (expected {dep.condition})"
            for dep in (rule.execution_order_dependencies if rule else [])
            if not dep.is_satisfied(program.stage_results.get(dep.stage_name)) and dep.condition != "always"
        ]

        context = []
        if failed_deps:
            context.append(f"Regular deps failed: {failed_deps}")
        if failed_exec_deps:
            context.append(f"Execution-order deps not satisfied: {failed_exec_deps}")

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
        rule = self.transition_rules.get(stage_name)
        return {
            "regular": list(rule.regular_dependencies) if rule else [],
            "execution_order": [
                {"stage": dep.stage_name, "condition": dep.condition}
                for dep in (rule.execution_order_dependencies if rule else [])
            ],
        }

    def validate_dependencies(self, available_stages: Set[str]) -> List[str]:
        errors = []
        for stage_name, rule in self.transition_rules.items():
            for dep in rule.regular_dependencies:
                if dep not in available_stages:
                    errors.append(f"Stage '{stage_name}' has regular dependency on non-existent stage '{dep}'")
            for exec_dep in rule.execution_order_dependencies:
                if exec_dep.stage_name not in available_stages:
                    errors.append(f"Stage '{stage_name}' has execution order dependency on non-existent stage '{exec_dep.stage_name}'")
        return errors