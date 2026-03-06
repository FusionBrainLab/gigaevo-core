"""DAG integration tests for Optuna program-output bypass.

Exercises the routing stages (OptunaPayloadBridge, PayloadResolver) within
a real DAG run, verifying both the success-cascade (Optuna succeeds →
CallProgramFunction skipped) and the failure-cascade (Optuna fails →
CallProgramFunction runs as fallback).

Uses mock stages that mimic the relevant I/O types.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from gigaevo.programs.core_types import (
    StageIO,
    StageState,
    VoidInput,
)
from gigaevo.programs.dag.automata import (
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import AnyContainer, Box
from gigaevo.programs.stages.optimization.optuna.models import (
    OptunaOptimizationOutput,
)
from gigaevo.programs.stages.optimization.optuna.routing import (
    OptunaPayloadBridge,
    PayloadResolver,
)
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Mock I/O types
# ---------------------------------------------------------------------------


class ValidatorPayloadInput(StageIO):
    """Mimics CallValidatorFunction.InputsModel (payload field)."""

    payload: AnyContainer
    context: Optional[AnyContainer] = None


class ValidatorOutput(StageIO):
    """Captures the payload that was received."""

    received_payload: Any = None


# ---------------------------------------------------------------------------
# Mock stages
# ---------------------------------------------------------------------------


class MockOptunaSuccess(Stage):
    """Mimics OptunaOptStage that completes successfully with captured output."""

    InputsModel = VoidInput
    OutputModel = OptunaOptimizationOutput

    PROGRAM_OUTPUT: Any = [1, 2, 3]

    async def compute(self, program: Program) -> OptunaOptimizationOutput:
        return OptunaOptimizationOutput(
            optimized_code="def solve(): return 42",
            best_scores={"score": 99.0},
            best_params={"x": 1.0},
            n_params=1,
            n_trials=10,
            search_space_summary=[],
            best_program_output=self.PROGRAM_OUTPUT,
        )


class MockOptunaFail(Stage):
    """Mimics OptunaOptStage that fails (e.g. timeout)."""

    InputsModel = VoidInput
    OutputModel = OptunaOptimizationOutput

    async def compute(self, program: Program) -> OptunaOptimizationOutput:
        raise RuntimeError("Optuna timed out")


class MockCallProgram(Stage):
    """Mimics CallProgramFunction — produces Box[Any]."""

    InputsModel = VoidInput
    OutputModel = Box[Any]

    PROGRAM_OUTPUT: Any = "fallback_output"
    call_count: int = 0

    async def compute(self, program: Program) -> Box[Any]:
        MockCallProgram.call_count += 1
        return Box[Any](data=self.PROGRAM_OUTPUT)


class MockValidator(Stage):
    """Mimics CallValidatorFunction — consumes payload, records it."""

    InputsModel = ValidatorPayloadInput
    OutputModel = ValidatorOutput
    received: Any = None

    async def compute(self, program: Program) -> ValidatorOutput:
        MockValidator.received = self.params.payload.data
        return ValidatorOutput(received_payload=self.params.payload.data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dag(
    nodes: dict,
    edges: list,
    state_manager,
    *,
    exec_deps: dict | None = None,
    **kwargs,
) -> DAG:
    return DAG(
        nodes=nodes,
        data_flow_edges=edges,
        execution_order_deps=exec_deps,
        state_manager=state_manager,
        writer=NullWriter(),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _reset_counters():
    MockCallProgram.call_count = 0
    MockValidator.received = None
    yield
    MockCallProgram.call_count = 0
    MockValidator.received = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOptunaBypassSuccessCascade:
    """Optuna succeeds → bridge captures output → resolver picks it →
    CallProgramFunction SKIPPED → validator runs with captured payload."""

    async def test_success_cascade(self, state_manager, make_program):
        program = make_program()

        nodes = {
            "OptunaOptStage": MockOptunaSuccess(timeout=5.0),
            "OptunaPayloadBridge": OptunaPayloadBridge(timeout=5.0),
            "CallProgramFunction": MockCallProgram(timeout=5.0),
            "PayloadResolver": PayloadResolver(timeout=5.0),
            "CallValidator": MockValidator(timeout=5.0),
        }
        for s in nodes.values():
            s.__class__.cache_handler = NO_CACHE

        edges = [
            DataFlowEdge.create(
                "OptunaOptStage", "OptunaPayloadBridge", "optuna_output"
            ),
            DataFlowEdge.create(
                "OptunaPayloadBridge", "PayloadResolver", "optuna_payload"
            ),
            DataFlowEdge.create(
                "CallProgramFunction", "PayloadResolver", "program_payload"
            ),
            DataFlowEdge.create("PayloadResolver", "CallValidator", "payload"),
        ]

        exec_deps = {
            "CallProgramFunction": [
                ExecutionOrderDependency.on_failure("OptunaOptStage"),
            ],
        }

        dag = _make_dag(nodes, edges, state_manager, exec_deps=exec_deps)
        await dag.run(program)

        results = program.stage_results

        # Optuna succeeded
        assert results["OptunaOptStage"].status == StageState.COMPLETED
        # Bridge extracted the payload
        assert results["OptunaPayloadBridge"].status == StageState.COMPLETED
        # CallProgramFunction was SKIPPED (on_failure dep, but Optuna succeeded)
        assert results["CallProgramFunction"].status == StageState.SKIPPED
        # Resolver picked optuna path
        assert results["PayloadResolver"].status == StageState.COMPLETED
        # Validator ran with the captured program output
        assert results["CallValidator"].status == StageState.COMPLETED
        assert MockValidator.received == MockOptunaSuccess.PROGRAM_OUTPUT
        # CallProgramFunction was never called
        assert MockCallProgram.call_count == 0


class TestOptunaBypassFallbackCascade:
    """Optuna fails → bridge SKIPPED → CallProgramFunction runs →
    resolver picks program_payload → validator runs normally."""

    async def test_fallback_cascade(self, state_manager, make_program):
        program = make_program()

        nodes = {
            "OptunaOptStage": MockOptunaFail(timeout=5.0),
            "OptunaPayloadBridge": OptunaPayloadBridge(timeout=5.0),
            "CallProgramFunction": MockCallProgram(timeout=5.0),
            "PayloadResolver": PayloadResolver(timeout=5.0),
            "CallValidator": MockValidator(timeout=5.0),
        }
        for s in nodes.values():
            s.__class__.cache_handler = NO_CACHE

        edges = [
            DataFlowEdge.create(
                "OptunaOptStage", "OptunaPayloadBridge", "optuna_output"
            ),
            DataFlowEdge.create(
                "OptunaPayloadBridge", "PayloadResolver", "optuna_payload"
            ),
            DataFlowEdge.create(
                "CallProgramFunction", "PayloadResolver", "program_payload"
            ),
            DataFlowEdge.create("PayloadResolver", "CallValidator", "payload"),
        ]

        exec_deps = {
            "CallProgramFunction": [
                ExecutionOrderDependency.on_failure("OptunaOptStage"),
            ],
        }

        dag = _make_dag(nodes, edges, state_manager, exec_deps=exec_deps)
        await dag.run(program)

        results = program.stage_results

        # Optuna failed
        assert results["OptunaOptStage"].status == StageState.FAILED
        # Bridge skipped (required dep impossible)
        assert results["OptunaPayloadBridge"].status == StageState.SKIPPED
        # CallProgramFunction ran (on_failure dep satisfied)
        assert results["CallProgramFunction"].status == StageState.COMPLETED
        assert MockCallProgram.call_count == 1
        # Resolver picked fallback path
        assert results["PayloadResolver"].status == StageState.COMPLETED
        # Validator ran with program output
        assert results["CallValidator"].status == StageState.COMPLETED
        assert MockValidator.received == MockCallProgram.PROGRAM_OUTPUT
