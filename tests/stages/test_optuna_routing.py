"""Tests for OptunaPayloadBridge and PayloadResolver routing stages."""

from __future__ import annotations

from gigaevo.programs.core_types import StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import AnyContainer
from gigaevo.programs.stages.optimization.optuna.models import (
    OptunaOptimizationOutput,
)
from gigaevo.programs.stages.optimization.optuna.routing import (
    OptunaPayloadBridge,
    PayloadResolver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


def _optuna_output(*, best_program_output=None) -> OptunaOptimizationOutput:
    return OptunaOptimizationOutput(
        optimized_code="def solve(): return 42",
        best_scores={"score": 1.0},
        best_params={"x": 1.0},
        n_params=1,
        n_trials=10,
        search_space_summary=[],
        best_program_output=best_program_output,
    )


# ---------------------------------------------------------------------------
# OptunaPayloadBridge
# ---------------------------------------------------------------------------


class TestOptunaPayloadBridge:
    async def test_success_with_program_output(self):
        """Bridge succeeds when best_program_output is non-None."""
        stage = OptunaPayloadBridge(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs(
            {"optuna_output": _optuna_output(best_program_output=[1, 2, 3])}
        )
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == [1, 2, 3]

    async def test_fails_when_program_output_is_none(self):
        """Bridge raises when best_program_output is None."""
        stage = OptunaPayloadBridge(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"optuna_output": _optuna_output(best_program_output=None)})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED

    async def test_preserves_complex_output(self):
        """Bridge preserves dict/nested program output."""
        payload = {"points": [[1, 2], [3, 4]], "metadata": {"n": 2}}
        stage = OptunaPayloadBridge(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs(
            {"optuna_output": _optuna_output(best_program_output=payload)}
        )
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == payload


# ---------------------------------------------------------------------------
# PayloadResolver
# ---------------------------------------------------------------------------


class TestPayloadResolver:
    async def test_prefers_optuna_payload(self):
        """When both available, optuna_payload wins."""
        stage = PayloadResolver(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs(
            {
                "optuna_payload": AnyContainer(data="from_optuna"),
                "program_payload": AnyContainer(data="from_program"),
            }
        )
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == "from_optuna"

    async def test_falls_back_to_program_payload(self):
        """When optuna_payload is None, program_payload is used."""
        stage = PayloadResolver(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs(
            {
                "optuna_payload": None,
                "program_payload": AnyContainer(data="from_program"),
            }
        )
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == "from_program"

    async def test_fails_when_both_none(self):
        """No payload source available → FAILED."""
        stage = PayloadResolver(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs(
            {
                "optuna_payload": None,
                "program_payload": None,
            }
        )
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
