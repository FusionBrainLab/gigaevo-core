"""Tests for OptunaOptimizationStage.partial_result() — graceful-shutdown salvage.

When ``execute()`` is cancelled by the stage ``timeout`` after the Optuna
baseline trial has already populated the in-memory ``best_*`` state, the
salvage hook must return an ``OptunaOptimizationOutput`` so the stage reports
COMPLETED instead of throwing away usable optimization progress.

If the timeout trips before the baseline trial completes, no useful state
exists yet → ``partial_result`` returns None → the stage falls through to the
existing FAILED path.
"""

from __future__ import annotations

import asyncio
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.programs.core_types import StageState
from gigaevo.programs.program import Program
from gigaevo.programs.stages.optimization.optuna import (
    OptunaOptimizationOutput,
    OptunaOptimizationStage,
    OptunaSearchSpace,
    ParamSpec,
)
import gigaevo.programs.stages.optimization.optuna.stage as _optuna_stage_module


def _mock_llm(search_space: OptunaSearchSpace) -> MagicMock:
    structured_mock = AsyncMock()
    structured_mock.ainvoke = AsyncMock(return_value=search_space)
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_mock)
    return llm


@pytest.fixture
def quick_validator(tmp_path):
    vpath = tmp_path / "validator.py"
    vpath.write_text(
        textwrap.dedent("""\
            def validate(output):
                return {"score": float(output)}
        """)
    )
    return vpath


def _single_param_search_space() -> OptunaSearchSpace:
    return OptunaSearchSpace(
        parameters=[
            ParamSpec(
                name="x",
                initial_value=5.0,
                param_type="float",
                low=0.0,
                high=10.0,
                reason="test param",
            )
        ],
        modifications=[],
        reasoning="single-param test",
    )


_PARAMETERIZED_CODE = textwrap.dedent("""\
    def run_code():
        return _optuna_params["x"]
""")


def _build_stage(validator, *, timeout: float = 1.0) -> OptunaOptimizationStage:
    stage = OptunaOptimizationStage(
        llm=_mock_llm(_single_param_search_space()),
        validator_path=validator,
        score_key="score",
        n_trials=20,
        max_parallel=4,
        eval_timeout=1,
        timeout=timeout,
    )
    stage._apply_modifications = MagicMock(return_value=_PARAMETERIZED_CODE)

    async def _no_measure(*_args, **_kwargs):
        return 0.01

    stage._measure_baseline_runtime = _no_measure
    return stage


class TestOptunaPartialResultDirect:
    """Unit-level tests calling ``partial_result()`` directly."""

    async def test_returns_none_when_no_best_state(self, quick_validator):
        """Fresh stage with no compute() run → partial_result must return None."""
        stage = _build_stage(quick_validator)
        program = Program(code="def run_code():\n    return 5.0\n")
        result = await stage.partial_result(program)
        assert result is None

    async def test_returns_output_when_state_populated(self, quick_validator):
        """With ``_best_*`` populated, partial_result builds a valid output."""
        stage = _build_stage(quick_validator)
        program = Program(code="def run_code():\n    return 5.0\n")

        param_specs = _single_param_search_space().parameters
        stage._best_value = 5.0
        stage._best_scores = {"score": 5.0}
        stage._best_params = {"x": 5.0}
        stage._best_prog_output = 5.0
        stage._best_param_specs = param_specs
        stage._best_parameterized_code = _PARAMETERIZED_CODE

        result = await stage.partial_result(program)

        assert isinstance(result, OptunaOptimizationOutput)
        assert result.best_program_output == 5.0
        assert result.best_scores == {"score": 5.0}
        assert result.best_params == {"x": 5.0}
        assert result.n_params == 1
        assert "_optuna_params" not in result.optimized_code


class TestOptunaSalvageIntegration:
    """End-to-end via execute() — internal deadline neutralised so the OUTER
    ``asyncio.wait_for`` actually fires and the salvage path executes."""

    async def test_salvage_returns_completed_with_baseline_result(
        self, quick_validator, monkeypatch
    ):
        # Push the internal _DEADLINE far into the future so _run_optuna keeps
        # launching trials past the stage timeout; the outer asyncio.wait_for
        # is the one that cancels compute().
        monkeypatch.setattr(_optuna_stage_module, "_DEADLINE_GRACE_S", -3600)

        program = Program(code="def run_code():\n    return 5.0\n")
        stage = _build_stage(quick_validator, timeout=1.0)

        n_calls = {"i": 0}

        async def _baseline_ok_then_block(*_args, **_kwargs):
            n_calls["i"] += 1
            if n_calls["i"] == 1:
                return {"score": 5.0}, 5.0, None
            await asyncio.sleep(60.0)
            return None, None, "should never reach"

        stage._evaluate_single = _baseline_ok_then_block

        stage.attach_inputs({})
        result = await stage.execute(program)

        assert result.status == StageState.COMPLETED, (
            f"Expected salvage→COMPLETED, got {result.status}; "
            f"error={getattr(result, 'error', None)}"
        )
        assert result.output is not None
        assert isinstance(result.output, OptunaOptimizationOutput)
        assert result.output.best_program_output == 5.0
        assert result.output.best_scores.get("score") == 5.0
        assert "_optuna_params" not in result.output.optimized_code

    async def test_no_salvage_when_baseline_never_finishes(
        self, quick_validator, monkeypatch
    ):
        """Timeout fires before baseline completes → no best state → FAILED."""
        monkeypatch.setattr(_optuna_stage_module, "_DEADLINE_GRACE_S", -3600)

        program = Program(code="def run_code():\n    return 5.0\n")
        stage = _build_stage(quick_validator, timeout=0.5)

        async def _always_blocks(*_args, **_kwargs):
            await asyncio.sleep(60.0)
            return None, None, "blocked"

        stage._evaluate_single = _always_blocks

        stage.attach_inputs({})
        result = await stage.execute(program)

        assert result.status == StageState.FAILED, (
            f"Expected FAILED (no salvage state), got {result.status}"
        )
