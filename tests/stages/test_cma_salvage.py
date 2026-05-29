"""Tests for CMANumericalOptimizationStage.partial_result() — graceful-shutdown salvage.

When ``execute()`` is cancelled by the stage ``timeout`` after at least the
baseline evaluation has populated the in-memory ``_best_*`` state, the salvage
hook must return a ``CMAOptimizationOutput`` so the stage reports COMPLETED
instead of throwing away usable optimization progress.

If the timeout trips before any state is populated (constants not yet
extracted), ``partial_result`` returns None → FAILED via the existing path.
"""

from __future__ import annotations

import asyncio
import textwrap

import pytest

from gigaevo.programs.core_types import StageState
from gigaevo.programs.program import Program
from gigaevo.programs.stages.optimization.cma import (
    CMANumericalOptimizationStage,
    CMAOptimizationOutput,
)


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


_PROGRAM_CODE = textwrap.dedent("""\
    def run_code():
        a = 2.5
        b = 3.5
        return a + b
""")


def _build_stage(validator, *, timeout: float = 1.0) -> CMANumericalOptimizationStage:
    return CMANumericalOptimizationStage(
        validator_path=validator,
        score_key="score",
        minimize=True,
        max_generations=50,
        max_parallel=2,
        eval_timeout=1,
        timeout=timeout,
    )


class TestCMAPartialResultDirect:
    """Unit-level tests calling ``partial_result()`` directly."""

    async def test_returns_none_when_no_state(self, quick_validator):
        """Fresh stage with no compute() run → partial_result must return None."""
        stage = _build_stage(quick_validator)
        program = Program(code=_PROGRAM_CODE)
        result = await stage.partial_result(program)
        assert result is None

    async def test_returns_output_when_state_populated(self, quick_validator):
        """With ``_best_*`` populated, partial_result builds a valid output."""
        from gigaevo.programs.stages.optimization.cma import (
            _extract_constants,
            _parameterize,
        )

        stage = _build_stage(quick_validator)
        program = Program(code=_PROGRAM_CODE)

        tree, constants = _extract_constants(
            _PROGRAM_CODE,
            skip_zero_one=stage.skip_zero_one,
            skip_integers=stage.skip_integers,
        )
        _ = _parameterize(tree, constants)
        initial = [c.value for c in constants]

        stage._best_tree = tree
        stage._best_constants = constants
        stage._best_solution = [2.7, 3.3]
        stage._best_scores = {"score": 6.0}
        stage._best_generation = 5
        stage._best_initial_constants = list(initial)

        result = await stage.partial_result(program)

        assert isinstance(result, CMAOptimizationOutput)
        assert result.best_scores == {"score": 6.0}
        assert result.optimized_constants == [2.7, 3.3]
        assert result.n_constants == len(constants)
        assert result.n_generations == 5


class TestCMASalvageIntegration:
    """End-to-end via execute() — evaluator hangs so OUTER ``asyncio.wait_for``
    fires and salvage path executes."""

    async def test_salvage_returns_completed_after_baseline(
        self, quick_validator, monkeypatch
    ):
        program = Program(code=_PROGRAM_CODE)
        stage = _build_stage(quick_validator, timeout=1.0)

        call_count = {"i": 0}

        async def _baseline_ok_then_block(**kwargs):
            call_count["i"] += 1
            if call_count["i"] == 1:
                return {"score": 6.0}, None
            await asyncio.sleep(60.0)
            return None, "blocked"

        # _measure_baseline / baseline evaluation goes through utils.evaluate_single
        # while subsequent generation evals go through stage._evaluate_single.
        # Patch both: baseline returns fast, generation evals block.
        monkeypatch.setattr(
            "gigaevo.programs.stages.optimization.cma.evaluate_single",
            _baseline_ok_then_block,
        )

        async def _slow_eval(*_args, **_kwargs):
            await asyncio.sleep(60.0)
            return None, "blocked"

        stage._evaluate_single = _slow_eval

        stage.attach_inputs({})
        result = await stage.execute(program)

        assert result.status == StageState.COMPLETED, (
            f"Expected salvage→COMPLETED, got {result.status}; "
            f"error={getattr(result, 'error', None)}"
        )
        assert result.output is not None
        assert isinstance(result.output, CMAOptimizationOutput)
        assert result.output.best_scores.get("score") == 6.0
        assert result.output.n_constants == 2

    async def test_no_salvage_when_baseline_never_finishes(
        self, quick_validator, monkeypatch
    ):
        """If the timeout fires before the baseline eval populates ``_best_scores``,
        no useful state exists → ``partial_result`` returns None → FAILED."""
        program = Program(code=_PROGRAM_CODE)
        stage = _build_stage(quick_validator, timeout=0.3)

        async def _always_blocks(*_args, **_kwargs):
            await asyncio.sleep(60.0)
            return None, "blocked"

        monkeypatch.setattr(
            "gigaevo.programs.stages.optimization.cma.evaluate_single",
            _always_blocks,
        )

        stage.attach_inputs({})
        result = await stage.execute(program)

        assert result.status == StageState.FAILED, (
            f"Expected FAILED (no salvage state), got {result.status}"
        )
