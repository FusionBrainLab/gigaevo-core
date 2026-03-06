"""Edge-case and boundary tests for gigaevo/programs/stages/optimization/cma.py

Covers:
1. _should_extract filtering (skip_integers, min/max_abs_value, booleans).
2. _extract_constants (float, negative, no constants, skip_zero_one).
3. _substitute round-trip for int originals and negated values.
4. _ConstantSubstitutor._coerce round-trip for int originals.
5. _ConstantInfo frozen model and defaults.
6. _evaluate_population adaptive penalty via actual production code.
7. _fallback_penalty sign convention via actual production CMANumericalOptimizationStage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.programs.stages.optimization.cma import (
    CMANumericalOptimizationStage,
    _ConstantInfo,
    _ConstantSubstitutor,
    _extract_constants,
    _parameterize,
    _should_extract,
    _substitute,
)

# ═══════════════════════════════════════════════════════════════════════════
# _should_extract filtering
# ═══════════════════════════════════════════════════════════════════════════


class TestShouldExtract:
    def test_bool_never_extracted(self) -> None:
        """Booleans should never be extracted regardless of value."""
        assert (
            _should_extract(
                1.0,
                skip_zero_one=False,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=True,
            )
            is False
        )
        assert (
            _should_extract(
                0.0,
                skip_zero_one=False,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=False,
            )
            is False
        )

    def test_skip_zero_one(self) -> None:
        assert (
            _should_extract(
                0.0,
                skip_zero_one=True,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=0.0,
            )
            is False
        )
        assert (
            _should_extract(
                1.0,
                skip_zero_one=True,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=1.0,
            )
            is False
        )
        assert (
            _should_extract(
                -1.0,
                skip_zero_one=True,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=-1.0,
            )
            is False
        )
        # Other values pass
        assert (
            _should_extract(
                2.0,
                skip_zero_one=True,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=2.0,
            )
            is True
        )

    def test_skip_integers(self) -> None:
        assert (
            _should_extract(
                5.0,
                skip_zero_one=False,
                skip_integers=True,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=5,
            )
            is False
        )
        # Float that looks like int should still be extracted
        assert (
            _should_extract(
                5.0,
                skip_zero_one=False,
                skip_integers=True,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=5.0,
            )
            is True
        )

    def test_min_abs_value(self) -> None:
        assert (
            _should_extract(
                0.001,
                skip_zero_one=False,
                skip_integers=False,
                min_abs_value=0.01,
                max_abs_value=None,
                raw_value=0.001,
            )
            is False
        )
        assert (
            _should_extract(
                0.1,
                skip_zero_one=False,
                skip_integers=False,
                min_abs_value=0.01,
                max_abs_value=None,
                raw_value=0.1,
            )
            is True
        )

    def test_max_abs_value(self) -> None:
        assert (
            _should_extract(
                1e10,
                skip_zero_one=False,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=1e6,
                raw_value=1e10,
            )
            is False
        )
        assert (
            _should_extract(
                100.0,
                skip_zero_one=False,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=1e6,
                raw_value=100.0,
            )
            is True
        )

    def test_normal_float_extracted(self) -> None:
        assert (
            _should_extract(
                3.14,
                skip_zero_one=False,
                skip_integers=False,
                min_abs_value=None,
                max_abs_value=None,
                raw_value=3.14,
            )
            is True
        )


# ═══════════════════════════════════════════════════════════════════════════
# _extract_constants
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractConstants:
    def test_extracts_float_constants(self) -> None:
        code = "x = 3.14\ny = 2.71"
        tree, constants = _extract_constants(code)
        values = [c.value for c in constants]
        assert 3.14 in values
        assert 2.71 in values

    def test_extracts_negative_constants(self) -> None:
        code = "x = -5.0"
        tree, constants = _extract_constants(code)
        assert len(constants) >= 1
        neg_consts = [c for c in constants if c.is_negated]
        assert len(neg_consts) >= 1
        # Value is stored as the raw constant value (positive) from the Constant
        # node inside UnaryOp(-); CMA stores it as -abs for negated constants
        assert abs(neg_consts[0].value) == pytest.approx(5.0)

    def test_no_constants_returns_empty(self) -> None:
        code = "def f(): pass"
        tree, constants = _extract_constants(code)
        assert constants == []

    def test_skip_zero_one_filters(self) -> None:
        code = "x = 0.0\ny = 1.0\nz = 2.0"
        _, consts = _extract_constants(code, skip_zero_one=True)
        values = [c.value for c in consts]
        assert 0.0 not in values
        assert 1.0 not in values
        assert 2.0 in values


# ═══════════════════════════════════════════════════════════════════════════
# _substitute round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestSubstituteRoundTrip:
    def test_substitute_preserves_int_type(self) -> None:
        """Integer constants should remain int after substitution (for range())."""
        code = "n = 10\nfor i in range(n): pass"
        tree, constants = _extract_constants(code, skip_integers=False)
        int_consts = [c for c in constants if c.was_int]
        if int_consts:
            new_values = [c.value for c in constants]
            # Replace 10 with 15 (but as float from CMA-ES)
            for i, c in enumerate(constants):
                if c.value == 10.0:
                    new_values[i] = 15.3  # CMA returns float
            result = _substitute(tree, constants, new_values)
            # Should have int 15, not float 15.3
            assert "15" in result
            # Should not have 15.3
            assert "15.3" not in result

    def test_substitute_negative_value(self) -> None:
        code = "x = -3.14"
        tree, constants = _extract_constants(code)
        new_values = [-2.5 if c.is_negated else c.value for c in constants]
        result = _substitute(tree, constants, new_values)
        compiled = compile(result, "<test>", "exec")
        assert compiled is not None

    def test_parameterize_produces_valid_python(self) -> None:
        code = "a = 2.5\nb = -1.3"
        tree, constants = _extract_constants(code)
        if constants:
            param_code = _parameterize(tree, constants)
            compile(param_code, "<test>", "exec")
            assert "_cma_params" in param_code


# ═══════════════════════════════════════════════════════════════════════════
# _ConstantSubstitutor._coerce
# ═══════════════════════════════════════════════════════════════════════════


class TestConstantSubstitutorCoerce:
    def test_coerce_float_to_int_rounds(self) -> None:
        assert _ConstantSubstitutor._coerce(3.7, was_int=True) == 4
        assert isinstance(_ConstantSubstitutor._coerce(3.7, was_int=True), int)

    def test_coerce_float_stays_float(self) -> None:
        result = _ConstantSubstitutor._coerce(3.7, was_int=False)
        assert result == 3.7
        assert isinstance(result, float)

    def test_coerce_negative_rounds_correctly(self) -> None:
        assert _ConstantSubstitutor._coerce(-2.6, was_int=True) == -3


# ═══════════════════════════════════════════════════════════════════════════
# _ConstantInfo
# ═══════════════════════════════════════════════════════════════════════════


class TestConstantInfo:
    def test_frozen_model(self) -> None:
        """ConstantInfo is frozen (immutable)."""
        info = _ConstantInfo(value=3.14, lineno=1, col_offset=4)
        with pytest.raises(Exception):
            info.value = 2.0  # type: ignore

    def test_defaults(self) -> None:
        info = _ConstantInfo(value=1.0, lineno=1, col_offset=0)
        assert info.is_negated is False
        assert info.was_int is False

    def test_was_int_flag(self) -> None:
        info = _ConstantInfo(value=5.0, lineno=1, col_offset=0, was_int=True)
        assert info.was_int is True


# ═══════════════════════════════════════════════════════════════════════════
# _evaluate_population — tests the actual production penalty logic
# ═══════════════════════════════════════════════════════════════════════════


def _make_cma_stage(
    tmp_path: Path,
    *,
    minimize: bool = False,
    adaptive_penalty: bool = True,
    penalty_fitness: float | None = None,
) -> CMANumericalOptimizationStage:
    """Create a real CMA stage with a dummy validator file."""
    validator = tmp_path / "validator.py"
    validator.write_text("def validate(output): return {'score': 1.0}\n")
    return CMANumericalOptimizationStage(
        validator_path=validator,
        score_key="score",
        minimize=minimize,
        adaptive_penalty=adaptive_penalty,
        penalty_fitness=penalty_fitness,
        timeout=300,  # required by Stage base class
    )


class TestEvaluatePopulationPenalty:
    """Test adaptive penalty via the actual _evaluate_population method.

    Instead of reimplementing the penalty formula locally, we mock
    _evaluate_single and call the real _evaluate_population to verify
    the penalty is always strictly worse than valid fitnesses in CMA space.
    """

    @pytest.mark.asyncio
    async def test_adaptive_penalty_normal_scores_maximize(
        self, tmp_path: Path
    ) -> None:
        """Normal scores with maximize: penalty must exceed all valid CMA fitnesses."""
        stage = _make_cma_stage(tmp_path, minimize=False)

        async def mock_eval_single(code, candidate, ctx):
            score = candidate[0]  # use first param as score
            return {"score": score}, None

        stage._evaluate_single = mock_eval_single  # type: ignore[assignment]

        population = [[0.8], [0.9], [0.95]]
        fitnesses, n_ok = await stage._evaluate_population("", population, None)

        assert n_ok == 3
        # All fitnesses should be negated scores (maximize → CMA minimizes)
        assert len(fitnesses) == 3
        for f, p in zip(fitnesses, population):
            assert f == pytest.approx(-p[0])

    @pytest.mark.asyncio
    async def test_adaptive_penalty_with_failures(self, tmp_path: Path) -> None:
        """Failed evaluations get the adaptive penalty, which is worse than all valid."""
        stage = _make_cma_stage(tmp_path, minimize=False)

        call_count = 0

        async def mock_eval_single(code, candidate, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return None, "eval failed"  # second evaluation fails
            return {"score": candidate[0]}, None

        stage._evaluate_single = mock_eval_single  # type: ignore[assignment]

        population = [[0.8], [0.9], [0.95]]
        fitnesses, n_ok = await stage._evaluate_population("", population, None)

        assert n_ok == 2  # only 2 succeeded
        # The failed one (index 1) should have a penalty worse than all valid
        valid_fitnesses = [fitnesses[0], fitnesses[2]]
        penalty_fitness = fitnesses[1]
        assert penalty_fitness > max(valid_fitnesses), (
            f"Penalty {penalty_fitness} not > worst valid {max(valid_fitnesses)}"
        )

    @pytest.mark.asyncio
    async def test_adaptive_penalty_near_zero_scores(self, tmp_path: Path) -> None:
        """Near-zero scores: penalty gap should still be meaningful."""
        stage = _make_cma_stage(tmp_path, minimize=False)

        call_count = 0

        async def mock_eval_single(code, candidate, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                return None, "fail"
            return {"score": candidate[0]}, None

        stage._evaluate_single = mock_eval_single  # type: ignore[assignment]

        population = [[0.001], [0.002], [0.003]]
        fitnesses, n_ok = await stage._evaluate_population("", population, None)

        assert n_ok == 2
        valid = [f for i, f in enumerate(fitnesses) if i != 2]
        penalty = fitnesses[2]
        gap = penalty - max(valid)
        assert gap > 0.001, f"Penalty gap {gap} too small for near-zero scores"

    @pytest.mark.asyncio
    async def test_adaptive_penalty_all_zero_scores_uses_scale_fallback(
        self, tmp_path: Path
    ) -> None:
        """All zero scores: scale should fall back to 1.0 via the `or 1.0` guard."""
        stage = _make_cma_stage(tmp_path, minimize=False)

        call_count = 0

        async def mock_eval_single(code, candidate, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                return None, "fail"
            return {"score": 0.0}, None

        stage._evaluate_single = mock_eval_single  # type: ignore[assignment]

        population = [[0.0], [0.0], [0.0]]
        fitnesses, n_ok = await stage._evaluate_population("", population, None)

        assert n_ok == 2
        penalty = fitnesses[2]
        # Valid CMA fitnesses are -0.0 = 0.0; penalty = 0.0 + 3.0 * 1.0 = 3.0
        assert penalty == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_all_evaluations_fail_uses_fallback_penalty(
        self, tmp_path: Path
    ) -> None:
        """When every evaluation fails, the fallback penalty is used."""
        stage = _make_cma_stage(tmp_path, minimize=False, penalty_fitness=0.0)

        async def mock_eval_single(code, candidate, ctx):
            return None, "all fail"

        stage._evaluate_single = mock_eval_single  # type: ignore[assignment]

        population = [[0.8], [0.9]]
        fitnesses, n_ok = await stage._evaluate_population("", population, None)

        assert n_ok == 0
        # All should get the fallback penalty
        assert all(f == stage._fallback_penalty for f in fitnesses)

    @pytest.mark.asyncio
    async def test_adaptive_penalty_minimize_mode(self, tmp_path: Path) -> None:
        """minimize=True: CMA sees raw scores. Penalty must exceed worst valid."""
        stage = _make_cma_stage(tmp_path, minimize=True)

        call_count = 0

        async def mock_eval_single(code, candidate, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return None, "fail"
            return {"score": candidate[0]}, None

        stage._evaluate_single = mock_eval_single  # type: ignore[assignment]

        population = [[0.1], [0.2], [0.3]]
        fitnesses, n_ok = await stage._evaluate_population("", population, None)

        assert n_ok == 2
        valid = [fitnesses[0], fitnesses[2]]
        penalty = fitnesses[1]
        # In minimize mode, raw scores are CMA fitnesses — penalty must be higher
        assert penalty > max(valid)


# ═══════════════════════════════════════════════════════════════════════════
# Sign convention — tests the actual _fallback_penalty and _evaluate_population
# ═══════════════════════════════════════════════════════════════════════════


class TestSignConvention:
    """Verify that minimize/maximize signs are handled correctly in the
    actual CMANumericalOptimizationStage, not in a local reimplementation.
    """

    def test_fallback_penalty_minimize_true(self, tmp_path: Path) -> None:
        """minimize=True with penalty_fitness: _fallback_penalty = penalty_fitness."""
        stage = _make_cma_stage(tmp_path, minimize=True, penalty_fitness=1e6)
        assert stage._fallback_penalty == 1e6

    def test_fallback_penalty_minimize_false(self, tmp_path: Path) -> None:
        """minimize=False with penalty_fitness: _fallback_penalty = -penalty_fitness."""
        stage = _make_cma_stage(tmp_path, minimize=False, penalty_fitness=1e6)
        assert stage._fallback_penalty == -1e6

    def test_fallback_penalty_default(self, tmp_path: Path) -> None:
        """No penalty_fitness: _fallback_penalty defaults to 1e18."""
        stage = _make_cma_stage(tmp_path)
        assert stage._fallback_penalty == 1e18

    @pytest.mark.asyncio
    async def test_maximize_negates_scores_in_cma_space(self, tmp_path: Path) -> None:
        """minimize=False: valid CMA fitnesses should be negated scores."""
        stage = _make_cma_stage(tmp_path, minimize=False)

        async def mock_eval_single(code, candidate, ctx):
            return {"score": candidate[0]}, None

        stage._evaluate_single = mock_eval_single  # type: ignore[assignment]

        population = [[0.3], [0.7]]
        fitnesses, n_ok = await stage._evaluate_population("", population, None)

        assert n_ok == 2
        assert fitnesses[0] == pytest.approx(-0.3)
        assert fitnesses[1] == pytest.approx(-0.7)

    @pytest.mark.asyncio
    async def test_minimize_preserves_raw_scores_in_cma_space(
        self, tmp_path: Path
    ) -> None:
        """minimize=True: valid CMA fitnesses should equal raw scores."""
        stage = _make_cma_stage(tmp_path, minimize=True)

        async def mock_eval_single(code, candidate, ctx):
            return {"score": candidate[0]}, None

        stage._evaluate_single = mock_eval_single  # type: ignore[assignment]

        population = [[0.3], [0.7]]
        fitnesses, n_ok = await stage._evaluate_population("", population, None)

        assert n_ok == 2
        assert fitnesses[0] == pytest.approx(0.3)
        assert fitnesses[1] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_penalty_sign_consistent_with_valid_fitnesses(
        self, tmp_path: Path
    ) -> None:
        """Penalty must be in the same sign space as valid fitnesses — always
        worse (higher in CMA-minimize space) regardless of minimize setting."""
        for minimize in (True, False):
            stage = _make_cma_stage(tmp_path, minimize=minimize)

            call_count = 0

            async def mock_eval_single(code, candidate, ctx):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    return None, "fail"
                return {"score": candidate[0]}, None

            stage._evaluate_single = mock_eval_single  # type: ignore[assignment]
            call_count = 0  # reset for each iteration

            population = [[0.5], [0.6], [0.7]]
            fitnesses, n_ok = await stage._evaluate_population("", population, None)

            assert n_ok == 2
            valid = [fitnesses[0], fitnesses[2]]
            penalty = fitnesses[1]
            assert penalty > max(valid), (
                f"minimize={minimize}: penalty {penalty} not > worst valid {max(valid)}"
            )
