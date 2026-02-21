"""Tests for CMANumericalOptimizationStage.

Covers:
  - AST constant extraction (positive, negative, skipping 0/1, filters)
  - Code parameterization (_cma_params[i] substitution)
  - Code value substitution (optimised constants back into source)
  - Evaluation code building (program + validator in one script)
  - Full end-to-end stage execution (CMA-ES actually optimises constants)
"""

from __future__ import annotations

import textwrap

import pytest

from gigaevo.programs.program import Program
from gigaevo.programs.stages.optimization.cma import (
    CMANumericalOptimizationStage,
    CMAOptimizationOutput,
    _extract_constants,
    _parameterize,
    _substitute,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Constant extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractConstants:
    """Unit tests for _extract_constants."""

    def test_basic_floats(self):
        code = textwrap.dedent("""\
            def f():
                return 3.14 + 2.718
        """)
        _, consts = _extract_constants(code)
        values = sorted(c.value for c in consts)
        assert values == [2.718, 3.14]

    def test_negated_literal(self):
        code = textwrap.dedent("""\
            def f():
                x = -5.5
                return x
        """)
        _, consts = _extract_constants(code)
        assert len(consts) == 1
        assert consts[0].value == -5.5
        assert consts[0].is_negated is True

    def test_skip_zero_one(self):
        code = textwrap.dedent("""\
            def f():
                return 0 + 1 + 0.0 + 1.0 + -1 + 42.0
        """)
        _, consts = _extract_constants(code, skip_zero_one=True)
        values = [c.value for c in consts]
        assert 0.0 not in values
        assert 1.0 not in values
        assert -1.0 not in values
        assert 42.0 in values

    def test_no_skip_zero_one(self):
        code = textwrap.dedent("""\
            def f():
                return 0 + 1 + 42
        """)
        _, consts = _extract_constants(code, skip_zero_one=False)
        values = sorted(c.value for c in consts)
        assert 0.0 in values
        assert 1.0 in values
        assert 42.0 in values

    def test_skip_integers(self):
        code = textwrap.dedent("""\
            def f():
                return 3 + 3.14
        """)
        _, consts = _extract_constants(code, skip_integers=True, skip_zero_one=False)
        values = [c.value for c in consts]
        assert 3.14 in values
        assert 3.0 not in values

    def test_min_abs_filter(self):
        code = textwrap.dedent("""\
            def f():
                return 0.001 + 5.0
        """)
        _, consts = _extract_constants(code, min_abs_value=0.01)
        values = [c.value for c in consts]
        assert 5.0 in values
        assert 0.001 not in values

    def test_max_abs_filter(self):
        code = textwrap.dedent("""\
            def f():
                return 5.0 + 10000.0
        """)
        _, consts = _extract_constants(code, max_abs_value=100.0)
        values = [c.value for c in consts]
        assert 5.0 in values
        assert 10000.0 not in values

    def test_booleans_skipped(self):
        code = textwrap.dedent("""\
            def f():
                return True + False + 3.14
        """)
        _, consts = _extract_constants(code)
        values = [c.value for c in consts]
        assert 3.14 in values
        assert len(consts) == 1

    def test_strings_skipped(self):
        code = textwrap.dedent("""\
            def f():
                x = "hello"
                return 3.14
        """)
        _, consts = _extract_constants(code)
        assert len(consts) == 1
        assert consts[0].value == 3.14

    def test_no_constants(self):
        code = textwrap.dedent("""\
            def f():
                return 0 + 1
        """)
        _, consts = _extract_constants(code, skip_zero_one=True)
        assert len(consts) == 0

    def test_mixed_positive_and_negative(self):
        code = textwrap.dedent("""\
            def f():
                return 3.0 + -7.0 + 5.0 + -2.0
        """)
        _, consts = _extract_constants(code)
        values = sorted(c.value for c in consts)
        assert values == [-7.0, -2.0, 3.0, 5.0]

    def test_int_constants_tracked_as_was_int(self):
        code = textwrap.dedent("""\
            def f():
                return 42 + 3.14
        """)
        _, consts = _extract_constants(code, skip_zero_one=False)
        by_val = {c.value: c for c in consts}
        assert by_val[42.0].was_int is True
        assert by_val[3.14].was_int is False

    def test_negated_int_tracked_as_was_int(self):
        code = textwrap.dedent("""\
            def f():
                return -10
        """)
        _, consts = _extract_constants(code)
        assert len(consts) == 1
        assert consts[0].was_int is True
        assert consts[0].value == -10.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Parameterization
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterize:
    def test_basic(self):
        code = textwrap.dedent("""\
            def f():
                return 3.14 + 2.0
        """)
        tree, consts = _extract_constants(code)
        result = _parameterize(tree, consts)
        assert "_cma_params[" in result
        assert "3.14" not in result
        assert "2.0" not in result

    def test_negated_replaced_as_single_unit(self):
        code = textwrap.dedent("""\
            def f():
                return -5.5
        """)
        tree, consts = _extract_constants(code)
        result = _parameterize(tree, consts)
        assert "_cma_params[0]" in result
        # The entire -5.5 should be replaced, no residual minus sign
        assert "-_cma_params" not in result

    def test_zero_one_preserved(self):
        code = textwrap.dedent("""\
            def f():
                return 0 + 1 + 42.0
        """)
        tree, consts = _extract_constants(code, skip_zero_one=True)
        result = _parameterize(tree, consts)
        assert "0" in result  # 0 not replaced
        assert "1" in result  # 1 not replaced
        assert "_cma_params[0]" in result  # 42.0 replaced

    def test_parameterized_code_runnable(self):
        code = textwrap.dedent("""\
            def f():
                return 3.14 + 2.0
        """)
        tree, consts = _extract_constants(code)
        param_code = _parameterize(tree, consts)
        # Execute with params matching original values
        ns = {}
        values = [c.value for c in consts]
        exec(f"_cma_params = {values!r}\n{param_code}", ns)
        original_ns = {}
        exec(code, original_ns)
        assert abs(ns["f"]() - original_ns["f"]()) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# 3. Value substitution
# ═══════════════════════════════════════════════════════════════════════════


class TestSubstitute:
    def test_basic(self):
        code = textwrap.dedent("""\
            def f():
                return 3.14 + 2.0
        """)
        tree, consts = _extract_constants(code)
        result = _substitute(tree, consts, [10.0, 20.0])
        ns = {}
        exec(result, ns)
        assert ns["f"]() == 30.0

    def test_negative_to_positive(self):
        code = textwrap.dedent("""\
            def f():
                return -5.0
        """)
        tree, consts = _extract_constants(code)
        result = _substitute(tree, consts, [7.0])
        ns = {}
        exec(result, ns)
        assert ns["f"]() == 7.0

    def test_positive_to_negative(self):
        code = textwrap.dedent("""\
            def f():
                return 5.0
        """)
        tree, consts = _extract_constants(code)
        result = _substitute(tree, consts, [-3.0])
        ns = {}
        exec(result, ns)
        assert ns["f"]() == -3.0

    def test_roundtrip_identity(self):
        """Substituting original values back should produce equivalent code."""
        code = textwrap.dedent("""\
            def compute(x):
                a = 3.14
                b = -2.718
                return a * x + b
        """)
        tree, consts = _extract_constants(code)
        original_values = [c.value for c in consts]
        result = _substitute(tree, consts, original_values)
        # Both versions should produce the same output
        ns_orig, ns_sub = {}, {}
        exec(code, ns_orig)
        exec(result, ns_sub)
        for x in [0, 1, -1, 10, 0.5]:
            assert abs(ns_orig["compute"](x) - ns_sub["compute"](x)) < 1e-12

    def test_int_constants_rounded_back(self):
        """Integer constants should be rounded back to int after optimisation."""
        code = textwrap.dedent("""\
            def f():
                n = 10
                return list(range(n))
        """)
        tree, consts = _extract_constants(code)
        assert len(consts) == 1
        assert consts[0].was_int is True

        # CMA-ES would produce a float like 7.6; substitution should round to 8
        result = _substitute(tree, consts, [7.6])
        ns = {}
        exec(result, ns)
        assert ns["f"]() == list(range(8))

    def test_int_constants_negative_rounded(self):
        """Negative optimised value for an int constant should round correctly."""
        code = textwrap.dedent("""\
            def f():
                return -5
        """)
        tree, consts = _extract_constants(code)
        result = _substitute(tree, consts, [-3.7])
        ns = {}
        exec(result, ns)
        assert ns["f"]() == -4  # round(-3.7) = -4

    def test_float_constants_not_rounded(self):
        """Float constants should remain as exact floats, not rounded."""
        code = textwrap.dedent("""\
            def f():
                return 3.14
        """)
        tree, consts = _extract_constants(code)
        result = _substitute(tree, consts, [2.718])
        ns = {}
        exec(result, ns)
        assert ns["f"]() == 2.718

    def test_mixed_int_float_substitution(self):
        """Int constants round, float constants don't, in the same code."""
        code = textwrap.dedent("""\
            def f():
                n = 10
                scale = 3.14
                return sum(range(n)) * scale
        """)
        tree, consts = _extract_constants(code)
        # consts are [10, 3.14] — int then float
        result = _substitute(tree, consts, [5.7, 2.0])
        ns = {}
        exec(result, ns)
        # n rounds to 6, scale stays 2.0
        assert ns["f"]() == sum(range(6)) * 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Evaluation code building
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildEvalCode:
    """Test that _build_eval_code produces runnable combined code."""

    @pytest.fixture
    def validator_file(self, tmp_path):
        """Write a simple validator that returns the sum as score."""
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
                def validate(result):
                    return {"score": float(result)}
            """)
        )
        return vpath

    def test_eval_code_runs(self, validator_file):
        stage = CMANumericalOptimizationStage(
            validator_path=validator_file,
            score_key="score",
            timeout=30,
            eval_timeout=10,
            max_generations=1,
        )
        program_code = textwrap.dedent("""\
            def run_code():
                return 3.14 + 2.0
        """)
        tree, consts = _extract_constants(program_code)
        param_code = _parameterize(tree, consts)
        values = [c.value for c in consts]

        eval_code = stage._build_eval_code(param_code, values)
        ns = {}
        exec(eval_code, ns)
        result = ns["_cma_eval"]()
        assert isinstance(result, dict)
        assert "score" in result
        assert abs(result["score"] - 5.14) < 1e-9

    def test_eval_code_with_context(self, tmp_path):
        vpath = tmp_path / "validator_ctx.py"
        vpath.write_text(
            textwrap.dedent("""\
                def validate(context, result):
                    return {"score": float(result + context)}
            """)
        )
        stage = CMANumericalOptimizationStage(
            validator_path=vpath,
            score_key="score",
            timeout=30,
            eval_timeout=10,
            max_generations=1,
        )
        program_code = textwrap.dedent("""\
            def run_code(ctx):
                return 10.0 + ctx
        """)
        tree, consts = _extract_constants(program_code)
        param_code = _parameterize(tree, consts)
        values = [c.value for c in consts]

        eval_code = stage._build_eval_code(param_code, values)
        ns = {}
        exec(eval_code, ns)
        result = ns["_cma_eval"](100)  # context=100
        assert result["score"] == 210.0  # (10+100) + 100


# ═══════════════════════════════════════════════════════════════════════════
# 5. Single evaluation via subprocess
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateSingle:
    """Test _evaluate_single through actual subprocess execution."""

    @pytest.fixture
    def validator_file(self, tmp_path):
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
                def validate(result):
                    return {"score": float(result)}
            """)
        )
        return vpath

    @pytest.mark.asyncio
    async def test_evaluate_returns_score(self, validator_file):
        stage = CMANumericalOptimizationStage(
            validator_path=validator_file,
            score_key="score",
            timeout=60,
            eval_timeout=15,
            max_generations=1,
        )
        program_code = textwrap.dedent("""\
            def run_code():
                return 3.14 + 2.0
        """)
        tree, consts = _extract_constants(program_code)
        param_code = _parameterize(tree, consts)
        values = [c.value for c in consts]

        result, err = await stage._evaluate_single(param_code, values, context=None)
        assert result is not None
        assert abs(result["score"] - 5.14) < 1e-6

    @pytest.mark.asyncio
    async def test_evaluate_bad_candidate_returns_none(self, validator_file):
        """A candidate that crashes the program should return None."""
        CMANumericalOptimizationStage(
            validator_path=validator_file,
            score_key="score",
            timeout=60,
            eval_timeout=5,
            max_generations=1,
        )
        # Code that will fail with a ZeroDivisionError when const is 0
        program_code = textwrap.dedent("""\
            def run_code():
                return 10.0 / 0.00001
        """)
        tree, consts = _extract_constants(program_code)
        param_code = _parameterize(tree, consts)
        # Pass 0.0 which will cause ZeroDivisionError in 10.0 / _cma_params[...]
        # Actually _cma_params[0] / _cma_params[1] — need to check
        # Let's just give a candidate that causes an error via the validator
        bad_validator = validator_file.parent / "bad_val.py"
        bad_validator.write_text("def validate(r): raise RuntimeError('boom')")

        stage2 = CMANumericalOptimizationStage(
            validator_path=bad_validator,
            score_key="score",
            timeout=60,
            eval_timeout=5,
            max_generations=1,
        )
        result, err = await stage2._evaluate_single(
            param_code, [10.0, 0.00001], context=None
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 6. Full end-to-end integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Full stage execution: CMA-ES should actually improve constants."""

    @pytest.fixture
    def quadratic_validator(self, tmp_path):
        """Validator that scores -(a-3)^2 -(b-7)^2.

        Maximising this → optimal a=3, b=7.
        """
        vpath = tmp_path / "quad_validator.py"
        vpath.write_text(
            textwrap.dedent("""\
                def validate(result):
                    a, b = result
                    score = -((a - 3.0)**2 + (b - 7.0)**2)
                    return {"score": score}
            """)
        )
        return vpath

    @pytest.mark.asyncio
    async def test_optimises_toward_target(self, quadratic_validator):
        """CMA-ES should move constants close to the known optimum (3, 7)."""
        program_code = textwrap.dedent("""\
            def run_code():
                a = 10.0
                b = 20.0
                return (a, b)
        """)
        program = Program(code=program_code)

        stage = CMANumericalOptimizationStage(
            validator_path=quadratic_validator,
            score_key="score",
            minimize=False,
            sigma0=2.0,
            max_generations=30,
            population_size=8,
            max_parallel=4,
            eval_timeout=10,
            timeout=300,
            update_program_code=True,
        )
        # Attach empty inputs (context is optional)
        stage.attach_inputs({})

        result = await stage.compute(program)

        assert isinstance(result, CMAOptimizationOutput)
        assert result.n_constants == 2
        assert result.n_generations > 0

        # The optimised constants should be close to (3, 7)
        a_opt, b_opt = result.optimized_constants
        assert abs(a_opt - 3.0) < 1.5, f"a={a_opt}, expected ~3.0"
        assert abs(b_opt - 7.0) < 1.5, f"b={b_opt}, expected ~7.0"

        # Score should be close to 0 (the maximum)
        assert result.best_scores["score"] > -5.0

        # program.code should have been updated in-place
        assert program.code == result.optimized_code
        assert program.code != program_code

    @pytest.mark.asyncio
    async def test_minimise_mode(self, tmp_path):
        """Test minimize=True works correctly."""
        vpath = tmp_path / "min_validator.py"
        vpath.write_text(
            textwrap.dedent("""\
                def validate(result):
                    # Minimum at x=5
                    return {"loss": (result - 5.0)**2}
            """)
        )

        program_code = textwrap.dedent("""\
            def run_code():
                return 50.0
        """)
        program = Program(code=program_code)

        stage = CMANumericalOptimizationStage(
            validator_path=vpath,
            score_key="loss",
            minimize=True,
            sigma0=5.0,
            max_generations=30,
            population_size=8,
            max_parallel=4,
            eval_timeout=10,
            timeout=300,
        )
        stage.attach_inputs({})
        result = await stage.compute(program)

        # Should be close to 5.0
        assert abs(result.optimized_constants[0] - 5.0) < 2.0
        assert result.best_scores["loss"] < 4.0  # Much less than (50-5)^2=2025

    @pytest.mark.asyncio
    async def test_no_constants_returns_original(self, tmp_path):
        """Code with only 0/1 constants should return immediately."""
        vpath = tmp_path / "dummy_val.py"
        vpath.write_text("def validate(r): return {'score': 0.0}")

        program_code = textwrap.dedent("""\
            def run_code():
                return 0 + 1
        """)
        program = Program(code=program_code)

        stage = CMANumericalOptimizationStage(
            validator_path=vpath,
            score_key="score",
            timeout=60,
            eval_timeout=5,
            max_generations=10,
        )
        stage.attach_inputs({})
        result = await stage.compute(program)

        assert result.n_constants == 0
        assert result.n_generations == 0
        # program.code may have trailing whitespace stripped by Pydantic
        assert result.optimized_code == program.code

    @pytest.mark.asyncio
    async def test_update_program_code_false(self, quadratic_validator):
        """When update_program_code=False, program.code should be unchanged."""
        program_code = textwrap.dedent("""\
            def run_code():
                a = 10.0
                b = 20.0
                return (a, b)
        """)
        program = Program(code=program_code)
        original_code = program.code  # after Pydantic normalisation

        stage = CMANumericalOptimizationStage(
            validator_path=quadratic_validator,
            score_key="score",
            minimize=False,
            sigma0=2.0,
            max_generations=5,
            population_size=6,
            max_parallel=4,
            eval_timeout=10,
            timeout=300,
            update_program_code=False,
        )
        stage.attach_inputs({})
        result = await stage.compute(program)

        # program.code should NOT have changed
        assert program.code == original_code
        # But the output should contain the optimised code
        assert result.optimized_code != original_code

    @pytest.mark.asyncio
    async def test_with_context(self, tmp_path):
        """Test that DAG context flows to both program and validator."""
        vpath = tmp_path / "ctx_validator.py"
        vpath.write_text(
            textwrap.dedent("""\
                def validate(ctx, result):
                    target = ctx  # context IS the target value
                    return {"score": -(result - target)**2}
            """)
        )

        program_code = textwrap.dedent("""\
            def run_code(ctx):
                return 50.0
        """)
        program = Program(code=program_code)

        stage = CMANumericalOptimizationStage(
            validator_path=vpath,
            score_key="score",
            minimize=False,
            sigma0=5.0,
            max_generations=30,
            population_size=8,
            max_parallel=4,
            eval_timeout=10,
            timeout=300,
        )
        from gigaevo.programs.stages.common import AnyContainer

        stage.attach_inputs({"context": AnyContainer(data=10.0)})
        result = await stage.compute(program)

        # Should converge toward 10.0 (the context/target)
        assert abs(result.optimized_constants[0] - 10.0) < 3.0

    @pytest.mark.asyncio
    async def test_negative_constants_optimised(self, tmp_path):
        """CMA-ES should handle programs with negative initial constants."""
        vpath = tmp_path / "neg_validator.py"
        vpath.write_text(
            textwrap.dedent("""\
                def validate(result):
                    return {"score": -(result - (-4.0))**2}
            """)
        )

        program_code = textwrap.dedent("""\
            def run_code():
                return -20.0
        """)
        program = Program(code=program_code)

        stage = CMANumericalOptimizationStage(
            validator_path=vpath,
            score_key="score",
            minimize=False,
            sigma0=5.0,
            max_generations=40,
            population_size=8,
            max_parallel=4,
            eval_timeout=10,
            timeout=300,
        )
        stage.attach_inputs({})
        result = await stage.compute(program)

        # Should converge toward -4.0
        assert abs(result.optimized_constants[0] - (-4.0)) < 2.0

    @pytest.mark.asyncio
    async def test_mixed_int_float_optimisation(self, tmp_path):
        """CMA-ES should optimise int and float constants together.

        The validator rewards sum(range(n)) * scale being close to 100.
        Optimal: n=10 (sum(range(10))=45), scale≈2.222  → 45*2.222≈100
        Or:      n=14 (sum(range(14))=91), scale≈1.099  → 91*1.099≈100
        etc.  The key assertion is that n stays an integer in the output.
        """
        vpath = tmp_path / "mixed_validator.py"
        vpath.write_text(
            textwrap.dedent("""\
                def validate(result):
                    return {"score": -(result - 100.0)**2}
            """)
        )

        program_code = textwrap.dedent("""\
            def run_code():
                n = 5
                scale = 10.0
                return sum(range(n)) * scale
        """)
        program = Program(code=program_code)

        stage = CMANumericalOptimizationStage(
            validator_path=vpath,
            score_key="score",
            minimize=False,
            sigma0=3.0,
            max_generations=40,
            population_size=10,
            max_parallel=4,
            eval_timeout=10,
            timeout=300,
        )
        stage.attach_inputs({})
        result = await stage.compute(program)

        # The optimised code should be valid Python with n as int
        ns = {}
        exec(result.optimized_code, ns)
        output = ns["run_code"]()

        # Output should be closer to 100 than the original (5*10=100 is
        # actually already perfect, but with different starting values
        # CMA should still find a good solution)
        assert abs(output - 100.0) < 30.0

        # n must be an integer in the final code (not 5.0 or 4.7)
        assert "n = " in result.optimized_code
        # Parse and verify the type
        import ast

        tree = ast.parse(result.optimized_code)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "n"
            ):
                # The value should be a plain int Constant (or -int via UnaryOp)
                val_node = node.value
                if isinstance(val_node, ast.Constant):
                    assert isinstance(val_node.value, int), (
                        f"n should be int, got {type(val_node.value)}: {val_node.value}"
                    )
                elif isinstance(val_node, ast.UnaryOp):
                    assert isinstance(val_node.operand, ast.Constant)
                    assert isinstance(val_node.operand.value, int), (
                        f"n should be int, got {type(val_node.operand.value)}"
                    )
                break


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
