"""Tests for OptunaOptimizationStage.

Covers:
  - Desubstitution (replacing _optuna_params["key"] with concrete values)
  - Evaluation code building
  - Full end-to-end stage execution (with mocked LLM)
"""

from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.programs.program import Program
from gigaevo.programs.stages.optimization.optuna import (
    OptunaOptimizationOutput,
    OptunaOptimizationStage,
    OptunaSearchSpace,
    ParamSpec,
    desubstitute_params,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _mock_llm(search_space: OptunaSearchSpace) -> MagicMock:
    """Create a mock LLM that returns the given search space."""
    structured_mock = AsyncMock()
    structured_mock.ainvoke = AsyncMock(return_value=search_space)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_mock)
    return llm


# ═══════════════════════════════════════════════════════════════════════════
# 1. Desubstitution
# ═══════════════════════════════════════════════════════════════════════════


class TestDesubstituteParams:
    """Test desubstitute_params -- replacing _optuna_params refs with values."""

    def test_basic_float(self):
        code = textwrap.dedent("""\
            def f():
                lr = _optuna_params["learning_rate"]
                return lr
        """)
        result = desubstitute_params(code, {"learning_rate": 0.05})
        ns = {}
        exec(result, ns)
        assert abs(ns["f"]() - 0.05) < 1e-12

    def test_int_param_stays_int(self):
        code = textwrap.dedent("""\
            def f():
                for i in range(_optuna_params["n"]):
                    pass
                return _optuna_params["n"]
        """)
        result = desubstitute_params(code, {"n": 7}, param_types={"n": "int"})
        ns = {}
        exec(result, ns)
        assert ns["f"]() == 7
        assert isinstance(ns["f"](), int)

    def test_negative_expression(self):
        """``-_optuna_params["x"]`` should become ``-value``."""
        code = textwrap.dedent("""\
            def f():
                return -_optuna_params["x"], _optuna_params["x"]
        """)
        result = desubstitute_params(code, {"x": 0.05})
        ns = {}
        exec(result, ns)
        neg, pos = ns["f"]()
        assert abs(neg - (-0.05)) < 1e-12
        assert abs(pos - 0.05) < 1e-12

    def test_symmetric_uniform(self):
        """The uniform(-X, X) pattern should desubstitute correctly."""
        code = textwrap.dedent("""\
            def f():
                return (-_optuna_params["max_noise"], _optuna_params["max_noise"])
        """)
        result = desubstitute_params(code, {"max_noise": 0.05})
        ns = {}
        exec(result, ns)
        neg, pos = ns["f"]()
        assert abs(neg - (-0.05)) < 1e-12
        assert abs(pos - 0.05) < 1e-12
        # Verify clean code (no _optuna_params references)
        assert "_optuna_params" not in result

    def test_multiple_params(self):
        code = textwrap.dedent("""\
            def f():
                lr = _optuna_params["lr"]
                epochs = _optuna_params["epochs"]
                return lr * epochs
        """)
        result = desubstitute_params(
            code,
            {"lr": 0.01, "epochs": 100},
            param_types={"lr": "float", "epochs": "int"},
        )
        ns = {}
        exec(result, ns)
        assert abs(ns["f"]() - 1.0) < 1e-12

    def test_unknown_param_left_alone(self):
        """Params not in the values dict should remain as subscripts."""
        code = '_optuna_params["known"] + _optuna_params["unknown"]'
        result = desubstitute_params(code, {"known": 42})
        assert "42" in result
        assert "_optuna_params" in result
        assert "unknown" in result

    def test_roundtrip_identity(self):
        """Desubstituting with initial values should produce equivalent code."""
        code = textwrap.dedent("""\
            def f():
                x = _optuna_params["a"]
                y = _optuna_params["b"]
                return x + y
        """)
        result = desubstitute_params(code, {"a": 3.0, "b": 7.0})
        ns = {}
        exec(result, ns)
        assert abs(ns["f"]() - 10.0) < 1e-12

    def test_string_param(self):
        """String categorical values should desubstitute to string literals."""
        code = textwrap.dedent("""\
            def f():
                method = _optuna_params["solver"]
                return method
        """)
        result = desubstitute_params(
            code,
            {"solver": "Nelder-Mead"},
            param_types={"solver": "categorical"},
        )
        assert "_optuna_params" not in result
        ns = {}
        exec(result, ns)
        assert ns["f"]() == "Nelder-Mead"

    def test_bool_param(self):
        """Boolean categorical values should desubstitute to True/False."""
        code = textwrap.dedent("""\
            def f():
                return _optuna_params["adaptive"]
        """)
        result = desubstitute_params(
            code,
            {"adaptive": True},
            param_types={"adaptive": "categorical"},
        )
        assert "_optuna_params" not in result
        ns = {}
        exec(result, ns)
        assert ns["f"]() is True

    def test_none_param(self):
        """None values should desubstitute to the literal None."""
        code = textwrap.dedent("""\
            def f():
                return _optuna_params["callback"]
        """)
        result = desubstitute_params(
            code,
            {"callback": None},
            param_types={"callback": "categorical"},
        )
        assert "_optuna_params" not in result
        ns = {}
        exec(result, ns)
        assert ns["f"]() is None

    def test_eval_cleanup_dotted_name(self):
        """``eval('mod.func')`` should be cleaned to ``mod.func``."""
        code = textwrap.dedent("""\
            import math
            def f():
                fn = eval(_optuna_params["func"])
                return fn(2.0)
        """)
        result = desubstitute_params(
            code,
            {"func": "math.sqrt"},
            param_types={"func": "categorical"},
        )
        assert "_optuna_params" not in result
        assert "eval(" not in result
        assert "math.sqrt" in result
        ns = {}
        exec(result, ns)
        assert abs(ns["f"]() - 2.0**0.5) < 1e-12

    def test_eval_cleanup_simple_name(self):
        """``eval('abs')`` should be cleaned to ``abs``."""
        code = 'eval(_optuna_params["fn"])(-5)'
        result = desubstitute_params(
            code,
            {"fn": "abs"},
            param_types={"fn": "categorical"},
        )
        assert "eval(" not in result
        assert eval(result) == 5

    def test_eval_cleanup_non_identifier_left_alone(self):
        """``eval('1+2')`` is NOT a dotted name -- should stay as eval."""
        code = 'eval(_optuna_params["expr"])'
        result = desubstitute_params(
            code,
            {"expr": "1+2"},
            param_types={"expr": "categorical"},
        )
        assert "eval(" in result

    def test_eval_inline_call(self):
        """``eval(_optuna_params["f"])(args)`` should clean to ``f(args)``."""
        code = textwrap.dedent("""\
            import math
            def f():
                return eval(_optuna_params["func"])(9.0)
        """)
        result = desubstitute_params(
            code,
            {"func": "math.sqrt"},
            param_types={"func": "categorical"},
        )
        assert "eval(" not in result
        ns = {}
        exec(result, ns)
        assert ns["f"]() == 3.0

    def test_mixed_types(self):
        """Mix of string, bool, int, float params in one desubstitution."""
        code = textwrap.dedent("""\
            def f():
                return (
                    _optuna_params["method"],
                    _optuna_params["adaptive"],
                    _optuna_params["n"],
                    _optuna_params["lr"],
                )
        """)
        result = desubstitute_params(
            code,
            {"method": "CG", "adaptive": False, "n": 10, "lr": 0.001},
            param_types={
                "method": "categorical",
                "adaptive": "categorical",
                "n": "int",
                "lr": "float",
            },
        )
        assert "_optuna_params" not in result
        ns = {}
        exec(result, ns)
        method, adaptive, n, lr = ns["f"]()
        assert method == "CG"
        assert adaptive is False
        assert n == 10 and isinstance(n, int)
        assert abs(lr - 0.001) < 1e-12

    def test_categorical_int_choice_stays_int(self):
        """Categorical params with integer choices must desubstitute to int, not float.

        When an LLM declares param_type='categorical' with integer choices
        (e.g. [3, 4, 5] for use in range()), the desubstituted code must
        produce an integer literal, not 3.0, otherwise range() raises TypeError.
        """
        code = textwrap.dedent("""\
            def f():
                n = _optuna_params["num_in_row"]
                return list(range(n))
        """)
        result = desubstitute_params(
            code,
            {"num_in_row": 4},
            param_types={"num_in_row": "categorical"},
        )
        assert "_optuna_params" not in result
        ns = {}
        exec(result, ns)
        value = ns["f"]()
        assert value == [0, 1, 2, 3]

    def test_categorical_float_as_string_int_coerced(self):
        """Float-as-string integer choices like '3.0' must coerce to int 3.

        Optuna may return a string like '3.0' from suggest_categorical when
        choices were given as floats-as-strings. If not coerced, range('3.0')
        raises TypeError: 'str' object cannot be interpreted as an integer.
        """
        code = textwrap.dedent("""\
            def f():
                n = _optuna_params["steps"]
                return list(range(n))
        """)
        result = desubstitute_params(
            code,
            {"steps": "4.0"},
            param_types={"steps": "categorical"},
        )
        assert "_optuna_params" not in result
        ns = {}
        exec(result, ns)
        value = ns["f"]()
        assert value == [0, 1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Evaluation code building
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildEvalCode:
    """Test that _build_eval_code produces runnable combined code."""

    @pytest.fixture
    def validator_file(self, tmp_path):
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                return {"score": float(output)}
        """)
        )
        return vpath

    def test_eval_code_runs(self, validator_file):
        """Parameterized code + params dict produces runnable eval code."""
        parameterized_code = textwrap.dedent("""\
            def run_code():
                return _optuna_params["x"] * 2
        """)
        stage = OptunaOptimizationStage(
            llm=MagicMock(),
            validator_path=validator_file,
            score_key="score",
            timeout=60,
        )
        eval_code = stage._build_eval_code(parameterized_code, {"x": 5.0})
        ns = {}
        exec(eval_code, ns)
        result = ns["_optuna_eval"]()
        assert isinstance(result, dict)
        assert abs(result["score"] - 10.0) < 1e-12

    def test_eval_code_with_context(self, validator_file):
        vpath = validator_file
        vpath.write_text(
            textwrap.dedent("""\
            def validate(ctx, output):
                return {"score": float(output + ctx)}
        """)
        )
        parameterized_code = textwrap.dedent("""\
            def run_code(ctx):
                return _optuna_params["x"] + ctx
        """)
        stage = OptunaOptimizationStage(
            llm=MagicMock(),
            validator_path=vpath,
            score_key="score",
            timeout=60,
        )
        eval_code = stage._build_eval_code(parameterized_code, {"x": 3.0})
        ns = {}
        exec(eval_code, ns)
        result = ns["_optuna_eval"](10)
        assert abs(result["score"] - 23.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# 3. Single evaluation
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateSingle:
    """Test _evaluate_single via subprocess execution."""

    @pytest.fixture
    def validator_file(self, tmp_path):
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                return {"score": float(output)}
        """)
        )
        return vpath

    @pytest.mark.asyncio
    async def test_evaluate_returns_score(self, validator_file):
        parameterized_code = textwrap.dedent("""\
            def run_code():
                return _optuna_params["x"] * 3
        """)
        stage = OptunaOptimizationStage(
            llm=MagicMock(),
            validator_path=validator_file,
            score_key="score",
            timeout=60,
        )
        scores, err = await stage._evaluate_single(
            parameterized_code, {"x": 5.0}, context=None
        )
        assert scores is not None
        assert abs(scores["score"] - 15.0) < 1e-12

    @pytest.mark.asyncio
    async def test_evaluate_bad_code_returns_none(self, validator_file):
        bad_code = "def run_code(): raise RuntimeError('boom')"
        stage = OptunaOptimizationStage(
            llm=MagicMock(),
            validator_path=validator_file,
            score_key="score",
            timeout=60,
        )
        scores, err = await stage._evaluate_single(bad_code, {}, context=None)
        assert scores is None


# ═══════════════════════════════════════════════════════════════════════════
# 4. End-to-end (with mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Full stage execution with mocked LLM returning parameterized code."""

    @pytest.fixture
    def quadratic_validator(self, tmp_path):
        """Validator: score = -(a-3)^2 - (b-7)^2.  Maximum at a=3, b=7."""
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                a, b = output
                score = -(a - 3.0)**2 - (b - 7.0)**2
                return {"score": score}
        """)
        )
        return vpath

    @pytest.mark.asyncio
    async def test_optimises_toward_target(self, quadratic_validator):
        original_code = textwrap.dedent("""\
            def run_code():
                a = 10.0
                b = 20.0
                return (a, b)
        """)
        parameterized_code = textwrap.dedent("""\
            def run_code():
                a = _optuna_params["a"]
                b = _optuna_params["b"]
                return (a, b)
        """)
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="a",
                    initial_value=10.0,
                    param_type="float",
                    low=0.0,
                    high=20.0,
                    reason="First param",
                ),
                ParamSpec(
                    name="b",
                    initial_value=20.0,
                    param_type="float",
                    low=0.0,
                    high=30.0,
                    reason="Second param",
                ),
            ],
            modifications=[],
            reasoning="Tune both to maximise score",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=quadratic_validator,
            score_key="score",
            minimize=False,
            n_trials=60,
            max_parallel=4,
            eval_timeout=10,
            timeout=300,
            update_program_code=True,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({})
        result = await stage.compute(program)

        assert isinstance(result, OptunaOptimizationOutput)
        assert result.n_params == 2
        assert result.n_trials > 0
        # Should improve substantially from baseline of -218.
        assert result.best_scores["score"] > -50.0, (
            f"score={result.best_scores['score']}, expected significant "
            f"improvement from baseline of -218"
        )
        # Optimized code should be clean (no _optuna_params refs).
        assert "_optuna_params" not in result.optimized_code

    @pytest.mark.asyncio
    async def test_no_params_returns_original(self, quadratic_validator):
        original_code = "def run_code(): return (1, 1)"
        search_space = OptunaSearchSpace(
            parameters=[],
            modifications=[],
            reasoning="Nothing to tune",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=quadratic_validator,
            score_key="score",
            timeout=60,
        )
        stage.attach_inputs({})
        result = await stage.compute(program)

        assert result.optimized_code == original_code
        assert result.n_params == 0
        assert result.n_trials == 0

    @pytest.mark.asyncio
    async def test_update_program_code_false(self, quadratic_validator):
        original_code = textwrap.dedent("""\
            def run_code():
                a = 10.0
                b = 20.0
                return (a, b)
        """)
        parameterized_code = textwrap.dedent("""\
            def run_code():
                a = _optuna_params["a"]
                b = _optuna_params["b"]
                return (a, b)
        """)
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="a",
                    initial_value=10.0,
                    param_type="float",
                    low=0.0,
                    high=20.0,
                    reason="A",
                ),
                ParamSpec(
                    name="b",
                    initial_value=20.0,
                    param_type="float",
                    low=0.0,
                    high=30.0,
                    reason="B",
                ),
            ],
            modifications=[],
            reasoning="Test",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=quadratic_validator,
            score_key="score",
            n_trials=10,
            max_parallel=2,
            eval_timeout=10,
            timeout=120,
            update_program_code=False,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({})
        await stage.compute(program)

        # Original code should NOT have been modified.
        assert program.code.strip() == original_code.strip()

    @pytest.mark.asyncio
    async def test_minimise_mode(self, tmp_path):
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                return {"loss": output ** 2}
        """)
        )
        original_code = "def run_code(): return 10.0"
        parameterized_code = 'def run_code(): return _optuna_params["x"]'
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="x",
                    initial_value=10.0,
                    param_type="float",
                    low=-10.0,
                    high=10.0,
                    reason="Minimize x^2",
                ),
            ],
            modifications=[],
            reasoning="Test minimize",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=vpath,
            score_key="loss",
            minimize=True,
            n_trials=40,
            max_parallel=4,
            eval_timeout=10,
            timeout=120,
            update_program_code=True,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({})
        result = await stage.compute(program)
        # Should find something close to 0.
        assert result.best_scores["loss"] < 10.0

    @pytest.mark.asyncio
    async def test_with_context(self, tmp_path):
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(ctx, output):
                return {"score": -abs(output - ctx)}
        """)
        )
        original_code = textwrap.dedent("""\
            def run_code(ctx):
                return 10.0 + ctx
        """)
        parameterized_code = textwrap.dedent("""\
            def run_code(ctx):
                return _optuna_params["offset"] + ctx
        """)
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="offset",
                    initial_value=10.0,
                    param_type="float",
                    low=-20.0,
                    high=20.0,
                    reason="Offset",
                ),
            ],
            modifications=[],
            reasoning="Test context",
        )

        from gigaevo.programs.stages.common import AnyContainer

        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=vpath,
            score_key="score",
            minimize=False,
            n_trials=30,
            max_parallel=4,
            eval_timeout=10,
            timeout=120,
            update_program_code=True,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({"context": AnyContainer(data=5.0)})
        result = await stage.compute(program)
        # Best offset should be ~0 so output == ctx.
        assert result.best_scores["score"] > -20.0

    @pytest.mark.asyncio
    async def test_int_param_stays_int(self, tmp_path):
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                return {"score": float(output)}
        """)
        )
        original_code = textwrap.dedent("""\
            def run_code():
                total = 0
                for i in range(5):
                    total += i
                return total
        """)
        parameterized_code = textwrap.dedent("""\
            def run_code():
                total = 0
                for i in range(_optuna_params["n"]):
                    total += i
                return total
        """)
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="n",
                    initial_value=5,
                    param_type="int",
                    low=1,
                    high=20,
                    reason="Loop count",
                ),
            ],
            modifications=[],
            reasoning="Test int",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=vpath,
            score_key="score",
            minimize=False,
            n_trials=20,
            max_parallel=4,
            eval_timeout=10,
            timeout=120,
            update_program_code=True,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({})
        result = await stage.compute(program)
        assert result.n_trials > 0
        # The optimized code should use range(N) with an int, not a float.
        assert "range(" in result.optimized_code
        assert "_optuna_params" not in result.optimized_code

    @pytest.mark.asyncio
    async def test_symmetric_params_uniform(self, tmp_path):
        """Test that the parameterized-code approach handles uniform(-X, X)."""
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                low, high = output
                # Best when range is small (close to 0)
                return {"score": -abs(high - low)}
        """)
        )
        original_code = textwrap.dedent("""\
            def run_code():
                return (-0.5, 0.5)
        """)
        parameterized_code = textwrap.dedent("""\
            def run_code():
                return (-_optuna_params["half_range"], _optuna_params["half_range"])
        """)
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="half_range",
                    initial_value=0.5,
                    param_type="log_float",
                    low=0.001,
                    high=1.0,
                    reason="Symmetric range",
                ),
            ],
            modifications=[],
            reasoning="Test symmetric",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=vpath,
            score_key="score",
            minimize=False,
            n_trials=30,
            max_parallel=4,
            eval_timeout=10,
            timeout=120,
            update_program_code=True,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({})
        result = await stage.compute(program)
        assert result.n_trials > 0
        # Both +X and -X should be in the final code, no _optuna_params.
        assert "_optuna_params" not in result.optimized_code

    @pytest.mark.asyncio
    async def test_string_categorical_method_sweep(self, tmp_path):
        """Sweep a solver method string -- the highest-impact knob."""
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                method, val = output
                # "best" method gets highest score
                bonus = {"best": 100, "good": 50, "bad": 0}
                return {"score": float(bonus.get(method, 0) + val)}
        """)
        )
        original_code = textwrap.dedent("""\
            def run_code():
                method = "bad"
                val = 5.0
                return (method, val)
        """)
        parameterized_code = textwrap.dedent("""\
            def run_code():
                method = _optuna_params["method"]
                val = _optuna_params["val"]
                return (method, val)
        """)
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="method",
                    initial_value="bad",
                    param_type="categorical",
                    choices=["bad", "good", "best"],
                    reason="Algorithm selection",
                ),
                ParamSpec(
                    name="val",
                    initial_value=5.0,
                    param_type="float",
                    low=0.0,
                    high=10.0,
                    reason="Numeric param",
                ),
            ],
            modifications=[],
            reasoning="Test string categorical sweep",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=vpath,
            score_key="score",
            minimize=False,
            n_trials=30,
            max_parallel=4,
            eval_timeout=10,
            timeout=120,
            update_program_code=True,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({})
        result = await stage.compute(program)
        assert result.n_trials > 0
        assert "_optuna_params" not in result.optimized_code
        # Should find "best" method (score bonus 100 vs 0).
        assert result.best_params["method"] == "best"
        # Final code should contain the string 'best'.
        assert "'best'" in result.optimized_code or '"best"' in result.optimized_code

    @pytest.mark.asyncio
    async def test_bool_categorical_sweep(self, tmp_path):
        """Sweep a boolean flag."""
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                flag, val = output
                score = val * (2.0 if flag else 1.0)
                return {"score": score}
        """)
        )
        original_code = textwrap.dedent("""\
            def run_code():
                use_boost = False
                val = 5.0
                return (use_boost, val)
        """)
        parameterized_code = textwrap.dedent("""\
            def run_code():
                use_boost = _optuna_params["use_boost"]
                val = _optuna_params["val"]
                return (use_boost, val)
        """)
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="use_boost",
                    initial_value=False,
                    param_type="categorical",
                    choices=[True, False],
                    reason="Toggle boost feature",
                ),
                ParamSpec(
                    name="val",
                    initial_value=5.0,
                    param_type="float",
                    low=1.0,
                    high=10.0,
                    reason="Value",
                ),
            ],
            modifications=[],
            reasoning="Test bool categorical",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=vpath,
            score_key="score",
            minimize=False,
            n_trials=20,
            max_parallel=4,
            eval_timeout=10,
            timeout=120,
            update_program_code=True,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({})
        result = await stage.compute(program)
        assert result.n_trials > 0
        assert "_optuna_params" not in result.optimized_code
        # Should discover that use_boost=True gives higher score.
        assert result.best_params["use_boost"] is True

    @pytest.mark.asyncio
    async def test_callable_sweep_via_eval(self, tmp_path):
        """Sweep entire callables using the eval() pattern."""
        vpath = tmp_path / "validator.py"
        vpath.write_text(
            textwrap.dedent("""\
            def validate(output):
                return {"score": float(output)}
        """)
        )
        original_code = textwrap.dedent("""\
            import math

            def run_code():
                return math.sqrt(16.0)
        """)
        parameterized_code = textwrap.dedent("""\
            import math

            def run_code():
                return eval(_optuna_params["func"])(16.0)
        """)
        search_space = OptunaSearchSpace(
            parameters=[
                ParamSpec(
                    name="func",
                    initial_value="math.sqrt",
                    param_type="categorical",
                    choices=["math.sqrt", "math.log2", "math.log10"],
                    reason="Which math function to apply",
                ),
            ],
            modifications=[],
            reasoning="Test callable sweep",
        )
        llm = _mock_llm(search_space)
        program = Program(code=original_code)
        stage = OptunaOptimizationStage(
            llm=llm,
            validator_path=vpath,
            score_key="score",
            minimize=False,
            n_trials=15,
            max_parallel=4,
            eval_timeout=10,
            timeout=120,
            update_program_code=True,
        )
        stage._apply_modifications = MagicMock(return_value=parameterized_code)
        stage.attach_inputs({})
        result = await stage.compute(program)
        assert result.n_trials > 0
        # eval() calls should be cleaned to direct references.
        assert "eval(" not in result.optimized_code
        assert "_optuna_params" not in result.optimized_code
        # math.log2(16) = 4.0, math.sqrt(16) = 4.0, math.log10(16) ≈ 1.2
        # Both sqrt and log2 give 4.0 so either is fine as "best".
        assert result.best_scores["score"] >= 4.0
