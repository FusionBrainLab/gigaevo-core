"""Tests for complexity stages: compute_numerical_complexity, compute_complexity_score,
GetCodeLengthStage, and ComputeComplexityStage."""

from __future__ import annotations

import pytest

from gigaevo.programs.core_types import StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.complexity import (
    _COMPLEXITY_CAPS,
    _COMPLEXITY_WEIGHTS,
    ComputeComplexityStage,
    GetCodeLengthStage,
    compute_complexity_score,
    compute_numerical_complexity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(code: str = "def solve(): return 42") -> Program:
    return Program(code=code, state=ProgramState.RUNNING)


# ---------------------------------------------------------------------------
# TestComputeNumericalComplexity
# ---------------------------------------------------------------------------


class TestComputeNumericalComplexity:
    def test_empty_string(self):
        """ast.parse('') succeeds; all counts should be 0."""
        result = compute_numerical_complexity("")
        assert result["call_count"] == 0
        assert result["binop_count"] == 0
        assert result["loop_count"] == 0
        assert result["condition_count"] == 0
        assert result["function_def_count"] == 0
        assert result["class_def_count"] == 0

    def test_syntax_error_raises(self):
        """Invalid syntax → SyntaxError from ast.parse."""
        with pytest.raises(SyntaxError):
            compute_numerical_complexity("def f(:\n    pass")

    def test_simple_function(self):
        """Simple function: 1 function def, 0 loops, 0 conditions."""
        code = "def f():\n    return 1"
        result = compute_numerical_complexity(code)
        assert result["function_def_count"] == 1
        assert result["loop_count"] == 0
        assert result["condition_count"] == 0

    def test_nested_loops(self):
        """Nested loops should register 2 loop counts."""
        code = "for i in range(10):\n    for j in range(10):\n        pass"
        result = compute_numerical_complexity(code)
        assert result["loop_count"] == 2
        assert result["call_count"] == 2  # range(10) x2

    def test_class_with_methods(self):
        """Class with method: 1 class, 1 function."""
        code = "class C:\n    def method(self):\n        pass"
        result = compute_numerical_complexity(code)
        assert result["class_def_count"] == 1
        assert result["function_def_count"] == 1

    def test_if_elif_else(self):
        """If/elif counts as 2 conditions (if + elif are both ast.If)."""
        code = "x = 1\nif x > 0:\n    pass\nelif x < 0:\n    pass\nelse:\n    pass"
        result = compute_numerical_complexity(code)
        assert result["condition_count"] == 2  # if + elif

    def test_binary_operations(self):
        """Binary operations counted."""
        code = "x = 1 + 2 * 3"
        result = compute_numerical_complexity(code)
        assert result["binop_count"] == 2  # + and *

    def test_subscript_operations(self):
        """Subscript (indexing) counted."""
        code = "x = [1,2,3]\ny = x[0]"
        result = compute_numerical_complexity(code)
        assert result["subscript_count"] == 1

    def test_unique_identifiers(self):
        """Unique identifiers counted correctly."""
        code = "x = 1\ny = 2\nz = x + y"
        result = compute_numerical_complexity(code)
        assert result["unique_identifiers"] >= 3  # x, y, z at minimum

    def test_depth_tracking_balanced(self):
        """max_depth for deeply nested code > max_depth for flat code."""
        flat_code = "x = 1\ny = 2\nz = 3"
        nested_code = "def f():\n    for i in range(10):\n        if i > 0:\n            x = i + 1"
        flat_result = compute_numerical_complexity(flat_code)
        nested_result = compute_numerical_complexity(nested_code)
        assert nested_result["max_depth"] > flat_result["max_depth"]

    def test_ast_entropy_positive(self):
        """Non-trivial code has positive AST entropy."""
        code = "def f(x):\n    if x > 0:\n        return x * 2\n    return 0"
        result = compute_numerical_complexity(code)
        assert result["ast_entropy"] > 0.0

    def test_total_nodes_is_sum_of_counted_types(self):
        """total_nodes equals sum of all counted node types."""
        code = "def f():\n    for i in range(10):\n        x = i + 1"
        result = compute_numerical_complexity(code)
        expected = (
            result["call_count"]
            + result["binop_count"]
            + result["subscript_count"]
            + result["loop_count"]
            + result["condition_count"]
            + result["function_def_count"]
            + result["class_def_count"]
        )
        assert result["total_nodes"] == expected

    def test_async_function_counted(self):
        """async def counts as function_def."""
        code = "async def f():\n    pass"
        result = compute_numerical_complexity(code)
        assert result["function_def_count"] == 1

    def test_while_loop_counted(self):
        """while loop counts as loop."""
        code = "while True:\n    break"
        result = compute_numerical_complexity(code)
        assert result["loop_count"] == 1


# ---------------------------------------------------------------------------
# TestComputeComplexityScore
# ---------------------------------------------------------------------------


class TestComputeComplexityScore:
    def test_empty_features(self):
        """Empty features dict → score 0.0."""
        assert compute_complexity_score({}) == 0.0

    def test_zero_features(self):
        """All zero features → score 0.0."""
        features = {k: 0 for k in _COMPLEXITY_WEIGHTS}
        assert compute_complexity_score(features) == 0.0

    def test_caps_applied(self):
        """Features above caps are clamped."""
        # Set all features to absurdly high values
        features = {k: 999999 for k in _COMPLEXITY_WEIGHTS}
        score = compute_complexity_score(features)
        # Expected: sum of (cap * weight) for each key
        expected = sum(
            _COMPLEXITY_CAPS[k] * _COMPLEXITY_WEIGHTS[k] for k in _COMPLEXITY_WEIGHTS
        )
        assert score == pytest.approx(expected)

    def test_single_feature(self):
        """Score with one feature set."""
        features = {"loop_count": 5}
        score = compute_complexity_score(features)
        assert score == pytest.approx(5 * _COMPLEXITY_WEIGHTS["loop_count"])

    def test_missing_keys_treated_as_zero(self):
        """Keys not in features dict treated as 0."""
        features = {"call_count": 10}
        score = compute_complexity_score(features)
        assert score == pytest.approx(10 * _COMPLEXITY_WEIGHTS["call_count"])

    def test_none_value_treated_as_zero(self):
        """None feature values treated as 0 via `or 0` pattern."""
        features = {"loop_count": None}
        assert compute_complexity_score(features) == 0.0


# ---------------------------------------------------------------------------
# TestGetCodeLengthStage
# ---------------------------------------------------------------------------


class TestGetCodeLengthStage:
    async def test_code_length_matches(self):
        """Output contains code_length equal to len(program.code)."""
        code = "def solve(): return 42"
        stage = GetCodeLengthStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({})
        prog = _prog(code)
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert result.output.data["code_length"] == float(len(code))

    async def test_short_code(self):
        """Short code → code_length matches len(code)."""
        code = "x"
        stage = GetCodeLengthStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({})
        prog = _prog(code)
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert result.output.data["code_length"] == float(len(code))


# ---------------------------------------------------------------------------
# TestComputeComplexityStage
# ---------------------------------------------------------------------------


class TestComputeComplexityStage:
    async def test_returns_all_expected_keys(self):
        """Output contains all expected complexity keys."""
        code = "def f():\n    for i in range(10):\n        if i > 0:\n            x = i + 1"
        stage = ComputeComplexityStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({})
        result = await stage.execute(_prog(code))

        assert result.status == StageState.COMPLETED
        expected_keys = {
            "call_count",
            "binop_count",
            "subscript_count",
            "loop_count",
            "condition_count",
            "function_def_count",
            "class_def_count",
            "unique_identifiers",
            "max_depth",
            "ast_entropy",
            "total_nodes",
            "complexity_score",
            "negative_complexity_score",
        }
        assert set(result.output.data.keys()) == expected_keys

    async def test_negative_complexity_score(self):
        """negative_complexity_score == -complexity_score."""
        code = "def f():\n    for i in range(10):\n        pass"
        stage = ComputeComplexityStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({})
        result = await stage.execute(_prog(code))

        data = result.output.data
        assert data["negative_complexity_score"] == -data["complexity_score"]

    async def test_syntax_error_fails(self):
        """Invalid syntax → stage FAILED."""
        stage = ComputeComplexityStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({})
        result = await stage.execute(_prog("def f(:\n    pass"))

        assert result.status == StageState.FAILED

    async def test_trivial_code(self):
        """Trivial code (no control flow) → low complexity, score near 0."""
        stage = ComputeComplexityStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({})
        result = await stage.execute(_prog("x = 1"))

        assert result.status == StageState.COMPLETED
        assert result.output.data["loop_count"] == 0
        assert result.output.data["condition_count"] == 0
        assert result.output.data["function_def_count"] == 0
