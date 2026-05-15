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

    def test_total_nodes_matches_ast_walk(self):
        """total_nodes is the size of the AST (matching `ast.walk`), not the
        sum of 7 specific visitor counts. The prior buggy sum under-reported
        by ~5-10x on any non-trivial program."""
        import ast as _ast

        code = "def f():\n    for i in range(10):\n        x = i + 1"
        result = compute_numerical_complexity(code)
        expected = sum(1 for _ in _ast.walk(_ast.parse(code)))
        assert result["total_nodes"] == expected
        # Sanity: should be >> the old sum-of-7-counts, which was 4 here.
        assert result["total_nodes"] >= 10

    def test_total_nodes_empty_module_is_one(self):
        """ast.parse('') returns a Module node, so total_nodes == 1 (not 0
        as the previous sum-of-counts approach reported)."""
        result = compute_numerical_complexity("")
        assert result["total_nodes"] == 1

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

    # -- modern syntax visitors --

    def test_match_statement_counts_each_case_as_condition(self):
        """Each case branch in a match statement is a discrete condition,
        mirroring how if/elif chains are counted."""
        code = "def f(x):\n    match x:\n        case 1: return 'a'\n        case 2: return 'b'\n        case _: return 'c'\n"
        result = compute_numerical_complexity(code)
        # 3 case branches → condition_count contribution of 3
        assert result["condition_count"] == 3

    def test_match_with_no_match_keeps_condition_count_unchanged(self):
        """Code without `match` is unaffected by the new visit_Match."""
        code = "def f(x):\n    if x > 0:\n        return 1\n    return 0\n"
        result = compute_numerical_complexity(code)
        # One If → 1 condition. Adding visit_Match must not change this.
        assert result["condition_count"] == 1

    def test_async_for_counts_as_loop(self):
        """async-for must be counted alongside regular for/while."""
        code = "async def f(items):\n    async for x in items:\n        pass\n"
        result = compute_numerical_complexity(code)
        assert result["loop_count"] == 1

    def test_for_count_unchanged_by_async_for_addition(self):
        """Regular for loop count must not change after adding visit_AsyncFor."""
        code = "for i in range(5):\n    pass\n"
        result = compute_numerical_complexity(code)
        assert result["loop_count"] == 1

    # -- comprehensions are loops in disguise --

    def test_list_comp_counts_as_loop(self):
        result = compute_numerical_complexity("[x for x in items]")
        assert result["loop_count"] == 1

    def test_set_dict_gen_comprehensions_count_as_loops(self):
        for code in (
            "{x for x in items}",
            "{x: y for x, y in items}",
            "(x for x in items)",
        ):
            assert compute_numerical_complexity(code)["loop_count"] == 1, code

    def test_list_comp_inner_if_counts_as_condition(self):
        """`[x for x in items if x > 0]` → 1 loop + 1 condition."""
        result = compute_numerical_complexity("[x for x in items if x > 0]")
        assert result["loop_count"] == 1
        assert result["condition_count"] == 1

    def test_nested_generators_count_separately(self):
        """`[x for x in a for y in b]` has two for clauses → 2 loops."""
        result = compute_numerical_complexity("[x for x in a for y in b]")
        assert result["loop_count"] == 2

    # -- ternary --

    def test_ifexp_counts_as_condition(self):
        result = compute_numerical_complexity("x = 1 if cond else 2")
        assert result["condition_count"] == 1

    def test_nested_ifexp_counts_each(self):
        result = compute_numerical_complexity("x = (1 if a else 2) if b else 3")
        assert result["condition_count"] == 2

    # -- lambda --

    def test_lambda_counts_as_function_def(self):
        result = compute_numerical_complexity("f = lambda x: x * 2")
        assert result["function_def_count"] == 1

    def test_def_unchanged_by_lambda_addition(self):
        """Regression: a plain `def` (no lambda) still counts as exactly 1."""
        result = compute_numerical_complexity("def f():\n    return 1\n")
        assert result["function_def_count"] == 1

    # -- try / except --

    def test_try_each_handler_counts_as_condition(self):
        code = (
            "try:\n    a()\nexcept ValueError:\n    pass\nexcept TypeError:\n    pass\n"
        )
        result = compute_numerical_complexity(code)
        assert result["condition_count"] == 2

    def test_trystar_handler_counts_as_condition(self):
        """PEP 654 `except*` shares Try's branching semantics."""
        code = "try:\n    a()\nexcept* ValueError:\n    pass\n"
        result = compute_numerical_complexity(code)
        assert result["condition_count"] == 1

    def test_try_else_and_finally_do_not_inflate_condition_count(self):
        """Regression: `else:` and `finally:` clauses are not handlers and
        must not add to condition_count."""
        code = (
            "try:\n    a()\n"
            "except ValueError:\n    pass\n"
            "else:\n    pass\n"
            "finally:\n    pass\n"
        )
        result = compute_numerical_complexity(code)
        assert result["condition_count"] == 1

    # -- integration: a realistic ~30-line pipeline exercising every visitor --

    def test_realistic_pipeline_full_feature_vector(self):
        """End-to-end check across all visitor methods on a realistic program.

        Expected values were cross-validated against an independent
        ``ast.walk``-based reference implementation and match exactly. They
        also match on every torch + transformers file ≥4k LOC tested.
        """
        program = '''
class DataPipeline:
    """Processes records."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.cache: dict[str, list[float]] = {}

    def filter_values(self, values):
        return [v for v in values if v > self.threshold]

    async def aggregate(self, sources):
        out = {}
        async for src in sources:
            try:
                data = await src.read()
            except IOError:
                continue
            except ValueError:
                data = []
            squared = list(map(lambda x: x * x, data))
            match src.kind:
                case "primary":
                    out["primary"] = sum(squared)
                case "fallback":
                    out["fallback"] = sum(squared) if squared else 0
                case _:
                    pass
            for k, v in out.items():
                if v < 0:
                    out[k] = 0
        return out
'''
        result = compute_numerical_complexity(program)
        assert result["loop_count"] == 3  # async-for, list-comp, for
        assert (
            result["condition_count"] == 8
        )  # match(3) + try(2) + comp-if(1) + ternary(1) + if(1)
        assert (
            result["function_def_count"] == 4
        )  # __init__, filter_values, aggregate, lambda
        assert result["class_def_count"] == 1
        assert result["total_nodes"] == 183


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
