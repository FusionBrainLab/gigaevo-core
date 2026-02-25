"""Tests for gigaevo/programs/stages/optimization/utils.py

Covers:
  - format_value_for_source: bool, str, int, float, None, int-type param, precision
  - make_numeric_const_node: positive/negative int/float, zero, precision formatting
  - read_validator: missing file, valid file
  - build_eval_code: basic, with preamble, with capture_program_output
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gigaevo.exceptions import ValidationError
from gigaevo.programs.stages.optimization.utils import (
    build_eval_code,
    format_value_for_source,
    make_numeric_const_node,
    read_validator,
)

# ═══════════════════════════════════════════════════════════════════════════
# format_value_for_source
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatValueForSource:
    def test_bool_true(self) -> None:
        """Booleans must NOT be treated as numeric (bool is subclass of int)."""
        result = format_value_for_source(True, "flag", {})
        assert result == "True"

    def test_bool_false(self) -> None:
        result = format_value_for_source(False, "flag", {})
        assert result == "False"

    def test_none_value(self) -> None:
        result = format_value_for_source(None, "x", {})
        assert result == "None"

    def test_string_value(self) -> None:
        result = format_value_for_source("hello", "name", {})
        assert result == "'hello'"

    def test_list_value(self) -> None:
        result = format_value_for_source([1, 2, 3], "items", {})
        assert result == "[1, 2, 3]"

    def test_tuple_value(self) -> None:
        result = format_value_for_source((1, 2), "pair", {})
        assert result == "(1, 2)"

    def test_int_value(self) -> None:
        result = format_value_for_source(42, "n", {})
        assert result == "42"

    def test_float_value(self) -> None:
        result = format_value_for_source(3.14159, "pi", {})
        assert result == "3.14159"

    def test_int_type_param_coerces_float_to_int(self) -> None:
        """When param_type is 'int', a float value should be coerced to int repr."""
        result = format_value_for_source(5.0, "k", {"k": "int"})
        assert result == "5"

    def test_int_type_param_rounds_float(self) -> None:
        result = format_value_for_source(5.7, "k", {"k": "int"})
        assert result == "6"

    def test_float_zero(self) -> None:
        result = format_value_for_source(0.0, "x", {"x": "float"})
        assert result == "0.0"

    def test_negative_int(self) -> None:
        result = format_value_for_source(-3, "x", {})
        assert result == "-3"

    def test_negative_float(self) -> None:
        result = format_value_for_source(-0.5, "lr", {"lr": "float"})
        assert result == "-0.5"

    def test_precision_controls_output(self) -> None:
        """Custom precision should limit significant digits."""
        result = format_value_for_source(3.141592653589793, "pi", {}, precision=3)
        assert result == "3.14"

    def test_large_float(self) -> None:
        result = format_value_for_source(1e10, "big", {"big": "float"})
        # Should produce scientific notation or exact repr depending on precision
        parsed = eval(result)
        assert parsed == pytest.approx(1e10)

    def test_very_small_float(self) -> None:
        result = format_value_for_source(1e-8, "tiny", {"tiny": "float"})
        parsed = eval(result)
        assert parsed == pytest.approx(1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# make_numeric_const_node
# ═══════════════════════════════════════════════════════════════════════════


class TestMakeNumericConstNode:
    def _dummy_node(self) -> ast.AST:
        """Create a dummy AST node with location info."""
        node = ast.Constant(value=0)
        node.lineno = 1
        node.col_offset = 0
        node.end_lineno = 1
        node.end_col_offset = 1
        return node

    def test_positive_int(self) -> None:
        result = make_numeric_const_node(42, is_int=True, src_node=self._dummy_node())
        assert isinstance(result, ast.Constant)
        assert result.value == 42
        assert isinstance(result.value, int)

    def test_negative_int(self) -> None:
        """Negative values produce UnaryOp(USub, Constant(abs))."""
        result = make_numeric_const_node(-5, is_int=True, src_node=self._dummy_node())
        assert isinstance(result, ast.UnaryOp)
        assert isinstance(result.op, ast.USub)
        assert result.operand.value == 5

    def test_positive_float(self) -> None:
        result = make_numeric_const_node(
            3.14, is_int=False, src_node=self._dummy_node()
        )
        assert isinstance(result, ast.Constant)
        assert isinstance(result.value, float)
        assert result.value == pytest.approx(3.14)

    def test_negative_float(self) -> None:
        result = make_numeric_const_node(
            -2.5, is_int=False, src_node=self._dummy_node()
        )
        assert isinstance(result, ast.UnaryOp)
        assert isinstance(result.op, ast.USub)
        assert result.operand.value == pytest.approx(2.5)

    def test_zero_int(self) -> None:
        result = make_numeric_const_node(0, is_int=True, src_node=self._dummy_node())
        assert isinstance(result, ast.Constant)
        assert result.value == 0

    def test_zero_float(self) -> None:
        """Zero float skips precision formatting (v != 0 check)."""
        result = make_numeric_const_node(0.0, is_int=False, src_node=self._dummy_node())
        assert isinstance(result, ast.Constant)
        assert result.value == 0.0

    def test_float_coerced_to_int(self) -> None:
        """is_int=True with float input rounds to int."""
        result = make_numeric_const_node(7.9, is_int=True, src_node=self._dummy_node())
        assert isinstance(result, ast.Constant)
        assert result.value == 8
        assert isinstance(result.value, int)

    def test_precision_applied_to_float(self) -> None:
        result = make_numeric_const_node(
            3.141592653589793,
            is_int=False,
            src_node=self._dummy_node(),
            precision=3,
        )
        assert isinstance(result, ast.Constant)
        assert result.value == pytest.approx(3.14)

    def test_inf_float_skips_precision(self) -> None:
        """math.isfinite(inf) is False → precision formatting skipped."""
        result = make_numeric_const_node(
            float("inf"), is_int=False, src_node=self._dummy_node()
        )
        assert isinstance(result, ast.Constant)
        assert result.value == float("inf")

    def test_negative_inf_float(self) -> None:
        result = make_numeric_const_node(
            float("-inf"), is_int=False, src_node=self._dummy_node()
        )
        assert isinstance(result, ast.UnaryOp)
        assert isinstance(result.op, ast.USub)
        assert result.operand.value == float("inf")

    def test_location_preserved(self) -> None:
        src = self._dummy_node()
        result = make_numeric_const_node(1, is_int=True, src_node=src)
        assert result.lineno == src.lineno
        assert result.col_offset == src.col_offset


# ═══════════════════════════════════════════════════════════════════════════
# read_validator
# ═══════════════════════════════════════════════════════════════════════════


class TestReadValidator:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="not found"):
            read_validator(tmp_path / "nonexistent.py")

    def test_valid_file(self, tmp_path: Path) -> None:
        vpath = tmp_path / "val.py"
        vpath.write_text("def validate(x): return {'score': 1.0}")
        result = read_validator(vpath)
        assert "validate" in result


# ═══════════════════════════════════════════════════════════════════════════
# build_eval_code
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildEvalCode:
    def test_basic_structure(self) -> None:
        code = build_eval_code(
            validator_code="def validate(x): return {'score': 1.0}",
            program_code="def run_code(): return 42",
            function_name="run_code",
            validator_fn="validate",
            eval_fn_name="_test_eval",
        )
        assert "import base64" in code
        assert "def _test_eval" in code
        assert "def run_code" in code
        # Default: returns only val result
        assert "return _val_result" in code
        assert "return (_val_result, _prog_result)" not in code

    def test_with_preamble(self) -> None:
        code = build_eval_code(
            validator_code="def validate(x): return {'score': 1.0}",
            program_code="def run_code(): return _params[0]",
            function_name="run_code",
            validator_fn="validate",
            eval_fn_name="_eval",
            preamble_lines=["_params = [42.0]", "import math"],
        )
        assert "_params = [42.0]" in code
        assert "import math" in code

    def test_capture_program_output(self) -> None:
        code = build_eval_code(
            validator_code="def validate(x): return {'score': 1.0}",
            program_code="def run_code(): return 42",
            function_name="run_code",
            validator_fn="validate",
            eval_fn_name="_eval",
            capture_program_output=True,
        )
        assert "return (_val_result, _prog_result)" in code

    def test_context_dispatch(self) -> None:
        """Generated code handles both with-context and without-context calls."""
        code = build_eval_code(
            validator_code="def validate(x): return {'score': 1.0}",
            program_code="def run_code(): return 42",
            function_name="run_code",
            validator_fn="validate",
            eval_fn_name="_eval",
        )
        # Should have context-aware branching
        assert "if _args:" in code
        assert "_val_fn(_args[0], _prog_result)" in code
        assert "_val_fn(_prog_result)" in code

    def test_generated_code_is_valid_python(self) -> None:
        """The generated code must be syntactically valid."""
        code = build_eval_code(
            validator_code="def validate(x): return {'score': 1.0}",
            program_code="def run_code(): return 42",
            function_name="run_code",
            validator_fn="validate",
            eval_fn_name="_eval",
            preamble_lines=["_p = 1"],
            capture_program_output=True,
        )
        # Should not raise SyntaxError
        compile(code, "<test>", "exec")

    def test_validator_encoded_as_base64(self) -> None:
        """Validator code is embedded as base64, not raw (avoids escaping issues)."""
        code = build_eval_code(
            validator_code='def validate(x):\n    return {"score": x}',
            program_code="def run_code(): return 1",
            function_name="run_code",
            validator_fn="validate",
            eval_fn_name="_eval",
        )
        assert "base64.b64decode" in code
        # The raw validator should NOT appear verbatim
        assert 'def validate(x):\n    return {"score": x}' not in code
