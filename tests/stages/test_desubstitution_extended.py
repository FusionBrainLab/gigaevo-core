"""Extended tests for gigaevo/programs/stages/optimization/optuna/desubstitution.py

Targets uncovered lines: reindent_to_match_block edge cases,
_find_matching_close_paren no-match / string escapes, _clean_eval_in_source
with nested parens, _coerce_param_value with containers/tuples, and
desubstitute_params with the AST path (add_tuned_comment=False).
"""

from __future__ import annotations

import textwrap

from gigaevo.programs.stages.optimization.optuna.desubstitution import (
    _build_line_offsets,
    _clean_eval_in_source,
    _coerce_param_value,
    _find_matching_close_paren,
    coerce_params,
    desubstitute_params,
    reindent_to_match_block,
    strip_line_number_prefix,
)

# ═══════════════════════════════════════════════════════════════════════════
# _coerce_param_value
# ═══════════════════════════════════════════════════════════════════════════


class TestCoerceParamValue:
    def test_int_string(self) -> None:
        assert _coerce_param_value("3") == 3

    def test_float_as_int_string(self) -> None:
        """'3.0' should be coerced to int 3."""
        assert _coerce_param_value("3.0") == 3
        assert isinstance(_coerce_param_value("3.0"), int)

    def test_true_string(self) -> None:
        assert _coerce_param_value("True") is True

    def test_false_string(self) -> None:
        assert _coerce_param_value("False") is False

    def test_plain_string_stays(self) -> None:
        """A non-parseable string stays as-is."""
        assert _coerce_param_value("hello") == "hello"

    def test_list_string(self) -> None:
        assert _coerce_param_value("[2, 3]") == [2, 3]

    def test_tuple_string(self) -> None:
        assert _coerce_param_value("(1, 2)") == (1, 2)

    def test_list_with_string_int_elements(self) -> None:
        """Elements inside list strings are recursively coerced."""
        # After literal_eval, the list contains floats — which are not re-coerced
        # because coerce only converts strings to int via literal_eval.
        # But a list of strings like '["3", "4"]' will coerce each element.
        result = _coerce_param_value('["3", "4"]')
        assert result == [3, 4]
        assert all(isinstance(x, int) for x in result)

    def test_nested_list(self) -> None:
        result = _coerce_param_value("[[1, 2], [3, 4]]")
        assert result == [[1, 2], [3, 4]]

    def test_non_string_list(self) -> None:
        """A Python list (not string) is recursively coerced."""
        result = _coerce_param_value([3.0, "5", 2.0])
        assert result == [3, 5, 2]

    def test_non_string_tuple(self) -> None:
        result = _coerce_param_value((3.0, "hello"))
        assert result == (3, "hello")
        assert isinstance(result, tuple)

    def test_plain_int_passthrough(self) -> None:
        assert _coerce_param_value(42) == 42

    def test_plain_float_passthrough(self) -> None:
        assert _coerce_param_value(3.14) == 3.14

    def test_none_passthrough(self) -> None:
        assert _coerce_param_value(None) is None

    def test_invalid_string_stays(self) -> None:
        """Strings that cause SyntaxError in literal_eval stay as-is."""
        assert _coerce_param_value("[invalid") == "[invalid"


class TestCoerceParams:
    def test_dict_coercion(self) -> None:
        result = coerce_params({"k": "3", "lr": 0.01, "method": "adam"})
        assert result == {"k": 3, "lr": 0.01, "method": "adam"}

    def test_empty_dict(self) -> None:
        assert coerce_params({}) == {}


# ═══════════════════════════════════════════════════════════════════════════
# _find_matching_close_paren
# ═══════════════════════════════════════════════════════════════════════════


class TestFindMatchingCloseParen:
    def test_simple_match(self) -> None:
        code = "eval(42)"
        # open_pos points to the '('
        result = _find_matching_close_paren(code, 4)
        assert result == 7

    def test_nested_parens(self) -> None:
        code = "eval(func(x))"
        #       0123456789012
        result = _find_matching_close_paren(code, 4)
        assert result == 12

    def test_no_match_returns_none(self) -> None:
        """Unbalanced parens → None."""
        code = "eval(42"
        result = _find_matching_close_paren(code, 4)
        assert result is None

    def test_string_with_paren_inside(self) -> None:
        """Parens inside strings should be ignored."""
        code = "eval(')')"
        result = _find_matching_close_paren(code, 4)
        assert result == 8

    def test_escaped_quote_in_string(self) -> None:
        """Backslash-escaped quotes inside strings handled correctly."""
        code = r"eval('\'')"
        result = _find_matching_close_paren(code, 4)
        assert result is not None

    def test_double_quoted_string(self) -> None:
        code = 'eval("hello")'
        #       0123456789012
        result = _find_matching_close_paren(code, 4)
        assert result == 12

    def test_empty_parens(self) -> None:
        code = "eval()"
        result = _find_matching_close_paren(code, 4)
        assert result == 5


# ═══════════════════════════════════════════════════════════════════════════
# _clean_eval_in_source
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanEvalInSource:
    def test_dotted_name(self) -> None:
        """eval('math.sqrt') → math.sqrt"""
        code = "f = eval('math.sqrt')"
        result = _clean_eval_in_source(code)
        assert result == "f = math.sqrt"

    def test_single_name(self) -> None:
        """eval('print') → print"""
        code = "f = eval('print')"
        result = _clean_eval_in_source(code)
        assert result == "f = print"

    def test_literal_list(self) -> None:
        """eval([2, 3]) → [2, 3]"""
        code = "x = eval([2, 3])"
        result = _clean_eval_in_source(code)
        assert result == "x = [2, 3]"

    def test_literal_number(self) -> None:
        """eval(42) → 42"""
        code = "x = eval(42)"
        result = _clean_eval_in_source(code)
        assert result == "x = 42"

    def test_non_identifier_string_stays(self) -> None:
        """eval('Nelder-Mead') is NOT a dotted identifier → stays as-is."""
        code = "x = eval('Nelder-Mead')"
        result = _clean_eval_in_source(code)
        assert result == "x = eval('Nelder-Mead')"

    def test_no_eval_returns_unchanged(self) -> None:
        code = "x = 42"
        result = _clean_eval_in_source(code)
        assert result == "x = 42"

    def test_multiple_evals(self) -> None:
        code = "a = eval('math.sin'); b = eval('math.cos')"
        result = _clean_eval_in_source(code)
        assert "eval" not in result
        assert "math.sin" in result
        assert "math.cos" in result

    def test_unbalanced_eval_paren(self) -> None:
        """Unbalanced paren after eval( → skip (no crash)."""
        code = "x = eval('oops"
        result = _clean_eval_in_source(code)
        assert result == code

    def test_empty_eval(self) -> None:
        """eval() with nothing inside → skip (empty inner)."""
        code = "x = eval()"
        result = _clean_eval_in_source(code)
        # Empty inner is skipped, so unchanged
        assert result == code

    def test_eval_with_non_literal_expression_stays(self) -> None:
        """eval(some_var) is not a literal → stays."""
        code = "x = eval(some_var)"
        result = _clean_eval_in_source(code)
        assert result == code


# ═══════════════════════════════════════════════════════════════════════════
# strip_line_number_prefix
# ═══════════════════════════════════════════════════════════════════════════


class TestStripLineNumberPrefix:
    def test_numbered_lines(self) -> None:
        lines = ["  1 | x = 1", "  2 | y = 2", " 10 | z = 3"]
        result = strip_line_number_prefix(lines)
        assert result == ["x = 1", "y = 2", "z = 3"]

    def test_no_prefix_passthrough(self) -> None:
        lines = ["x = 1", "y = 2"]
        result = strip_line_number_prefix(lines)
        assert result == ["x = 1", "y = 2"]

    def test_empty_list(self) -> None:
        assert strip_line_number_prefix([]) == []

    def test_mixed_prefix_and_no_prefix(self) -> None:
        """Each line is independently processed."""
        lines = ["  1 | x = 1", "plain line", "  3 | z = 3"]
        result = strip_line_number_prefix(lines)
        assert result[0] == "x = 1"
        assert result[1] == "plain line"
        assert result[2] == "z = 3"


# ═══════════════════════════════════════════════════════════════════════════
# reindent_to_match_block
# ═══════════════════════════════════════════════════════════════════════════


class TestReindentToMatchBlock:
    def test_basic_reindent(self) -> None:
        original = ["    x = 1", "    y = 2"]
        replacement = ["a = 10", "b = 20"]
        result = reindent_to_match_block(replacement, original)
        assert result == ["    a = 10", "    b = 20"]

    def test_empty_original_returns_replacement(self) -> None:
        """With no original lines, return replacement unchanged."""
        replacement = ["x = 1", "y = 2"]
        result = reindent_to_match_block(replacement, [])
        assert result == replacement

    def test_all_blank_original_returns_replacement(self) -> None:
        """Original lines that are all blank → no base indent determinable."""
        replacement = ["x = 1"]
        result = reindent_to_match_block(replacement, ["", "  "])
        # All original lines are blank (strip() == ""), so orig_indents is empty
        assert result == replacement

    def test_blank_replacement_lines_preserved(self) -> None:
        """Blank lines in replacement stay blank."""
        original = ["    x = 1"]
        replacement = ["a = 1", "", "b = 2"]
        result = reindent_to_match_block(replacement, original)
        assert result[1] == ""

    def test_nested_indent_preserved(self) -> None:
        """Relative indent within replacement block is preserved."""
        original = ["        outer"]
        replacement = ["if True:", "    inner()"]
        result = reindent_to_match_block(replacement, original)
        assert result[0] == "        if True:"
        assert result[1] == "            inner()"

    def test_replacement_already_correctly_indented(self) -> None:
        """If replacement indent matches, output == replacement."""
        original = ["    x = 1"]
        replacement = ["    y = 2", "    z = 3"]
        result = reindent_to_match_block(replacement, original)
        assert result == ["    y = 2", "    z = 3"]

    def test_all_blank_replacement(self) -> None:
        """All blank replacement lines → returned as-is."""
        original = ["    x = 1"]
        replacement = ["", ""]
        result = reindent_to_match_block(replacement, original)
        assert result == ["", ""]


# ═══════════════════════════════════════════════════════════════════════════
# _build_line_offsets
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildLineOffsets:
    def test_single_line_no_newline(self) -> None:
        offsets = _build_line_offsets("hello")
        # line 1 starts at 0, line 2 (virtual) at 5
        assert offsets == [0, 5]

    def test_multi_line(self) -> None:
        offsets = _build_line_offsets("ab\ncd\n")
        # line 1: offset 0 ("ab\n"), line 2: offset 3 ("cd\n"), line 3: offset 6
        assert offsets == [0, 3, 6]

    def test_empty_string(self) -> None:
        offsets = _build_line_offsets("")
        # Empty string has no lines from splitlines(keepends=True), so only [0]
        assert offsets == [0]


# ═══════════════════════════════════════════════════════════════════════════
# desubstitute_params (AST path — add_tuned_comment=False)
# ═══════════════════════════════════════════════════════════════════════════


class TestDesubstituteParamsASTPath:
    """Test the ast.unparse path (add_tuned_comment=False)."""

    def test_basic_float_replacement(self) -> None:
        code = 'x = _optuna_params["lr"]'
        result = desubstitute_params(
            code, {"lr": 0.01}, {"lr": "float"}, add_tuned_comment=False
        )
        assert "_optuna_params" not in result
        assert "0.01" in result

    def test_int_replacement(self) -> None:
        code = 'n = _optuna_params["k"]'
        result = desubstitute_params(
            code, {"k": 5}, {"k": "int"}, add_tuned_comment=False
        )
        assert "_optuna_params" not in result
        assert "5" in result

    def test_string_replacement(self) -> None:
        code = 'method = _optuna_params["method"]'
        result = desubstitute_params(
            code, {"method": "adam"}, {"method": "categorical"}, add_tuned_comment=False
        )
        assert "_optuna_params" not in result
        assert "'adam'" in result

    def test_none_replacement(self) -> None:
        code = 'x = _optuna_params["val"]'
        result = desubstitute_params(
            code, {"val": None}, {"val": "categorical"}, add_tuned_comment=False
        )
        assert "None" in result

    def test_bool_replacement(self) -> None:
        code = 'flag = _optuna_params["flag"]'
        result = desubstitute_params(
            code, {"flag": True}, {"flag": "categorical"}, add_tuned_comment=False
        )
        assert "True" in result

    def test_negative_float(self) -> None:
        code = 'x = _optuna_params["x"]'
        result = desubstitute_params(
            code, {"x": -0.5}, {"x": "float"}, add_tuned_comment=False
        )
        assert "-0.5" in result

    def test_eval_cleanup_dotted_name(self) -> None:
        """eval('math.sqrt') in the AST path should be cleaned to math.sqrt."""
        code = 'f = eval(_optuna_params["fn"])'
        result = desubstitute_params(
            code, {"fn": "math.sqrt"}, {"fn": "categorical"}, add_tuned_comment=False
        )
        assert "eval" not in result
        assert "math.sqrt" in result

    def test_multiple_params(self) -> None:
        code = textwrap.dedent("""\
            lr = _optuna_params["lr"]
            bs = _optuna_params["bs"]
        """)
        result = desubstitute_params(
            code,
            {"lr": 0.001, "bs": 32},
            {"lr": "float", "bs": "int"},
            add_tuned_comment=False,
        )
        assert "_optuna_params" not in result
        assert "0.001" in result
        assert "32" in result

    def test_unknown_param_left_intact(self) -> None:
        """Params not in the values dict stay as _optuna_params references."""
        code = 'x = _optuna_params["known"]; y = _optuna_params["unknown"]'
        result = desubstitute_params(
            code, {"known": 42}, {"known": "int"}, add_tuned_comment=False
        )
        assert "42" in result
        assert "_optuna_params" in result  # "unknown" stays


class TestDesubstituteParamsSourcePath:
    """Test the source-level path (add_tuned_comment=True, default)."""

    def test_tuned_comment_appended(self) -> None:
        code = 'x = _optuna_params["lr"]'
        result = desubstitute_params(code, {"lr": 0.01}, {"lr": "float"})
        assert "# tuned (Optuna)" in result

    def test_tuned_comment_not_duplicated(self) -> None:
        """If line already has a tuned comment, don't add another."""
        code = 'x = _optuna_params["lr"]  # tuned'
        result = desubstitute_params(code, {"lr": 0.01}, {"lr": "float"})
        assert result.count("# tuned") == 1

    def test_eval_cleanup_in_source_path(self) -> None:
        """eval('scipy.optimize.minimize') cleaned in source-level path."""
        code = 'f = eval(_optuna_params["solver"])'
        result = desubstitute_params(
            code,
            {"solver": "scipy.optimize.minimize"},
            {"solver": "categorical"},
            add_tuned_comment=True,
        )
        assert "eval" not in result
        assert "scipy.optimize.minimize" in result
