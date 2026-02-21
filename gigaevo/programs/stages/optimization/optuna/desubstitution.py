"""AST transforms for Optuna parameter desubstitution.

Replaces ``_optuna_params["key"]`` subscripts with concrete values,
cleans up ``eval('dotted.name')`` patterns, and provides source-level
helpers for line-number stripping and re-indentation.
"""

from __future__ import annotations

import ast
import copy
import re
from typing import Any, Optional

from gigaevo.programs.stages.optimization.optuna.models import (
    _DEFAULT_PRECISION,
    _OPTUNA_PARAMS_NAME,
)
from gigaevo.programs.stages.optimization.utils import (
    INT_LIKE_STR_RE,
    coerce_int_like_string,
    format_value_for_source,
    make_numeric_const_node,
)

# ---------------------------------------------------------------------------
# Param value coercion (int-like strings, recursive into containers)
# ---------------------------------------------------------------------------


def _coerce_param_value(value: Any) -> Any:
    """Coerce int-like strings to int; recurse into lists and tuples.

    Handles both ``"3"`` (pure integer string) and ``"3.0"`` (float-as-string
    integer) so that categorical choices used in ``range()`` or indexing stay
    as Python ``int`` after desubstitution.
    """
    if isinstance(value, str):
        coerced = coerce_int_like_string(value)
        if not isinstance(coerced, str):
            return coerced
        # Also catch float-as-string integers like "3.0" / "-4.0"
        try:
            f = float(value)
            if f == int(f):
                return int(f)
        except ValueError:
            pass
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_coerce_param_value(x) for x in value)
    return value


def _coerce_params(values: dict[str, Any]) -> dict[str, Any]:
    """Recursively coerce int-like strings to int in param values.

    Categorical choices like ["4","5","6"] or list params with string elements
    can cause TypeError when used in range(k) or similar. This ensures
    int-like strings become actual ints throughout nested structures.
    """
    return {k: _coerce_param_value(v) for k, v in values.items()}


# ---------------------------------------------------------------------------
# AST node transformer -- desubstitute _optuna_params references
# ---------------------------------------------------------------------------


class _ParamDesubstitutor(ast.NodeTransformer):
    """Replace ``_optuna_params["key"]`` subscripts with concrete values.

    Handles numeric, string, boolean, and ``None`` values.  For numerics,
    ``int`` params are coerced to ``int`` (so ``range(n)`` stays valid),
    and negative values emit ``UnaryOp(USub, Constant(abs_val))``.
    """

    def __init__(
        self,
        values: dict[str, Any],
        param_types: dict[str, str],
        line_offsets: list[int] | None = None,
    ):
        self._values = values
        self._param_types = param_types
        self._line_offsets = line_offsets
        self._tuned_linenos: set[int] = set()
        # (start_char, end_char, value_str) for comment-accurate replacement in source
        self._tuned_spans: list[tuple[int, int, str]] = []

    def _is_param_subscript(self, node: ast.AST) -> Optional[str]:
        """Return the param name if *node* is ``_optuna_params["key"]``."""
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == _OPTUNA_PARAMS_NAME
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            return node.slice.value
        return None

    def _make_const(self, value: Any, src_node: ast.AST, param_name: str) -> ast.AST:
        """Create an AST literal node for *value*.

        - ``str`` / ``bool`` / ``None`` → ``ast.Constant`` directly
          (check ``bool`` before ``int`` since ``bool`` is a subclass!)
        - ``list`` / ``tuple`` → recurse to coerce elements, then ``ast.Constant``
        - ``int`` param type → coerced to ``int``
        - ``float`` param type → kept as ``float``
        - Negative numerics → ``UnaryOp(USub, Constant(abs))``
        - ``str`` that looks like an integer (e.g. categorical "5") → ``int`` so range() works
        """
        # Non-numeric types: emit directly, except int-like strings.
        if value is None or isinstance(value, bool):
            node = ast.Constant(value=value)
            return ast.copy_location(node, src_node)
        if isinstance(value, str):
            if INT_LIKE_STR_RE.match(value.strip()):
                node = ast.Constant(value=int(value))
                return ast.copy_location(node, src_node)
            node = ast.Constant(value=value)
            return ast.copy_location(node, src_node)
        if isinstance(value, (list, tuple)):
            coerced = type(value)(_coerce_param_value(x) for x in value)
            node = ast.Constant(value=coerced)
            return ast.copy_location(node, src_node)

        # Numeric: delegate to shared helper.
        # Preserve integer values as int regardless of declared ptype (e.g.
        # categorical params whose choices are integers must not become 3.0
        # because range() / indexing requires int, not float).
        ptype = self._param_types.get(param_name, "float")
        is_int = (ptype == "int") or isinstance(value, int)
        return make_numeric_const_node(
            value, is_int, src_node, precision=_DEFAULT_PRECISION
        )

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Process subscript nodes, replacing param references."""
        name = self._is_param_subscript(node)
        if name is not None and name in self._values:
            self._tuned_linenos.add(node.lineno)
            if (
                self._line_offsets is not None
                and hasattr(node, "end_lineno")
                and node.end_lineno is not None
                and node.end_col_offset is not None
            ):
                start = self._line_offsets[node.lineno - 1] + node.col_offset
                end = self._line_offsets[node.end_lineno - 1] + node.end_col_offset
                value_str = format_value_for_source(
                    self._values[name], name, self._param_types
                )
                self._tuned_spans.append((start, end, value_str))
            return self._make_const(self._values[name], node, name)
        self.generic_visit(node)
        return node


# ---------------------------------------------------------------------------
# eval() cleanup
# ---------------------------------------------------------------------------

#: Pattern matching a valid Python dotted name (e.g. ``scipy.optimize.minimize``).
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")

#: Matches eval('dotted.name') or eval("dotted.name") for source-level cleanup.
_EVAL_STRING_RE = re.compile(
    r"\beval\s*\(\s*([\"'])([^\"']+)\1\s*\)",
)


def _clean_eval_in_source(code: str) -> str:
    """Replace ``eval('dotted.name')`` / ``eval("dotted.name")`` with the dotted name in source.

    Only replaces when the string content matches _DOTTED_NAME_RE, so comments and
    line structure are preserved (unlike parse → _EvalCleaner → unparse).
    """

    def repl(m: re.Match[str]) -> str:
        inner = m.group(2)
        return inner if _DOTTED_NAME_RE.match(inner) else m.group(0)

    return _EVAL_STRING_RE.sub(repl, code)


def _dotted_name_to_ast(name: str, src_node: ast.AST) -> ast.AST:
    """Convert a dotted name string like ``a.b.c`` to an AST ``Attribute`` chain."""
    parts = name.split(".")
    result: ast.AST = ast.Name(id=parts[0], ctx=ast.Load())
    for part in parts[1:]:
        result = ast.Attribute(value=result, attr=part, ctx=ast.Load())
    return ast.copy_location(result, src_node)


class _EvalCleaner(ast.NodeTransformer):
    """Clean up ``eval('dotted.name')`` → ``dotted.name`` after desubstitution.

    When a categorical parameter holds a callable reference like
    ``"scipy.optimize.minimize"``, the parameterized code uses
    ``eval(_optuna_params["solver"])``.  After desubstitution this
    becomes ``eval('scipy.optimize.minimize')``, which is functional
    but ugly.  This pass replaces it with the direct name reference.

    Only applies when the ``eval`` argument is a string constant that
    matches a valid Python dotted identifier.
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Process call nodes, cleaning up ``eval('dotted.name')``."""
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "eval"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _DOTTED_NAME_RE.match(node.args[0].value)
        ):
            return _dotted_name_to_ast(node.args[0].value, node)
        return node


# ---------------------------------------------------------------------------
# Source-level helpers
# ---------------------------------------------------------------------------


def _build_line_offsets(source: str) -> list[int]:
    """Return list of character offsets of the start of each line (1-based index)."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


#: Matches optional spaces, digits, "|", optional space at start of line (numbered code format).
_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*\|\s*")


def _strip_line_number_prefix(lines: list[str]) -> list[str]:
    """Remove a leading ``N | ``-style prefix from each line if present.

    If the LLM copies the numbered format into parameterized_snippet, this
    strips it so we never insert line numbers into the source.
    """
    return [_LINE_NUMBER_PREFIX_RE.sub("", line) for line in lines]


def _reindent_to_match_block(
    replacement_lines: list[str], original_lines: list[str]
) -> list[str]:
    """Re-indent replacement lines so the block has the same base indent as the original.

    The LLM often returns parameterized_snippet with no or wrong indentation. We take
    the minimum indent of the original block as the base and apply it to the
    replacement, preserving relative indentation within the replacement.
    """
    if not original_lines:
        return replacement_lines
    # Base indent: minimum leading spaces in non-blank original lines
    orig_indents = [
        len(line) - len(line.lstrip()) for line in original_lines if line.strip()
    ]
    if not orig_indents:
        return replacement_lines
    base_indent_len = min(orig_indents)

    repl_indents = [
        len(line) - len(line.lstrip()) for line in replacement_lines if line.strip()
    ]
    min_repl = min(repl_indents) if repl_indents else 0

    result = []
    for line in replacement_lines:
        if not line.strip():
            result.append(line)
            continue
        current = len(line) - len(line.lstrip())
        content = line.lstrip()
        new_indent_len = base_indent_len + (current - min_repl)
        result.append(" " * max(0, new_indent_len) + content)
    return result


# ---------------------------------------------------------------------------
# Public API -- desubstitute_params
# ---------------------------------------------------------------------------


def desubstitute_params(
    parameterized_code: str,
    values: dict[str, Any],
    param_types: dict[str, str] | None = None,
    add_tuned_comment: bool = True,
) -> str:
    """Replace ``_optuna_params["key"]`` references with concrete *values*.

    Also cleans up ``eval('dotted.name')`` patterns left behind when a
    categorical parameter selects a callable (e.g. a solver function).
    If *add_tuned_comment* is True, appends ``# tuned (Optuna)`` at the end of
    each line where a parameter was substituted (using original source spans
    so the comment stays on the correct line).
    """
    param_types = param_types or {}
    values = _coerce_params(values)
    line_offsets = (
        _build_line_offsets(parameterized_code) if add_tuned_comment else None
    )
    tree = ast.parse(parameterized_code)
    desub = _ParamDesubstitutor(values, param_types, line_offsets=line_offsets)
    new_tree = desub.visit(copy.deepcopy(tree))
    # Clean up eval('dotted.name') → dotted.name
    new_tree = _EvalCleaner().visit(new_tree)
    ast.fix_missing_locations(new_tree)

    if add_tuned_comment and desub._tuned_spans:
        # Replace in original source by span so values land on the correct lines;
        # then add comment at end of each affected line (not inline, to avoid
        # breaking e.g. eval(_optuna_params["x"]) with a comment inside the call).
        code = parameterized_code
        for start, end, value_str in sorted(desub._tuned_spans, key=lambda x: -x[0]):
            code = code[:start] + value_str + code[end:]
        # Which lines (1-based) had a substitution
        tuned_linenos = set()
        for start, _end, _ in desub._tuned_spans:
            for i in range(len(line_offsets) - 1):
                if line_offsets[i] <= start < line_offsets[i + 1]:
                    tuned_linenos.add(i + 1)
                    break
        lines = code.splitlines(keepends=True)
        for i in range(len(lines)):
            if (i + 1) in tuned_linenos and " # tuned" not in lines[i].rstrip():
                stripped = lines[i].rstrip("\n")
                lines[i] = (
                    stripped.rstrip()
                    + "  # tuned (Optuna)"
                    + ("\n" if lines[i].endswith("\n") else "")
                )
        code = "".join(lines)
        # Eval cleanup on source so line numbers (and comments) stay correct.
        return _clean_eval_in_source(code)
    return ast.unparse(new_tree)
