"""AST transforms for Optuna parameter desubstitution.

Replaces ``_optuna_params["key"]`` subscripts with concrete values,
cleans up ``eval('dotted.name')`` patterns, and provides source-level
helpers for line-number stripping and re-indentation.
"""

from __future__ import annotations

import ast
import bisect
import copy
import re
from typing import Any

from gigaevo.programs.stages.optimization.optuna.models import (
    _DEFAULT_PRECISION,
    _OPTUNA_PARAMS_NAME,
)
from gigaevo.programs.stages.optimization.utils import (
    format_value_for_source,
    make_numeric_const_node,
)

# ---------------------------------------------------------------------------
# Param value coercion (int-like strings, recursive into containers)
# ---------------------------------------------------------------------------


def _coerce_param_value(value: Any) -> Any:
    """Coerce string parameter values to their Python-native types.

    Uses ``ast.literal_eval`` as a universal parser, which handles:
    - ``"3"`` → ``3`` (int)
    - ``"3.0"`` → ``3`` (float-as-int, coerced so ``range()`` works)
    - ``"[2, 3]"`` → ``[2, 3]`` (containers)
    - ``"(1, 2)"`` → ``(1, 2)`` (tuples)
    - ``"True"`` → ``True`` (booleans)
    - ``"hello"`` → ``"hello"`` (plain strings stay as-is)
    """
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        # Coerce whole-number floats to int so range()/indexing works
        if isinstance(parsed, float) and parsed == int(parsed):
            return int(parsed)
        # Recurse into containers to coerce elements (e.g. "3" inside a list)
        if isinstance(parsed, (list, tuple)):
            return type(parsed)(_coerce_param_value(x) for x in parsed)
        return parsed
    if isinstance(value, (list, tuple)):
        return type(value)(_coerce_param_value(x) for x in value)
    return value


def coerce_params(values: dict[str, Any]) -> dict[str, Any]:
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

    def _is_param_subscript(self, node: ast.AST) -> str | None:
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

        Values are already coerced by ``_coerce_params`` before this method
        is called, so strings are plain strings and containers already hold
        native Python types.

        - ``str`` / ``bool`` / ``None`` / ``list`` / ``tuple`` → ``ast.Constant``
        - Numeric → delegated to :func:`make_numeric_const_node`
        """
        if value is None or isinstance(value, (bool, str)):
            node = ast.Constant(value=value)
            return ast.copy_location(node, src_node)
        if isinstance(value, (list, tuple)):
            # list/tuple not valid for ast.Constant; build as ast.List/ast.Tuple
            elts: list[ast.expr] = [ast.Constant(value=v) for v in value]
            container: ast.expr
            if isinstance(value, tuple):
                container = ast.Tuple(elts=elts, ctx=ast.Load())
            else:
                container = ast.List(elts=elts, ctx=ast.Load())
            return ast.copy_location(container, src_node)

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
#
# Two implementations exist because each desubstitution path needs one:
#   - Source-level (_clean_eval_in_source): used when add_tuned_comment=True,
#     operates on the raw source string so that LLM-authored comments and
#     line numbers are preserved (ast.parse strips comments).
#   - AST-level (_EvalCleaner): used when add_tuned_comment=False, operates
#     on the AST that gets fed to ast.unparse.
#
# After coerce_params(), values are already native Python types (lists, ints,
# etc.), so the only eval() patterns that remain post-desubstitution are:
#   - eval('dotted.name')  → dotted.name   (callable categorical)
#   - eval(<literal>)      → <literal>      (coerced container/numeric)
#   - eval('non-identifier') stays as-is    (e.g. eval('Nelder-Mead'))
# ---------------------------------------------------------------------------

#: Pattern matching a valid Python dotted name (e.g. ``scipy.optimize.minimize``).
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


def _find_matching_close_paren(code: str, open_pos: int) -> int | None:
    """Find the closing ``)``, handling nesting and string literals."""
    depth = 1
    i = open_pos + 1
    while i < len(code):
        c = code[i]
        if c in ('"', "'"):
            i += 1
            while i < len(code) and code[i] != c:
                if code[i] == "\\":
                    i += 1
                i += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


#: Matches the start of an ``eval(`` call.
_EVAL_CALL_RE = re.compile(r"\beval\s*\(")


def _clean_eval_in_source(code: str) -> str:
    """Remove unnecessary ``eval()`` wrappers from desubstituted source.

    After coercion, only two patterns appear:

    - ``eval('dotted.name')`` → ``dotted.name``  (callable categorical)
    - ``eval(<literal>)``     → ``<literal>``     (coerced container/numeric)
    """
    parts: list[str] = []
    last_end = 0
    for m in _EVAL_CALL_RE.finditer(code):
        open_paren = m.end() - 1
        close_paren = _find_matching_close_paren(code, open_paren)
        if close_paren is None:
            continue
        inner = code[open_paren + 1 : close_paren].strip()
        if not inner:
            continue

        replacement = None
        if inner[0] in ('"', "'"):
            # Quoted string: eval('math.sqrt') — only strip if dotted name
            try:
                string_val = ast.literal_eval(inner)
            except (ValueError, SyntaxError):
                continue
            if isinstance(string_val, str) and _DOTTED_NAME_RE.match(string_val):
                replacement = string_val
        else:
            # Unquoted: eval([2, 3]), eval((1, 2)), eval(42), etc.
            try:
                ast.literal_eval(inner)
                replacement = inner
            except (ValueError, SyntaxError):
                pass

        if replacement is not None:
            parts.append(code[last_end : m.start()])
            parts.append(replacement)
            last_end = close_paren + 1
    if parts:
        parts.append(code[last_end:])
        return "".join(parts)
    return code


class _EvalCleaner(ast.NodeTransformer):
    """Remove unnecessary ``eval()`` wrappers in the AST.

    Used only when ``add_tuned_comment=False`` (the ``ast.unparse`` path).
    After coercion, handles:

    - ``eval(Constant('dotted.name'))`` → ``Attribute`` chain
    - ``eval(<literal_node>)``          → the literal node directly
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "eval"
            and len(node.args) == 1
            and not node.keywords
        ):
            return node

        arg = node.args[0]

        # eval('dotted.name') → Attribute chain
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if _DOTTED_NAME_RE.match(arg.value):
                # Build a.b.c → Attribute(Attribute(Name("a"), "b"), "c")
                parts = arg.value.split(".")
                result: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
                for part in parts[1:]:
                    result = ast.Attribute(value=result, attr=part, ctx=ast.Load())
                return ast.copy_location(result, node)
            return node

        # eval(<literal>) — already a valid AST node, just strip eval()
        if isinstance(arg, (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return ast.copy_location(arg, node)
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


def strip_line_number_prefix(lines: list[str]) -> list[str]:
    """Remove a leading ``N | ``-style prefix from each line if present.

    If the LLM copies the numbered format into parameterized_snippet, this
    strips it so we never insert line numbers into the source.
    """
    return [_LINE_NUMBER_PREFIX_RE.sub("", line) for line in lines]


def reindent_to_match_block(
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
    values = coerce_params(values)

    tree = ast.parse(parameterized_code)
    desub = _ParamDesubstitutor(
        values,
        param_types,
        line_offsets=_build_line_offsets(parameterized_code)
        if add_tuned_comment
        else None,
    )
    new_tree = desub.visit(copy.deepcopy(tree))

    if add_tuned_comment and desub._tuned_spans:
        # Source-level path: preserves comments and line numbers.
        # Replace _optuna_params spans with concrete values in the original source.
        code = parameterized_code
        for start, end, value_str in sorted(desub._tuned_spans, key=lambda x: -x[0]):
            code = code[:start] + value_str + code[end:]
        # Append "# tuned (Optuna)" on each affected line.
        assert desub._line_offsets is not None  # set when line_offsets kwarg provided
        tuned_linenos = {
            bisect.bisect_right(desub._line_offsets, start)
            for start, _end, _ in desub._tuned_spans
        }
        lines = code.splitlines(keepends=True)
        for i in range(len(lines)):
            if (i + 1) in tuned_linenos and " # tuned" not in lines[i].rstrip():
                stripped = lines[i].rstrip("\n")
                lines[i] = (
                    stripped.rstrip()
                    + "  # tuned (Optuna)"
                    + ("\n" if lines[i].endswith("\n") else "")
                )
        return _clean_eval_in_source("".join(lines))

    # AST path: used when add_tuned_comment=False or no spans were substituted.
    new_tree = _EvalCleaner().visit(new_tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
