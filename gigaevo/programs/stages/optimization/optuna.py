"""LLM-guided Optuna hyperparameter optimization stage.

An LLM analyses program code, identifies meaningful hyperparameters, and
produces a **parameterized version** of the code where tuneable constants
are replaced by references to ``_optuna_params["name"]``.  Optuna then
tunes those parameters asynchronously by injecting different values into
the parameterized code.

Key design:

* **Parameterized code** -- the LLM rewrites the program so that each
  tuneable constant (including derived / linked constants like ``-X`` and
  ``X`` in ``uniform(-X, X)``) references a central ``_optuna_params``
  dict.  This cleanly handles multi-occurrence and derived constants
  without AST position tracking.
* **Rich search spaces** -- supports int, float, log-float and categorical.
* **Desubstitution** -- after optimization the ``_optuna_params["..."]``
  references are replaced with concrete best values via a simple AST
  transform, producing clean final code.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Literal, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
import optuna
from pydantic import BaseModel, Field

from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.core_types import StageIO
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.optimization.utils import (
    OptimizationInput,
    build_eval_code,
    evaluate_single,
    read_validator,
)
from gigaevo.programs.stages.stage_registry import StageRegistry

# ---------------------------------------------------------------------------
# Pydantic models -- LLM structured output
# ---------------------------------------------------------------------------

_PARAM_TYPES = Literal["float", "int", "log_float", "categorical"]

#: Union of all value types a parameter can hold.
_ParamValue = Union[float, int, str, bool]

#: Name of the params dict injected into the parameterized code.
_OPTUNA_PARAMS_NAME = "_optuna_params"

#: Default float precision for display and suggestion rounding.
_DEFAULT_PRECISION = 6

#: String that looks like an integer (for categoricals like ["4","5","6"] used in range()).
_INT_LIKE_STR_RE = re.compile(r"^-?\d+$")


def _format_value_for_source(
    value: Any, param_name: str, param_types: dict[str, str]
) -> str:
    """Format *value* as it would appear in Python source (for comment placement)."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        # Emit as int literal when string looks like an integer (e.g. categorical
        # choices ["4","5","6"]) so range(num_points) and similar work.
        if _INT_LIKE_STR_RE.match(value.strip()):
            return repr(int(value))
        return repr(value)
    ptype = param_types.get(param_name, "float")
    if ptype == "int":
        v = int(round(value)) if isinstance(value, float) else int(value)
        return repr(v)
    v = float(value) if not isinstance(value, float) else value
    if isinstance(v, float) and v != 0 and math.isfinite(v):
        v = float(f"{v:.{_DEFAULT_PRECISION}g}")
    if v < 0:
        return f"-{_format_value_for_source(-v, param_name, param_types)}"
    return repr(v)


class ParamSpec(BaseModel):
    """One independent tuneable parameter proposed by the LLM.

    Supports numeric parameters (float, int, log_float) as well as
    categorical parameters whose choices can be strings, booleans,
    or numbers -- enabling algorithm selection, method sweeps, and
    feature toggles.
    """

    name: str = Field(
        description=(
            "Short, snake_case identifier for this parameter "
            "(e.g. 'learning_rate', 'num_iterations', 'solver_method')."
        )
    )
    initial_value: _ParamValue = Field(
        description=(
            "The current / default value of this parameter.  "
            "Can be a number, string, or boolean."
        ),
    )
    param_type: _PARAM_TYPES = Field(
        description=(
            "Search-space type: 'float' for continuous, 'int' for discrete "
            "integer, 'log_float' for log-uniform continuous, "
            "'categorical' for a finite set of choices (numbers, strings, "
            "or booleans)."
        )
    )
    low: Optional[float] = Field(
        default=None,
        description="Lower bound (required for float / int / log_float).",
    )
    high: Optional[float] = Field(
        default=None,
        description="Upper bound (required for float / int / log_float).",
    )
    choices: Optional[list[_ParamValue]] = Field(
        default=None,
        description=(
            "List of allowed values (required for categorical).  "
            "Can include strings, booleans, and numbers."
        ),
    )
    reason: str = Field(
        description="One-sentence explanation of why this parameter is tuneable.",
    )


class CodeModification(BaseModel):
    """A single patch to apply to the original code."""

    original_snippet: str = Field(
        description=(
            "Exact copy of the code block to be replaced.  Must match the "
            "original source indentation and content exactly. Include "
            "surrounding lines if needed to ensure uniqueness."
        )
    )
    parameterized_snippet: str = Field(
        description=(
            "The replacement code block containing _optuna_params references. "
            "Must maintain the same indentation level as the original."
        )
    )


class OptunaSearchSpace(BaseModel):
    """Structured search-space proposal returned by the LLM.

    Contains parameter specifications and a list of patches to apply
    to the code.
    """

    parameters: list[ParamSpec] = Field(
        description="List of independent tuneable parameters.",
    )
    modifications: list[CodeModification] = Field(
        description="List of code patches to inject parameters.",
    )
    new_imports: list[str] = Field(
        default_factory=list,
        description=(
            "List of new import statements required by the parameters (e.g. "
            "'import numpy as np')."
        ),
    )
    reasoning: str = Field(
        description=(
            "Very brief (1-2 sentences) overall strategy: why these parameters matter and "
            "what trade-offs tuning them explores."
        ),
    )


# ---------------------------------------------------------------------------
# Stage output
# ---------------------------------------------------------------------------


class OptunaOptimizationOutput(StageIO):
    """Output produced by :class:`OptunaOptimizationStage`."""

    optimized_code: str
    best_scores: dict[str, float]
    best_params: dict[str, Any]
    n_params: int
    n_trials: int
    search_space_summary: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# AST helper -- desubstitute _optuna_params references
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
            if _INT_LIKE_STR_RE.match(value.strip()):
                node = ast.Constant(value=int(value))
                return ast.copy_location(node, src_node)
            node = ast.Constant(value=value)
            return ast.copy_location(node, src_node)

        # Numeric: coerce based on declared param type.
        ptype = self._param_types.get(param_name, "float")
        if ptype == "int":
            v: int | float = (
                int(round(value)) if isinstance(value, float) else int(value)
            )
        else:
            v = float(value) if not isinstance(value, float) else value
            if isinstance(v, float) and v != 0 and math.isfinite(v):
                v = float(f"{v:.{_DEFAULT_PRECISION}g}")

        if v < 0:
            inner = ast.Constant(value=type(v)(-v))
            node = ast.UnaryOp(op=ast.USub(), operand=inner)
        else:
            node = ast.Constant(value=v)
        return ast.copy_location(node, src_node)

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
                value_str = _format_value_for_source(
                    self._values[name], name, self._param_types
                )
                self._tuned_spans.append((start, end, value_str))
            return self._make_const(self._values[name], node, name)
        self.generic_visit(node)
        return node


#: Pattern matching a valid Python dotted name (e.g. ``scipy.optimize.minimize``).
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")

#: Matches eval('dotted.name') or eval("dotted.name") for source-level cleanup.
_EVAL_STRING_RE = re.compile(
    r"\beval\s*\(\s*([\"'])([^\"']+)\1\s*\)",
)


def _clean_eval_in_source(code: str) -> str:
    """Replace ``eval('dotted.name')`` / ``eval(\"dotted.name\")`` with the dotted name in source.

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


def _build_line_offsets(source: str) -> list[int]:
    """Return list of character offsets of the start of each line (1-based index)."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


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


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Your goal is to **parametrize** the given Python code: replace existing tuneable \
values (literals, method args) with references to ``_optuna_params["name"]``. \
An optimizer will then sample those parameters and run the code with different \
values. You return a structured response: a list of **parameters** (name, type, \
bounds/choices, initial value) and a list of **modifications** (patches). Each \
patch replaces one contiguous block of source with the same block where literals \
are replaced by ``_optuna_params["name"]``. Every name used in patches must appear \
in the parameters list.

**Rules**
- **No new logic**: Only replace values that already exist. Do not add branches, variables, or control flow.
- **Non-overlapping patches**: No two ``original_snippet``s may share any line. If two parameters are in the same block (e.g. two assignments under one comment), use one modification with one ``original_snippet`` (the whole block) and one ``parameterized_snippet`` (both values as ``_optuna_params["..."]``). Overlapping or adjacent separate patches are rejected.
- **original_snippet**: Byte-for-byte copy of the source: same comments (e.g. ``# 1. Jitter`` not ``# Jitter``), same numbers (``0.01`` not ``1e-2``), same variable names. Include 1–2 context lines so the snippet matches exactly one location. Do not skip lines or use ``...``.
- **parameterized_snippet**: Same lines and indentation as ``original_snippet``; only replace literals with ``_optuna_params["name"]``. Write every ``_optuna_params["name"]`` in full (including ``"]``). Do not drop the next line (e.g. keep ``x = res.x`` after ``minimize(...)``) or remove closing ``)``/commas from calls.
- **Imports**: If needed for ``eval()`` etc., list in ``new_imports`` only.
- **Seeds**: Do not parameterize seeds (e.g. ``random.seed(42)``).

**Search space**: Use ``float``, ``int``, ``log_float`` (positive bounds), or ``categorical``. For ``int`` use integer ``low``/``high``; for ``categorical`` include the current value. Each parameter needs an ``initial_value`` (the value currently in the code).

**Examples**
1) Single numeric: copy the exact lines (e.g. comment + ``lr = 0.01``); replace the value with ``_optuna_params["learning_rate"]``. Add a parameter with name ``learning_rate``, type ``float``, and ``initial_value`` 0.01. Add a comment or context so the snippet matches only one place.
2) Two parameters in one block → one modification: one ``original_snippet`` with the full block, one ``parameterized_snippet`` with both ``_optuna_params["margin"]`` and ``_optuna_params["w_overlap"]``. Include both in the parameters list.
3) Swapping callables: ``ret = eval(_optuna_params["integrator"])(func, 0, 1)``; set ``new_imports`` if needed; add a ``categorical`` parameter for the integrator.

**Avoid**: Overlapping or adjacent separate patches; changing variable names or shortening comments in ``original_snippet``; missing ``"]`` or closing ``)`` in ``parameterized_snippet``; extra indentation; patching a repeated line (e.g. ``x = 10``) without context; using a name in a patch that is not in the parameters list. Prefer fewer, high-impact parameters.

**Output length**: Keep reasoning to 3–4 sentences and use at most a few parameters (e.g. <= 10). Short snippets only; do not repeat full file contents.
"""

_USER_PROMPT_TEMPLATE = """\
Parametrize the code below: (1) list **parameters** (name, type, bounds/choices, initial_value) and (2) list **modifications** (patches). Each patch: ``original_snippet`` = exact copy of a block that appears only once in the file; ``parameterized_snippet`` = same block with tuneable values replaced by ``_optuna_params["name"]``. Every name in patches must have a matching entry in parameters. Keep reasoning and snippets minimal.

Code:
```python
{code}
```
{task_description_section}\
"""


# ---------------------------------------------------------------------------
# The Stage
# ---------------------------------------------------------------------------


@StageRegistry.register(
    description="LLM-guided hyperparameter optimization using Optuna"
)
class OptunaOptimizationStage(Stage):
    """Analyse program code with an LLM, then tune identified hyperparameters
    with Optuna.

    **How it works**

    1. An LLM analyses the program source and returns a structured search
       space together with a **parameterized version** of the code where
       tuneable constants are replaced by ``_optuna_params["name"]``
       references.
    2. Optuna runs ``n_trials`` asynchronous trials, each injecting
       different parameter values into the parameterized code and
       evaluating through an external validator script.
    3. The best parameter values are substituted back into the
       parameterized code (replacing ``_optuna_params["name"]`` with
       concrete literals) to produce clean ``optimized_code``.

    **Validator contract**

    Same as :class:`CMANumericalOptimizationStage` -- the validator Python
    file must define a function (default ``validate``) returning a dict
    that contains *score_key*.

    Parameters
    ----------
    llm : MultiModelRouter
        LLM wrapper for structured output calls.
    validator_path : Path
        Path to the validator ``.py`` file.
    score_key : str
        Key in the validator's returned dict to optimise.
    minimize : bool
        If ``True`` minimise *score_key*; otherwise maximise (default).
    n_trials : int
        Number of Optuna trials (default ``50``).
    max_parallel : int
        Maximum concurrent evaluation sub-processes (default ``8``).
    eval_timeout : int
        Timeout in seconds for each evaluation (default ``30``).
    function_name : str
        Function to call inside the program (default ``"run_code"``).
    validator_fn : str
        Function to call inside the validator (default ``"validate"``).
    update_program_code : bool
        If ``True`` (default), overwrite ``program.code`` in-place.
    add_tuned_comment : bool
        If ``True`` (default), append ``# tuned (Optuna)`` on lines where a
        parameter was substituted, so future LLM mutations know it was hyperparameter-tuned.
    task_description : str | None
        Optional task description forwarded to the LLM.
    python_path : list[Path] | None
        Extra ``sys.path`` entries for evaluation sub-processes.
    max_memory_mb : int | None
        Per-evaluation RSS memory cap in MB.
    """

    InputsModel = OptimizationInput
    OutputModel = OptunaOptimizationOutput

    def __init__(
        self,
        *,
        llm: MultiModelRouter,
        validator_path: Path,
        score_key: str,
        minimize: bool = False,
        n_trials: int = 50,
        max_parallel: int = 8,
        eval_timeout: int = 30,
        function_name: str = "run_code",
        validator_fn: str = "validate",
        update_program_code: bool = True,
        add_tuned_comment: bool = True,
        task_description: str | None = None,
        python_path: list[Path] | None = None,
        max_memory_mb: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._validator_code = read_validator(validator_path)

        self.llm = llm
        self.score_key = score_key
        self.minimize = minimize
        self.n_trials = n_trials
        self.max_parallel = max_parallel
        self.eval_timeout = eval_timeout
        self.function_name = function_name
        self.validator_fn = validator_fn
        self.update_program_code = update_program_code
        self.add_tuned_comment = add_tuned_comment
        self.task_description = task_description
        self.python_path = python_path or []
        self.max_memory_mb = max_memory_mb

    # ------------------------------------------------------------------
    # Phase 1: LLM analysis
    # ------------------------------------------------------------------

    def _apply_modifications(
        self, original_code: str, search_space: OptunaSearchSpace
    ) -> str:
        """Apply the LLM's suggested patches to the original code.

        Uses exact matching first, then falls back to a whitespace-flexible
        fuzzy match. Patches are applied from bottom to top so that earlier
        replacements do not invalidate positions of later ones (and duplicate
        snippets are only applied once).

        Parameters
        ----------
        original_code : str
            The original program source code.
        search_space : OptunaSearchSpace
            The search space and modifications proposed by the LLM.

        Returns
        -------
        str
            The parameterized code with ``_optuna_params`` references.

        Raises
        ------
        ValueError
            If the resulting parameterized code has syntax errors.
        """
        # Pass 1: find all (start, end, replacement) in original_code; dedupe by span
        replacements: list[tuple[int, int, str]] = []
        seen_spans: set[tuple[int, int]] = set()
        snippet_not_found = False

        def _ensure_trailing_newline(text: str, span_end: int) -> str:
            """If source has newline at span end, ensure replacement ends with newline."""
            if (
                span_end > 0
                and span_end <= len(original_code)
                and original_code[span_end - 1] == "\n"
            ):
                return text if text.endswith("\n") else text + "\n"
            return text

        for mod in search_space.modifications:
            original = mod.original_snippet
            parameterized = mod.parameterized_snippet

            if not original.strip():
                continue

            # 1. Try exact string match (search in original_code)
            count = original_code.count(original)
            if count == 1:
                start = original_code.index(original)
                end = start + len(original)
                if (start, end) not in seen_spans:
                    seen_spans.add((start, end))
                    # Normalize indentation to match original span (avoid "unexpected indent")
                    span_text = original_code[start:end]
                    span_lines = span_text.splitlines()
                    base_indent = ""
                    min_indent_len = float("inf")
                    for line in span_lines:
                        if line.strip():
                            indent_match = re.match(r"^[ \t]*", line)
                            if indent_match:
                                ind = indent_match.group(0)
                                if len(ind) < min_indent_len:
                                    min_indent_len = len(ind)
                                    base_indent = ind
                    clean_snippet = textwrap.dedent(parameterized).strip("\n")
                    repl = textwrap.indent(clean_snippet, base_indent)
                    repl = _ensure_trailing_newline(repl, end)
                    replacements.append((start, end, repl))
                continue

            if count > 1:
                logger.warning(
                    "[Optuna] Snippet found {} times (ambiguous), skipping:\n{}",
                    count,
                    original,
                )
                snippet_not_found = True
                continue

            # 2. Try fuzzy match (whitespace flexible)
            lines = [line.strip() for line in original.splitlines() if line.strip()]
            if not lines:
                continue

            pattern_parts = []
            for line in lines:
                # 1. Build regex manually: all non-alphanumeric chars are flexible
                content = line.strip()
                res = ""
                for char in content:
                    if char.isalnum() or char == "_":
                        res += char
                    elif char.isspace():
                        # Use \s* instead of \s+ to be flexible with source formatting
                        res += r"\s*"
                    else:
                        # Escape the character for regex and add optional whitespace
                        res += r"\s*" + re.escape(char) + r"\s*"

                # 2. Collapse redundant \s* or \s+ (match literal \s and \s* in res)
                res = re.sub(r"(\\\\s[*+])+", r"\\s*", res)

                # 3. Allow for anything before/after the content on the same line
                # (trailing space, optional comma/semicolon, optional comment)
                pattern_parts.append(
                    r"[ \t]*" + res + r"[ \t\r]*(?:[,;])?[ \t\r]*(?:#.*)?"
                )

            # Join lines with a pattern that allows for:
            # 1. Direct transitions (newline + whitespace)
            # 2. Intervening blank lines or comment-only lines that the LLM might have skipped.
            intervening_gap = r"(?:\s*(?:\#[^\n]*)?\r?\n)*\s*"
            pattern_str = r"(?m)" + intervening_gap.join(pattern_parts)

            try:
                pattern = re.compile(pattern_str)
                matches = list(pattern.finditer(original_code))
            except re.error as e:
                logger.warning("[Optuna] Failed to compile regex for snippet: {}", e)
                matches = []

            if len(matches) == 1:
                m = matches[0]
                start, end = m.start(), m.end()
                if (start, end) in seen_spans:
                    continue
                seen_spans.add((start, end))

                match_text = m.group(0)
                match_lines = match_text.splitlines()
                # Use minimum indent in the match so replacement aligns with block top
                base_indent = ""
                min_indent_len = float("inf")
                for line in match_lines:
                    if line.strip():
                        indent_match = re.match(r"^[ \t]*", line)
                        if indent_match:
                            ind = indent_match.group(0)
                            if len(ind) < min_indent_len:
                                min_indent_len = len(ind)
                                base_indent = ind

                clean_snippet = textwrap.dedent(parameterized).strip("\n")
                indented_snippet = textwrap.indent(clean_snippet, base_indent)
                indented_snippet = _ensure_trailing_newline(indented_snippet, end)
                replacements.append((start, end, indented_snippet))
                continue

            if len(matches) > 1:
                logger.warning(
                    "[Optuna] Fuzzy match found {} times (ambiguous), skipping:\n{}",
                    len(matches),
                    original,
                )
                snippet_not_found = True
                continue

            snippet_not_found = True
            logger.warning(
                "[Optuna] Snippet not found in code (tried exact & fuzzy):\n{}\n"
                "--- Regex pattern used ---\n{}",
                original,
                pattern_str,
            )

        if snippet_not_found:
            raise ValueError(
                "One or more snippets were not found or were ambiguous; "
                "refusing to apply partial patches."
            )

        # Reject overlapping spans; do not apply partial patches.
        sorted_by_start = sorted(replacements, key=lambda x: x[0])
        for i in range(len(sorted_by_start) - 1):
            _, e1 = sorted_by_start[i][:2]
            s2, _ = sorted_by_start[i + 1][:2]
            if e1 > s2:
                raise ValueError(
                    "Overlapping replacement spans detected (e.g. {} > {}); "
                    "refusing to apply. Each modification's original_snippet must not "
                    "overlap any other (use separate, non-overlapping code blocks).".format(
                        e1, s2
                    )
                )

        # Pass 2: apply from bottom to top so indices remain valid
        code = original_code
        for start, end, text in sorted(replacements, key=lambda x: -x[1]):
            next_ch = code[end : end + 1] if end < len(code) else ""
            if not text.endswith("\n") and next_ch and next_ch != "\n":
                text = text + "\n"
            code = code[:start] + text + code[end:]

        # Prepend new imports
        if search_space.new_imports:
            imports_str = "\n".join(search_space.new_imports)
            code = f"{imports_str}\n{code}"

        # Validate syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.error(
                "[Optuna] Parameterized code has syntax error: {}\nCode snippet around error:\n{}",
                e,
                "\n".join(code.splitlines()[max(0, e.lineno - 5) : e.lineno + 5])
                if e.lineno
                else "Unknown location",
            )
            raise ValueError(f"Parameterized code syntax error: {e}")

        return code

    async def _analyze_code(self, code: str) -> OptunaSearchSpace:
        """Call the LLM to propose a search space for *code*.

        Parameters
        ----------
        code : str
            The source code to analyze.

        Returns
        -------
        OptunaSearchSpace
            The proposed parameters and code modifications.
        """
        task_section = ""
        if self.task_description:
            task_section = f"\nTask description:\n{self.task_description}\n"

        user_msg = _USER_PROMPT_TEMPLATE.format(
            code=code,
            task_description_section=task_section,
        )

        structured_llm = self.llm.with_structured_output(OptunaSearchSpace)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        result = await structured_llm.ainvoke(messages)
        return result

    # ------------------------------------------------------------------
    # Phase 2: Optuna evaluation
    # ------------------------------------------------------------------

    def _build_eval_code(self, parameterized_code: str, params: dict[str, Any]) -> str:
        """Compose a self-contained script: params dict + program + validator.

        Parameters
        ----------
        parameterized_code : str
            The code containing ``_optuna_params`` references.
        params : dict[str, Any]
            The specific parameter values to inject for this evaluation.

        Returns
        -------
        str
            A complete Python script ready for execution.
        """
        return build_eval_code(
            validator_code=self._validator_code,
            program_code=parameterized_code,
            function_name=self.function_name,
            validator_fn=self.validator_fn,
            eval_fn_name="_optuna_eval",
            preamble_lines=[f"{_OPTUNA_PARAMS_NAME} = {params!r}"],
        )

    async def _evaluate_single(
        self,
        parameterized_code: str,
        params: dict[str, Any],
        context: Optional[dict[str, Any]],
    ) -> tuple[Optional[dict[str, float]], Optional[str]]:
        """Run one trial and return (score_dict, error_message).

        Parameters
        ----------
        parameterized_code : str
            The code to evaluate.
        params : dict[str, Any]
            Parameters for this trial.
        context : Optional[dict[str, Any]]
            Optional evaluation context.

        Returns
        -------
        tuple[Optional[dict[str, float]], Optional[str]]
            A tuple of (scores, error_message).
        """
        eval_code = self._build_eval_code(parameterized_code, params)
        return await evaluate_single(
            eval_code=eval_code,
            eval_fn_name="_optuna_eval",
            context=context,
            score_key=self.score_key,
            python_path=self.python_path,
            timeout=self.eval_timeout,
            max_memory_mb=self.max_memory_mb,
            log_tag="Optuna",
        )

    async def _run_optuna(
        self,
        parameterized_code: str,
        param_specs: list[ParamSpec],
        context: Optional[dict[str, Any]],
        pid: str,
    ) -> tuple[dict[str, Any], dict[str, float], int]:
        """Run Optuna optimization.

        Parameters
        ----------
        parameterized_code : str
            The code to optimize.
        param_specs : list[ParamSpec]
            Specifications of parameters to tune.
        context : Optional[dict[str, Any]]
            Optional evaluation context.
        pid : str
            Short program ID for logging.

        Returns
        -------
        tuple[dict[str, Any], dict[str, float], int]
            Best parameters, best scores, and number of successful trials.
        """
        direction = "minimize" if self.minimize else "maximize"

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(),
        )

        sem = asyncio.Semaphore(self.max_parallel)

        best_scores: dict[str, float] = {}
        best_value: float | None = None
        best_params: dict[str, Any] = {p.name: p.initial_value for p in param_specs}

        def _is_better(score: float) -> bool:
            if best_value is None:
                return True
            if direction == "minimize":
                return score < best_value
            return score > best_value

        async def _objective(trial: optuna.trial.Trial) -> float:
            nonlocal best_scores, best_value, best_params

            values: dict[str, Any] = {}
            for p in param_specs:
                if p.param_type == "float":
                    v = trial.suggest_float(p.name, p.low, p.high)
                    if v != 0 and math.isfinite(v):
                        v = float(f"{v:.{_DEFAULT_PRECISION}g}")
                    values[p.name] = v
                elif p.param_type == "int":
                    values[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high))
                elif p.param_type == "log_float":
                    v = trial.suggest_float(p.name, p.low, p.high, log=True)
                    if v != 0 and math.isfinite(v):
                        v = float(f"{v:.{_DEFAULT_PRECISION}g}")
                    values[p.name] = v
                elif p.param_type == "categorical":
                    values[p.name] = trial.suggest_categorical(p.name, p.choices)

            logger.trace(
                "[Optuna][{}][trial {}] Evaluating: {}", pid, trial.number, values
            )

            async with sem:
                # Log when evaluation actually starts (semaphore acquired), not when trial was asked
                logger.debug(
                    "[Optuna][{}] Trial {}/{} started (evaluating)",
                    pid,
                    trial.number + 1,
                    self.n_trials,
                )
                scores, error = await self._evaluate_single(
                    parameterized_code, values, context
                )

            if scores is None:
                raise optuna.TrialPruned(f"Evaluation failed: {error}")

            score = float(scores[self.score_key])

            if _is_better(score):
                best_value = score
                best_scores = scores
                best_params = dict(values)

            return score

        failure_reasons: list[str] = []
        n_completed = 0
        _completed_lock = asyncio.Lock()

        async def _log_progress() -> None:
            nonlocal n_completed
            async with _completed_lock:
                n_completed += 1
                if n_completed % 10 == 0 or n_completed == self.n_trials:
                    logger.info(
                        "[Optuna][{}] Progress: {}/{} trials run, best {}={:.{prec}g}",
                        pid,
                        n_completed,
                        self.n_trials,
                        self.score_key,
                        best_value if best_value is not None else float("nan"),
                        prec=_DEFAULT_PRECISION,
                    )

        async def _run_trial(trial_number: int) -> None:
            trial = study.ask()
            k = trial_number + 1
            try:
                value = await _objective(trial)
                study.tell(trial, value)
                logger.debug(
                    "[Optuna][{}] Trial {}/{} completed, {}={:.{prec}g}",
                    pid,
                    k,
                    self.n_trials,
                    self.score_key,
                    value,
                    prec=_DEFAULT_PRECISION,
                )
                await _log_progress()
            except optuna.TrialPruned as e:
                # Capture the prune reason (error message)
                reason = str(e)
                if reason not in failure_reasons:
                    failure_reasons.append(reason)
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                logger.debug("[Optuna][{}] Trial {}/{} pruned", pid, k, self.n_trials)
                await _log_progress()
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if reason not in failure_reasons:
                    failure_reasons.append(reason)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                logger.debug(
                    "[Optuna][{}] Trial {}/{} failed: {}",
                    pid,
                    k,
                    self.n_trials,
                    reason,
                )
                await _log_progress()

        # Evaluate baseline (parameterized code with initial values).
        baseline_values = {p.name: p.initial_value for p in param_specs}

        # We need the full error here, so we call evaluate_single directly
        # instead of self._evaluate_single which drops the error message.
        baseline_eval_code = self._build_eval_code(parameterized_code, baseline_values)
        baseline_result, baseline_err = await evaluate_single(
            eval_code=baseline_eval_code,
            eval_fn_name="_optuna_eval",
            context=context,
            score_key=self.score_key,
            python_path=self.python_path,
            timeout=self.eval_timeout,
            max_memory_mb=self.max_memory_mb,
            log_tag="Optuna",
        )

        if baseline_result is not None:
            baseline_score = float(baseline_result[self.score_key])
            if _is_better(baseline_score):
                best_value = baseline_score
                best_scores = baseline_result
                best_params = dict(baseline_values)
            logger.info(
                "[Optuna][{}] Baseline {}={:.{prec}f}",
                pid,
                self.score_key,
                baseline_score,
                prec=_DEFAULT_PRECISION,
            )
        else:
            # Enhanced logging for baseline failure
            logger.info(
                "[Optuna][{}] Baseline evaluation failed (original parameters invalid). "
                "Proceeding with optimization to find valid parameters.\n"
                "Error details: {}",
                pid,
                baseline_err or "Unknown error (check debug logs)",
            )

        # Run trials.
        logger.info(
            "[Optuna][{}] Running {} trials (up to {} in parallel)...",
            pid,
            self.n_trials,
            self.max_parallel,
        )
        tasks = [asyncio.create_task(_run_trial(i)) for i in range(self.n_trials)]
        await asyncio.gather(*tasks, return_exceptions=True)

        n_complete = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )

        if n_complete == 0:
            reasons_str = "\n".join(f"- {r}" for r in failure_reasons[:5])
            if len(failure_reasons) > 5:
                reasons_str += f"\n- ... and {len(failure_reasons) - 5} more"

            logger.warning(
                "[Optuna][{}] No trials completed successfully; "
                "returning original code.\nCommon errors:\n{}",
                pid,
                reasons_str,
            )
            return best_params, best_scores, 0

        # Re-evaluate best to get full scores.
        final_scores, _ = await self._evaluate_single(
            parameterized_code, best_params, context
        )
        if final_scores is not None:
            best_scores = final_scores

        logger.debug(
            "[Optuna][{}] Best trial: {} {}={}",
            pid,
            best_params,
            self.score_key,
            best_value,
        )

        return best_params, best_scores, n_complete

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    async def compute(self, program: Program) -> OptunaOptimizationOutput:
        """Analyze code with LLM and tune hyperparameters using Optuna.

        Parameters
        ----------
        program : Program
            The program to optimize.

        Returns
        -------
        OptunaOptimizationOutput
            Results including optimized code, best parameters, and trial stats.
        """
        code = program.code
        pid = program.id[:8]

        # 1. LLM analysis
        logger.debug("[Optuna][{}] Analysing code with LLM...", pid)
        try:
            search_space = await self._analyze_code(code)
            parameterized_code = self._apply_modifications(code, search_space)
        except Exception as exc:
            logger.warning(
                "[Optuna][{}] LLM analysis or patching failed: {}; returning original code",
                pid,
                exc,
            )
            return OptunaOptimizationOutput(
                optimized_code=code,
                best_scores={},
                best_params={},
                n_params=0,
                n_trials=0,
                search_space_summary=[],
            )

        if not search_space.parameters:
            logger.info(
                "[Optuna][{}] LLM found no tuneable parameters; "
                "returning original code.",
                pid,
            )
            return OptunaOptimizationOutput(
                optimized_code=code,
                best_scores={},
                best_params={},
                n_params=0,
                n_trials=0,
                search_space_summary=[],
            )

        param_specs = search_space.parameters
        # parameterized_code is already computed in try-block above
        n = len(param_specs)

        logger.debug(
            "[Optuna][{}] LLM proposed {} parameters: {}",
            pid,
            n,
            [p.name for p in param_specs],
        )
        logger.debug("[Optuna][{}] LLM reasoning: {}", pid, search_space.reasoning)

        # 2. Resolve context
        ctx = self.params.context.data if self.params.context is not None else None

        # 3. Run Optuna
        best_params, best_scores, n_complete = await self._run_optuna(
            parameterized_code, param_specs, ctx, pid
        )

        # 4. Build optimised code (desubstitute params into clean code)
        param_types = {p.name: p.param_type for p in param_specs}
        optimized_code = desubstitute_params(
            parameterized_code,
            best_params,
            param_types,
            add_tuned_comment=self.add_tuned_comment,
        )

        # 5. Optionally update program in-place
        if self.update_program_code:
            program.code = optimized_code

        # 6. Summary
        search_summary = [
            {
                "name": p.name,
                "param_type": p.param_type,
                "initial_value": p.initial_value,
                "optimized_value": best_params.get(p.name),
                "low": p.low,
                "high": p.high,
                "choices": p.choices,
            }
            for p in param_specs
        ]

        display_score = (
            float(best_scores[self.score_key])
            if self.score_key in best_scores
            else None
        )
        logger.info(
            "[Optuna][{}] == Done ==  trials={}/{} params={} {}={}  updated={}",
            pid,
            n_complete,
            self.n_trials,
            n,
            self.score_key,
            f"{display_score:.{_DEFAULT_PRECISION}f}"
            if display_score is not None
            else "N/A",
            self.update_program_code,
        )

        return OptunaOptimizationOutput(
            optimized_code=optimized_code,
            best_scores=best_scores,
            best_params=best_params,
            n_params=n,
            n_trials=n_complete,
            search_space_summary=search_summary,
        )
