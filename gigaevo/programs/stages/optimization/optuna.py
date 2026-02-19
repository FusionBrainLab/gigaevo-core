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


def _coerce_params(values: dict[str, Any]) -> dict[str, Any]:
    """Recursively coerce int-like strings to int in param values.

    Categorical choices like ["4","5","6"] or list params with string elements
    can cause TypeError when used in range(k) or similar. This ensures
    int-like strings become actual ints throughout nested structures.
    """
    result: dict[str, Any] = {}
    for k, v in values.items():
        result[k] = _coerce_param_value(v)
    return result


def _coerce_param_value(value: Any) -> Any:
    """Coerce int-like strings to int; recurse into lists and tuples."""
    if isinstance(value, str):
        if _INT_LIKE_STR_RE.match(value.strip()):
            return int(value)
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_coerce_param_value(x) for x in value)
    return value


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
    """A single patch applied to a specific line range."""

    start_line: int = Field(
        description="The 1-indexed starting line number of the block to replace."
    )
    end_line: int = Field(
        description="The 1-indexed ending line number of the block to replace (inclusive)."
    )
    parameterized_snippet: str = Field(
        description=(
            "The replacement code block with _optuna_params references. Use relative "
            "indentation only: first line has no leading spaces; indent following lines "
            "relative to that (e.g. 4 spaces per nesting level)."
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


class OptunaOptimizationConfig(BaseModel):
    """Configuration for Optuna features and sampler settings."""

    # Reproducibility
    random_state: Optional[int] = Field(
        default=None,
        description="Random seed for the sampler. Set for reproducible runs.",
    )

    # TPE sampler
    n_startup_trials: Optional[int] = Field(
        default=None,
        description="Number of random trials run before TPE (in addition to n_trials). Total runs = n_startup_trials + n_trials. If None, uses min(25, max(10, n_trials // 2)).",
    )
    multivariate: bool = Field(
        default=True,
        description="Use multivariate TPE to model parameter correlations.",
    )

    # Dynamic Feature Importance
    importance_freezing: bool = Field(
        default=True, description="Enable freezing of low-impact parameters."
    )
    importance_check_at: Optional[int] = Field(
        default=None,
        description="Number of trials after which to check importance (always enforced to be after TPE phase, i.e. >= n_startup_trials + 1). If None, uses total_trials // 3.",
    )
    min_trials_for_importance: int = Field(
        default=10, description="Minimum trials before importance check."
    )
    importance_threshold_ratio: float = Field(
        default=0.1,
        description="Ratio of average importance below which a parameter is frozen.",
    )
    importance_absolute_threshold: float = Field(
        default=0.01,
        description="Absolute importance value below which a parameter is frozen.",
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
            if _INT_LIKE_STR_RE.match(value.strip()):
                node = ast.Constant(value=int(value))
                return ast.copy_location(node, src_node)
            node = ast.Constant(value=value)
            return ast.copy_location(node, src_node)
        if isinstance(value, (list, tuple)):
            coerced = type(value)(_coerce_param_value(x) for x in value)
            node = ast.Constant(value=coerced)
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


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Your goal is to **parametrize** the given Python code: replace existing tuneable \
values (literals, method args) with references to ``_optuna_params["name"]``. \
An optimizer will then sample those parameters and run the code with different \
values. You return a structured response: a list of **parameters** (name, type, \
bounds/choices, initial value) and a list of **modifications** (patches). Each \
patch identifies a line range (start_line to end_line, 1-indexed) and provides \
the ``parameterized_snippet`` to replace it.

**Rules**
- **Prioritize Impact**: Focus on parameterizing constants and literals that are \
likely to have a high impact on the target metric ({score_key}).
- **No new logic**: Only replace values that already exist. Do not add branches, variables, or control flow.
- **Line Ranges**: Use the line numbers from the provided code. `start_line` and `end_line` are inclusive.
- **Non-overlapping patches**: No two modifications may overlap their line ranges. If multiple parameters are in the same block, use one modification for the whole block.
- **parameterized_snippet**: Use **relative indentation only**: the first line must have no leading spaces; indent each following line relative to that (e.g. 4 spaces per nesting level). Do NOT include line numbers or a ``N | `` prefix. The block will be aligned to the original location automatically.
- **Imports**: If needed for ``eval()`` etc., list in ``new_imports`` only.
- **Seeds**: Do not parameterize seeds (e.g. ``random.seed(42)``).

**Search space**: Use ``float``, ``int``, ``log_float`` (positive bounds), or ``categorical``. Each parameter needs an ``initial_value`` (the value currently in the code).

**Examples**
1) Single line: `start_line: 10, end_line: 10, parameterized_snippet: "lr = _optuna_params['learning_rate']"` (no leading spaces).
2) Multi-line block (lines 15-17): snippet starts at column 0; inner lines use relative indent, e.g. "rows = _optuna_params['rows']\\nfor row in range(rows):\\n    v = (row + _optuna_params['offset']) / rows".

**Output length**: Keep reasoning to 3–4 sentences and use at most a few parameters (e.g. <= 10).
"""

_USER_PROMPT_TEMPLATE = """\
Parametrize the code below: (1) list **parameters** (name, type, bounds/choices, initial_value) and (2) list **modifications** (patches) using line ranges.

Code (with line numbers):
```python
{numbered_code}
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
        config: Optional[OptunaOptimizationConfig] = None,
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
        self.config = config or OptunaOptimizationConfig()

    # ------------------------------------------------------------------
    # Phase 1: LLM analysis
    # ------------------------------------------------------------------

    def _apply_modifications(
        self, original_code: str, search_space: OptunaSearchSpace
    ) -> str:
        """Apply the LLM's suggested line-range patches to the original code.

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
            If line ranges are invalid or if the resulting code has syntax errors.
        """
        lines = original_code.splitlines()
        num_lines = len(lines)
        mods = sorted(search_space.modifications, key=lambda x: x.start_line)

        for i, mod in enumerate(mods):
            if mod.start_line < 1 or mod.end_line > num_lines:
                raise ValueError(
                    f"Line range {mod.start_line}-{mod.end_line} out of bounds "
                    f"(1-{num_lines})"
                )
            if mod.start_line > mod.end_line:
                raise ValueError(
                    f"Invalid range: start_line {mod.start_line} > end_line {mod.end_line}"
                )
            if i > 0 and mod.start_line <= mods[i - 1].end_line:
                raise ValueError(
                    f"Overlapping line ranges: {mods[i - 1].start_line}-{mods[i - 1].end_line} "
                    f"and {mod.start_line}-{mod.end_line}"
                )

        new_lines = list(lines)
        for mod in reversed(mods):
            start_idx = mod.start_line - 1
            end_idx = mod.end_line
            replacement_lines = mod.parameterized_snippet.splitlines()
            # Defensive: strip any "N | " prefix if the LLM copied the numbered format
            replacement_lines = _strip_line_number_prefix(replacement_lines)
            # Re-indent to match the original block so we never get "unexpected indent"
            original_block = lines[start_idx:end_idx]
            replacement_lines = _reindent_to_match_block(
                replacement_lines, original_block
            )
            new_lines[start_idx:end_idx] = replacement_lines

        code = "\n".join(new_lines)
        if original_code.endswith("\n") and not code.endswith("\n"):
            code += "\n"

        if search_space.new_imports:
            imports_str = "\n".join(search_space.new_imports)
            code = f"{imports_str}\n{code}"

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
        # Provide line-numbered code to the LLM for precise patching
        lines = code.splitlines()
        numbered_code = "\n".join(
            f"{i + 1:4d} | {line}" for i, line in enumerate(lines)
        )

        task_section = ""
        if self.task_description:
            task_section = f"\nTask description:\n{self.task_description}\n"

        user_msg = _USER_PROMPT_TEMPLATE.format(
            numbered_code=numbered_code,
            task_description_section=task_section,
        )

        structured_llm = self.llm.with_structured_output(OptunaSearchSpace)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(score_key=self.score_key)),
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
        # Coerce int-like strings so range(k) etc. work when k comes from categorical/initial_value
        params = _coerce_params(params)
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
        tuple[dict[str, Any], dict[str, float], int, int]
            Best parameters, best scores, number of successful trials, and total trials run.
        """
        direction = "minimize" if self.minimize else "maximize"

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # TPE with configurable startup trials and multivariate
        from_config = self.config.n_startup_trials is not None
        n_startup = (
            self.config.n_startup_trials
            if from_config
            else min(25, max(10, self.n_trials // 2))
        )
        # Total trials = startup (random) + n_trials (TPE); startup trials are extra, not counted in n_trials.
        total_trials = n_startup + self.n_trials
        logger.debug(
            "[Optuna][{}] TPE sampler: n_startup_trials={} ({}), total_trials={} ({} + {} TPE)",
            pid,
            n_startup,
            "from config" if from_config else "default min(25, max(10, n_trials//2))",
            total_trials,
            n_startup,
            self.n_trials,
        )
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup,
            multivariate=self.config.multivariate,
            seed=self.config.random_state,
        )
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
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

        # Run importance only after TPE phase has produced at least one completed trial.
        importance_check_at = (
            self.config.importance_check_at
            if self.config.importance_check_at is not None
            else max(10, total_trials // 3)
        )
        importance_check_at = max(importance_check_at, n_startup + 1)
        frozen_params: dict[str, Any] = {}
        _importance_lock = asyncio.Lock()

        async def _objective(trial: optuna.trial.Trial) -> float:
            nonlocal best_scores, best_value, best_params

            values: dict[str, Any] = {}
            async with _importance_lock:
                current_frozen = dict(frozen_params)

            for p in param_specs:
                if p.name in current_frozen:
                    values[p.name] = current_frozen[p.name]
                    continue

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
                status = "random" if trial.number <= n_startup else "TPE"
                logger.debug(
                    "[Optuna][{}] Trial {}/{} started (evaluating, mode={})",
                    pid,
                    trial.number + 1,
                    total_trials,
                    status,
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

                # Dynamic Feature Importance: freeze unimportant parameters
                if (
                    self.config.importance_freezing
                    and n_completed == importance_check_at
                    and len(param_specs) > 3
                ):
                    try:
                        completed_trials = [
                            t
                            for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ]
                        if (
                            len(completed_trials)
                            >= self.config.min_trials_for_importance
                        ):
                            importances = optuna.importance.get_param_importances(study)
                            # Only freeze if the parameter is statistically insignificant
                            # (i.e., its importance is a tiny fraction of the average expected importance)
                            avg_importance = 1.0 / len(importances)
                            threshold = (
                                avg_importance * self.config.importance_threshold_ratio
                            )

                            async with _importance_lock:
                                for name, imp in importances.items():
                                    if (
                                        imp < threshold
                                        or imp
                                        < self.config.importance_absolute_threshold
                                    ):
                                        # Freeze at baseline value (initial_value)
                                        frozen_val = next(
                                            p.initial_value
                                            for p in param_specs
                                            if p.name == name
                                        )
                                        frozen_params[name] = frozen_val
                                        logger.info(
                                            "[Optuna][{}] Freezing low-impact parameter '{}' (importance={:.3f}, thresh={:.3f}) at baseline",
                                            pid,
                                            name,
                                            imp,
                                            threshold,
                                        )
                    except Exception as e:
                        logger.debug("[Optuna][{}] Importance check failed: {}", pid, e)

                if n_completed % 10 == 0 or n_completed == total_trials:
                    logger.info(
                        "[Optuna][{}] Progress: {}/{} trials run, best {}={:.{prec}g}",
                        pid,
                        n_completed,
                        total_trials,
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
                    total_trials,
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
                logger.debug("[Optuna][{}] Trial {}/{} pruned", pid, k, total_trials)
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
                    total_trials,
                    reason,
                )
                await _log_progress()

        # 1. Evaluate baseline (parameterized code with initial values).
        baseline_values = {p.name: p.initial_value for p in param_specs}

        # Enqueue the baseline values so Optuna starts by evaluating the current code.
        study.enqueue_trial(baseline_values)

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

        # Run trials: total = n_startup (random) + n_trials (TPE).
        logger.info(
            "[Optuna][{}] Running {} trials total ({} random + {} TPE, up to {} in parallel)...",
            pid,
            total_trials,
            n_startup,
            self.n_trials,
            self.max_parallel,
        )
        tasks = [asyncio.create_task(_run_trial(i)) for i in range(total_trials)]
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
            return best_params, best_scores, 0, total_trials

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

        return best_params, best_scores, n_complete, total_trials

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
        best_params, best_scores, n_complete, total_trials = await self._run_optuna(
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
            "[Optuna][{}] == Done ==  trials={}/{} (+ baseline) params={} {}={}  updated={}",
            pid,
            n_complete,
            total_trials,
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
