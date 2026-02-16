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

    def __init__(self, values: dict[str, Any], param_types: dict[str, str]):
        self._values = values
        self._param_types = param_types

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
        """
        # Non-numeric types: emit directly.
        if value is None or isinstance(value, (str, bool)):
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
            return self._make_const(self._values[name], node, name)
        self.generic_visit(node)
        return node


#: Pattern matching a valid Python dotted name (e.g. ``scipy.optimize.minimize``).
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


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


def desubstitute_params(
    parameterized_code: str,
    values: dict[str, Any],
    param_types: dict[str, str] | None = None,
) -> str:
    """Replace ``_optuna_params["key"]`` references with concrete *values*.

    Also cleans up ``eval('dotted.name')`` patterns left behind when a
    categorical parameter selects a callable (e.g. a solver function).

    Parameters
    ----------
    parameterized_code : str
        Code containing ``_optuna_params["key"]`` references.
    values : dict[str, Any]
        Mapping of parameter name to concrete value.
    param_types : dict[str, str] | None
        Mapping of parameter name to type string (``"int"``, ``"float"``,
        etc.).  Used to coerce values.  If ``None``, all values are kept
        as-is.
    """
    tree = ast.parse(parameterized_code)
    new_tree = _ParamDesubstitutor(values, param_types or {}).visit(copy.deepcopy(tree))
    # Clean up eval('dotted.name') → dotted.name
    new_tree = _EvalCleaner().visit(new_tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a world-class optimization engineer. Your goal is to improve \
Python programs by identifying existing tuneable values and replacing them \
with parameters.

**CRITICAL RULES:**
1.  **NO NEW LOGIC**: Parameterize ONLY values that already exist (literals, \
    method args). Do NOT add new logic, variables, or control flow.
2.  **PRESERVE STRUCTURE**: The parameterized code must match the original \
    structure. **DO NOT** break string literals or multi-line statements in \
    ways that create syntax errors. **Ensure parentheses and brackets are balanced.**
3.  **USE PATCHES**: Do NOT return the full file. Return only specific \
    code blocks that change.
    - ``original_snippet``: Must be a faithful copy of the source lines to change. \
      **Include surrounding context lines** (e.g., comments or unique neighbor lines) \
      to ensure the snippet matches **ONLY ONE** location in the file. \
      **KEEP CONTEXT MINIMAL** (1-2 lines) to avoid hitting token limits, unless more is needed for uniqueness.
    - ``parameterized_snippet``: The replacement lines using \
      ``_optuna_params["name"]``. Must be syntactically valid Python. \
      **Ensure indentation matches the original snippet exactly.**
4.  **IMPORTS**: If new parameters require imports (e.g. for ``eval()``), \
    list them in ``new_imports``. Do NOT add them to the snippets.
5.  **NO PLACEHOLDERS**: Do NOT use `...` or comments like `# ...` in \
    ``original_snippet`` unless they explicitly exist in the source code.

**Common Pitfalls to AVOID:**
- **Unbalanced Parentheses**: When parameterizing inside a function call, ensure you don't accidentally remove a closing parenthesis or comma.
- **Indentation Errors**: The ``parameterized_snippet`` must fit perfectly into the surrounding code's indentation.
- **String Literals**: Do not leave string literals unterminated.

**Process:**
1.  Identify tuneable constants (floats, ints, method strings, booleans).
    - **Note**: Avoid parameterizing seeds (e.g., ``random.seed(42)``) as they \
      should remain fixed for reproducible evaluations.
2.  Propose a search space (``float``, ``int``, ``log_float``, ``categorical``).
3.  Create a patch for each location:
    - Copy the original lines (plus unique context).
    - **Deduplication**: If a line like ``threshold = 0.5`` appears multiple times, \
      you **MUST** include the surrounding function name or a unique comment \
      to ensure the patch hits the correct spot.
    - Create the replacement lines with ``_optuna_params["param_name"]``.
    - **Verify Uniqueness**: If the snippet appears multiple times (e.g., inside a loop \
      or repeated function calls), expand it to include more context lines until it is unique.

**Example 1 -- Numeric Parameter:**
Original code:
```python
    # Training configuration
    lr = 0.01
    optimizer = Adam(lr=lr)
```

Modification:
- original_snippet:
```python
    # Training configuration
    lr = 0.01
```
- parameterized_snippet:
```python
    # Training configuration
    lr = _optuna_params["learning_rate"]
```

**Example 2 -- Method String (with context for uniqueness):**
Original code:
```python
    # First minimization call
    res = minimize(fun, x0, method="L-BFGS-B")

    # ... later in code ...
    # Second minimization call (we only want to tune the first one)
    res2 = minimize(fun2, x0, method="L-BFGS-B")
```

Modification (tuning ONLY the first call):
- original_snippet:
```python
    # First minimization call
    res = minimize(fun, x0, method="L-BFGS-B")
```
- parameterized_snippet:
```python
    # First minimization call
    res = minimize(fun, x0, method=_optuna_params["method"])
```

**Example 3 -- Callable Sweep (using eval):**
Original code:
```python
    ret = scipy.integrate.quad(func, 0, 1)
```

Modification:
- original_snippet:
```python
    ret = scipy.integrate.quad(func, 0, 1)
```
- parameterized_snippet:
```python
    ret = eval(_optuna_params["integrator"])(func, 0, 1)
```
- new_imports: ["import scipy.integrate"]

Use this ``eval(_optuna_params["..."])`` pattern whenever different functions with \
compatible signatures could be swapped. Common examples:
- Optimizers: ``scipy.optimize.minimize`` vs ``scipy.optimize.differential_evolution``
- Solvers: ``np.linalg.solve`` vs ``scipy.linalg.solve``
- Distance metrics, interpolation functions, etc.

**What NOT to do (forbidden):**
- BAD: Original has ``points.append(center)``. Do NOT add a parameter \
  ``include_center`` and rewrite as ``if _optuna_params["include_center"]: \
  points.append(center) else: ...``. That adds new logic.
- BAD: Introducing any new variable or branch that does not exist in the \
  original. Only REPLACE existing values in place.
- BAD: Patching a common line like ``x = 10`` without context if it appears \
  multiple times in the file. This will cause an ambiguous match error.
- GOOD: Include unique anchors (like function names or comments) to ensure a \
  single match. For example:
  ```python
    # Training configuration
    threshold = _optuna_params["threshold"]
  ```
- BAD: Using ``...`` as a placeholder in ``original_snippet``. It MUST be \
  exact code content.

**Guidelines:**
- Prefer FEWER, high-impact parameters over many marginal ones.
- Algorithm/method strings already in the code are often high-impact.
- Set ranges grounded in domain knowledge. For ``int`` types, **ALWAYS** ensure \
  ``low`` and ``high`` are integers to avoid type errors in the program.
- For ``log_float``, both ``low`` and ``high`` must be positive.
- For ``categorical``, include the current value in the choices list.
- **Uniqueness**: If you are patching a common value (e.g., `n=10`), you **MUST** \
  include the function signature or surrounding comments in the snippet to avoid \
  ambiguity.
"""

_USER_PROMPT_TEMPLATE = """\
Analyze this code and propose tuneable parameters. Return a list of \
modifications (patches) to inject ``_optuna_params`` references.

**IMPORTANT**:
- Ensure ``original_snippet`` matches **EXACTLY ONE** location in the file. \
  Use surrounding lines or comments as anchors to ensure uniqueness.
- The ``parameterized_snippet`` must replace the ``original_snippet`` and \
  contain the ``_optuna_params["name"]`` reference.
- **Keep reasoning concise** and **snippets minimal** to avoid output length limits.

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
        fuzzy match.

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
        code = original_code

        for mod in search_space.modifications:
            original = mod.original_snippet
            parameterized = mod.parameterized_snippet

            if not original.strip():
                continue

            # 1. Try exact string match
            count = code.count(original)
            if count == 1:
                code = code.replace(original, parameterized)
                continue

            if count > 1:
                logger.warning(
                    "[Optuna] Snippet found {} times (ambiguous), skipping:\n{}",
                    count,
                    original,
                )
                continue

            # 2. Try fuzzy match (whitespace flexible)
            # We normalize the snippet by stripping leading/trailing blank lines
            # and matching with a regex that is flexible about internal whitespace.
            lines = [line for line in original.splitlines() if line.strip()]
            if not lines:
                continue

            pattern_parts = []
            for line in lines:
                # Escape the content of the line
                content = re.escape(line.strip())
                # Replace escaped spaces with \s+ to be flexible about internal spacing
                content = content.replace(r"\ ", r"\s+")
                # Match line with flexible indentation
                pattern_parts.append(r"^[ \t]*" + content + r"[ \t]*$")

            # Join with flexible newline matching
            pattern_str = r"(?m)" + r"\n[ \t\r\n]*".join(pattern_parts)

            try:
                pattern = re.compile(pattern_str)
                matches = list(pattern.finditer(code))
            except re.error as e:
                logger.warning("[Optuna] Failed to compile regex for snippet: {}", e)
                matches = []

            if len(matches) == 1:
                # Found unique fuzzy match!
                m = matches[0]

                # Detect indentation of the matched block in the source code
                match_text = m.group(0)
                first_line = match_text.splitlines()[0]
                first_line_indent_match = re.match(r"^[ \t]*", first_line)
                base_indent = (
                    first_line_indent_match.group(0) if first_line_indent_match else ""
                )

                # Normalize the parameterized snippet:
                # 1. Dedent it to remove common indentation.
                # 2. Indent it with the detected base_indent.
                clean_snippet = textwrap.dedent(parameterized).strip("\n")
                indented_snippet = textwrap.indent(clean_snippet, base_indent)

                # Ensure string literals are not broken by dedent/indent if they span multiple lines
                # This is a heuristic: if the snippet contains triple quotes, we should be careful.
                # But typically, parameterized snippets are short.

                code = code[: m.start()] + indented_snippet + code[m.end() :]
                continue

            elif len(matches) > 1:
                logger.warning(
                    "[Optuna] Fuzzy match found {} times (ambiguous), skipping:\n{}",
                    len(matches),
                    original,
                )
                continue

            # If we reached here, neither exact nor fuzzy match worked
            logger.warning(
                "[Optuna] Snippet not found in code (tried exact & fuzzy):\n{}\n"
                "--- Regex pattern used ---\n{}",
                original,
                pattern_str,
            )

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
            # Log the full code to debug file if needed, but here we just raise
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

            async with sem:
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

        async def _run_trial(trial_number: int) -> None:
            trial = study.ask()
            try:
                value = await _objective(trial)
                study.tell(trial, value)
            except optuna.TrialPruned as e:
                # Capture the prune reason (error message)
                reason = str(e)
                if reason not in failure_reasons:
                    failure_reasons.append(reason)
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if reason not in failure_reasons:
                    failure_reasons.append(reason)
                logger.warning(
                    "[Optuna][{}] trial {} failed: {}",
                    pid,
                    trial_number,
                    exc,
                )
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

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

        logger.info(
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
        logger.info("[Optuna][{}] Analysing code with LLM...", pid)
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

        logger.info(
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
            parameterized_code, best_params, param_types
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
            "[Optuna][{}] == Done ==  trials={} params={} {}={}  updated={}",
            pid,
            n_complete,
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
