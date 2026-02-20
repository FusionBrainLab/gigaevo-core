"""Shared utilities for optimization stages (CMA-ES, Optuna, etc.).

Contains common functionality:
- Validator file loading
- Evaluation code building (program + validator in a self-contained script)
- Subprocess-based single-evaluation runner
- Shared input model (``context`` DAG input)
"""

from __future__ import annotations

import ast
import asyncio
import base64
import math
from pathlib import Path
import re
from typing import Any, Optional, Sequence

from loguru import logger

from gigaevo.exceptions import ValidationError
from gigaevo.programs.core_types import StageIO
from gigaevo.programs.stages.common import AnyContainer
from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)

# ---------------------------------------------------------------------------
# Shared numeric / AST helpers
# ---------------------------------------------------------------------------

#: Matches a string that looks like an integer (e.g. "5", "-42").
INT_LIKE_STR_RE = re.compile(r"^-?\d+$")


def coerce_int_like_string(value: str) -> str | int:
    """Coerce a string to ``int`` if it looks like an integer, else return as-is.

    Examples::

        >>> coerce_int_like_string("5")
        5
        >>> coerce_int_like_string("hello")
        'hello'
    """
    if INT_LIKE_STR_RE.match(value.strip()):
        return int(value)
    return value


def format_value_for_source(
    value: Any,
    param_name: str,
    param_types: dict[str, str],
    precision: int = 6,
) -> str:
    """Format *value* as it would appear in Python source (for comment placement).

    Handles ``None``, ``bool``, ``str`` (with int-like coercion),
    ``int`` parameters, and ``float`` parameters (with precision formatting).
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        if INT_LIKE_STR_RE.match(value.strip()):
            return repr(int(value))
        return repr(value)
    if isinstance(value, (list, tuple)):
        return repr(value)
    ptype = param_types.get(param_name, "float")
    if ptype == "int":
        v = int(round(value)) if isinstance(value, float) else int(value)
        return repr(v)
    v = float(value) if not isinstance(value, float) else value
    if isinstance(v, float) and v != 0 and math.isfinite(v):
        v = float(f"{v:.{precision}g}")
    if v < 0:
        return f"-{format_value_for_source(-v, param_name, param_types, precision)}"
    return repr(v)


def make_numeric_const_node(
    value: int | float,
    is_int: bool,
    src_node: ast.AST,
    precision: int = 6,
) -> ast.AST:
    """Create an AST Constant (or UnaryOp(USub, Constant)) for a numeric value.

    Handles negative values, int coercion, and float precision formatting.
    Used by both CMA and Optuna desubstitution.
    """
    if is_int:
        v: int | float = round(value) if isinstance(value, float) else int(value)
    else:
        v = float(value) if not isinstance(value, float) else value
        if v != 0 and math.isfinite(v):
            v = float(f"{v:.{precision}g}")

    if v < 0:
        inner = ast.Constant(value=-v)
        node: ast.AST = ast.UnaryOp(op=ast.USub(), operand=inner)
    else:
        node = ast.Constant(value=v)
    return ast.copy_location(node, src_node)


# ---------------------------------------------------------------------------
# Shared stage input model
# ---------------------------------------------------------------------------


class OptimizationInput(StageIO):
    """Common input model for optimization stages.

    The only (optional) DAG input is ``context`` -- an :class:`AnyContainer`
    forwarded to both the program function and the validator.
    """

    context: Optional[AnyContainer]


# ---------------------------------------------------------------------------
# Validator loading
# ---------------------------------------------------------------------------


def read_validator(validator_path: Path) -> str:
    """Read and return validator source code.

    Raises :class:`ValidationError` if the file does not exist or cannot
    be read.
    """
    p = Path(validator_path)
    if not p.exists():
        raise ValidationError(f"Validator file not found: {p}")
    try:
        return p.read_text(encoding="utf-8")
    except OSError as e:
        raise ValidationError(f"Failed to read validator file: {e}") from e


# ---------------------------------------------------------------------------
# Evaluation code building
# ---------------------------------------------------------------------------


def build_eval_code(
    *,
    validator_code: str,
    program_code: str,
    function_name: str,
    validator_fn: str,
    eval_fn_name: str,
    preamble_lines: list[str] | None = None,
) -> str:
    """Compose a self-contained script that runs *program -> validator*.

    Parameters
    ----------
    validator_code : str
        Raw Python source of the validator file.
    program_code : str
        Python source of the program (possibly parameterised).
    function_name : str
        Function to call inside the program code.
    validator_fn : str
        Function to call inside the validator code.
    eval_fn_name : str
        Name of the top-level evaluation wrapper function in the
        generated script (e.g. ``"_cma_eval"`` or ``"_optuna_eval"``).
    preamble_lines : list[str] | None
        Extra lines to insert before the program code (e.g.
        ``["_cma_params = [1.0, 2.0]"]``).
    """
    validator_b64 = base64.b64encode(validator_code.encode("utf-8")).decode("ascii")

    lines: list[str] = ["import base64, types"]

    if preamble_lines:
        lines.extend(preamble_lines)

    lines.extend(
        [
            program_code,
            "",
            "_val_ns = {}",
            (
                "exec(compile(base64.b64decode("
                f'"{validator_b64}"'
                ').decode(), "<validator>", "exec"), _val_ns)'
            ),
            "",
            f"def {eval_fn_name}(*_args):",
            f"    _prog_result = {function_name}(*_args)",
            f"    _val_fn = _val_ns['{validator_fn}']",
            "    if _args:",
            "        return _val_fn(_args[0], _prog_result)",
            "    return _val_fn(_prog_result)",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single evaluation via subprocess
# ---------------------------------------------------------------------------


async def evaluate_single(
    *,
    eval_code: str,
    eval_fn_name: str,
    context: Any,
    score_key: str,
    python_path: Sequence[Path],
    timeout: int,
    max_memory_mb: int | None,
    log_tag: str = "Opt",
) -> tuple[dict[str, float] | None, str | None]:
    """Run one evaluation in a subprocess and return (score_dict, error_msg).

    Returns ``(None, "error message")`` on failure.

    Parameters
    ----------
    eval_code : str
        Self-contained Python script (from :func:`build_eval_code`).
    eval_fn_name : str
        Name of the evaluation function inside *eval_code*.
    context : Any
        Optional context forwarded as the first positional argument.
    score_key : str
        Key that must be present in the returned dict.
    python_path : Sequence[Path]
        Extra ``sys.path`` entries for the subprocess.
    timeout : int
        Per-evaluation timeout in seconds.
    max_memory_mb : int | None
        RSS memory cap in MB.
    log_tag : str
        Short prefix for log messages (e.g. ``"CMA"``, ``"Optuna"``).
    """
    args: list[Any] = [context] if context is not None else []

    try:
        result, _, _ = await run_exec_runner(
            code=eval_code,
            function_name=eval_fn_name,
            args=args,
            kwargs={},
            python_path=list(python_path),
            timeout=timeout,
            max_memory_mb=max_memory_mb,
        )
        if isinstance(result, tuple):  # with artifact
            result = result[0]

        if isinstance(result, dict) and score_key in result:
            return result, None

        msg = f"Unexpected result type: {type(result).__name__} (expected dict with key '{score_key}')"
        logger.warning("[{}] {}", log_tag, msg)
        return None, msg

    except asyncio.TimeoutError:
        logger.trace("[{}] single evaluation timed out", log_tag)
        return None, "Timeout"
    except ExecRunnerError as exc:
        last_line = (exc.stderr or "").strip().rsplit("\n", 1)[-1]
        logger.trace("[{}] eval failed: {} | {}", log_tag, exc, last_line)
        # Return the actual error message so the caller can log it if critical
        return None, f"{exc} | {last_line}"
