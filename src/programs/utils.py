import asyncio
import base64
from datetime import datetime, timezone
from pathlib import Path
import pickle
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.programs.stages.state import ProgramStageResult, StageState

EXECUTION_SUCCESS_SIGNAL = "EXECUTION_SUCCESS"


def dedent_code(code: str) -> str:
    """Remove leading indentation from user code."""
    return textwrap.dedent(code).strip()


def format_error_for_llm(
    error: Union[str, Exception, Dict[str, Any]],
    stderr: str = "",
    context: Optional[str] = None,
    stage_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format error information in a standardized way for LLM parsing.

    Args:
        error: The error (string, exception, or dict)
        stderr: Standard error output
        context: Additional context about the error
        stage_name: Name of the stage where error occurred

    Returns:
        Standardized error dictionary with consistent fields
    """
    if isinstance(error, dict) and "error_message" in error:
        # Already formatted
        return error

    if isinstance(error, Exception):
        error_message = str(error)
        error_type = type(error).__name__

        # Get traceback if available
        try:
            tb_lines = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            full_traceback = "".join(tb_lines).strip()
        except:
            full_traceback = ""
    else:
        error_message = str(error) if error else "Unknown error"
        error_type = "UnknownError"
        full_traceback = ""

    formatted_error = {
        "error_message": error_message,
        "error_type": error_type,
        "stderr": stderr.strip() if stderr else "",
        "stage_name": stage_name or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if full_traceback:
        formatted_error["traceback"] = full_traceback

    if context:
        formatted_error["context"] = context

    return formatted_error


def pretty_print_error(error_dict: Dict[str, Any]) -> str:
    """
    Create a human-readable error message from error dictionary.
    This is designed to be easily parseable by LLMs.
    """
    if not isinstance(error_dict, dict):
        return str(error_dict)

    parts = []

    # Main error message
    error_msg = error_dict.get("error_message", "Unknown error")
    error_type = error_dict.get("error_type", "Error")
    stage_name = error_dict.get("stage_name", "unknown")

    parts.append(f"[{stage_name}] {error_type}: {error_msg}")

    # Context if available
    if "context" in error_dict:
        parts.append(f"Context: {error_dict['context']}")

    # Stderr if available and not empty
    stderr = error_dict.get("stderr", "").strip()
    if stderr:
        parts.append(f"Standard Error Output:\n{stderr}")

    # Traceback if available
    if "traceback" in error_dict:
        parts.append(f"Full Traceback:\n{error_dict['traceback']}")

    return "\n\n".join(parts)


def build_stage_result(
    status: StageState,
    started_at: datetime,
    output: Optional[Any] = None,
    error: Optional[Union[str, Exception, Dict[str, Any]]] = None,
    stderr: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    stage_name: Optional[str] = None,
    context: Optional[str] = None,
) -> ProgramStageResult:
    """
    Construct a standardized stage result with comprehensive error handling.

    Args:
        status: Stage status
        started_at: When the stage started
        output: Stage output (only for COMPLETED status)
        error: Error information (string, exception, or dict)
        stderr: Standard error output
        metadata: Additional metadata
        stage_name: Name of the stage (for error formatting)
        context: Additional context about the execution
    """
    formatted_error = None
    if error is not None:
        formatted_error = format_error_for_llm(
            error=error, stderr=stderr, context=context, stage_name=stage_name
        )

    return ProgramStageResult(
        status=status,
        output=output if status == StageState.COMPLETED else None,
        error=formatted_error,
        metadata=metadata or {},
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
    )


async def run_python_snippet(
    code: str,
    started_at: datetime,
    timeout: int,
    stage_name: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> ProgramStageResult:
    """Runs a Python snippet in a subprocess and decodes its output."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"Subprocess timed out after {timeout}s, attempting termination"
        )

        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            logger.debug("Subprocess terminated gracefully")
        except asyncio.TimeoutError:
            # If graceful termination fails, force kill
            logger.warning(
                "Graceful termination failed, force killing subprocess"
            )
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
                logger.debug("Subprocess force killed")
            except asyncio.TimeoutError:
                # Last resort - log and continue (process might be in uninterruptible state)
                logger.error(
                    "Failed to kill subprocess - process may be stuck in uninterruptible state"
                )
            except ProcessLookupError:
                # Process already died
                logger.debug("Subprocess already terminated")
        except ProcessLookupError:
            # Process already died
            logger.debug("Subprocess already terminated")

        logger.error("Subprocess timed out and was terminated.")
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error={
                "error_type": "TimeoutError",
                "error_message": f"Execution timed out after {timeout} seconds.",
                "stage_name": stage_name or "unknown",
                "context": "Subprocess execution timeout",
            },
            metadata={
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc),
                "duration": timeout,
                "exit_code": -1,  # Conventional exit code for timeout
            },
        )

    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    stderr_str = stderr.decode("utf-8", errors="replace").strip()

    lines = stdout_str.splitlines()
    if proc.returncode == 0 and EXECUTION_SUCCESS_SIGNAL in lines:
        try:
            encoded = lines[-1]  # assume last line is base64 pickle string
            result = pickle.loads(base64.b64decode(encoded))
            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=result,
                metadata={"stdout": stdout_str, "stderr": stderr_str},
                stage_name=stage_name,
            )
        except Exception as e:
            logger.exception("Failed to decode result from subprocess.")
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=e,
                stderr=stderr_str,
                metadata={"stdout": stdout_str, "stderr": stderr_str},
                stage_name=stage_name,
                context="Failed to decode subprocess result",
            )
    else:
        # Try to extract meaningful error type/message from stderr
        error_type = "UnknownError"
        error_message = f"Subprocess failed with exit code {proc.returncode}"

        if stderr_str:
            # Use first line containing 'Error' or entire last line
            for line in stderr_str.splitlines():
                if "Error" in line or "Error:" in line:
                    error_message = line.strip()
                    break

            if "SyntaxError" in stderr_str:
                error_type = "SyntaxError"
            elif "timeout" in stderr_str.lower():
                error_type = "TimeoutError"
            elif "not callable" in stderr_str or "not found" in stderr_str:
                error_type = "FunctionNotFound"

        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error={
                "error_message": error_message,
                "error_type": error_type,
                "stderr": stderr_str,
            },
            metadata={
                "stdout": stdout_str,
                "stderr": stderr_str,
                "exit_code": proc.returncode,
            },
            stage_name=stage_name,
            context=f"Process exited with non-zero code: {proc.returncode}",
        )


def construct_exec_code(
    user_code: str,
    function_name: str,
    input_b64: Optional[str] = None,
    input_file_path: Optional[str] = None,
    python_path: Optional[List[Path]] = None,
) -> str:
    """
    Construct executable Python code to run a specific function from user code,
    with optional input passed via base64 or file path, and sys.path injection.
    """
    path_inserts = ""
    if python_path:
        joined = "\n".join(
            [f"sys.path.insert(0, r'{str(p)}')" for p in python_path]
        )
        path_inserts = f"""
# Add to sys.path
{joined}
"""

    input_loading = ""
    if input_file_path:
        input_loading = f"""
# Load input from file
with open(r'{input_file_path}', 'rb') as f:
    input_obj = pickle.load(f)
"""
    elif input_b64:
        input_loading = f"""
# Decode input
import base64
input_obj = pickle.loads(base64.b64decode(r'''{input_b64}'''))
"""

    return f"""
import sys, traceback, pickle, base64
{path_inserts}
{user_code}
{input_loading}

try:
    if '{function_name}' not in globals() or not callable(globals()['{function_name}']):
        raise ValueError("Function '{function_name}' not found or not callable")
    result = globals()['{function_name}']({'' if input_b64 is None and input_file_path is None else 'input_obj'})
    result_b64 = base64.b64encode(pickle.dumps(result)).decode('utf-8')
    print('{EXECUTION_SUCCESS_SIGNAL}')
    print(result_b64)
except Exception as e:
    print('TRACEBACK_START', file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print('TRACEBACK_END', file=sys.stderr)
    sys.exit(1)
"""
