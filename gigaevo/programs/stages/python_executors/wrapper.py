from __future__ import annotations

import asyncio
from pathlib import Path
import pickle
import sys
from typing import Any, Sequence


class ExecRunnerError(Exception):
    """Child process failed. Carries returncode and stderr text."""

    def __init__(self, *, returncode: int, stderr: str, stdout_bytes: bytes):
        super().__init__(f"exec_runner failed (exit={returncode})")
        self.returncode = returncode
        self.stderr = stderr
        self.stdout_bytes = stdout_bytes


async def run_exec_runner(
    *,
    code: str,
    function_name: str,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    python_path: Sequence[Path] | None = None,
    timeout: int,
    cwd: Path | None = None,
) -> tuple[Any, bytes, str]:
    """
    Run gigaevo.programs.exec_runner as a subprocess.
    Returns: (result_object, raw_stdout_bytes, stderr_text)
    Raises: ExecRunnerError on non-zero exit, asyncio.TimeoutError on timeout.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "gigaevo.programs.stages.python_executors.exec_runner",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
    )

    payload = {
        "code": code,
        "function_name": function_name,
        "python_path": [str(p) for p in (python_path or [])],
        "args": list(args or []),
        "kwargs": dict(kwargs or {}),
    }
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=data), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except Exception:
                pass
        raise

    stderr_text = stderr.decode("utf-8", errors="replace")

    if proc.returncode == 0:
        try:
            value = pickle.loads(stdout)
        except Exception as e:
            raise ExecRunnerError(
                returncode=0, stderr=f"Invalid pickle payload: {e}", stdout_bytes=stdout
            )
        return value, stdout, stderr_text

    raise ExecRunnerError(
        returncode=proc.returncode, stderr=stderr_text, stdout_bytes=stdout
    )
