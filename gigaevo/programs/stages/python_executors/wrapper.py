from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import cloudpickle


class ExecRunnerError(Exception):
    """Child process failed. Carries returncode and stderr text."""

    def __init__(self, *, returncode: int, stderr: str, stdout_bytes: bytes):
        super().__init__(f"exec_runner failed (exit={returncode})")
        self.returncode = returncode
        self.stderr = stderr
        self.stdout_bytes = stdout_bytes


def _find_runner_in_repo() -> Path:
    """Build path for tools/exec_runner.py."""
    cur = Path(__file__).resolve()
    parent = cur.parent.parent.parent.parent.parent
    return parent / "tools" / "exec_runner.py"


def _prepend_sys_path(paths: Sequence[Path | str] | None) -> None:
    if not paths:
        return
    for path in paths:
        candidate = str(path)
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)


async def run_exec_runner(
    *,
    code: str,
    function_name: str,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    python_path: Sequence[Path] | None = None,
    timeout: int,
    cwd: Path | None = None,
    runner_path: Path | None = None,
) -> tuple[Any, bytes, str]:
    """
    Run the lightweight exec runner as a subprocess.

    Returns: (result_object, raw_stdout_bytes, stderr_text)
    Raises: ExecRunnerError on non-zero exit, asyncio.TimeoutError on timeout.
    """
    script = str(runner_path or _find_runner_in_repo())

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if python_path:
        python_path_entries = [str(p) for p in python_path]
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            python_path_entries + ([existing_pythonpath] if existing_pythonpath else [])
        )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        env=env,
    )

    payload = {
        "code": code,
        "function_name": function_name,
        "python_path": [str(p) for p in (python_path or [])],
        "args": list(args or []),
        "kwargs": dict(kwargs or {}),
    }
    data = cloudpickle.dumps(payload, protocol=cloudpickle.DEFAULT_PROTOCOL)

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
            _prepend_sys_path(python_path)
            value = cloudpickle.loads(stdout)
        except Exception as e:
            raise ExecRunnerError(
                returncode=0,
                stderr=f"Invalid cloudpickle payload: {e}\n[stderr]\n{stderr_text}",
                stdout_bytes=stdout,
            )
        return value, stdout, stderr_text

    raise ExecRunnerError(
        returncode=proc.returncode, stderr=stderr_text, stdout_bytes=stdout
    )
