#!/usr/bin/env python3
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import linecache
import os
from pathlib import Path
import struct
import sys
import traceback
import types
from typing import Any

import cloudpickle

_CODE_FILENAME = "user_code.py"


def _register_source(filename: str, source: str) -> None:
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)


def _load_module_from_code(
    code: str, *, mod_name: str = "user_code"
) -> types.ModuleType:
    _register_source(_CODE_FILENAME, code)
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    code_obj = compile(code, _CODE_FILENAME, "exec")
    exec(code_obj, mod.__dict__)
    return mod


def _prepend_sys_path(paths: list[str] | None) -> None:
    if not paths:
        return
    for p in paths:
        if p and p not in sys.path:
            sys.path.insert(0, str(Path(p).resolve()))


def _ensure_cwd_in_path() -> None:
    """
    Ensure the current working directory is in sys.path.
    This allows imports like 'import problems.some_module' when running
    from the project root.
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _write_code_context(tb: BaseException, *, out: io.TextIOBase) -> None:
    try:
        extracted = traceback.extract_tb(tb.__traceback__)
        user_frames = [f for f in extracted if f.filename == _CODE_FILENAME]
        if not user_frames:
            return
        last = user_frames[-1]
        lineno = last.lineno
        lines = linecache.getlines(_CODE_FILENAME)
        if not lines:
            return
        start = max(1, lineno - 3)
        end = min(len(lines), lineno + 3)
        print(f"\nCode context ({_CODE_FILENAME}:{lineno}):", file=out)
        for i in range(start, end + 1):
            prefix = ">>" if i == lineno else "  "
            print(f"{prefix} {i:4d}: {lines[i - 1].rstrip()}", file=out)
    except Exception as e:
        print(f"Error writing code context: {e}", file=out)


def _format_syntax_error(e: SyntaxError) -> str:
    buf = io.StringIO()
    print("Traceback (most recent call last):", file=buf)
    print(f'  File "{e.filename}", line {e.lineno}', file=buf)
    if e.text:
        line = e.text.rstrip("\n")
        print(f"    {line}", file=buf)
        if e.offset and 1 <= e.offset <= len(line) + 1:
            print("    " + " " * (e.offset - 1) + "^", file=buf)
    print(f"{e.__class__.__name__}: {e.msg}", file=buf)
    return buf.getvalue()


def _run_one(payload: dict[str, Any]) -> tuple[Any | None, dict[str, Any] | None]:
    """
    Execute one payload. Returns (result, None) on success or (None, error_dict) on failure.
    error_dict has _error=True, stderr=str, returncode=int.
    """
    captured = io.StringIO()
    try:
        _ensure_cwd_in_path()

        code: str = payload["code"]
        fn_name: str = payload["function_name"]
        py_path: list[str] = payload.get("python_path", [])
        args: list[Any] = payload.get("args", [])
        kwargs: dict[str, Any] = payload.get("kwargs", {})

        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise TypeError("Payload must contain 'args': list and 'kwargs': dict")

        _prepend_sys_path(py_path)
        mod = _load_module_from_code(code)
        fn = getattr(mod, fn_name, None)
        if not callable(fn):
            raise ValueError(f"Function '{fn_name}' not found or not callable")

        with redirect_stdout(captured), redirect_stderr(captured):
            result = fn(*args, **kwargs)

        printed = captured.getvalue()
        if printed:
            sys.stderr.write(printed)
            sys.stderr.flush()

        cloudpickle.register_pickle_by_value(mod)
        return (result, None)

    except SyntaxError as e:
        printed = captured.getvalue()
        buf = io.StringIO()
        if printed:
            buf.write("[captured stdout/stderr before error]\n")
            buf.write(printed)
        buf.write(_format_syntax_error(e))
        return (None, {"_error": True, "stderr": buf.getvalue(), "returncode": 1})

    except Exception as e:
        printed = captured.getvalue()
        buf = io.StringIO()
        if printed:
            buf.write("[captured stdout/stderr before error]\n")
            buf.write(printed)
        traceback.print_exception(type(e), e, e.__traceback__, file=buf)
        _write_code_context(e, out=buf)
        return (None, {"_error": True, "stderr": buf.getvalue(), "returncode": 1})


def _worker_loop() -> None:
    """Run a persistent loop: read length-prefixed payloads, execute, write length-prefixed responses."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        len_buf = stdin.read(4)
        if not len_buf or len(len_buf) < 4:
            break
        (n,) = struct.unpack(">I", len_buf)
        if n == 0:
            break
        payload_bytes = stdin.read(n)
        if len(payload_bytes) < n:
            break
        payload = cloudpickle.loads(payload_bytes)
        result, error = _run_one(payload)
        if error is not None:
            body = cloudpickle.dumps(error)
        else:
            body = cloudpickle.dumps(result)
        stdout.write(struct.pack(">I", len(body)))
        stdout.write(body)
        stdout.flush()


def main() -> None:
    if "--worker" in sys.argv:
        _worker_loop()
        return

    try:
        _ensure_cwd_in_path()
        payload: dict[str, Any] = cloudpickle.load(sys.stdin.buffer)
        result, error = _run_one(payload)
        if error is not None:
            sys.stderr.write(error["stderr"])
            sys.stderr.flush()
            sys.exit(error["returncode"])
        cloudpickle.dump(result, sys.stdout.buffer)
        sys.stdout.buffer.flush()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()