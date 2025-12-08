#!/usr/bin/env python3
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import linecache
import os
from pathlib import Path
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


def main() -> None:
    captured = io.StringIO()
    try:
        _ensure_cwd_in_path()

        payload: dict[str, Any] = cloudpickle.load(sys.stdin.buffer)

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
        cloudpickle.dump(result, sys.stdout.buffer)
        sys.stdout.buffer.flush()

    except SyntaxError as e:
        printed = captured.getvalue()
        if printed:
            sys.stderr.write("[captured stdout/stderr before error]\n")
            sys.stderr.write(printed)
        sys.stderr.write(_format_syntax_error(e))
        sys.stderr.flush()
        sys.exit(1)

    except Exception as e:
        printed = captured.getvalue()
        if printed:
            sys.stderr.write("[captured stdout/stderr before error]\n")
            sys.stderr.write(printed)

        traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
        _write_code_context(e, out=sys.stderr)

        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
