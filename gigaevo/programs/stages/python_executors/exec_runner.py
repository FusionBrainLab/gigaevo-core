"""Worker-side user-code execution.

Loaded inside a loky worker by :func:`gigaevo.programs.stages.python_executors.wrapper.run_exec_runner`.
Loads a user-supplied code string as a synthetic ``user_code`` module, calls
the named function, and returns ``(value, None)`` on success or
``(None, error_dict)`` on failure.  All process-level isolation (subprocess
lifetime, IPC, timeouts, memory) lives in :mod:`wrapper`; this module only
deals with the in-worker concerns: sys.path, env mutation, source-registered
tracebacks, and user-exception capture.
"""

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
    """Inject *source* into linecache so tracebacks show the user's lines."""
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
    """Insert each *path* at the head of ``sys.path``, deduplicating by
    resolved location.  Order of the input list is preserved at the front
    (left-most wins after iteration)."""
    if not paths:
        return
    resolved_cache: list[tuple[str, str]] = []
    for entry in sys.path:
        try:
            resolved_cache.append((entry, str(Path(entry).resolve())))
        except OSError:
            resolved_cache.append((entry, entry))

    for raw_path in reversed(paths):
        if not raw_path:
            continue
        resolved = str(Path(raw_path).resolve())
        sys.path[:] = [
            entry for entry, normalized in resolved_cache if normalized != resolved
        ]
        sys.path.insert(0, resolved)
        resolved_cache = []
        for entry in sys.path:
            try:
                resolved_cache.append((entry, str(Path(entry).resolve())))
            except OSError:
                resolved_cache.append((entry, entry))


def _iter_top_level_module_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    names: set[str] = set()
    for child in path.iterdir():
        if child.name == "__pycache__":
            continue
        if child.is_file() and child.suffix == ".py" and child.stem != "__init__":
            names.add(child.stem)
        elif child.is_dir() and (child / "__init__.py").is_file():
            names.add(child.name)
    return names


def _module_file_path(module: types.ModuleType) -> Path | None:
    filename = getattr(module, "__file__", None)
    if not filename:
        return None
    try:
        return Path(filename).resolve()
    except OSError:
        return None


def _module_belongs_to_path(module: types.ModuleType, path: Path, name: str) -> bool:
    module_file = _module_file_path(module)
    if module_file is None:
        return False
    candidates = [
        candidate.resolve()
        for candidate in (path / f"{name}.py", path / name / "__init__.py")
        if candidate.exists()
    ]
    return module_file in candidates


def _clear_shadowed_top_level_modules(paths: list[str] | None) -> None:
    """Drop ``sys.modules`` entries that a freshly-prepended *paths* would
    shadow.  Without this, a long-lived worker keeps the first-loaded
    version of a top-level module even after the project root changes."""
    if not paths:
        return
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        for name in _iter_top_level_module_names(path):
            existing = sys.modules.get(name)
            if existing is None:
                continue
            if _module_belongs_to_path(existing, path, name):
                continue
            sys.modules.pop(name, None)


def _ensure_cwd_in_path() -> None:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _write_code_context(tb: BaseException, *, out: io.TextIOBase) -> None:
    """Append a ±3-line excerpt of the user's source around the failing line."""
    try:
        extracted = traceback.extract_tb(tb.__traceback__)
        user_frames = [f for f in extracted if f.filename == _CODE_FILENAME]
        if not user_frames:
            return
        lineno = user_frames[-1].lineno
        lines = linecache.getlines(_CODE_FILENAME)
        if not lines or lineno is None:
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


_ENV_MISSING = object()


def _apply_env(env: dict[str, Any]) -> dict[str, Any]:
    old: dict[str, Any] = {}
    for k, v in env.items():
        old[k] = os.environ.get(k, _ENV_MISSING)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    return old


def _restore_env(old: dict[str, Any]) -> None:
    for k, v in old.items():
        if v is _ENV_MISSING:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _error_envelope(message: str, returncode: int = 1) -> dict[str, Any]:
    return {"_error": True, "stderr": message, "returncode": returncode}


def _run_one(payload: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
    """Run a single user-code call.

    Returns ``(value, None)`` on success or ``(None, error_dict)`` on
    failure.  ``error_dict`` has the shape
    ``{"_error": True, "stderr": <traceback text>, "returncode": int}``.
    """
    captured = io.StringIO()
    old_env: dict[str, Any] | None = None
    try:
        _ensure_cwd_in_path()

        code: str = payload["code"]
        fn_name: str = payload["function_name"]
        py_path: list[str] = payload.get("python_path", [])
        args: list[Any] = payload.get("args", [])
        kwargs: dict[str, Any] = payload.get("kwargs", {})
        env_updates: dict[str, Any] = payload.get("env", {}) or {}

        if env_updates:
            old_env = _apply_env(env_updates)

        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise TypeError("Payload must contain 'args': list and 'kwargs': dict")

        _prepend_sys_path(py_path)
        _clear_shadowed_top_level_modules(py_path)
        mod = _load_module_from_code(code)
        fn = getattr(mod, fn_name, None)
        if not callable(fn):
            raise ValueError(f"Function '{fn_name}' not found or not callable")

        with redirect_stdout(captured), redirect_stderr(captured):
            result = fn(*args, **kwargs)

        # User-visible prints surface in the worker's real stderr (so they
        # appear in worker logs); they are not returned to the parent.
        printed = captured.getvalue()
        if printed:
            sys.stderr.write(printed)
            sys.stderr.flush()

        # Register the synthetic module so cloudpickle embeds user-defined
        # classes by value; otherwise the parent cannot unpickle a result
        # that references them (the parent has no 'user_code' module).
        cloudpickle.register_pickle_by_value(mod)
        return (result, None)

    except SyntaxError as e:
        buf = io.StringIO()
        if printed := captured.getvalue():
            buf.write("[captured stdout/stderr before error]\n")
            buf.write(printed)
        buf.write(_format_syntax_error(e))
        return (None, _error_envelope(buf.getvalue()))

    except BaseException as e:
        # Catch BaseException so user sys.exit() / KeyboardInterrupt becomes
        # a structured error rather than tearing down the worker.
        buf = io.StringIO()
        if printed := captured.getvalue():
            buf.write("[captured stdout/stderr before error]\n")
            buf.write(printed)
        traceback.print_exception(type(e), e, e.__traceback__, file=buf)
        _write_code_context(e, out=buf)
        return (None, _error_envelope(buf.getvalue()))

    finally:
        if old_env is not None:
            _restore_env(old_env)
