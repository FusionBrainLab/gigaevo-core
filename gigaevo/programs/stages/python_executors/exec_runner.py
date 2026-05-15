"""Worker-side user-code execution.

Loaded inside a loky worker by :func:`gigaevo.programs.stages.python_executors.wrapper.run_exec_runner`.
Loads a user-supplied code string as a synthetic ``user_code`` module, calls
the named function, and returns ``(value, None)`` on success or
``(None, WorkerError)`` on failure.  All process-level isolation (subprocess
lifetime, IPC, timeouts) lives in :mod:`wrapper`; this module only deals
with the in-worker concerns: sys.path, env mutation, source-registered
tracebacks, and user-exception capture.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class WorkerCall:
    """Picklable description of a single user-code invocation."""

    code: str
    function_name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    python_path: list[str] = field(default_factory=list)
    env: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerError:
    """Structured failure: ``stderr`` is the formatted traceback."""

    stderr: str
    returncode: int = 1


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


def _prepend_sys_path(paths: list[str]) -> None:
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


def _clear_shadowed_top_level_modules(paths: list[str]) -> None:
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


@contextmanager
def _scoped_env(updates: dict[str, str | None]):
    """Apply *updates* to ``os.environ`` for the duration of the block.

    A ``None`` value means "unset this variable for the block".  Original
    values (or absence) are restored on exit, even if the block raises.
    """
    if not updates:
        yield
        return

    old: dict[str, Any] = {}
    for k, v in updates.items():
        old[k] = os.environ.get(k, _ENV_MISSING)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is _ENV_MISSING:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _format_error(prefix_text: str, captured: str = "") -> str:
    buf = io.StringIO()
    if captured:
        buf.write("[captured stdout/stderr before error]\n")
        buf.write(captured)
    buf.write(prefix_text)
    return buf.getvalue()


def _run_one(call: WorkerCall) -> tuple[Any, WorkerError | None]:
    """Run a single user-code call inside a (possibly reused) worker.

    Catches ``BaseException`` so ``sys.exit()``/``KeyboardInterrupt`` become
    structured errors instead of tearing down the worker.  Snapshots and
    restores ``cwd`` and ``sys.path`` so user-code mutations don't leak into
    the next caller; ``sys.modules`` is intentionally left alone (matches
    CPython ``-c`` semantics).
    """
    captured = io.StringIO()
    try:
        cwd_before: str | None = os.getcwd()
    except OSError:
        cwd_before = None
    sys_path_before = list(sys.path)

    try:
        _ensure_cwd_in_path()

        if not isinstance(call.args, list) or not isinstance(call.kwargs, dict):
            raise TypeError("WorkerCall.args must be list and .kwargs must be dict")

        _prepend_sys_path(call.python_path)
        _clear_shadowed_top_level_modules(call.python_path)

        with _scoped_env(call.env):
            mod = _load_module_from_code(call.code)
            fn = getattr(mod, call.function_name, None)
            if not callable(fn):
                raise ValueError(
                    f"Function '{call.function_name}' not found or not callable"
                )

            with redirect_stdout(captured), redirect_stderr(captured):
                result = fn(*call.args, **call.kwargs)

            printed = captured.getvalue()
            if printed:
                sys.stderr.write(printed)
                sys.stderr.flush()

        # The parent has no ``user_code`` module; register-by-value embeds
        # any user-defined classes into the pickle stream.  Set-keyed-by-name
        # so this is idempotent across calls.
        cloudpickle.register_pickle_by_value(mod)
        return (result, None)

    except SyntaxError as e:
        return (
            None,
            WorkerError(stderr=_format_error(_format_syntax_error(e), captured.getvalue())),
        )

    except BaseException as e:
        buf = io.StringIO()
        traceback.print_exception(type(e), e, e.__traceback__, file=buf)
        _write_code_context(e, out=buf)
        return (
            None,
            WorkerError(stderr=_format_error(buf.getvalue(), captured.getvalue())),
        )

    finally:
        sys.path[:] = sys_path_before
        if cwd_before is not None:
            with contextlib.suppress(OSError):
                os.chdir(cwd_before)
