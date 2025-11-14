from __future__ import annotations

import resource
import sys
import traceback
import types
from typing import Any, Dict, List

import cloudpickle


def _load_module_from_code(code: str, name: str = "user_code") -> types.ModuleType:
    mod = types.ModuleType(name)
    compiled = compile(code, "<user_code>", "exec")
    exec(compiled, mod.__dict__)
    return mod


def _prepend_sys_path(paths: list[str] | None) -> None:
    if not paths:
        return
    for p in paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _set_memory_limit(max_memory_mb: int | None) -> None:
    """
    Set memory limit for the current process to prevent RAM exhaustion.
    Args:
        max_memory_mb: Maximum memory in megabytes, or None for no limit.
    """
    if max_memory_mb is None:
        return

    max_bytes = max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, max_bytes))


def main() -> None:
    try:
        payload: Dict[str, Any] = cloudpickle.loads(sys.stdin.buffer.read())
        code: str = payload["code"]
        fn_name: str = payload["function_name"]
        py_path: List[str] = payload.get("python_path", [])
        args: List[Any] = payload.get("args", [])
        kwargs: Dict[str, Any] = payload.get("kwargs", {})
        max_memory_mb: int | None = payload.get("max_memory_mb")

        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise TypeError("Payload must contain 'args': list and 'kwargs': dict")

        # Set memory limit BEFORE executing any user code
        _set_memory_limit(max_memory_mb)

        _prepend_sys_path(py_path)
        mod = _load_module_from_code(code)
        fn = getattr(mod, fn_name, None)
        if not callable(fn):
            raise ValueError(f"Function '{fn_name}' not found or not callable")

        result = fn(*args, **kwargs)

        sys.stdout.buffer.write(cloudpickle.dumps(result))
        sys.stdout.buffer.flush()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
