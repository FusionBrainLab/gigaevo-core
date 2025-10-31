# gigaevo/programs/exec_runner.py
from __future__ import annotations

import pickle
import sys
import traceback
import types
from typing import Any, Dict, List


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


def main() -> None:
    try:
        payload: Dict[str, Any] = pickle.loads(sys.stdin.buffer.read())
        code: str = payload["code"]
        fn_name: str = payload["function_name"]
        py_path: List[str] = payload.get("python_path", [])
        args: List[Any] = payload.get("args", [])
        kwargs: Dict[str, Any] = payload.get("kwargs", {})

        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise TypeError("Payload must contain 'args': list and 'kwargs': dict")

        _prepend_sys_path(py_path)
        mod = _load_module_from_code(code)
        fn = getattr(mod, fn_name, None)
        if not callable(fn):
            raise ValueError(f"Function '{fn_name}' not found or not callable")

        result = fn(*args, **kwargs)

        sys.stdout.buffer.write(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
        sys.stdout.buffer.flush()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
