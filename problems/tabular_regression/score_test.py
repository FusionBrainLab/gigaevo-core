"""End-of-evolution TEST scorer.

Usage:
    python score_test.py initial_programs/prog8.py
    python score_test.py /path/to/elite.py

Loads `entrypoint()` from the given file, calls validate.py's `score_on_test`,
and prints the test RMSE. This is the unbiased held-out number — invoke only
AFTER evolution finishes, never as a search signal.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from validate import score_on_test  # noqa: E402


def _load_entrypoint(prog_path: Path):
    spec = importlib.util.spec_from_file_location(prog_path.stem, prog_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {prog_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "entrypoint"):
        raise RuntimeError(f"{prog_path} does not define entrypoint()")
    return mod.entrypoint()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: score_test.py <program.py> [<program.py> ...]", file=sys.stderr)
        return 2

    rows = []
    for arg in sys.argv[1:]:
        path = Path(arg).resolve()
        factory = _load_entrypoint(path)
        result = score_on_test(factory)
        rows.append((path.name, result["test_rmse"]))

    width = max((len(n) for n, _ in rows), default=10)
    print(f"{'program':<{width}}  test_rmse")
    print("-" * (width + 12))
    for name, rmse in rows:
        print(f"{name:<{width}}  {rmse:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
