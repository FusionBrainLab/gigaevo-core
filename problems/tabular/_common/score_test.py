"""End-of-evolution TEST scorer (standalone; never a search signal).

Usage:
    python problems/tabular/<dataset>/score_test.py initial_programs/prog1.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

# NOTE: strip the filename first, THEN resolve the (real) dataset dir.
# Path(__file__).resolve().parent would follow the symlink into _common.
_DATASET_DIR = Path(__file__).parent.resolve()
_COMMON_DIR = _DATASET_DIR.parent / "_common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from tabular_problem import build  # noqa: E402

_PROBLEM = build(_DATASET_DIR.name)


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
    for arg in sys.argv[1:]:
        path = Path(arg).resolve()
        result = _PROBLEM.score_on_test(_load_entrypoint(path))
        metrics = "  ".join(f"{k}={v:.5f}" for k, v in result.items())
        print(f"{path.name}: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
