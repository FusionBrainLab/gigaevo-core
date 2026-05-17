"""Subprocess-based sweep runner.

Spawns one ``python run.py <experiment> <overrides...>`` process per
combination produced by a sweep-definition module. Each subprocess
gets a fresh Python interpreter, so module-level state never leaks
between runs.

A sweep file is a Python module exporting ``define_sweep() ->
list[list[str]]``: each inner list is the argv slice to forward to
``run.py`` for one run. See ``sweeps/`` for shipped examples.

Invocation::

    python -m gigaevo.sweep experiments/base.py sweeps/seeds.py
    python -m gigaevo.sweep experiments/base.py sweeps/seeds.py --parallel 4
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_one(args: tuple[Path, list[str]]) -> int:
    experiment, overrides = args
    cmd = [sys.executable, str(REPO_ROOT / "run.py"), str(experiment), *overrides]
    return subprocess.run(cmd).returncode


def _load_sweep(path: Path) -> list[list[str]]:
    spec = importlib.util.spec_from_file_location("_gigaevo_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import sweep module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "define_sweep"):
        raise AttributeError(
            f"{path} must export define_sweep() -> list[list[str]]"
        )
    runs = module.define_sweep()
    if not isinstance(runs, list) or not all(
        isinstance(r, list) and all(isinstance(x, str) for x in r) for r in runs
    ):
        raise TypeError(
            f"{path}.define_sweep() must return list[list[str]]; got "
            f"{type(runs).__name__}"
        )
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gigaevo.sweep",
        description=(
            "Run an experiment over a parameter sweep, "
            "one subprocess per combination."
        ),
    )
    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to the experiment file (exports build() -> ExperimentConfig)",
    )
    parser.add_argument(
        "sweep",
        type=Path,
        help="Path to a Python module exporting define_sweep() -> list[list[str]]",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum concurrent subprocesses (default: 1, sequential)",
    )
    parsed = parser.parse_args(argv)

    runs = _load_sweep(parsed.sweep)
    work = [(parsed.experiment, ovs) for ovs in runs]

    if parsed.parallel <= 1:
        results = [_run_one(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=parsed.parallel) as pool:
            results = list(pool.map(_run_one, work))

    failures = sum(1 for rc in results if rc != 0)
    total = len(runs)
    if failures:
        print(
            f"Sweep finished: {failures}/{total} runs failed",
            file=sys.stderr,
        )
        return 1
    print(f"Sweep finished: {total}/{total} runs OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
