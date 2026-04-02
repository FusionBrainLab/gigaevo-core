from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType
from typing import Any

import yaml

PROBLEM_DIR = Path(__file__).resolve().parent


def _find_wrapper_path(start_dir: Path) -> Path:
    """Locate the executor wrapper even if the problem is nested under subfolders."""
    relative_wrapper = Path("gigaevo/programs/stages/python_executors/wrapper.py")
    for candidate_root in [start_dir, *start_dir.parents]:
        candidate = candidate_root / relative_wrapper
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate gigaevo executor wrapper relative to "
        f"{start_dir}. Expected to find {relative_wrapper} in a parent directory."
    )


WRAPPER_PATH = _find_wrapper_path(PROBLEM_DIR)


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_exec_runner():
    wrapper_module = _load_module(
        "algotune_lyapunov_stability_exec_wrapper", WRAPPER_PATH
    )
    return wrapper_module.run_exec_runner


def _load_runtime_config(metrics_path: Path) -> tuple[int, int]:
    try:
        data = yaml.safe_load(metrics_path.read_text()) or {}
    except Exception:
        return 5, 3

    runtime_cfg = data.get("runtime_evaluation", {})
    timing_repetitions = max(1, int(runtime_cfg.get("timing_repetitions", 5)))
    warmup_repetitions = max(0, int(runtime_cfg.get("warmup_repetitions", 3)))
    return timing_repetitions, warmup_repetitions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local program against the algotune_lyapunov_stability context "
            "and validator, then compute the same speed metrics used by the "
            "algotune_speed pipeline."
        )
    )
    parser.add_argument(
        "--program",
        type=Path,
        default=PROBLEM_DIR / "initial_programs" / "baseline.py",
        help="Path to the program file containing entrypoint(context).",
    )
    parser.add_argument(
        "--timing-repetitions",
        type=int,
        default=None,
        help="Override metrics.yaml timing_repetitions.",
    )
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        default=None,
        help="Override metrics.yaml warmup_repetitions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as compact JSON only.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each isolated entrypoint run.",
    )
    return parser


async def _execute_entrypoint(
    *,
    code: str,
    context: dict[str, Any],
    timeout: int,
    run_exec_runner: Any,
) -> Any:
    outputs, _, _ = await run_exec_runner(
        code=code,
        function_name="entrypoint",
        args=[context],
        python_path=[PROBLEM_DIR],
        timeout=timeout,
    )
    return outputs


async def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    program_path = args.program.resolve()
    if not program_path.exists():
        raise FileNotFoundError(f"Program file not found: {program_path}")

    if str(PROBLEM_DIR) not in sys.path:
        sys.path.insert(0, str(PROBLEM_DIR))

    context_module = _load_module(
        "algotune_lyapunov_stability_context", PROBLEM_DIR / "context.py"
    )
    validate_module = _load_module(
        "algotune_lyapunov_stability_validate", PROBLEM_DIR / "validate.py"
    )
    program_module = _load_module("algotune_lyapunov_stability_program", program_path)

    if not hasattr(program_module, "entrypoint"):
        raise AttributeError(f"{program_path} does not define entrypoint(context).")

    run_exec_runner = _load_run_exec_runner()

    timing_repetitions, warmup_repetitions = _load_runtime_config(
        PROBLEM_DIR / "metrics.yaml"
    )
    if args.timing_repetitions is not None:
        timing_repetitions = max(1, int(args.timing_repetitions))
    if args.warmup_repetitions is not None:
        warmup_repetitions = max(0, int(args.warmup_repetitions))

    context = context_module.build_context()
    program_code = program_path.read_text()

    outputs = await _execute_entrypoint(
        code=program_code,
        context=context,
        timeout=args.timeout,
        run_exec_runner=run_exec_runner,
    )
    metrics = dict(validate_module.validate(context, outputs))

    for _ in range(warmup_repetitions):
        await _execute_entrypoint(
            code=program_code,
            context=context,
            timeout=args.timeout,
            run_exec_runner=run_exec_runner,
        )

    timed_runs: list[float] = []
    for _ in range(timing_repetitions):
        started = time.perf_counter()
        await _execute_entrypoint(
            code=program_code,
            context=context,
            timeout=args.timeout,
            run_exec_runner=run_exec_runner,
        )
        timed_runs.append(time.perf_counter() - started)

    execution_time_sec = statistics.median(timed_runs)
    metrics["execution_time_sec"] = float(execution_time_sec)
    metrics["fitness"] = float(1.0 / max(execution_time_sec, 1.0e-9))
    metrics["timing_repetitions"] = float(timing_repetitions)
    metrics["warmup_repetitions"] = float(warmup_repetitions)
    metrics["program_path"] = str(program_path)

    if args.json:
        print(json.dumps(metrics, sort_keys=True))
    else:
        print(f"Program: {program_path}")
        print(f"Warm-up repetitions: {warmup_repetitions}")
        print(f"Timing repetitions: {timing_repetitions}")
        print(json.dumps(metrics, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
