from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from gigaevo.problems.context import ProblemContext

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS_ROOT = ROOT / "problems" / "algotune"

TASKS = [
    ("algotune_qp", "quadratic"),
    ("algotune_markowitz", "markowitz"),
    ("algotune_lp_box", "boxed linear"),
    ("algotune_chebyshev_center", "chebyshev-center"),
    ("algotune_power_control", "power-control"),
    ("algotune_battery_scheduling", "battery-scheduling"),
]


def _load_module(problem_dir: Path, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(problem_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize(("task_name", "description_snippet"), TASKS)
def test_problem_context_is_valid(task_name: str, description_snippet: str) -> None:
    problem_dir = PROBLEMS_ROOT / task_name
    pc = ProblemContext(problem_dir)
    pc.validate(add_context=True)

    assert pc.is_contextual is True
    assert pc.metrics_context.get_primary_key() == "fitness"
    assert description_snippet in pc.task_description.lower()


@pytest.mark.parametrize(("task_name", "_description_snippet"), TASKS)
def test_baseline_solves_fixed_suite(task_name: str, _description_snippet: str) -> None:
    problem_dir = PROBLEMS_ROOT / task_name
    context_module = _load_module(
        problem_dir,
        f"{task_name}_context",
        problem_dir / "context.py",
    )
    baseline_module = _load_module(
        problem_dir,
        f"{task_name}_baseline",
        problem_dir / "initial_programs" / "baseline.py",
    )
    validate_module = _load_module(
        problem_dir,
        f"{task_name}_validate",
        problem_dir / "validate.py",
    )

    context = context_module.build_context()
    outputs = baseline_module.entrypoint(context)
    metrics = validate_module.validate(context, outputs)

    assert metrics["is_valid"] == 1.0
    assert metrics["exact_case_fraction"] == pytest.approx(1.0)
