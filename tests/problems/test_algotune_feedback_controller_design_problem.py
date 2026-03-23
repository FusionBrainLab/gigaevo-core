from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from gigaevo.problems.context import ProblemContext

ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = ROOT / "problems" / "algotune_feedback_controller_design"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROBLEM_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_problem_context_is_valid() -> None:
    pc = ProblemContext(PROBLEM_DIR)
    pc.validate(add_context=True)

    assert pc.is_contextual is True
    assert pc.metrics_context.get_primary_key() == "fitness"
    assert "feedback-controller design" in pc.task_description.lower()


def test_baseline_solves_fixed_suite() -> None:
    context_module = _load_module(
        "algotune_feedback_controller_design_context", PROBLEM_DIR / "context.py"
    )
    baseline_module = _load_module(
        "algotune_feedback_controller_design_baseline",
        PROBLEM_DIR / "initial_programs" / "baseline.py",
    )
    validate_module = _load_module(
        "algotune_feedback_controller_design_validate", PROBLEM_DIR / "validate.py"
    )

    context = context_module.build_context()
    outputs = baseline_module.entrypoint(context)
    metrics = validate_module.validate(context, outputs)

    assert metrics["is_valid"] == 1.0
    assert metrics["exact_case_fraction"] == pytest.approx(1.0)
