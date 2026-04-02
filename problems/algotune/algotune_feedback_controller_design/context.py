from __future__ import annotations

from algotune_feedback_controller_design_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_feedback_controller_design",
        "source_task": "AlgoTune feedback_controller_design",
        "mode": "full",
        "cases": get_case_specs(),
    }
