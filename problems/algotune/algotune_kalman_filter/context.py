from __future__ import annotations

from algotune_kalman_filter_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_kalman_filter",
        "source_task": "AlgoTune kalman_filter",
        "mode": "full",
        "cases": get_case_specs(),
    }
