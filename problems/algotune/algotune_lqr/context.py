from __future__ import annotations

from algotune_lqr_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_lqr",
        "source_task": "AlgoTune lqr",
        "mode": "full",
        "cases": get_case_specs(),
    }
