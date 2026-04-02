from __future__ import annotations

from algotune_lp_box_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_lp_box",
        "source_task": "AlgoTune lp_box",
        "mode": "full",
        "cases": get_case_specs(),
    }
