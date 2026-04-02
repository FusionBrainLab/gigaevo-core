from __future__ import annotations

from algotune_qp_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_qp",
        "source_task": "AlgoTune qp",
        "mode": "full",
        "cases": get_case_specs(),
    }
