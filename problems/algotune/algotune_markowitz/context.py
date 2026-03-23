from __future__ import annotations

from algotune_markowitz_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_markowitz",
        "source_task": "AlgoTune markowitz",
        "mode": "full",
        "cases": get_case_specs(),
    }
