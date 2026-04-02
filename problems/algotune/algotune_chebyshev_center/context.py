from __future__ import annotations

from algotune_chebyshev_center_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_chebyshev_center",
        "source_task": "AlgoTune chebyshev_center",
        "mode": "full",
        "cases": get_case_specs(),
    }
