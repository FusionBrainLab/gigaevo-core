from __future__ import annotations

from algotune_pde_heat1d_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_pde_heat1d",
        "source_task": "AlgoTune pde_heat1d",
        "mode": "full",
        "cases": get_case_specs(),
    }
