from __future__ import annotations

from algotune_ode_lotkavolterra_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_ode_lotkavolterra",
        "source_task": "AlgoTune ode_lotkavolterra",
        "mode": "full",
        "cases": get_case_specs(),
    }
