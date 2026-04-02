from __future__ import annotations

from algotune_ode_seirs_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_ode_seirs",
        "source_task": "AlgoTune ode_seirs",
        "mode": "full",
        "cases": get_case_specs(),
    }
