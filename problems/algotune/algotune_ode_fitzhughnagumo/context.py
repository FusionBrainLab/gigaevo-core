from __future__ import annotations

from algotune_ode_fitzhughnagumo_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_ode_fitzhughnagumo",
        "source_task": "AlgoTune ode_fitzhughnagumo",
        "mode": "full",
        "cases": get_case_specs(),
    }
