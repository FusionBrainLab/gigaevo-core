from __future__ import annotations

from algotune_battery_scheduling_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_battery_scheduling",
        "source_task": "AlgoTune battery_scheduling",
        "mode": "full",
        "cases": get_case_specs(),
    }
