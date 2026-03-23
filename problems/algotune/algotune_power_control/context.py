from __future__ import annotations

from algotune_power_control_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_power_control",
        "source_task": "AlgoTune power_control",
        "mode": "full",
        "cases": get_case_specs(),
    }
