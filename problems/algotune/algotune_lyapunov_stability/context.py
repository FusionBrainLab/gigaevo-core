from __future__ import annotations

from algotune_lyapunov_stability_helper import get_case_specs


def build_context() -> dict[str, object]:
    """Return lightweight deterministic case metadata for program execution."""
    return {
        "task_name": "algotune_lyapunov_stability",
        "source_task": "AlgoTune lyapunov_stability",
        "mode": "full",
        "cases": get_case_specs(),
    }
