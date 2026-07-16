"""Shared experiment-log path resolution for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ManifestRun(Protocol):
    label: str
    log_path: str | None


def experiment_dir(experiment: str) -> Path:
    from gigaevo.experiment.manifest import experiment_dir as manifest_experiment_dir

    return manifest_experiment_dir(experiment)


def problem_metrics_path(problem_name: str) -> Path:
    """Resolve a problem's metric schema from the canonical project root."""
    from gigaevo.experiment.manifest import PROJ

    return PROJ / "problems" / problem_name / "metrics.yaml"


def run_log_path(experiment: str, run: ManifestRun) -> Path:
    """Resolve a run's configured log path relative to its experiment."""
    configured = Path(run.log_path or f"run_{run.label}.log")
    if configured.is_absolute():
        return configured
    return experiment_dir(experiment) / configured
