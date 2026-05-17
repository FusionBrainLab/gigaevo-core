"""Typed entry point for the evolutionary search runtime.

Replaces ``run.py`` once the Hydra cutover lands. Today both coexist:
``run.py`` runs the legacy YAML path; ``python -m gigaevo.cli
experiments/X.py`` runs the typed path. The parity matrix
(:mod:`tests.integration.test_hydra_parity`) exercises both on every PR
until the cutover completes.

The CLI is intentionally thin: ~100 LOC of explicit construction with
no decorator magic, no chdir, no module singletons. Configuration
loads through :func:`build_experiment` (Pydantic-validated),
CLI overrides apply via tyro (auto-generated from the model field
tree), the resolved config dumps to JSON for reproducibility, and the
typed graph is handed to the engine.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from gigaevo.config.experiment_loader import build_experiment
from gigaevo.config.schemas.experiment import ExperimentConfig

# ``run_experiment`` is the existing async workflow in run.py. The
# cutover (hydra-3.5) renames cli.py to run.py and the legacy
# DictConfig-shaped run_experiment is rewritten to accept
# ExperimentConfig directly. Today the CLI imports the legacy callable
# via a lazy import to avoid pulling Hydra into typed code paths used
# by tests; the import happens inside ``main`` so importing this module
# never triggers Hydra registration.

_RUN_OBJECT_GRAPH_PENDING = (
    "build_object_graph(cfg) ships in hydra-1.10; until then the CLI "
    "validates configuration and writes the resolved tree to disk but "
    "does not invoke the engine. Use python run.py for end-to-end "
    "execution against the legacy Hydra path."
)


def _parse_initial_args(
    argv: list[str],
) -> tuple[Path, bool, list[str]]:
    """Parse the experiment-path + dry-run prefix; forward the remainder
    to tyro for nested field overrides."""
    parser = argparse.ArgumentParser(
        prog="gigaevo",
        description="Evolutionary search runtime — typed entry point",
    )
    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to an experiment Python file that exports build() -> ExperimentConfig",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, validate, and dump the resolved config without invoking the engine",
    )
    parsed, overrides = parser.parse_known_args(argv)
    return parsed.experiment, parsed.dry_run, overrides


def _apply_tyro_overrides(
    baseline: ExperimentConfig, override_args: list[str]
) -> ExperimentConfig:
    """Apply ``--key value`` overrides via tyro, re-running every Pydantic
    validator against the merged configuration."""
    if not override_args:
        return baseline

    import tyro

    return tyro.cli(
        ExperimentConfig,
        default=baseline,
        args=override_args,
        prog="gigaevo overrides",
    )


def _dump_resolved_config(cfg: ExperimentConfig) -> Path:
    """Write ``config.json`` under ``output_dir/experiment_id`` and
    return the absolute path. The dump is the reproducibility record:
    given identical inputs, two runs share an output directory."""
    out_dir = (cfg.output_dir / cfg.experiment_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(cfg.model_dump_json(indent=2))
    logger.info(
        "Resolved config dumped to {} (experiment_id={})",
        config_path,
        cfg.experiment_id,
    )
    return config_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code so the function is
    usable from both a script entry and from in-process integration
    tests that want to assert exit semantics."""
    if argv is None:
        argv = sys.argv[1:]

    experiment_path, dry_run, override_args = _parse_initial_args(argv)

    load_dotenv()

    baseline = build_experiment(experiment_path)
    cfg = _apply_tyro_overrides(baseline, override_args)

    config_path = _dump_resolved_config(cfg)

    if dry_run:
        logger.info(
            "Dry run complete. Validated config at {}. Engine invocation skipped.",
            config_path,
        )
        return 0

    logger.warning(_RUN_OBJECT_GRAPH_PENDING)

    try:
        from gigaevo.config.object_graph import run_with_config
    except ImportError:
        logger.warning(
            "object_graph not yet available (hydra-1.10 deliverable); "
            "skipping engine invocation. Config validation succeeded."
        )
        return 0

    return asyncio.run(run_with_config(cfg))


if __name__ == "__main__":
    sys.exit(main())
