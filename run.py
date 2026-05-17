"""Typed entry point for the evolutionary search runtime.

The CLI is intentionally thin: explicit construction with no decorator
magic, no chdir, no module singletons. Configuration loads through
:func:`build_experiment` (Pydantic-validated), CLI overrides apply via
tyro (auto-generated from the model field tree), the resolved config
dumps to JSON for reproducibility, and the typed object graph is
handed to :func:`gigaevo.config.object_graph.run_with_config`.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

from gigaevo.config.experiment_loader import build_experiment
from gigaevo.config.schemas.experiment import ExperimentConfig


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

    from gigaevo.config.object_graph import run_with_config

    return asyncio.run(run_with_config(cfg))


if __name__ == "__main__":
    sys.exit(main())
