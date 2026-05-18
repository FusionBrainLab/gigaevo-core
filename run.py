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
import contextlib
import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
from loguru import logger

from gigaevo.config.experiment_loader import build_experiment
from gigaevo.config.schemas.experiment import ExperimentConfig


def _build_initial_parser() -> argparse.ArgumentParser:
    """Build the argparse layer that owns the experiment-path positional
    and the ``--dry-run`` switch. ``add_help`` stays off so a trailing
    ``--help`` after the experiment path reaches the tyro layer and
    prints the typed-override field tree."""
    parser = argparse.ArgumentParser(
        prog="gigaevo",
        description="Evolutionary search runtime — typed entry point",
        add_help=False,
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        type=Path,
        help="Path to an experiment Python file that exports build() -> ExperimentConfig",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, validate, and dump the resolved config without invoking the engine",
    )
    parser.add_argument(
        "-h",
        "--help",
        dest="help",
        action="store_true",
        help="Print this help; with an experiment argument, also print the tyro field tree",
    )
    return parser


def _parse_initial_args(
    argv: list[str],
) -> tuple[Path | None, bool, bool, list[str]]:
    """Parse the experiment-path + dry-run prefix; forward the remainder
    to tyro for nested field overrides.

    Returns ``(experiment_path, dry_run, help_requested, overrides)``.
    """
    parser = _build_initial_parser()
    parsed, overrides = parser.parse_known_args(argv)
    return parsed.experiment, parsed.dry_run, parsed.help, overrides


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
    given identical inputs, two runs share an output directory.

    Two concurrent sweep workers can resolve to the same
    ``experiment_id`` (overrides that don't affect the hashed fields).
    The write is performed via ``tempfile.NamedTemporaryFile`` +
    ``os.replace`` so a half-written ``config.json`` is never visible
    to a peer reader and last-writer-wins semantics hold without data
    corruption.
    """
    out_dir = (cfg.output_dir / cfg.experiment_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    payload = cfg.model_dump_json(indent=2)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out_dir,
            prefix=".config.",
            suffix=".json.tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, config_path)
        tmp_path = None
    finally:
        # On any failure between tempfile creation and replace, the
        # ``.config.*.tmp`` entry would otherwise accumulate in the
        # output directory across retries.
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
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

    experiment_path, dry_run, help_requested, override_args = _parse_initial_args(
        argv
    )

    if help_requested and experiment_path is None:
        _build_initial_parser().print_help()
        return 0

    if experiment_path is None:
        _build_initial_parser().print_usage(sys.stderr)
        print(
            "gigaevo: error: the following arguments are required: experiment",
            file=sys.stderr,
        )
        return 2

    load_dotenv()

    baseline = build_experiment(experiment_path)

    if help_requested:
        _build_initial_parser().print_help()
        print()
        import tyro

        tyro.cli(
            ExperimentConfig,
            default=baseline,
            args=["--help"],
            prog="gigaevo overrides",
        )
        return 0

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
