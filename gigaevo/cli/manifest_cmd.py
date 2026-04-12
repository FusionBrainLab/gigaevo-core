"""Manifest subcommand group: read, write, and gate experiment.yaml fields.

Replaces inline ``PYTHONPATH=. python -c "from tools.experiment.manifest import ..."``
snippets in experiment lifecycle skills with proper CLI calls.

Usage examples::

    gigaevo -e hover/foo manifest get status
    gigaevo -e hover/foo manifest get runs
    gigaevo -e hover/foo manifest get launch.watchdog_pid
    gigaevo -e hover/foo manifest set status running
    gigaevo -e hover/foo manifest update launch.watchdog_pid 12345
    gigaevo -e hover/foo manifest gate implemented
    gigaevo -e hover/foo manifest pr-description --push
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import click


def _require_experiment(ctx: click.Context) -> str:
    """Extract experiment name from context or exit with error."""
    experiment: str | None = ctx.obj.get("experiment")
    if not experiment:
        click.echo("Error: manifest commands require --experiment / -e flag.", err=True)
        ctx.exit(1)
        raise SystemExit(1)
    return experiment


def _coerce_value(raw_value: str) -> Any:
    """Auto-convert string value to appropriate Python type.

    Conversion order: bool literals, null/None, int, float, string.
    """
    lower = raw_value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none"):
        return None
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


def _traverse_raw(raw: dict[str, Any], dotted_path: str) -> Any:
    """Walk a dotted path through nested dicts. Raises KeyError if not found."""
    parts = dotted_path.split(".")
    current: Any = raw
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _set_nested(raw: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a value at a dotted path, creating intermediate dicts as needed."""
    parts = dotted_path.split(".")
    current = raw
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group()
@click.pass_context
def manifest(ctx: click.Context) -> None:
    """Read, write, and gate experiment.yaml fields."""
    pass


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

_KNOWN_SCALAR_FIELDS = {"status", "max_generations", "branch", "task", "name"}


@manifest.command()
@click.argument("field")
@click.pass_context
def get(ctx: click.Context, field: str) -> None:
    """Read a manifest field by name or dotted path.

    Special fields: status, runs, max_generations, stopping_rule, servers.
    Dotted paths (e.g. launch.watchdog_pid) traverse the raw YAML dict.
    """
    from gigaevo.cli.output_formatter import OutputFormatter

    experiment = _require_experiment(ctx)

    from tools.experiment.manifest import load_manifest

    manifest_obj = load_manifest(experiment)
    formatter: OutputFormatter = ctx.obj["formatter"]

    if field == "runs":
        rows = [
            {
                "Label": run.label,
                "DB": run.db,
                "Prefix": run.prefix,
                "Pipeline": run.pipeline,
                "PID": run.pid if run.pid is not None else "-",
            }
            for run in manifest_obj.runs
        ]
        formatter.echo(rows, columns=["Label", "DB", "Prefix", "Pipeline", "PID"], title="Runs")
        return

    if field == "servers":
        for server in manifest_obj.servers:
            click.echo(server)
        return

    if field in _KNOWN_SCALAR_FIELDS:
        click.echo(getattr(manifest_obj, field))
        return

    if field == "stopping_rule":
        stopping_rule = manifest_obj.config.get("stopping_rule")
        if stopping_rule is None:
            stopping_rule = manifest_obj._raw.get("config", {}).get("stopping_rule")
        if stopping_rule is None:
            click.echo("Error: Field not found: stopping_rule", err=True)
            ctx.exit(1)
            return
        click.echo(stopping_rule)
        return

    try:
        value = _traverse_raw(manifest_obj._raw, field)
    except KeyError:
        click.echo(f"Error: Field not found: {field}", err=True)
        ctx.exit(1)
        return

    if isinstance(value, dict):
        click.echo(json.dumps(value, indent=2, default=str))
    elif isinstance(value, list):
        click.echo(json.dumps(value, indent=2, default=str))
    else:
        click.echo(value)


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------


@manifest.command("set")
@click.argument("field")
@click.argument("value")
@click.pass_context
def set_field(ctx: click.Context, field: str, value: str) -> None:
    """Write a top-level field. Currently only supports 'status'.

    For arbitrary fields, use 'manifest update' instead.
    """
    experiment = _require_experiment(ctx)

    if field != "status":
        click.echo(
            f"Error: 'set' only supports 'status'. "
            f"Use 'manifest update {field} {value}' for arbitrary fields.",
            err=True,
        )
        ctx.exit(1)
        return

    from tools.experiment.manifest import set_status

    updated = set_status(experiment, value)
    click.echo(f"Status updated: {updated.status}")


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@manifest.command()
@click.argument("path")
@click.argument("value")
@click.pass_context
def update(ctx: click.Context, path: str, value: str) -> None:
    """Write any field by dotted path (e.g. launch.watchdog_pid 12345).

    Auto-converts values: integers, floats, booleans (true/false),
    null/None, or keeps as string.
    """
    experiment = _require_experiment(ctx)

    coerced = _coerce_value(value)

    from tools.experiment.manifest import update_manifest

    def updater(raw: dict[str, Any]) -> None:
        _set_nested(raw, path, coerced)

    update_manifest(experiment, updater)
    click.echo(f"Updated {path} = {coerced!r}")


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


@manifest.command()
@click.argument("expected_status")
@click.pass_context
def gate(ctx: click.Context, expected_status: str) -> None:
    """Assert experiment status matches expected value.

    Exits 0 on match (GATE PASSED), exits 1 on mismatch (BLOCKED).
    """
    experiment = _require_experiment(ctx)

    from tools.experiment.manifest import load_manifest

    manifest_obj = load_manifest(experiment)

    if manifest_obj.status == expected_status:
        click.echo(
            f"GATE PASSED: {manifest_obj.name} status={manifest_obj.status} "
            f"({len(manifest_obj.runs)} runs, max_gen={manifest_obj.max_generations})"
        )
        return

    click.echo(
        f"BLOCKED: status={manifest_obj.status}, expected {expected_status}",
        err=True,
    )
    ctx.exit(1)


# ---------------------------------------------------------------------------
# pr-description
# ---------------------------------------------------------------------------


@manifest.command("pr-description")
@click.option("--push/--no-push", default=False, help="Push description to GitHub PR.")
@click.pass_context
def pr_description(ctx: click.Context, push: bool) -> None:
    """Generate PR description from experiment.yaml.

    With --push, also updates the GitHub PR body via `gh pr edit`.
    """
    experiment = _require_experiment(ctx)

    from tools.experiment.manifest import generate_pr_description

    description = generate_pr_description(experiment)
    click.echo(description)

    if push:
        from tools.experiment.manifest import load_manifest

        manifest_obj = load_manifest(experiment)
        if manifest_obj.pr_number:
            subprocess.run(
                ["gh", "pr", "edit", str(manifest_obj.pr_number), "--body", description],
                check=True,
            )
            click.echo(f"PR #{manifest_obj.pr_number} description updated.")
        else:
            click.echo("Warning: No pr_number in manifest; --push skipped.", err=True)
