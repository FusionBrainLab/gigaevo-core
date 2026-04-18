"""GigaEvo CLI -- unified experiment monitoring and analysis."""

from __future__ import annotations

import importlib
from typing import Any

import click

from gigaevo.cli.output_formatter import OutputFormatter

# Lazy subcommand registry: name -> (module_path, attr_name)
_LAZY_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "status": ("gigaevo.cli.status", "status"),
    "trajectory": ("gigaevo.cli.trajectory", "trajectory"),
    "top": ("gigaevo.cli.top", "top"),
    "logs": ("gigaevo.cli.logs", "logs"),
    "plot": ("gigaevo.cli.plot_group", "plot"),
    "export": ("gigaevo.cli.export", "export"),
    "flush": ("gigaevo.cli.flush", "flush"),
    "watchdog": ("gigaevo.cli.watchdog_cmd", "watchdog"),
    "checkpoint": ("gigaevo.cli.checkpoint", "checkpoint"),
    "manifest": ("gigaevo.cli.manifest_cmd", "manifest"),
    "inspect": ("gigaevo.cli.inspect_cmd", "inspect_cmd"),
    "launch": ("gigaevo.cli.launch_cmd", "launch"),
    "events": ("gigaevo.cli.events_cmd", "events"),
}


class LazyGroup(click.Group):
    """Click group that imports subcommand modules only when invoked."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        rv = list(_LAZY_SUBCOMMANDS.keys())
        rv.extend(super().list_commands(ctx))
        return sorted(set(rv))

    def get_command(  # type: ignore[override]
        self, ctx: click.Context, cmd_name: str
    ) -> click.BaseCommand | None:
        # Check eagerly-registered commands first
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        # Lazy-load from registry
        if cmd_name in _LAZY_SUBCOMMANDS:
            module_path, attr_name = _LAZY_SUBCOMMANDS[cmd_name]
            mod = importlib.import_module(module_path)
            return getattr(mod, attr_name)
        return None


@click.group(cls=LazyGroup)
@click.option(
    "-e",
    "--experiment",
    type=str,
    default=None,
    help=(
        "Experiment name (task/name, e.g. 'heilbron/k5-budget-v3'). "
        "Reads experiments/<name>/experiment.yaml and auto-discovers runs."
    ),
)
@click.option(
    "-r",
    "--run",
    type=str,
    multiple=True,
    help=(
        "Run spec 'prefix@db[:label]' (e.g. 'adv_k5_1_G@1:K5_1_G'). "
        "Repeatable — pass multiple times to target several runs."
    ),
)
@click.option(
    "-f",
    "--format",
    "format_name",
    type=click.Choice(["table", "json", "csv", "markdown"], case_sensitive=False),
    default=None,
    help=(
        "Output format for tabular data. Auto-detects: table for TTY, "
        "json when piped. Subcommand-specific short flags (e.g. 'logs -f' "
        "for follow) do NOT collide — Click consumes global flags first."
    ),
)
@click.option(
    "-q", "--quiet", is_flag=True, default=False, help="Suppress non-error output."
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Emit debug-level progress messages.",
)
@click.option(
    "--redis-host",
    default="localhost",
    help="Redis server hostname (default: localhost).",
)
@click.option(
    "--redis-port", type=int, default=6379, help="Redis server port (default: 6379)."
)
@click.pass_context
def main(
    ctx: click.Context,
    experiment: str | None,
    run: tuple[str, ...],
    format_name: str | None,
    quiet: bool,
    verbose: bool,
    redis_host: str,
    redis_port: int,
) -> None:
    """GigaEvo CLI -- unified experiment monitoring and analysis.

    Run `gigaevo <subcommand> --help` for per-command documentation.
    Common workflows: `status` for run health, `trajectory`/`top` for
    fitness inspection, `plot` for visualisations, `logs` for tailing,
    `manifest` for experiment.yaml edits, `launch`/`watchdog` for
    lifecycle control, `flush` for teardown.

    \b
    Argument order
    --------------
    Global flags (-e, -r, -f, -q, -v, --redis-host, --redis-port) MUST
    appear BEFORE the subcommand. Subcommands may re-use short flags
    (e.g. `logs -f` for follow, `top -n 10` for top-N) — there's no
    collision because Click consumes global flags first.

    \b
    Examples
    --------
      gigaevo -e heilbron/k5-budget-v3 status
      gigaevo -e heilbron/k5-budget-v3 -f json trajectory --tail 20
      gigaevo -r adv_k5_1_G@1:K5_1_G -r adv_k5_1_D@2:K5_1_D status
      gigaevo -e heilbron/k5-budget-v3 top -n 5 --code
      gigaevo -e heilbron/k5-budget-v3 logs -f        # -f here = --follow
      gigaevo -e heilbron/k5-budget-v3 manifest get runs
      gigaevo flush --db 1 2 3 --confirm              # no -e/-r needed

    Target selection: pass `-e/--experiment` to auto-discover all runs
    from the manifest, OR `-r/--run` (repeatable) to target specific
    prefix@db pairs. Some commands (flush, inspect) ignore both.
    """
    ctx.ensure_object(dict)
    ctx.obj["formatter"] = OutputFormatter(format_name=format_name, quiet=quiet)
    ctx.obj["experiment"] = experiment
    ctx.obj["runs"] = run
    ctx.obj["redis_host"] = redis_host
    ctx.obj["redis_port"] = redis_port
    ctx.obj["verbose"] = verbose
