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
    "launch": ("gigaevo.cli.lifecycle", "launch"),
    "closeout": ("gigaevo.cli.lifecycle", "closeout"),
    "restart": ("gigaevo.cli.lifecycle", "restart"),
}


class LazyGroup(click.Group):
    """Click group that imports subcommand modules only when invoked."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        rv = list(_LAZY_SUBCOMMANDS.keys())
        rv.extend(super().list_commands(ctx))
        return sorted(set(rv))

    def get_command(
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
    help="Experiment name (task/name). Reads experiment.yaml for run discovery.",
)
@click.option(
    "-r",
    "--run",
    type=str,
    multiple=True,
    help="Run spec prefix@db[:label]. Repeatable.",
)
@click.option(
    "-f",
    "--format",
    "format_name",
    type=click.Choice(["table", "json", "csv", "markdown"], case_sensitive=False),
    default=None,
    help="Output format (auto-detects: table for terminal, json for pipe).",
)
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress output.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output.")
@click.option("--redis-host", default="localhost", help="Redis server hostname.")
@click.option("--redis-port", type=int, default=6379, help="Redis server port.")
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
    """GigaEvo CLI -- unified experiment monitoring and analysis."""
    ctx.ensure_object(dict)
    ctx.obj["formatter"] = OutputFormatter(format_name=format_name, quiet=quiet)
    ctx.obj["experiment"] = experiment
    ctx.obj["runs"] = run
    ctx.obj["redis_host"] = redis_host
    ctx.obj["redis_port"] = redis_port
    ctx.obj["verbose"] = verbose
