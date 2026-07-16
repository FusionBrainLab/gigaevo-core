"""GigaEvo CLI -- unified experiment monitoring and analysis."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import sys
from typing import Any

import click

from gigaevo.cli.output_formatter import OutputFormatter


@dataclass(frozen=True)
class LazyCommandSpec:
    module: str
    attribute: str
    help: str


# Click's default Group.format_commands imports every command for its short
# help. Keeping help in this registry lets root help remain genuinely lazy.
_LAZY_SUBCOMMANDS: dict[str, LazyCommandSpec] = {
    "status": LazyCommandSpec(
        "gigaevo.cli.status", "status", "Show live Redis run status and health."
    ),
    "trajectory": LazyCommandSpec(
        "gigaevo.cli.trajectory",
        "trajectory",
        "Show Redis metric frontier and mean histories.",
    ),
    "top": LazyCommandSpec(
        "gigaevo.cli.top", "top", "Inspect top programs from Redis or disk storage."
    ),
    "logs": LazyCommandSpec(
        "gigaevo.cli.logs", "logs", "List or tail experiment log files."
    ),
    "plot": LazyCommandSpec(
        "gigaevo.cli.plot_group",
        "plot",
        "Plot runs from Redis, disk storage, or exported CSV.",
    ),
    "export": LazyCommandSpec(
        "gigaevo.cli.export",
        "export",
        "Export Redis or disk program data to CSV.",
    ),
    "flush": LazyCommandSpec(
        "gigaevo.cli.flush", "flush", "Stop run processes and flush Redis databases."
    ),
    "watchdog": LazyCommandSpec(
        "gigaevo.cli.watchdog_cmd",
        "watchdog",
        "Run experiment monitoring and notifications.",
    ),
    "checkpoint": LazyCommandSpec(
        "gigaevo.cli.checkpoint",
        "checkpoint",
        "Collect a one-shot Redis status summary.",
    ),
    "manifest": LazyCommandSpec(
        "gigaevo.cli.manifest_cmd",
        "manifest",
        "Read and update experiment manifests.",
    ),
    "inspect": LazyCommandSpec(
        "gigaevo.cli.inspect_cmd", "inspect", "Discover GigaEvo prefixes in Redis."
    ),
    "launch": LazyCommandSpec(
        "gigaevo.cli.launch_cmd", "launch", "Validate and launch an experiment."
    ),
    "events": LazyCommandSpec(
        "gigaevo.cli.events_cmd", "events", "Audit and plot canonical run events."
    ),
    "profiler": LazyCommandSpec(
        "gigaevo.cli.profiler_cmd",
        "profiler",
        "Build profiling reports from run logs.",
    ),
    "metrics": LazyCommandSpec(
        "gigaevo.cli.metrics_cmd", "metrics", "Dump raw Redis metric histories."
    ),
    "memory": LazyCommandSpec(
        "gigaevo.cli.memory_cmd", "memory", "Analyze and calibrate memory ledgers."
    ),
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
            spec = _LAZY_SUBCOMMANDS[cmd_name]
            mod = importlib.import_module(spec.module)
            return getattr(mod, spec.attribute)
        return None

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        """Render root command help without importing lazy command modules."""
        commands: list[tuple[str, click.Command | None, str]] = []
        for name in self.list_commands(ctx):
            eager = super().get_command(ctx, name)
            if eager is not None:
                if not eager.hidden:
                    commands.append((name, eager, ""))
            elif name in _LAZY_SUBCOMMANDS:
                commands.append((name, None, _LAZY_SUBCOMMANDS[name].help))

        if not commands:
            return
        limit = formatter.width - 6 - max(len(name) for name, _, _ in commands)
        rows = [
            (
                name,
                static_help if command is None else command.get_short_help_str(limit),
            )
            for name, command, static_help in commands
        ]
        with formatter.section("Commands"):
            formatter.write_dl(rows)

    def invoke(self, ctx: click.Context) -> Any:
        try:
            result = super().invoke(ctx)
            # Force a late EPIPE (small output lingers in the block buffer until
            # exit) to surface here, not at the interpreter's shutdown flush.
            sys.stdout.flush()
            return result
        except BrokenPipeError:
            # Downstream reader (e.g. `gigaevo top | head`) closed the pipe:
            # redirect stdout to devnull so the shutdown flush can't re-raise,
            # then exit quietly like a Unix filter instead of dumping a traceback.
            try:
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            except (OSError, ValueError):
                pass
            raise SystemExit(0) from None
        except FileNotFoundError as exc:
            # Convert missing-manifest errors into a clean ClickException (exit 1,
            # no traceback) instead of an uncaught FileNotFoundError with exit 0.
            msg = str(exc)
            if "experiment.yaml" in msg:
                raise click.ClickException(msg) from exc
            raise


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
        "Run spec 'prefix@db[:label]' (e.g. 'adv_k5_1_G@1:K5_1_G') or a "
        "disk storage path 'outputs/run/storage[:label]' for storage=disk "
        "runs. Repeatable — pass multiple times to target several runs."
    ),
)
@click.option(
    "-f",
    "--format",
    "format_name",
    type=click.Choice(["table", "json", "csv", "markdown"], case_sensitive=False),
    default=None,
    help=(
        "Output format for tabular data (table|json|csv|markdown). "
        "Auto-detects: table for TTY, json when piped. Must appear "
        "BEFORE the subcommand — Click consumes global flags first, "
        "so subcommand short flags (e.g. `logs -f` for --follow) do "
        "not collide."
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
    appear BEFORE the subcommand. Click consumes them first, so
    subcommands can re-use the same short letters without collision:

    \b
       Global                       Subcommand-local (after name)
       ---------------------        -----------------------------
       -e/--experiment              --
       -r/--run                     --
       -f/--format                  logs -f  = --follow
                                    top/trajectory/manifest get -f = format override
       -q/--quiet                   --
       -v/--verbose                 --

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

    Target selection: pass `-e/--experiment` to load Redis runs from the
    manifest, OR repeat `-r/--run` to target Redis and/or disk runs.
    Labels must be unique when several runs are selected.

    \b
    Storage support
    ---------------
      Redis + disk   top, export, plot
      Redis only     status, trajectory, metrics, checkpoint
      Files/logs     logs, profiler, events
      No run target  flush, inspect

    Redis specs: prefix@db[:label], db[:label], or @db[:label].
    Disk specs: /path/to/storage[:label], ./relative/storage[:label],
    or any slash-containing relative path without '@'.
    """
    ctx.ensure_object(dict)
    ctx.obj["formatter"] = OutputFormatter(format_name=format_name, quiet=quiet)
    ctx.obj["experiment"] = experiment
    ctx.obj["runs"] = run
    ctx.obj["redis_host"] = redis_host
    ctx.obj["redis_port"] = redis_port
    ctx.obj["verbose"] = verbose
