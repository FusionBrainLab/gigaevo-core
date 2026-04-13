"""Watchdog subcommand -- start the WatchdogEngine for an experiment."""

from __future__ import annotations

import click


@click.command("watchdog")
@click.option(
    "--poll-interval",
    type=int,
    default=3600,
    help="Seconds between monitoring cycles.",
)
@click.option(
    "--max-generations",
    type=int,
    default=None,
    help="Stop after this many generations.",
)
@click.option(
    "--max-restarts",
    type=int,
    default=3,
    help="Max restart attempts on failure.",
)
@click.option(
    "--plugin",
    "plugin_name",
    type=str,
    default=None,
    help="Force a specific plugin (solo, adversarial, heilbron, prompt_coevo).",
)
@click.pass_context
def watchdog(
    ctx: click.Context,
    poll_interval: int,
    max_generations: int | None,
    max_restarts: int,
    plugin_name: str | None,
) -> None:
    """Start or manage the experiment watchdog."""
    experiment = ctx.obj.get("experiment")
    if not experiment:
        click.echo("Error: Watchdog requires --experiment flag.", err=True)
        ctx.exit(1)
        return

    # Lazy imports to keep CLI startup fast
    from gigaevo.cli.run_resolver import _load_metric_names
    from gigaevo.monitoring.experiment_monitor import RunConfig
    from gigaevo.monitoring.run_spec import RunSpec
    from gigaevo.monitoring.watchdog_config import WatchdogConfig
    from gigaevo.monitoring.watchdog_engine import WatchdogEngine
    from gigaevo.monitoring.watchdog_plugin import get_registry, resolve_plugin
    from tools.experiment.manifest import load_manifest

    manifest = load_manifest(experiment)

    # Resolve plugin
    if plugin_name:
        registry = get_registry()
        if plugin_name not in registry:
            click.echo(
                f"Error: Plugin '{plugin_name}' not found. "
                f"Available: {sorted(registry.keys())}",
                err=True,
            )
            ctx.exit(1)
            return
        plugin = registry[plugin_name]()
    else:
        plugin_cls = resolve_plugin(manifest=manifest)
        plugin = plugin_cls()

    # Build RunConfigs from manifest
    run_configs = []
    for run in manifest.runs:
        spec = RunSpec(prefix=run.prefix, db=run.db, label=run.label)
        metric_names = _load_metric_names(run.problem_name)
        rc = RunConfig(run_spec=spec, metric_names=metric_names, pid=run.pid)
        run_configs.append(rc)

    # Build config
    config = WatchdogConfig(
        poll_interval_s=poll_interval,
        max_restarts=max_restarts,
    )

    click.echo(
        f"Starting watchdog for {experiment} "
        f"({len(run_configs)} runs, poll={poll_interval}s)"
    )

    engine = WatchdogEngine(
        experiment_name=experiment,
        plugin=plugin,
        run_configs=run_configs,
        config=config,
        max_generations=max_generations or manifest.max_generations,
    )
    engine.run()
