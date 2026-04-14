"""Watchdog subcommand -- start the WatchdogEngine for an experiment."""

from __future__ import annotations

import os

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
    from gigaevo.monitoring.experiment_monitor import RunConfig
    import gigaevo.monitoring.plugins  # noqa: F401 — triggers @register decorators
    from gigaevo.monitoring.run_spec import RunSpec
    from gigaevo.monitoring.watchdog_config import WatchdogConfig
    from gigaevo.monitoring.watchdog_engine import WatchdogEngine
    from gigaevo.monitoring.watchdog_plugin import get_registry, resolve_plugin
    from tools.experiment.manifest import load_manifest

    manifest = load_manifest(experiment)

    # Auto-configure NO_PROXY from manifest servers
    no_proxy = os.environ.get("NO_PROXY", "")
    extra_hosts = list(manifest.servers) + ["api.github.com"]
    watchdog_manifest = getattr(manifest, "watchdog", None)
    if watchdog_manifest and watchdog_manifest.no_proxy_hosts:
        extra_hosts.extend(watchdog_manifest.no_proxy_hosts)
    for host in extra_hosts:
        if host not in no_proxy:
            no_proxy = ",".join(filter(None, [no_proxy, host]))
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    click.echo(f"  NO_PROXY: {no_proxy}")

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
        rc = RunConfig(run_spec=spec, pid=run.pid)
        run_configs.append(rc)

    # Build config -- CLI flags take precedence over manifest
    effective_poll = (
        poll_interval
        if poll_interval != 3600
        else (watchdog_manifest.poll_interval_s if watchdog_manifest else 3600)
    )
    effective_restarts = (
        max_restarts if max_restarts != 3 else (5)  # default from WatchdogConfig
    )
    config = WatchdogConfig(
        poll_interval_s=effective_poll,
        max_restarts=effective_restarts,
        plot_retries=(watchdog_manifest.plot_retries if watchdog_manifest else 3),
        plot_retry_delay_s=(
            watchdog_manifest.plot_retry_delay_s if watchdog_manifest else 30
        ),
        checkpoint_milestones=(
            tuple(watchdog_manifest.checkpoint_milestones)
            if watchdog_manifest
            else (0.1, 0.2, 0.5, 1.0)
        ),
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
