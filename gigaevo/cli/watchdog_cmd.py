"""Watchdog subcommand -- start the WatchdogEngine for an experiment."""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv


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

    # Load .env so Telegram credentials and HTTPS_PROXY are available without
    # the caller (launch.sh, subprocess.Popen) having to pre-source the shell env.
    # override=False keeps any pre-existing shell vars authoritative.
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        click.echo(f"  .env: loaded from {Path(dotenv_path)}")
    else:
        click.echo("  .env: not found (skipping)")

    # Lazy imports to keep CLI startup fast
    from gigaevo.experiment.manifest import load_manifest
    from gigaevo.monitoring.dispatcher import NotificationDispatcher
    from gigaevo.monitoring.experiment_monitor import RunConfig
    from gigaevo.monitoring.notifications import NotificationChannel
    import gigaevo.monitoring.plugins  # noqa: F401 — triggers @register decorators
    from gigaevo.monitoring.run_spec import RunSpec
    from gigaevo.monitoring.telegram_channel import TelegramChannel
    from gigaevo.monitoring.watchdog_config import WatchdogConfig
    from gigaevo.monitoring.watchdog_engine import WatchdogEngine
    from gigaevo.monitoring.watchdog_plugin import get_registry, resolve_plugin

    manifest = load_manifest(experiment)
    watchdog_manifest = manifest.control_plane.watchdog

    # Proxy (HTTPS_PROXY / NO_PROXY) comes from the user's shell / .env —
    # load_dotenv above already populated it. The watchdog no longer builds
    # its own NO_PROXY from manifest fields.
    click.echo(f"  NO_PROXY: {os.environ.get('NO_PROXY', '(unset)')}")

    # Resolve plugin class
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
        plugin_cls = registry[plugin_name]
    else:
        plugin_cls = resolve_plugin(manifest=manifest)

    # Wire plugin from manifest.control_plane.watchdog section — plot_metrics,
    # plot_commands, and sentinel_value all come from the yaml (see WatchdogSection).
    plugin_kwargs: dict = {
        "plot_metrics": list(watchdog_manifest.plot_metrics),
        "plot_commands": list(watchdog_manifest.plot_commands),
        "sentinel_value": watchdog_manifest.sentinel_value,
    }
    plugin = plugin_cls(**plugin_kwargs)
    click.echo(
        f"  Plugin: {plugin_cls.__name__} "
        f"metrics={plugin_kwargs['plot_metrics']} "
        f"sentinel={plugin_kwargs['sentinel_value']} "
        f"commands={len(plugin_kwargs['plot_commands'])}"
    )

    # Build RunConfigs from manifest
    run_configs = []
    metric_names = list(watchdog_manifest.plot_metrics) or ["fitness"]
    for run in manifest.contract.runs:
        spec = RunSpec(prefix=run.prefix, db=run.db, label=run.label, role=run.role)
        rc = RunConfig(run_spec=spec, metric_names=metric_names, pid=run.pid)
        run_configs.append(rc)

    # Build config -- CLI flags take precedence over manifest
    effective_poll = (
        poll_interval if poll_interval != 3600 else watchdog_manifest.poll_interval_s
    )
    effective_restarts = (
        max_restarts if max_restarts != 3 else (5)  # default from WatchdogConfig
    )
    config = WatchdogConfig(
        poll_interval_s=effective_poll,
        max_restarts=effective_restarts,
        plot_retries=watchdog_manifest.plot_retries,
        plot_retry_delay_s=watchdog_manifest.plot_retry_delay_s,
        checkpoint_milestones=tuple(watchdog_manifest.checkpoint_milestones),
    )

    # Build notification channels. Each channel's from_env() returns None if its
    # required env vars are missing -- caller (shell/launch.sh) is responsible for
    # having the environment loaded (e.g. sourcing .env) before invoking watchdog.
    channels: list[NotificationChannel] = [
        ch for ch in (TelegramChannel.from_env(),) if ch is not None
    ]
    click.echo(
        f"  Telegram: {'enabled' if any(isinstance(c, TelegramChannel) for c in channels) else 'disabled'}"
    )
    dispatcher = NotificationDispatcher(channels)

    click.echo(
        f"Starting watchdog for {experiment} "
        f"({len(run_configs)} runs, poll={poll_interval}s)"
    )

    engine = WatchdogEngine(
        experiment_name=experiment,
        plugin=plugin,
        run_configs=run_configs,
        config=config,
        max_generations=max_generations or manifest.contract.max_generations,
        dispatcher=dispatcher,
    )
    engine.run()
