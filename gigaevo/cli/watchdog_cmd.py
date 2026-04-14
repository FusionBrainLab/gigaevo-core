"""Watchdog subcommand -- start the WatchdogEngine for an experiment."""

from __future__ import annotations

import os
from pathlib import Path
import re

import click


def _get_github_token() -> str | None:
    """Read GitHub token from gh CLI config (~/.config/gh/hosts.yml)."""
    try:
        token_file = Path.home() / ".config/gh/hosts.yml"
        text = token_file.read_text()
        m = re.search(r"oauth_token:\s*(\S+)", text)
        return m.group(1) if m else None
    except Exception:
        return None


@click.command("watchdog")
@click.option(
    "--poll-interval",
    type=int,
    default=None,
    help="Seconds between monitoring cycles (default: from manifest or 3600).",
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
    default=None,
    help="Max restart attempts on failure (default: from manifest or 5).",
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
    poll_interval: int | None,
    max_generations: int | None,
    max_restarts: int | None,
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
    from gigaevo.monitoring.manifest import load_manifest
    from gigaevo.monitoring.run_spec import RunSpec
    from gigaevo.monitoring.watchdog_config import WatchdogConfig
    from gigaevo.monitoring.watchdog_engine import WatchdogEngine
    from gigaevo.monitoring.watchdog_plugin import get_registry, resolve_plugin

    manifest = load_manifest(experiment)
    wd_cfg = manifest.watchdog

    # Auto-configure NO_PROXY from manifest servers
    no_proxy = os.environ.get("NO_PROXY", "")
    extra_hosts = list(manifest.servers) + ["api.github.com"]
    extra_hosts.extend(wd_cfg.no_proxy_hosts)
    for host in extra_hosts:
        if host and host not in no_proxy:
            no_proxy = ",".join(filter(None, [no_proxy, host]))
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy

    # Resolve plugin: CLI flag > manifest.watchdog.plugin > manifest.watchdog_plugin > solo
    effective_plugin_name = plugin_name or wd_cfg.plugin or manifest.watchdog_plugin

    # Build plugin kwargs from manifest watchdog section
    plugin_kwargs: dict = {}
    if wd_cfg.sentinel_value is not None:
        plugin_kwargs["sentinel_value"] = wd_cfg.sentinel_value
    if wd_cfg.plot_metrics:
        plugin_kwargs["plot_metrics"] = list(wd_cfg.plot_metrics)
    if wd_cfg.plot_commands:
        plugin_kwargs["plot_commands"] = list(wd_cfg.plot_commands)

    if effective_plugin_name:
        registry = get_registry()
        if effective_plugin_name not in registry:
            click.echo(
                f"Error: Plugin '{effective_plugin_name}' not found. "
                f"Available: {sorted(registry.keys())}",
                err=True,
            )
            ctx.exit(1)
            return
        plugin = registry[effective_plugin_name](**plugin_kwargs)
    else:
        plugin_cls = resolve_plugin(manifest=manifest)
        plugin = plugin_cls(**plugin_kwargs)

    # Build RunConfigs from manifest
    run_configs = []
    for run in manifest.runs:
        spec = RunSpec(prefix=run.prefix, db=run.db, label=run.label)
        rc = RunConfig(run_spec=spec, pid=run.pid)
        run_configs.append(rc)

    # Build config: CLI flags take precedence over manifest watchdog section
    effective_poll = poll_interval or wd_cfg.poll_interval_s
    effective_restarts = max_restarts if max_restarts is not None else 5
    config = WatchdogConfig(
        poll_interval_s=effective_poll,
        max_restarts=effective_restarts,
        plot_retries=wd_cfg.plot_retries,
        plot_retry_delay_s=wd_cfg.plot_retry_delay_s,
        rolling_comment_threshold_hours=wd_cfg.rolling_comment_threshold_hours,
        checkpoint_milestones=tuple(wd_cfg.checkpoint_milestones),
    )

    # Build notification channels
    from gigaevo.monitoring.dispatcher import NotificationDispatcher
    from gigaevo.monitoring.notifications import NotificationChannel

    channels: list[NotificationChannel] = []

    # GitHub PR channel: requires token + PR number
    gh_token = _get_github_token()
    pr_number_val = manifest.experiment.pr_number
    branch = manifest.experiment.branch

    if gh_token and pr_number_val:
        from gigaevo.monitoring.github_pr_channel import GitHubPRChannel

        rolling_redis = None
        try:
            import redis as redis_lib

            rolling_redis = redis_lib.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
            )
        except Exception as exc:
            click.echo(f"  Rolling comment Redis: failed ({exc})", err=True)

        channels.append(
            GitHubPRChannel(
                repo="KhrulkovV/gigaevo-core-internal",
                pr_number=pr_number_val,
                token=gh_token,
                branch=branch,
                experiment_name=experiment,
                rolling_comment_redis=rolling_redis,
                rolling_comment_threshold_hours=config.rolling_comment_threshold_hours,
            )
        )
        click.echo(
            f"  GitHub PR channel: enabled "
            f"(PR #{pr_number_val}, rolling after {config.rolling_comment_threshold_hours}h)"
        )
    else:
        click.echo("  GitHub PR channel: disabled (no token or PR number)")

    dispatcher = NotificationDispatcher(channels)

    # Baseline for plugin SOTA comparison
    baseline = None
    if manifest.baseline.mean is not None:
        baseline = manifest.baseline.mean

    click.echo(
        f"Starting watchdog for {experiment} "
        f"({len(run_configs)} runs, poll={effective_poll}s)"
    )
    click.echo(f"  NO_PROXY: {no_proxy}")

    engine = WatchdogEngine(
        experiment_name=experiment,
        plugin=plugin,
        run_configs=run_configs,
        config=config,
        max_generations=max_generations or manifest.experiment.max_generations,
        dispatcher=dispatcher,
        baseline=baseline,
    )
    engine.run()
