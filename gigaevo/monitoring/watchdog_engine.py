"""WatchdogEngine -- generic monitoring loop with plugin dispatch.

Composes Phase 1 (ExperimentMonitor, AlertDetector), Phase 2
(NotificationDispatcher), and Phase 3 (WatchdogPlugin) into the
single run loop invoked via `gigaevo -e <exp> watchdog`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import resource
import signal
import time

from loguru import logger
import redis as redis_lib

from gigaevo.experiment.manifest import set_status
from gigaevo.monitoring.alerts import Alert, AlertDetector, AlertSeverity, AlertType
from gigaevo.monitoring.dispatcher import NotificationDispatcher
from gigaevo.monitoring.experiment_monitor import ExperimentMonitor, RunConfig
from gigaevo.monitoring.notifications import PlotAttachment, StatusUpdate
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_config import WatchdogConfig
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin

_log = logger.bind(component="watchdog_engine")


class WatchdogEngine:
    """Generic watchdog engine with plugin-based experiment handling.

    Usage:
        engine = WatchdogEngine(
            experiment_name="hover/test",
            plugin=SoloPlugin(),
            run_configs=[RunConfig(RunSpec.parse("prefix@4:A"))],
            config=WatchdogConfig(poll_interval_s=3600),
        )
        engine.run()
    """

    def __init__(
        self,
        experiment_name: str,
        plugin: WatchdogPlugin,
        run_configs: list[RunConfig],
        config: WatchdogConfig | None = None,
        max_generations: int | None = None,
        monitor: ExperimentMonitor | None = None,
        alert_detector: AlertDetector | None = None,
        dispatcher: NotificationDispatcher | None = None,
        heartbeat_redis: redis_lib.Redis | None = None,
        plot_dir: Path | None = None,
        baseline: float | None = None,
        excluded_events: list[str] | None = None,
    ):
        self.experiment_name = experiment_name
        self.plugin = plugin
        self.run_configs = list(run_configs)
        self.config = config or WatchdogConfig()
        self.max_generations = max_generations
        self._baseline = baseline
        self._monitor = monitor or ExperimentMonitor(
            redis_host=self.config.redis_host,
            redis_port=self.config.redis_port,
        )
        self._alert_detector = alert_detector or AlertDetector(
            excluded_events=excluded_events
        )
        self._dispatcher = dispatcher or NotificationDispatcher([])
        self._heartbeat_redis = heartbeat_redis
        self._plot_dir = plot_dir or Path(
            f"/tmp/watchdog_{experiment_name.replace('/', '_')}"
        )
        self._shutdown = False
        self._cycle_count = 0

        # Stagnation tracking: label -> list of frontier fitness values
        self._frontier_history: dict[str, list[float]] = {}

    def run(self) -> None:
        """Entry point: SIGTERM handler + outer restart loop."""
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        _log.info(
            f"Watchdog started: {self.experiment_name}, {len(self.run_configs)} runs"
        )

        for attempt in range(self.config.max_restarts):
            try:
                self._main_loop()
                return  # Clean exit (completion or SIGTERM)
            except Exception as exc:
                _log.error(
                    f"Watchdog crashed (attempt {attempt + 1}"
                    f"/{self.config.max_restarts}): {exc}"
                )
                if attempt < self.config.max_restarts - 1:
                    _log.info(f"Restarting in {self.config.restart_cooldown_s}s...")
                    time.sleep(self.config.restart_cooldown_s)

        # Max restarts exhausted -- send FINAL alert
        self._dispatch_final_alert()

    def _main_loop(self) -> None:
        """Run cycles until shutdown or completion."""
        self._plot_dir.mkdir(parents=True, exist_ok=True)

        while not self._shutdown:
            self._cycle_count += 1
            asyncio.run(self._cycle(self._cycle_count))

            if self._shutdown:
                break

            _log.info(f"Sleeping {self.config.poll_interval_s}s until next cycle")
            # Sleep in small increments to check _shutdown
            for _ in range(self.config.poll_interval_s):
                if self._shutdown:
                    break
                time.sleep(1)

    async def _cycle(self, cycle: int) -> None:
        """Single monitoring cycle."""
        _log.info(f"Cycle {cycle} starting")
        ts = datetime.now(tz=UTC)

        # 1. Heartbeat
        self._write_heartbeat()

        # 2. Collect snapshots
        snapshots = self._monitor.collect(self.run_configs)

        # 3. Alert detection
        alerts = list(self._alert_detector.check(snapshots))

        # 4. Stagnation detection (MON-06)
        stagnation_alerts = self._check_stagnation(snapshots)
        alerts.extend(stagnation_alerts)

        # 5. Generate plots (with retries per D-04)
        plots: list[PlotAttachment] = []
        for attempt in range(self.config.plot_retries):
            try:
                plots = self.plugin.generate_plots(snapshots, self._plot_dir, cycle)
                break  # Success
            except Exception as exc:
                _log.error(
                    f"Plot generation attempt {attempt + 1}"
                    f"/{self.config.plot_retries} failed: {exc}"
                )
                if attempt < self.config.plot_retries - 1:
                    _log.info(f"Retrying in {self.config.plot_retry_delay_s}s...")
                    time.sleep(self.config.plot_retry_delay_s)
            finally:
                self._close_matplotlib_figures()

        # 6. Format status
        try:
            self.plugin.format_status_body(
                snapshots, self.experiment_name, cycle, self.max_generations
            )
        except Exception as exc:
            _log.error(f"Status formatting failed: {exc}")

        # 6b. Format plugin-specific Telegram body
        telegram_body: str | None = None
        try:
            telegram_body = self.plugin.format_telegram_body(
                snapshots,
                self.experiment_name,
                cycle,
                self.max_generations,
                baseline=self._get_baseline(),
            )
        except Exception as exc:
            _log.error(f"Telegram body formatting failed: {exc}")

        # 7. Build StatusUpdate and dispatch
        update = StatusUpdate(
            experiment_name=self.experiment_name,
            snapshots=snapshots,
            alerts=alerts,
            plots=plots,
            max_generations=self.max_generations,
            timestamp=ts,
            telegram_body=telegram_body,
        )
        await self._dispatcher.dispatch(update)

        # 8. Redis checkpoint markers
        self._write_redis_checkpoint(snapshots, cycle)

        # 9. Completion detection
        if any(a.alert_type == AlertType.COMPLETION for a in alerts):
            self._write_completion(snapshots)
            try:
                set_status(self.experiment_name, "complete")
                _log.info("Manifest status transitioned to 'complete'")
            except Exception as exc:
                _log.error(f"Failed to set manifest status to complete: {exc}")
            self._shutdown = True

        # 10. Cleanup old plots
        self._cleanup_plots()

        # 11. Log memory
        self._log_memory()

        _log.info(
            f"Cycle {cycle} complete: {len(snapshots)} snapshots, {len(alerts)} alerts"
        )

    def _write_heartbeat(self) -> None:
        """Write heartbeat to Redis with TTL."""
        if self._heartbeat_redis is None:
            try:
                self._heartbeat_redis = redis_lib.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
            except Exception as exc:
                _log.error(f"Cannot connect to Redis for heartbeat: {exc}")
                return

        try:
            key = f"experiments:{self.experiment_name}:watchdog_heartbeat"
            self._heartbeat_redis.set(
                key,
                str(int(time.time())),
                ex=self.config.heartbeat_ttl_s,
            )
        except Exception as exc:
            _log.error(f"Heartbeat write failed: {exc}")

    def _check_stagnation(self, snapshots: list[RunSnapshot]) -> list[Alert]:
        """Track frontier fitness per run; alert if unchanged for stagnation_gens cycles."""
        alerts: list[Alert] = []
        for snap in snapshots:
            label = snap.run_spec.label
            fitness = snap.metrics.get("fitness")
            if fitness is None:
                continue

            if label not in self._frontier_history:
                self._frontier_history[label] = []
            self._frontier_history[label].append(fitness)

            history = self._frontier_history[label]
            n = self.config.stagnation_gens
            if len(history) >= n:
                window = history[-n:]
                if all(v == window[0] for v in window):
                    alerts.append(
                        Alert(
                            alert_type=AlertType.STALL,
                            severity=AlertSeverity.WARN,
                            run_label=label,
                            message=(
                                f"Run {label}: frontier fitness stagnant at {fitness} "
                                f"for {n} consecutive cycles."
                            ),
                            details={
                                "fitness": fitness,
                                "stagnation_cycles": n,
                                "generation": snap.generation,
                            },
                        )
                    )
        return alerts

    def _write_redis_checkpoint(self, snapshots: list[RunSnapshot], cycle: int) -> None:
        """Write Redis checkpoint markers at milestone percentages."""
        if self._heartbeat_redis is None or self.max_generations is None:
            return
        min_gen = min(
            (s.generation for s in snapshots if s.generation is not None),
            default=0,
        )
        milestones = [
            int(self.max_generations * p) for p in self.config.checkpoint_milestones
        ]
        for milestone in milestones:
            if min_gen >= milestone > 0:
                key = f"experiments:{self.experiment_name}:checkpoint:{milestone}"
                try:
                    if self._heartbeat_redis.exists(key):
                        continue
                    metrics = {
                        s.run_spec.label: {
                            "gen": s.generation,
                            "fitness": s.metrics.get("fitness"),
                        }
                        for s in snapshots
                    }
                    data = json.dumps(
                        {
                            "gen": milestone,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "metrics": metrics,
                        }
                    )
                    self._heartbeat_redis.set(key, data)
                except Exception as exc:
                    _log.error(
                        f"Checkpoint write failed for milestone {milestone}: {exc}"
                    )

    def _write_completion(self, snapshots: list[RunSnapshot]) -> None:
        """Write Redis completion marker when all runs finish."""
        if self._heartbeat_redis is None:
            return
        try:
            completion_data = json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "run_states": [
                        {
                            "label": s.run_spec.label,
                            "gen": s.generation,
                            "fitness": s.metrics.get("fitness"),
                        }
                        for s in snapshots
                    ],
                }
            )
            self._heartbeat_redis.set(
                f"experiments:{self.experiment_name}:completion",
                completion_data,
            )
        except Exception as exc:
            _log.error(f"Completion write failed: {exc}")

    def _get_baseline(self) -> float | None:
        """Return the SOTA baseline value for Telegram formatting.

        Uses the baseline passed at construction time.
        """
        return self._baseline

    def _cleanup_plots(self) -> None:
        """Remove oldest plot files if count exceeds max_plot_files."""
        if not self._plot_dir.exists():
            return
        files = sorted(self._plot_dir.glob("*.*"), key=lambda p: p.stat().st_mtime)
        excess = len(files) - self.config.max_plot_files
        if excess > 0:
            for f in files[:excess]:
                try:
                    f.unlink()
                except OSError as exc:
                    _log.warning(f"Cannot remove plot {f}: {exc}")

    def _log_memory(self) -> float:
        """Log current memory RSS in MB. Returns RSS in MB."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = usage.ru_maxrss / 1024  # Linux: KB -> MB
        _log.info(f"Memory RSS: {rss_mb:.1f} MB")
        return rss_mb

    def _close_matplotlib_figures(self) -> None:
        """Close all open matplotlib figures to prevent memory leaks."""
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:
            pass  # matplotlib not installed -- no figures to close

    def _dispatch_final_alert(self) -> None:
        """Send FINAL alert on max restarts -- best-effort to both channels."""
        _log.error(
            f"Max restarts ({self.config.max_restarts}) exhausted "
            f"for {self.experiment_name}"
        )
        alert = Alert(
            alert_type=AlertType.CRASH,
            severity=AlertSeverity.ERROR,
            run_label="watchdog",
            message=(
                f"WATCHDOG CRASHED -- max restarts ({self.config.max_restarts}) "
                f"reached for {self.experiment_name}. Manual intervention required."
            ),
        )
        update = StatusUpdate(
            experiment_name=self.experiment_name,
            alerts=[alert],
        )
        try:
            asyncio.run(self._dispatcher.dispatch(update))
        except Exception as exc:
            _log.error(f"Final alert dispatch failed: {exc}")

    def _sigterm_handler(self, signum, frame) -> None:
        """SIGTERM handler: set shutdown flag."""
        _log.info("SIGTERM received -- finishing current cycle then exiting")
        self._shutdown = True
