"""Integration test: WatchdogEngine + mock plugin + mock channels + fakeredis.

Exercises the full Phase 3 stack in one test to prove composition works.
This is NOT a unit test -- it intentionally couples components.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import fakeredis

from gigaevo.monitoring.alerts import Alert, AlertDetector, AlertType
from gigaevo.monitoring.dispatcher import NotificationDispatcher
from gigaevo.monitoring.experiment_monitor import ExperimentMonitor, RunConfig
from gigaevo.monitoring.notifications import (
    NotificationChannel,
    StatusUpdate,
)
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_config import WatchdogConfig
from gigaevo.monitoring.watchdog_engine import WatchdogEngine
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, get_registry

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_snapshot(label="A", gen=5, fitness=0.5):
    return RunSnapshot(
        run_spec=RunSpec(prefix="test/run", db=1, label=label),
        generation=gen,
        metrics={"fitness": fitness},
        total_programs=100,
        valid_programs=90,
        pid=12345,
        pid_alive=True,
    )


class StubPlugin(WatchdogPlugin):
    """Minimal concrete plugin for integration testing."""

    def __init__(self):
        self.generate_plots_calls: list[tuple] = []
        self.format_status_body_calls: list[tuple] = []

    def generate_plots(self, snapshots, output_dir, cycle):
        self.generate_plots_calls.append((snapshots, output_dir, cycle))
        return []

    def format_status_body(self, snapshots, experiment_name, cycle, max_generations):
        self.format_status_body_calls.append(
            (snapshots, experiment_name, cycle, max_generations)
        )
        return f"## Status cycle {cycle}\nAll {len(snapshots)} runs OK."


class StubChannel(NotificationChannel):
    """Records all notifications for assertion."""

    def __init__(self):
        self.status_updates: list[StatusUpdate] = []
        self.alerts_received: list[Alert] = []

    async def send_status(self, update):
        self.status_updates.append(update)
        return True

    async def send_alert(self, alert):
        self.alerts_received.append(alert)
        return True

    async def check_health(self):
        return True


# ── Integration Tests ────────────────────────────────────────────────────────


class TestFullCycleIntegration:
    """Engine executes one full cycle with all components wired together."""

    def test_single_cycle_end_to_end(self, tmp_path):
        """One cycle: heartbeat -> collect -> alert -> plot -> format -> dispatch."""
        server = fakeredis.FakeServer()
        heartbeat_redis = fakeredis.FakeRedis(
            server=server, db=0, decode_responses=True
        )

        monitor = MagicMock(spec=ExperimentMonitor)
        snap = _make_snapshot("A", gen=5, fitness=0.65)
        monitor.collect.return_value = [snap]

        detector = AlertDetector(max_generations=50)
        plugin = StubPlugin()
        channel = StubChannel()
        dispatcher = NotificationDispatcher([channel])

        config = WatchdogConfig(
            poll_interval_s=60,
            max_restarts=1,
            stagnation_gens=10,
        )

        engine = WatchdogEngine(
            experiment_name="test/integration",
            plugin=plugin,
            run_configs=[RunConfig(RunSpec(prefix="test/run", db=1, label="A"))],
            config=config,
            max_generations=50,
            monitor=monitor,
            alert_detector=detector,
            dispatcher=dispatcher,
            heartbeat_redis=heartbeat_redis,
            plot_dir=tmp_path / "plots",
        )

        asyncio.run(engine._cycle(cycle=1))

        # Verify full chain executed
        monitor.collect.assert_called_once()
        assert len(plugin.generate_plots_calls) == 1
        assert len(plugin.format_status_body_calls) == 1
        assert len(channel.status_updates) == 1

        # Verify heartbeat was written
        hb_key = "experiments:test/integration:watchdog_heartbeat"
        assert heartbeat_redis.exists(hb_key)

        # Verify StatusUpdate contents
        update = channel.status_updates[0]
        assert update.experiment_name == "test/integration"
        assert len(update.snapshots) == 1
        assert update.snapshots[0].run_spec.label == "A"
        assert update.max_generations == 50


class TestMultiCycleStagnation:
    """Stagnation detection across multiple cycles."""

    def test_stagnation_alert_after_n_cycles(self, tmp_path):
        """After stagnation_gens cycles with same fitness, alert appears."""
        monitor = MagicMock(spec=ExperimentMonitor)
        snap = _make_snapshot("A", gen=10, fitness=0.5)
        monitor.collect.return_value = [snap]

        detector = AlertDetector(max_generations=50)
        plugin = StubPlugin()
        channel = StubChannel()
        dispatcher = NotificationDispatcher([channel])

        config = WatchdogConfig(
            poll_interval_s=1,
            stagnation_gens=3,
        )

        server = fakeredis.FakeServer()
        heartbeat_redis = fakeredis.FakeRedis(
            server=server, db=0, decode_responses=True
        )

        engine = WatchdogEngine(
            experiment_name="test/stagnation",
            plugin=plugin,
            run_configs=[RunConfig(RunSpec(prefix="test/run", db=1, label="A"))],
            config=config,
            max_generations=50,
            monitor=monitor,
            alert_detector=detector,
            dispatcher=dispatcher,
            heartbeat_redis=heartbeat_redis,
            plot_dir=tmp_path / "plots",
        )

        for i in range(4):
            asyncio.run(engine._cycle(cycle=i + 1))

        all_alerts = []
        for update in channel.status_updates:
            all_alerts.extend(update.alerts)

        stagnation_alerts = [
            a
            for a in all_alerts
            if "stagnant" in a.message.lower() or "stagnation" in a.message.lower()
        ]
        assert len(stagnation_alerts) >= 1


class TestPluginRegistryCompleteness:
    """All 4 plugins are registered and importable."""

    def test_all_plugins_registered(self):
        import gigaevo.monitoring.plugins  # noqa: F401

        registry = get_registry()
        assert "solo" in registry
        assert "adversarial" in registry
        assert "prompt_coevo" in registry

    def test_each_plugin_is_watchdog_plugin_subclass(self):
        import gigaevo.monitoring.plugins  # noqa: F401

        registry = get_registry()
        for name, cls in registry.items():
            assert issubclass(cls, WatchdogPlugin), (
                f"{name} is not a WatchdogPlugin subclass"
            )


class TestAlertFlowIntegration:
    """Alerts from detector flow through to channel."""

    def test_crash_alert_reaches_channel(self, tmp_path):
        """A dead PID triggers a CRASH alert that reaches the channel."""
        dead_snap = RunSnapshot(
            run_spec=RunSpec(prefix="test", db=1, label="A"),
            generation=5,
            metrics={"fitness": 0.5},
            total_programs=100,
            valid_programs=90,
            pid=99999,
            pid_alive=False,
        )

        monitor = MagicMock(spec=ExperimentMonitor)
        monitor.collect.return_value = [dead_snap]

        detector = AlertDetector(max_generations=50)
        plugin = StubPlugin()
        channel = StubChannel()
        dispatcher = NotificationDispatcher([channel])

        server = fakeredis.FakeServer()
        heartbeat_redis = fakeredis.FakeRedis(
            server=server, db=0, decode_responses=True
        )

        engine = WatchdogEngine(
            experiment_name="test/crash",
            plugin=plugin,
            run_configs=[RunConfig(RunSpec(prefix="test", db=1, label="A"), pid=99999)],
            config=WatchdogConfig(stagnation_gens=100),
            max_generations=50,
            monitor=monitor,
            alert_detector=detector,
            dispatcher=dispatcher,
            heartbeat_redis=heartbeat_redis,
            plot_dir=tmp_path / "plots",
        )

        asyncio.run(engine._cycle(cycle=1))

        assert len(channel.status_updates) == 1
        update = channel.status_updates[0]
        crash_alerts = [a for a in update.alerts if a.alert_type == AlertType.CRASH]
        assert len(crash_alerts) == 1
        assert (
            "not alive" in crash_alerts[0].message.lower()
            or "PID" in crash_alerts[0].message
        )
