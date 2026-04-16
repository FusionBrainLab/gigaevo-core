"""Integration test: WatchdogEngine + mock plugin + mock channels + fakeredis.

Exercises the full Phase 3 stack in one test to prove composition works.
This is NOT a unit test -- it intentionally couples components.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import fakeredis
import yaml

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

        detector = AlertDetector()
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

        detector = AlertDetector()
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


# ── Fixture-driven tests ────────────────────────────────────────────────────

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "watchdog"


def _load_fixture(name: str):
    """Load a watchdog test fixture by name."""
    fixture_path = FIXTURE_DIR / name
    with open(fixture_path / "experiment.yaml") as f:
        manifest_data = yaml.safe_load(f)
    with open(fixture_path / "redis_data.json") as f:
        redis_data = json.load(f)
    return manifest_data, redis_data


def _snapshots_from_fixture(redis_data: dict) -> list[RunSnapshot]:
    """Create RunSnapshot objects from fixture redis_data.json."""
    snapshots = []
    for run in redis_data["runs"]:
        # Infer role from label (G/D suffix) for adversarial runs
        label = run["label"]
        role = None
        if label.endswith("_G"):
            role = "constructor"
        elif label.endswith("_D"):
            role = "improver"
        snap = RunSnapshot(
            run_spec=RunSpec(
                prefix=run["prefix"], db=run["db"], label=label, role=role
            ),
            generation=run.get("generation"),
            metrics={"fitness": run.get("fitness")},
            total_programs=run.get("total_programs"),
            valid_programs=run.get("valid_programs"),
            running_programs=run.get("running_programs"),
            pid=run.get("pid"),
            pid_alive=run.get("pid_alive"),
        )
        snapshots.append(snap)
    return snapshots


def _mock_subprocess_success(cmd, **kwargs):
    """Side effect that creates fake output PNGs for any plot command."""
    out_dir = None
    for i, arg in enumerate(cmd):
        if arg == "-o" and i + 1 < len(cmd):
            out_dir = Path(cmd[i + 1])
            break
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        if "arms-race" in cmd:
            (out_dir / "arms_race.png").write_bytes(b"fake-png")
        elif "comparison" in cmd:
            (out_dir / "evolution_runs_comparison.png").write_bytes(b"fake-png")
    return subprocess.CompletedProcess(cmd, 0, b"", b"")


class TestFixtureSoloFullCycle:
    """Full watchdog cycle with solo plugin using fixture data."""

    def test_solo_cycle_collect_plot_format_dispatch(self, tmp_path):
        manifest_data, redis_data = _load_fixture("solo_hover")
        snapshots = _snapshots_from_fixture(redis_data)

        from gigaevo.monitoring.plugins.solo import SoloPlugin

        plugin = SoloPlugin()
        channel = StubChannel()
        dispatcher = NotificationDispatcher([channel])
        monitor = MagicMock(spec=ExperimentMonitor)
        monitor.collect.return_value = snapshots

        server = fakeredis.FakeServer()
        heartbeat_redis = fakeredis.FakeRedis(
            server=server, db=0, decode_responses=True
        )

        engine = WatchdogEngine(
            experiment_name=manifest_data["experiment"]["name"],
            plugin=plugin,
            run_configs=[
                RunConfig(RunSpec(prefix=r["prefix"], db=r["db"], label=r["label"]))
                for r in redis_data["runs"]
            ],
            config=WatchdogConfig(poll_interval_s=60, stagnation_gens=100),
            max_generations=manifest_data["experiment"]["max_generations"],
            monitor=monitor,
            dispatcher=dispatcher,
            heartbeat_redis=heartbeat_redis,
            plot_dir=tmp_path / "plots",
            baseline=manifest_data.get("baseline", {}).get("mean"),
        )

        with patch(
            "gigaevo.monitoring.plugins.solo.subprocess.run",
            side_effect=_mock_subprocess_success,
        ):
            asyncio.run(engine._cycle(cycle=1))

        assert len(channel.status_updates) == 1
        update = channel.status_updates[0]
        assert update.experiment_name == "hover/solo-test"
        assert len(update.snapshots) == 2

    def test_solo_format_telegram_body(self):
        _, redis_data = _load_fixture("solo_hover")
        snapshots = _snapshots_from_fixture(redis_data)

        from gigaevo.monitoring.plugins.solo import SoloPlugin

        plugin = SoloPlugin()
        body = plugin.format_telegram_body(
            snapshots, "hover/solo-test", cycle=1, max_generations=25, baseline=0.76
        )
        assert body is not None
        assert "hover/solo-test" in body
        assert "SOTA" in body


class TestFixtureAdversarialFullCycle:
    """Full watchdog cycle with adversarial plugin using fixture data."""

    def test_adversarial_cycle_generates_two_plot_types(self, tmp_path):
        manifest_data, redis_data = _load_fixture("adversarial_heilbron")
        snapshots = _snapshots_from_fixture(redis_data)

        from gigaevo.monitoring.plugins.adversarial import AdversarialPlugin

        plugin = AdversarialPlugin()
        channel = StubChannel()
        dispatcher = NotificationDispatcher([channel])
        monitor = MagicMock(spec=ExperimentMonitor)
        monitor.collect.return_value = snapshots

        server = fakeredis.FakeServer()
        heartbeat_redis = fakeredis.FakeRedis(
            server=server, db=0, decode_responses=True
        )

        engine = WatchdogEngine(
            experiment_name="heilbron/adversarial-test",
            plugin=plugin,
            run_configs=[
                RunConfig(RunSpec(prefix=r["prefix"], db=r["db"], label=r["label"]))
                for r in redis_data["runs"]
            ],
            config=WatchdogConfig(poll_interval_s=60, stagnation_gens=100),
            max_generations=50,
            monitor=monitor,
            dispatcher=dispatcher,
            heartbeat_redis=heartbeat_redis,
            plot_dir=tmp_path / "plots",
            baseline=0.03449,
        )

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            side_effect=_mock_subprocess_success,
        ):
            asyncio.run(engine._cycle(cycle=1))

        assert len(channel.status_updates) == 1
        update = channel.status_updates[0]
        assert len(update.snapshots) == 4

    def test_adversarial_telegram_body_has_gd_sections(self):
        _, redis_data = _load_fixture("adversarial_heilbron")
        snapshots = _snapshots_from_fixture(redis_data)

        from gigaevo.monitoring.plugins.adversarial import AdversarialPlugin

        plugin = AdversarialPlugin()
        body = plugin.format_telegram_body(
            snapshots,
            "heilbron/adversarial-test",
            cycle=1,
            max_generations=50,
            baseline=0.03449,
        )
        assert body is not None
        assert "Constructor (G)" in body
        assert "Improver (D)" in body
        assert "SOTA" in body

    def test_adversarial_arms_race_command_args(self, tmp_path):
        _, redis_data = _load_fixture("adversarial_heilbron")
        snapshots = _snapshots_from_fixture(redis_data)

        from gigaevo.monitoring.plugins.adversarial import AdversarialPlugin

        plugin = AdversarialPlugin()

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as mock_run:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)

        arms_race_calls = [c for c in mock_run.call_args_list if "arms-race" in c[0][0]]
        assert len(arms_race_calls) == 1
        cmd = arms_race_calls[0][0][0]
        assert "--paired" in cmd


class TestFixturePromptCoevoFullCycle:
    """Full watchdog cycle with prompt co-evo plugin using fixture data."""

    def test_prompt_coevo_cycle_groups_populations(self, tmp_path):
        manifest_data, redis_data = _load_fixture("prompt_coevo")
        snapshots = _snapshots_from_fixture(redis_data)

        from gigaevo.monitoring.plugins.prompt_coevo import PromptCoevoPlugin

        plugin = PromptCoevoPlugin()
        channel = StubChannel()
        dispatcher = NotificationDispatcher([channel])
        monitor = MagicMock(spec=ExperimentMonitor)
        monitor.collect.return_value = snapshots

        server = fakeredis.FakeServer()
        heartbeat_redis = fakeredis.FakeRedis(
            server=server, db=0, decode_responses=True
        )

        engine = WatchdogEngine(
            experiment_name="hover/prompt-coevo-test",
            plugin=plugin,
            run_configs=[
                RunConfig(RunSpec(prefix=r["prefix"], db=r["db"], label=r["label"]))
                for r in redis_data["runs"]
            ],
            config=WatchdogConfig(poll_interval_s=60, stagnation_gens=100),
            max_generations=25,
            monitor=monitor,
            dispatcher=dispatcher,
            heartbeat_redis=heartbeat_redis,
            plot_dir=tmp_path / "plots",
            baseline=0.80,
        )

        with patch(
            "gigaevo.monitoring.plugins.prompt_coevo.subprocess.run",
            side_effect=_mock_subprocess_success,
        ):
            asyncio.run(engine._cycle(cycle=1))

        assert len(channel.status_updates) == 1
        update = channel.status_updates[0]
        assert len(update.snapshots) == 4

    def test_prompt_coevo_telegram_has_population_groups(self):
        _, redis_data = _load_fixture("prompt_coevo")
        snapshots = _snapshots_from_fixture(redis_data)

        from gigaevo.monitoring.plugins.prompt_coevo import PromptCoevoPlugin

        plugin = PromptCoevoPlugin()
        body = plugin.format_telegram_body(
            snapshots,
            "hover/prompt-coevo-test",
            cycle=1,
            max_generations=25,
            baseline=0.80,
        )
        assert body is not None
        assert "Code Population" in body
        assert "Prompt Population" in body
        assert "SOTA" in body


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

        detector = AlertDetector()
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
