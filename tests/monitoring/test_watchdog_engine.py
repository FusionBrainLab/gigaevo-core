"""Tests for WatchdogEngine core loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import fakeredis

from gigaevo.monitoring.alerts import Alert, AlertSeverity, AlertType
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_config import WatchdogConfig
from gigaevo.monitoring.watchdog_engine import WatchdogEngine


def _make_snapshot(label: str = "A", gen: int = 5, fitness: float = 0.5) -> RunSnapshot:
    return RunSnapshot(
        run_spec=RunSpec(prefix="test", db=1, label=label),
        generation=gen,
        metrics={"fitness": fitness},
        total_programs=100,
        valid_programs=90,
        pid=12345,
        pid_alive=True,
    )


def _make_plugin():
    """Create a mock WatchdogPlugin."""
    plugin = MagicMock()
    plugin.generate_plots.return_value = []
    plugin.format_status_body.return_value = "## Status\nAll good."
    plugin.extra_telegram_content.return_value = None
    plugin.extra_redis_queries.return_value = {}
    return plugin


def _make_engine(**overrides):
    """Create a WatchdogEngine with mocked dependencies."""
    monitor = MagicMock()
    monitor.collect.return_value = [_make_snapshot()]

    detector = MagicMock()
    detector.check.return_value = []

    dispatcher = MagicMock()
    dispatcher.dispatch = AsyncMock(return_value=MagicMock(all_succeeded=True))

    defaults = {
        "experiment_name": "test/exp",
        "plugin": _make_plugin(),
        "run_configs": [],
        "config": WatchdogConfig(),
        "monitor": monitor,
        "alert_detector": detector,
        "dispatcher": dispatcher,
    }
    defaults.update(overrides)
    return WatchdogEngine(**defaults)


class TestEngineConstruction:
    def test_constructs_with_required_args(self):
        engine = WatchdogEngine(
            experiment_name="hover/test",
            plugin=_make_plugin(),
            run_configs=[],
            config=WatchdogConfig(),
        )
        assert engine.experiment_name == "hover/test"

    def test_default_config(self):
        engine = WatchdogEngine(
            experiment_name="test",
            plugin=_make_plugin(),
            run_configs=[],
        )
        assert engine.config.poll_interval_s == 3600

    def test_max_generations_stored(self):
        engine = WatchdogEngine(
            experiment_name="test",
            plugin=_make_plugin(),
            run_configs=[],
            max_generations=50,
        )
        assert engine.max_generations == 50


class TestEngineCycle:
    """Test a single _cycle() execution."""

    def test_cycle_calls_collect_alerts_plots_format_dispatch(self):
        """Engine cycle: collect -> alerts -> plots -> format -> dispatch."""
        plugin = _make_plugin()
        monitor = MagicMock()
        snap = _make_snapshot()
        monitor.collect.return_value = [snap]

        detector = MagicMock()
        detector.check.return_value = []

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(return_value=MagicMock(all_succeeded=True))

        engine = WatchdogEngine(
            experiment_name="test",
            plugin=plugin,
            run_configs=[],
            config=WatchdogConfig(),
            monitor=monitor,
            alert_detector=detector,
            dispatcher=dispatcher,
        )

        asyncio.run(engine._cycle(cycle=1))

        monitor.collect.assert_called_once()
        detector.check.assert_called_once_with([snap])
        plugin.generate_plots.assert_called_once()
        plugin.format_status_body.assert_called_once()
        dispatcher.dispatch.assert_called_once()

    def test_cycle_passes_alerts_to_status_update(self):
        """Alerts from detector appear in the StatusUpdate sent to dispatcher."""
        plugin = _make_plugin()
        monitor = MagicMock()
        monitor.collect.return_value = [_make_snapshot()]

        alert = Alert(
            alert_type=AlertType.STALL,
            severity=AlertSeverity.WARN,
            run_label="A",
            message="stalled",
        )
        detector = MagicMock()
        detector.check.return_value = [alert]

        captured_update = []

        async def capture_dispatch(update):
            captured_update.append(update)
            return MagicMock(all_succeeded=True)

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(side_effect=capture_dispatch)

        engine = WatchdogEngine(
            experiment_name="test",
            plugin=plugin,
            run_configs=[],
            monitor=monitor,
            alert_detector=detector,
            dispatcher=dispatcher,
        )
        asyncio.run(engine._cycle(cycle=1))

        assert len(captured_update) == 1
        assert alert in captured_update[0].alerts

    def test_cycle_passes_plots_to_status_update(self):
        """Plots from plugin appear in the StatusUpdate."""
        from pathlib import Path

        from gigaevo.monitoring.notifications import PlotAttachment

        plot = PlotAttachment(path=Path("/tmp/test.png"), caption="test plot")
        plugin = _make_plugin()
        plugin.generate_plots.return_value = [plot]

        captured_update = []

        async def capture_dispatch(update):
            captured_update.append(update)
            return MagicMock(all_succeeded=True)

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(side_effect=capture_dispatch)

        engine = _make_engine(plugin=plugin, dispatcher=dispatcher)
        asyncio.run(engine._cycle(cycle=1))

        assert len(captured_update) == 1
        assert captured_update[0].plots == [plot]

    def test_cycle_survives_plot_generation_error(self):
        """If plugin.generate_plots raises, cycle still dispatches."""
        plugin = _make_plugin()
        plugin.generate_plots.side_effect = RuntimeError("plot boom")

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(return_value=MagicMock(all_succeeded=True))

        engine = _make_engine(plugin=plugin, dispatcher=dispatcher)
        asyncio.run(engine._cycle(cycle=1))

        dispatcher.dispatch.assert_called_once()

    def test_cycle_survives_format_error(self):
        """If plugin.format_status_body raises, cycle still dispatches."""
        plugin = _make_plugin()
        plugin.format_status_body.side_effect = RuntimeError("format boom")

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(return_value=MagicMock(all_succeeded=True))

        engine = _make_engine(plugin=plugin, dispatcher=dispatcher)
        asyncio.run(engine._cycle(cycle=1))

        dispatcher.dispatch.assert_called_once()


class TestHeartbeat:
    """Heartbeat written to Redis with correct TTL."""

    def test_heartbeat_writes_to_redis(self):
        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=0, decode_responses=True)

        engine = WatchdogEngine(
            experiment_name="hover/test",
            plugin=_make_plugin(),
            run_configs=[],
            config=WatchdogConfig(poll_interval_s=600, heartbeat_ttl_multiplier=2),
            heartbeat_redis=r,
        )
        engine._write_heartbeat()

        key = "experiments:hover/test:watchdog_heartbeat"
        assert r.exists(key)
        ttl = r.ttl(key)
        assert ttl > 0
        assert ttl <= 1200  # 600 * 2

    def test_heartbeat_value_is_timestamp(self):
        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=0, decode_responses=True)

        engine = WatchdogEngine(
            experiment_name="test/exp",
            plugin=_make_plugin(),
            run_configs=[],
            heartbeat_redis=r,
        )
        engine._write_heartbeat()

        key = "experiments:test/exp:watchdog_heartbeat"
        value = r.get(key)
        assert value is not None
        assert int(value) > 0


class TestPlotCleanup:
    """Bounded plot file retention."""

    def test_cleanup_removes_oldest_files(self, tmp_path):
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()
        for i in range(60):
            (plot_dir / f"plot_{i:03d}.png").write_text("x")

        engine = WatchdogEngine(
            experiment_name="test",
            plugin=_make_plugin(),
            run_configs=[],
            config=WatchdogConfig(max_plot_files=50),
            plot_dir=plot_dir,
        )
        engine._cleanup_plots()

        remaining = list(plot_dir.glob("*.png"))
        assert len(remaining) <= 50

    def test_cleanup_no_op_when_under_limit(self, tmp_path):
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()
        for i in range(5):
            (plot_dir / f"plot_{i:03d}.png").write_text("x")

        engine = WatchdogEngine(
            experiment_name="test",
            plugin=_make_plugin(),
            run_configs=[],
            config=WatchdogConfig(max_plot_files=50),
            plot_dir=plot_dir,
        )
        engine._cleanup_plots()

        remaining = list(plot_dir.glob("*.png"))
        assert len(remaining) == 5

    def test_cleanup_no_op_when_dir_missing(self, tmp_path):
        plot_dir = tmp_path / "nonexistent"
        engine = WatchdogEngine(
            experiment_name="test",
            plugin=_make_plugin(),
            run_configs=[],
            plot_dir=plot_dir,
        )
        engine._cleanup_plots()  # Should not raise


class TestSIGTERM:
    """SIGTERM handling."""

    def test_shutdown_flag_set_by_handler(self):
        engine = _make_engine()
        assert engine._shutdown is False
        engine._sigterm_handler(None, None)
        assert engine._shutdown is True


class TestStagnationDetection:
    """MON-06: No frontier improvement for N gens triggers alert."""

    def test_stagnation_detected_after_n_cycles(self):
        """If frontier fitness unchanged for stagnation_gens cycles, alert fires."""
        engine = _make_engine(config=WatchdogConfig(stagnation_gens=3))
        snap = _make_snapshot(label="A", gen=10, fitness=0.5)

        # First 2 cycles: no stagnation (need 3)
        alerts = engine._check_stagnation([snap])
        assert len(alerts) == 0
        alerts = engine._check_stagnation([snap])
        assert len(alerts) == 0

        # 3rd cycle: stagnation detected
        alerts = engine._check_stagnation([snap])
        assert len(alerts) == 1
        assert "stagnant" in alerts[0].message.lower()
        assert alerts[0].alert_type == AlertType.STALL

    def test_no_stagnation_when_fitness_changes(self):
        """If frontier fitness changes within window, no stagnation alert."""
        engine = _make_engine(config=WatchdogConfig(stagnation_gens=3))

        snap1 = _make_snapshot(label="A", gen=10, fitness=0.5)
        snap2 = _make_snapshot(label="A", gen=11, fitness=0.6)
        snap3 = _make_snapshot(label="A", gen=12, fitness=0.6)

        engine._check_stagnation([snap1])
        engine._check_stagnation([snap2])
        alerts = engine._check_stagnation([snap3])
        assert len(alerts) == 0

    def test_stagnation_per_run(self):
        """Each run tracked independently."""
        engine = _make_engine(config=WatchdogConfig(stagnation_gens=2))

        snap_a = _make_snapshot(label="A", fitness=0.5)
        snap_b = _make_snapshot(label="B", fitness=0.7)

        engine._check_stagnation([snap_a, snap_b])
        alerts = engine._check_stagnation([snap_a, snap_b])

        # Both should trigger (both stagnant for 2 cycles)
        labels = {a.run_label for a in alerts}
        assert "A" in labels
        assert "B" in labels


class TestFinalAlert:
    """On max restarts, posts FINAL alert to both channels."""

    def test_dispatch_final_alert_sends_crash_alert(self):
        captured = []

        async def capture(update):
            captured.append(update)
            return MagicMock(all_succeeded=True)

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(side_effect=capture)

        engine = _make_engine(dispatcher=dispatcher)
        engine._dispatch_final_alert()

        assert len(captured) == 1
        update = captured[0]
        assert len(update.alerts) == 1
        alert = update.alerts[0]
        assert alert.alert_type == AlertType.CRASH
        assert alert.severity == AlertSeverity.ERROR
        assert "WATCHDOG CRASHED" in alert.message


class TestRedisCheckpoint:
    """Redis checkpoint and completion marker writing."""

    def test_write_redis_checkpoint_at_milestone(self):
        """Writes checkpoint when min generation reaches milestone."""
        import json

        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=0, decode_responses=True)

        engine = WatchdogEngine(
            experiment_name="hover/test",
            plugin=_make_plugin(),
            run_configs=[],
            config=WatchdogConfig(checkpoint_milestones=(0.1, 0.5, 1.0)),
            max_generations=50,
            heartbeat_redis=r,
        )

        snap_a = _make_snapshot(label="A", gen=10, fitness=0.5)
        snap_b = _make_snapshot(label="B", gen=12, fitness=0.6)
        engine._write_redis_checkpoint([snap_a, snap_b], cycle=1)

        key = "experiments:hover/test:checkpoint:5"
        assert r.exists(key)
        data = json.loads(r.get(key))
        assert data["gen"] == 5
        assert "timestamp" in data
        assert "A" in data["metrics"]

    def test_write_redis_checkpoint_skips_when_no_redis(self):
        """No error when heartbeat_redis is None."""
        engine = _make_engine(max_generations=50)
        engine._heartbeat_redis = None
        snap = _make_snapshot(gen=10)
        engine._write_redis_checkpoint([snap], cycle=1)

    def test_write_redis_checkpoint_skips_when_no_max_gen(self):
        """No error when max_generations is None."""
        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=0, decode_responses=True)
        engine = WatchdogEngine(
            experiment_name="test",
            plugin=_make_plugin(),
            run_configs=[],
            max_generations=None,
            heartbeat_redis=r,
        )
        snap = _make_snapshot(gen=10)
        engine._write_redis_checkpoint([snap], cycle=1)
        assert len(list(r.scan_iter("experiments:*:checkpoint:*"))) == 0

    def test_write_redis_checkpoint_does_not_overwrite(self):
        """Existing checkpoint is not overwritten."""
        import json

        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=0, decode_responses=True)

        key = "experiments:test/exp:checkpoint:5"
        r.set(key, '{"gen": 5, "original": true}')

        engine = WatchdogEngine(
            experiment_name="test/exp",
            plugin=_make_plugin(),
            run_configs=[],
            config=WatchdogConfig(checkpoint_milestones=(0.1, 0.5, 1.0)),
            max_generations=50,
            heartbeat_redis=r,
        )
        snap = _make_snapshot(gen=10)
        engine._write_redis_checkpoint([snap], cycle=2)

        data = json.loads(r.get(key))
        assert data.get("original") is True

    def test_write_completion_marker(self):
        """Writes completion marker with run states."""
        import json

        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=0, decode_responses=True)

        engine = WatchdogEngine(
            experiment_name="hover/test",
            plugin=_make_plugin(),
            run_configs=[],
            heartbeat_redis=r,
        )

        snaps = [
            _make_snapshot(label="A", gen=50, fitness=0.8),
            _make_snapshot(label="B", gen=50, fitness=0.75),
        ]
        engine._write_completion(snaps)

        key = "experiments:hover/test:completion"
        assert r.exists(key)
        data = json.loads(r.get(key))
        assert "timestamp" in data
        assert len(data["run_states"]) == 2
        assert data["run_states"][0]["label"] == "A"
        assert data["run_states"][0]["gen"] == 50

    def test_write_completion_skips_when_no_redis(self):
        """No error when heartbeat_redis is None."""
        engine = _make_engine()
        engine._heartbeat_redis = None
        engine._write_completion([_make_snapshot()])


class TestCompletionShutdown:
    """Engine sets _shutdown when COMPLETION alert is detected."""

    def test_cycle_sets_shutdown_on_completion(self):
        """When alerts contain COMPLETION, engine sets _shutdown=True."""
        completion_alert = Alert(
            alert_type=AlertType.COMPLETION,
            severity=AlertSeverity.INFO,
            run_label="experiment",
            message="All done",
        )
        detector = MagicMock()
        detector.check.return_value = [completion_alert]

        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=0, decode_responses=True)

        engine = _make_engine(
            alert_detector=detector,
            heartbeat_redis=r,
        )
        engine.experiment_name = "test/exp"

        asyncio.run(engine._cycle(cycle=1))
        assert engine._shutdown is True

        key = "experiments:test/exp:completion"
        assert r.exists(key)


class TestMemoryLogging:
    """Memory RSS logged each cycle."""

    def test_log_memory_returns_rss_mb(self):
        engine = _make_engine()
        rss = engine._log_memory()
        assert isinstance(rss, float)
        assert rss > 0

    def test_close_matplotlib_figures(self):
        """_close_matplotlib_figures calls plt.close('all')."""
        engine = _make_engine()
        # Should not raise even if matplotlib is available
        engine._close_matplotlib_figures()
