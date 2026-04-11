"""Tests for SoloPlugin -- standard MAP-Elites watchdog plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from gigaevo.monitoring.notifications import PlotAttachment
from gigaevo.monitoring.plugins.solo import SoloPlugin
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, get_registry


def _make_snapshot(label="A", db=1, gen=10, fitness=0.65, pid=1234, alive=True):
    return RunSnapshot(
        run_spec=RunSpec(prefix="chains/hover/static_soft", db=db, label=label),
        generation=gen,
        metrics={"fitness": fitness},
        total_programs=200,
        valid_programs=180,
        pid=pid,
        pid_alive=alive,
    )


class TestSoloPluginRegistration:
    def test_registered_as_solo(self):
        assert "solo" in get_registry()
        assert get_registry()["solo"] is SoloPlugin

    def test_is_watchdog_plugin_subclass(self):
        assert issubclass(SoloPlugin, WatchdogPlugin)

    def test_instantiates(self):
        plugin = SoloPlugin()
        assert isinstance(plugin, WatchdogPlugin)


class TestSoloPluginGeneratePlots:
    def test_generates_comparison_plot(self, tmp_path):
        """Calls comparison.py subprocess and returns PlotAttachment."""
        plugin = SoloPlugin()
        snapshots = [_make_snapshot("A", 1), _make_snapshot("B", 2)]

        def fake_run(*args, **kwargs):
            out_dir = None
            cmd = args[0]
            for i, arg in enumerate(cmd):
                if arg == "--output-folder" and i + 1 < len(cmd):
                    out_dir = Path(cmd[i + 1])
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "evolution_runs_comparison.png").write_text("fake")
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=fake_run):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)

        assert len(plots) >= 1
        assert isinstance(plots[0], PlotAttachment)
        assert plots[0].path.exists()

    def test_empty_snapshots_returns_empty_list(self, tmp_path):
        plugin = SoloPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_subprocess_failure_returns_empty_list(self, tmp_path):
        """If comparison.py fails, returns empty list (no crash)."""
        plugin = SoloPlugin()
        snapshots = [_make_snapshot()]
        with patch("subprocess.run", side_effect=Exception("boom")):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert plots == []

    def test_builds_correct_run_args(self, tmp_path):
        """--run arguments use prefix@db:label format."""
        plugin = SoloPlugin()
        snapshots = [
            _make_snapshot("A", 4),
            _make_snapshot("B", 5),
        ]
        captured_cmd = []

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=capture_run):
            plugin.generate_plots(snapshots, tmp_path, cycle=1)

        run_indices = [i for i, x in enumerate(captured_cmd) if x == "--run"]
        assert len(run_indices) == 2


class TestSoloPluginFormatStatus:
    def test_format_includes_experiment_name(self):
        plugin = SoloPlugin()
        snapshots = [_make_snapshot()]
        body = plugin.format_status_body(
            snapshots, "hover/test", cycle=5, max_generations=25
        )
        assert "hover/test" in body

    def test_format_includes_cycle_number(self):
        plugin = SoloPlugin()
        body = plugin.format_status_body(
            [_make_snapshot()], "test", cycle=42, max_generations=25
        )
        assert "42" in body

    def test_format_includes_markdown_table(self):
        plugin = SoloPlugin()
        body = plugin.format_status_body(
            [_make_snapshot()], "test", cycle=1, max_generations=25
        )
        assert "| Run" in body or "Run" in body

    def test_format_empty_snapshots(self):
        plugin = SoloPlugin()
        body = plugin.format_status_body([], "test", cycle=1, max_generations=None)
        assert isinstance(body, str)
        assert len(body) > 0


class TestSoloPluginDefaults:
    def test_extra_telegram_content_returns_none(self):
        plugin = SoloPlugin()
        assert plugin.extra_telegram_content([]) is None

    def test_extra_redis_queries_returns_empty(self):
        plugin = SoloPlugin()
        assert plugin.extra_redis_queries() == {}
