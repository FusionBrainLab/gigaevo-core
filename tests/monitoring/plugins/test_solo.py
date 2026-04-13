"""Tests for SoloPlugin -- CLI-delegating plot generation + Telegram formatting."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

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
        running_programs=5,
    )


def _mock_subprocess_comparison(cmd, **kwargs):
    """Side effect that creates the comparison output PNG."""
    from pathlib import Path

    out_dir = None
    for i, arg in enumerate(cmd):
        if arg == "-o" and i + 1 < len(cmd):
            out_dir = Path(cmd[i + 1])
            break
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evolution_runs_comparison.png").write_bytes(b"fake-png")
    return subprocess.CompletedProcess(cmd, 0, b"", b"")


class TestSoloPluginRegistration:
    def test_registered_as_solo(self):
        assert "solo" in get_registry()
        assert get_registry()["solo"] is SoloPlugin

    def test_is_watchdog_plugin_subclass(self):
        assert issubclass(SoloPlugin, WatchdogPlugin)

    def test_instantiates(self):
        plugin = SoloPlugin()
        assert isinstance(plugin, WatchdogPlugin)


class TestSoloPluginUsesSubprocess:
    """Verify subprocess-based CLI delegation."""

    def test_has_subprocess_import(self):
        import inspect

        source = inspect.getsource(SoloPlugin)
        assert "subprocess" in source

    def test_no_ax_bar_calls(self):
        import inspect

        source = inspect.getsource(SoloPlugin)
        assert "ax.bar(" not in source


class TestSoloPluginGeneratePlots:
    def test_generates_comparison_plot(self, tmp_path):
        """Calls subprocess for comparison plot and returns PlotAttachment."""
        plugin = SoloPlugin()
        snapshots = [_make_snapshot("A", 1), _make_snapshot("B", 2)]

        with patch(
            "gigaevo.monitoring.plugins.solo.subprocess.run",
            side_effect=_mock_subprocess_comparison,
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)

        assert len(plots) == 1
        assert isinstance(plots[0], PlotAttachment)
        assert plots[0].path.exists()

    def test_empty_snapshots_returns_empty_list(self, tmp_path):
        plugin = SoloPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_subprocess_failure_returns_empty_list(self, tmp_path):
        plugin = SoloPlugin()
        snapshots = [_make_snapshot()]

        with patch(
            "gigaevo.monitoring.plugins.solo.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, b"", b"error"),
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert plots == []

    def test_subprocess_timeout_returns_empty_list(self, tmp_path):
        plugin = SoloPlugin()
        snapshots = [_make_snapshot()]

        with patch(
            "gigaevo.monitoring.plugins.solo.subprocess.run",
            side_effect=subprocess.TimeoutExpired([], 120),
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert plots == []

    def test_cycle_number_in_filename(self, tmp_path):
        plugin = SoloPlugin()

        with patch(
            "gigaevo.monitoring.plugins.solo.subprocess.run",
            side_effect=_mock_subprocess_comparison,
        ):
            plots = plugin.generate_plots([_make_snapshot()], tmp_path, cycle=42)
        assert len(plots) == 1
        assert "0042" in plots[0].path.name

    def test_command_includes_ema_smoothing(self, tmp_path):
        plugin = SoloPlugin()
        snapshots = [_make_snapshot()]

        with patch(
            "gigaevo.monitoring.plugins.solo.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as mock_run:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)

        cmd = mock_run.call_args[0][0]
        assert "comparison" in cmd
        assert "--smoothing" in cmd
        ema_idx = cmd.index("--smoothing")
        assert cmd[ema_idx + 1] == "ema"

    def test_run_args_built_from_snapshots(self, tmp_path):
        plugin = SoloPlugin()
        snapshots = [_make_snapshot("A", 1), _make_snapshot("B", 2)]

        with patch(
            "gigaevo.monitoring.plugins.solo.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as mock_run:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)

        cmd = mock_run.call_args[0][0]
        assert "-r" in cmd
        assert "chains/hover/static_soft@1:A" in cmd
        assert "chains/hover/static_soft@2:B" in cmd


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


class TestSoloPluginFormatTelegramBody:
    def test_contains_run_info(self):
        plugin = SoloPlugin()
        snapshots = [_make_snapshot("A", gen=10, fitness=0.65)]
        body = plugin.format_telegram_body(
            snapshots, "hover/test", cycle=1, max_generations=25
        )
        assert body is not None
        assert "hover/test" in body
        assert "A" in body
        assert "0.65000" in body

    def test_contains_baseline_when_set(self):
        plugin = SoloPlugin()
        body = plugin.format_telegram_body(
            [_make_snapshot()], "test", cycle=1, max_generations=25, baseline=0.76
        )
        assert body is not None
        assert "SOTA baseline" in body
        assert "0.76000" in body

    def test_no_baseline_when_none(self):
        plugin = SoloPlugin()
        body = plugin.format_telegram_body(
            [_make_snapshot()], "test", cycle=1, max_generations=25, baseline=None
        )
        assert body is not None
        assert "SOTA" not in body

    def test_stalled_flag(self):
        snap = RunSnapshot(
            run_spec=RunSpec(prefix="test", db=1, label="X"),
            generation=10,
            metrics={"fitness": 0.5},
            running_programs=0,
        )
        plugin = SoloPlugin()
        body = plugin.format_telegram_body(
            [snap], "test", cycle=1, max_generations=25
        )
        assert body is not None
        assert "! X" in body


class TestSoloPluginDefaults:
    def test_extra_telegram_content_returns_none(self):
        plugin = SoloPlugin()
        assert plugin.extra_telegram_content([]) is None

    def test_extra_redis_queries_returns_empty(self):
        plugin = SoloPlugin()
        assert plugin.extra_redis_queries() == {}
