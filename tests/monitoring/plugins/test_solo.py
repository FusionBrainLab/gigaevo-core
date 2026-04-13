"""Tests for SoloPlugin -- standard MAP-Elites watchdog plugin (inline matplotlib)."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

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


class TestSoloPluginNoSubprocess:
    """Verify no subprocess or tools/ references exist."""

    def test_no_subprocess_import(self):
        import inspect

        source = inspect.getsource(SoloPlugin)
        assert "subprocess" not in source

    def test_no_tools_reference(self):
        import inspect

        source = inspect.getsource(SoloPlugin)
        assert "tools/" not in source
        assert "_PROJ" not in source


class TestSoloPluginGeneratePlots:
    def test_generates_bar_chart(self, tmp_path):
        """Creates a bar chart PNG file using inline matplotlib."""
        plugin = SoloPlugin()
        snapshots = [_make_snapshot("A", 1), _make_snapshot("B", 2)]

        plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)

        assert len(plots) == 1
        assert isinstance(plots[0], PlotAttachment)
        assert plots[0].path.suffix == ".png"
        assert plots[0].path.exists()
        assert plots[0].path.stat().st_size > 0

    def test_empty_snapshots_returns_empty_list(self, tmp_path):
        plugin = SoloPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_no_fitness_values_returns_empty_list(self, tmp_path):
        """If all fitness values are None, returns empty."""
        plugin = SoloPlugin()
        snap = RunSnapshot(
            run_spec=RunSpec(prefix="test", db=1, label="X"),
            generation=5,
            metrics={},
            total_programs=10,
            valid_programs=8,
            pid=1000,
            pid_alive=True,
        )
        plots = plugin.generate_plots([snap], tmp_path, cycle=1)
        assert plots == []

    def test_matplotlib_failure_returns_empty_list(self, tmp_path):
        """If matplotlib raises, returns empty list (no crash)."""
        plugin = SoloPlugin()
        snapshots = [_make_snapshot()]
        with patch(
            "matplotlib.pyplot.subplots", side_effect=Exception("display error")
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert plots == []

    def test_plt_close_called_on_success(self, tmp_path):
        """plt.close(fig) is called for resource cleanup."""
        plugin = SoloPlugin()
        snapshots = [_make_snapshot()]

        with patch("matplotlib.pyplot.close") as mock_close:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)
        mock_close.assert_called()

    def test_plt_close_called_on_failure(self, tmp_path):
        """plt.close(fig) is called in finally even when save fails."""
        plugin = SoloPlugin()
        snapshots = [_make_snapshot()]

        with (
            patch("matplotlib.pyplot.close") as mock_close,
            patch(
                "matplotlib.figure.Figure.savefig", side_effect=Exception("save fail")
            ),
        ):
            plugin.generate_plots(snapshots, tmp_path, cycle=1)
        mock_close.assert_called()

    def test_cycle_number_in_filename(self, tmp_path):
        """Output filename includes zero-padded cycle number."""
        plugin = SoloPlugin()
        plots = plugin.generate_plots([_make_snapshot()], tmp_path, cycle=42)
        assert len(plots) == 1
        assert "0042" in plots[0].path.name


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
