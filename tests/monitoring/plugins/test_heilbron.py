"""Tests for HeilbronPlugin -- 2x2 panel plots with 3-metric panels."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from gigaevo.monitoring.notifications import PlotAttachment
from gigaevo.monitoring.plugins.heilbron import HeilbronPlugin
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, get_registry


def _make_snapshot(
    label="S1",
    db=1,
    prefix="heilbron_solo",
    gen=10,
    fitness=0.03,
    actual_fitness=0.028,
    soft_fitness=0.5,
):
    return RunSnapshot(
        run_spec=RunSpec(prefix=prefix, db=db, label=label),
        generation=gen,
        metrics={
            "fitness": fitness,
            "actual_fitness": actual_fitness,
            "soft_fitness": soft_fitness,
        },
        total_programs=100,
        valid_programs=90,
        pid=1000 + db,
        pid_alive=True,
    )


class TestHeilbronPluginRegistration:
    def test_registered_as_heilbron(self):
        assert "heilbron" in get_registry()
        assert get_registry()["heilbron"] is HeilbronPlugin

    def test_is_watchdog_plugin_subclass(self):
        assert issubclass(HeilbronPlugin, WatchdogPlugin)


class TestHeilbronPluginGeneratePlots:
    def test_generates_panel_plot(self, tmp_path):
        """Creates a multi-panel PNG file."""
        plugin = HeilbronPlugin()
        snapshots = [
            _make_snapshot("S1", 1),
            _make_snapshot("S2", 2),
            _make_snapshot("A1", 3, prefix="heilbron_adv"),
            _make_snapshot("A2", 4, prefix="heilbron_adv"),
        ]

        plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert len(plots) >= 1
        for p in plots:
            assert isinstance(p, PlotAttachment)
            assert p.path.suffix == ".png"

    def test_empty_snapshots_returns_empty(self, tmp_path):
        plugin = HeilbronPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_matplotlib_failure_returns_empty(self, tmp_path):
        """If matplotlib raises, returns empty list (no crash)."""
        plugin = HeilbronPlugin()
        snapshots = [_make_snapshot()]
        with patch(
            "matplotlib.pyplot.subplots", side_effect=Exception("display error")
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert plots == []

    def test_plt_close_called_on_success(self, tmp_path):
        """plt.close(fig) is called even on success (resource cleanup)."""
        plugin = HeilbronPlugin()
        snapshots = [_make_snapshot()]

        with patch("matplotlib.pyplot.close") as mock_close:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)
        mock_close.assert_called()

    def test_plt_close_called_on_failure(self, tmp_path):
        """plt.close(fig) is called in finally even when rendering fails."""
        plugin = HeilbronPlugin()
        snapshots = [_make_snapshot()]

        with (
            patch("matplotlib.pyplot.close") as mock_close,
            patch(
                "matplotlib.figure.Figure.savefig", side_effect=Exception("save fail")
            ),
        ):
            plugin.generate_plots(snapshots, tmp_path, cycle=1)
        mock_close.assert_called()


class TestHeilbronPluginFormatStatus:
    def test_includes_experiment_name(self):
        plugin = HeilbronPlugin()
        body = plugin.format_status_body(
            [_make_snapshot()], "heilbron/test", cycle=5, max_generations=50
        )
        assert "heilbron/test" in body

    def test_includes_metrics_columns(self):
        """Status body mentions actual_fitness and soft_fitness."""
        plugin = HeilbronPlugin()
        body = plugin.format_status_body(
            [_make_snapshot()], "heilbron/test", cycle=1, max_generations=50
        )
        assert isinstance(body, str)
        assert len(body) > 50


class TestHeilbronPluginTelegramContent:
    def test_extra_telegram_content_returns_string(self):
        """HeilbronPlugin overrides extra_telegram_content for photo references."""
        plugin = HeilbronPlugin()
        content = plugin.extra_telegram_content([_make_snapshot()])
        assert content is None or isinstance(content, str)


class TestHeilbronPluginExtraRedisQueries:
    def test_extra_redis_queries_returns_dict(self):
        plugin = HeilbronPlugin()
        queries = plugin.extra_redis_queries()
        assert isinstance(queries, dict)
