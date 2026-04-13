"""Tests for AdversarialPlugin -- multi-metric panel plots (inline matplotlib)."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from gigaevo.monitoring.notifications import PlotAttachment
from gigaevo.monitoring.plugins.adversarial import AdversarialPlugin
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


class TestAdversarialPluginRegistration:
    def test_registered_as_adversarial(self):
        assert "adversarial" in get_registry()
        assert get_registry()["adversarial"] is AdversarialPlugin

    def test_is_watchdog_plugin_subclass(self):
        assert issubclass(AdversarialPlugin, WatchdogPlugin)


class TestAdversarialPluginNoSubprocess:
    """Verify no subprocess or tools/ references exist."""

    def test_no_subprocess_import(self):
        import inspect

        source = inspect.getsource(AdversarialPlugin)
        assert "subprocess" not in source

    def test_no_tools_reference(self):
        import inspect

        source = inspect.getsource(AdversarialPlugin)
        assert "tools/" not in source
        assert "_PROJ" not in source


class TestAdversarialPluginInit:
    def test_default_plot_metrics(self):
        plugin = AdversarialPlugin()
        assert plugin._plot_metrics == ["fitness"]

    def test_custom_plot_metrics(self):
        plugin = AdversarialPlugin(
            plot_metrics=["fitness", "actual_fitness", "soft_fitness"]
        )
        assert plugin._plot_metrics == ["fitness", "actual_fitness", "soft_fitness"]


class TestAdversarialPluginRunGrouping:
    """AdversarialPlugin groups runs by prefix."""

    def test_groups_by_prefix(self):
        plugin = AdversarialPlugin()
        snapshots = [
            _make_snapshot("S1", 1, prefix="heilbron_solo"),
            _make_snapshot("S2", 2, prefix="heilbron_solo"),
            _make_snapshot("A1", 3, prefix="heilbron_adversarial_pop_a"),
            _make_snapshot("A2", 4, prefix="heilbron_adversarial_pop_b"),
        ]
        groups = plugin._group_runs(snapshots)
        assert len(groups) >= 2

    def test_single_group_when_same_prefix(self):
        plugin = AdversarialPlugin()
        snapshots = [
            _make_snapshot("A", 1, prefix="same"),
            _make_snapshot("B", 2, prefix="same"),
        ]
        groups = plugin._group_runs(snapshots)
        assert len(groups) == 1


class TestAdversarialPluginGeneratePlots:
    def test_generates_panel_plot(self, tmp_path):
        """Creates a multi-panel PNG file using inline matplotlib."""
        plugin = AdversarialPlugin(
            plot_metrics=["fitness", "actual_fitness", "soft_fitness"]
        )
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
            assert p.path.exists()
            assert p.path.stat().st_size > 0

    def test_empty_snapshots(self, tmp_path):
        plugin = AdversarialPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_matplotlib_failure_returns_empty(self, tmp_path):
        """If matplotlib raises, returns empty list (no crash)."""
        plugin = AdversarialPlugin()
        snapshots = [_make_snapshot()]
        with patch(
            "matplotlib.pyplot.subplots", side_effect=Exception("display error")
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert plots == []

    def test_plt_close_called_on_success(self, tmp_path):
        """plt.close(fig) is called even on success (resource cleanup)."""
        plugin = AdversarialPlugin()
        snapshots = [_make_snapshot()]

        with patch("matplotlib.pyplot.close") as mock_close:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)
        mock_close.assert_called()

    def test_plt_close_called_on_failure(self, tmp_path):
        """plt.close(fig) is called in finally even when rendering fails."""
        plugin = AdversarialPlugin()
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
        plugin = AdversarialPlugin()
        plots = plugin.generate_plots([_make_snapshot()], tmp_path, cycle=42)
        assert len(plots) == 1
        assert "0042" in plots[0].path.name


class TestAdversarialPluginFormatStatus:
    def test_format_includes_group_headers(self):
        plugin = AdversarialPlugin()
        snapshots = [
            _make_snapshot("S1", 1, prefix="heilbron_solo"),
            _make_snapshot("A1", 3, prefix="heilbron_adv"),
        ]
        body = plugin.format_status_body(
            snapshots, "adversarial/test", cycle=5, max_generations=50
        )
        assert "adversarial/test" in body
        assert isinstance(body, str)

    def test_format_empty_snapshots(self):
        plugin = AdversarialPlugin()
        body = plugin.format_status_body([], "test", cycle=1, max_generations=None)
        assert isinstance(body, str)


class TestAdversarialPluginTelegramContent:
    def test_extra_telegram_content_with_data(self):
        plugin = AdversarialPlugin()
        content = plugin.extra_telegram_content([_make_snapshot()])
        assert content is not None
        assert "fitness" in content

    def test_extra_telegram_content_empty(self):
        plugin = AdversarialPlugin()
        content = plugin.extra_telegram_content([])
        assert content is None

    def test_extra_telegram_uses_first_plot_metric(self):
        plugin = AdversarialPlugin(plot_metrics=["actual_fitness", "fitness"])
        content = plugin.extra_telegram_content([_make_snapshot()])
        assert content is not None
        assert "actual_fitness" in content
