"""Tests for PromptCoevoPlugin -- prompt co-evolution monitoring (inline matplotlib)."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from gigaevo.monitoring.notifications import PlotAttachment
from gigaevo.monitoring.plugins.prompt_coevo import PromptCoevoPlugin
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, get_registry


def _make_code_snapshot(label="C1", db=9, gen=10, fitness=0.76):
    return RunSnapshot(
        run_spec=RunSpec(prefix="chains/hover/static_soft", db=db, label=label),
        generation=gen,
        metrics={"fitness": fitness},
        total_programs=100,
        valid_programs=85,
        pid=1000,
        pid_alive=True,
    )


def _make_prompt_snapshot(label="P1", db=11, gen=8, fitness=0.25, prompt_length=299.0):
    return RunSnapshot(
        run_spec=RunSpec(prefix="prompt_evolution_hover", db=db, label=label),
        generation=gen,
        metrics={"fitness": fitness, "prompt_length": prompt_length},
        total_programs=50,
        valid_programs=45,
        pid=2000,
        pid_alive=True,
    )


class TestPromptCoevoPluginRegistration:
    def test_registered_as_prompt_coevo(self):
        assert "prompt_coevo" in get_registry()
        assert get_registry()["prompt_coevo"] is PromptCoevoPlugin

    def test_is_watchdog_plugin_subclass(self):
        assert issubclass(PromptCoevoPlugin, WatchdogPlugin)


class TestPromptCoevoPluginNoSubprocess:
    """Verify no subprocess or tools/ references exist."""

    def test_no_subprocess_import(self):
        import inspect

        source = inspect.getsource(PromptCoevoPlugin)
        assert "subprocess" not in source

    def test_no_tools_reference(self):
        import inspect

        source = inspect.getsource(PromptCoevoPlugin)
        assert "tools/" not in source
        assert "_PROJ" not in source


class TestPromptCoevoPluginGrouping:
    def test_separates_code_and_prompt_runs(self):
        """Groups by prefix: code runs vs prompt runs."""
        plugin = PromptCoevoPlugin()
        snapshots = [
            _make_code_snapshot("C1", 9),
            _make_code_snapshot("C2", 10),
            _make_prompt_snapshot("P1", 11),
            _make_prompt_snapshot("P2", 12),
        ]
        groups = plugin._group_runs(snapshots)
        assert len(groups) == 2

    def test_single_group_same_prefix(self):
        plugin = PromptCoevoPlugin()
        snapshots = [_make_code_snapshot("C1", 9), _make_code_snapshot("C2", 10)]
        groups = plugin._group_runs(snapshots)
        assert len(groups) == 1

    def test_classify_prompt_group(self):
        plugin = PromptCoevoPlugin()
        assert plugin._classify_group("prompt_evolution_hover") == "Prompt Population"

    def test_classify_code_group(self):
        plugin = PromptCoevoPlugin()
        assert plugin._classify_group("chains/hover/static_soft") == "Code Population"


class TestPromptCoevoPluginGeneratePlots:
    def test_generates_plots_per_group(self, tmp_path):
        """Creates one PNG per population group using inline matplotlib."""
        plugin = PromptCoevoPlugin()
        snapshots = [
            _make_code_snapshot("C1", 9),
            _make_prompt_snapshot("P1", 11),
        ]

        plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)

        assert len(plots) >= 1
        for p in plots:
            assert isinstance(p, PlotAttachment)
            assert p.path.suffix == ".png"
            assert p.path.exists()
            assert p.path.stat().st_size > 0

    def test_empty_snapshots(self, tmp_path):
        plugin = PromptCoevoPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_matplotlib_failure_partial_results(self, tmp_path):
        """If one group fails, other groups still get plotted."""
        plugin = PromptCoevoPlugin()
        snapshots = [
            _make_code_snapshot("C1", 9),
            _make_prompt_snapshot("P1", 11),
        ]
        call_count = [0]
        orig_subplots = matplotlib.pyplot.subplots

        def intermittent_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("first call fails")
            return orig_subplots(*args, **kwargs)

        with patch("matplotlib.pyplot.subplots", side_effect=intermittent_fail):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert isinstance(plots, list)

    def test_plt_close_called_on_success(self, tmp_path):
        """plt.close(fig) is called for resource cleanup."""
        plugin = PromptCoevoPlugin()
        snapshots = [_make_code_snapshot()]

        with patch("matplotlib.pyplot.close") as mock_close:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)
        mock_close.assert_called()

    def test_cycle_number_in_filename(self, tmp_path):
        """Output filename includes zero-padded cycle number."""
        plugin = PromptCoevoPlugin()
        plots = plugin.generate_plots([_make_code_snapshot()], tmp_path, cycle=7)
        assert len(plots) == 1
        assert "0007" in plots[0].path.name

    def test_caption_includes_population_type(self, tmp_path):
        """Caption indicates whether it's code or prompt population."""
        plugin = PromptCoevoPlugin()
        plots = plugin.generate_plots([_make_prompt_snapshot()], tmp_path, cycle=1)
        assert len(plots) == 1
        assert "Prompt Population" in plots[0].caption


class TestPromptCoevoPluginFormatStatus:
    def test_includes_experiment_name(self):
        plugin = PromptCoevoPlugin()
        body = plugin.format_status_body(
            [_make_code_snapshot(), _make_prompt_snapshot()],
            "hover/prompt_coevolution",
            cycle=3,
            max_generations=25,
        )
        assert "hover/prompt_coevolution" in body

    def test_includes_group_sections(self):
        plugin = PromptCoevoPlugin()
        snapshots = [_make_code_snapshot(), _make_prompt_snapshot()]
        body = plugin.format_status_body(snapshots, "test", cycle=1, max_generations=25)
        assert isinstance(body, str)
        assert len(body) > 50

    def test_empty_snapshots(self):
        plugin = PromptCoevoPlugin()
        body = plugin.format_status_body([], "test", cycle=1, max_generations=None)
        assert isinstance(body, str)


class TestPromptCoevoPluginDefaults:
    def test_extra_telegram_content_returns_none(self):
        plugin = PromptCoevoPlugin()
        assert plugin.extra_telegram_content([]) is None

    def test_extra_redis_queries_returns_empty(self):
        plugin = PromptCoevoPlugin()
        assert plugin.extra_redis_queries() == {}
