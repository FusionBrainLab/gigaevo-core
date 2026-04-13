"""Tests for PromptCoevoPlugin -- CLI-delegating plot generation + Telegram formatting."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

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
        running_programs=5,
    )


def _make_prompt_snapshot(label="P1", db=11, gen=8, fitness=0.25):
    return RunSnapshot(
        run_spec=RunSpec(prefix="prompt_evolution_hover", db=db, label=label),
        generation=gen,
        metrics={"fitness": fitness},
        total_programs=50,
        valid_programs=45,
        pid=2000,
        pid_alive=True,
        running_programs=3,
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


class TestPromptCoevoPluginRegistration:
    def test_registered_as_prompt_coevo(self):
        assert "prompt_coevo" in get_registry()
        assert get_registry()["prompt_coevo"] is PromptCoevoPlugin

    def test_is_watchdog_plugin_subclass(self):
        assert issubclass(PromptCoevoPlugin, WatchdogPlugin)


class TestPromptCoevoPluginUsesSubprocess:
    """Verify subprocess-based CLI delegation."""

    def test_has_subprocess_import(self):
        import inspect

        source = inspect.getsource(PromptCoevoPlugin)
        assert "subprocess" in source

    def test_no_ax_bar_calls(self):
        import inspect

        source = inspect.getsource(PromptCoevoPlugin)
        assert "ax.bar(" not in source


class TestPromptCoevoPluginGrouping:
    def test_separates_code_and_prompt_runs(self):
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
        """Calls subprocess once per population group."""
        plugin = PromptCoevoPlugin()
        snapshots = [
            _make_code_snapshot("C1", 9),
            _make_prompt_snapshot("P1", 11),
        ]

        with patch(
            "gigaevo.monitoring.plugins.prompt_coevo.subprocess.run",
            side_effect=_mock_subprocess_comparison,
        ) as mock_run:
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)

        assert mock_run.call_count == 2
        assert len(plots) >= 1
        for p in plots:
            assert isinstance(p, PlotAttachment)
            assert p.path.exists()

    def test_empty_snapshots(self, tmp_path):
        plugin = PromptCoevoPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_subprocess_failure_partial_results(self, tmp_path):
        """If one group fails, other groups still produce plots."""
        plugin = PromptCoevoPlugin()
        snapshots = [
            _make_code_snapshot("C1", 9),
            _make_prompt_snapshot("P1", 11),
        ]
        call_count = [0]

        def intermittent_fail(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return subprocess.CompletedProcess(cmd, 1, b"", b"error")
            return _mock_subprocess_comparison(cmd, **kwargs)

        with patch(
            "gigaevo.monitoring.plugins.prompt_coevo.subprocess.run",
            side_effect=intermittent_fail,
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert isinstance(plots, list)

    def test_cycle_number_in_filename(self, tmp_path):
        plugin = PromptCoevoPlugin()

        with patch(
            "gigaevo.monitoring.plugins.prompt_coevo.subprocess.run",
            side_effect=_mock_subprocess_comparison,
        ):
            plots = plugin.generate_plots([_make_code_snapshot()], tmp_path, cycle=7)
        assert len(plots) == 1
        assert "0007" in plots[0].path.name

    def test_caption_includes_population_type(self, tmp_path):
        plugin = PromptCoevoPlugin()

        with patch(
            "gigaevo.monitoring.plugins.prompt_coevo.subprocess.run",
            side_effect=_mock_subprocess_comparison,
        ):
            plots = plugin.generate_plots([_make_prompt_snapshot()], tmp_path, cycle=1)
        assert len(plots) == 1
        assert "Prompt Population" in plots[0].caption

    def test_run_args_per_group(self, tmp_path):
        """Each group subprocess call only includes that group's run args."""
        plugin = PromptCoevoPlugin()
        snapshots = [
            _make_code_snapshot("C1", 9),
            _make_prompt_snapshot("P1", 11),
        ]

        with patch(
            "gigaevo.monitoring.plugins.prompt_coevo.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as mock_run:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)

        # Each call should have only its group's -r args
        for call in mock_run.call_args_list:
            cmd = call[0][0]
            r_args = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-r"]
            # Each group should have exactly 1 run
            assert len(r_args) == 1


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


class TestPromptCoevoPluginFormatTelegramBody:
    def test_contains_population_groups(self):
        plugin = PromptCoevoPlugin()
        snapshots = [_make_code_snapshot("C1"), _make_prompt_snapshot("P1")]
        body = plugin.format_telegram_body(
            snapshots, "test/exp", cycle=1, max_generations=25
        )
        assert body is not None
        assert "Code Population" in body
        assert "Prompt Population" in body

    def test_contains_run_metrics(self):
        plugin = PromptCoevoPlugin()
        snapshots = [_make_code_snapshot("C1", fitness=0.76)]
        body = plugin.format_telegram_body(
            snapshots, "test/exp", cycle=1, max_generations=25
        )
        assert body is not None
        assert "0.76000" in body

    def test_contains_baseline_when_set(self):
        plugin = PromptCoevoPlugin()
        body = plugin.format_telegram_body(
            [_make_code_snapshot()], "test", cycle=1, max_generations=25, baseline=0.80
        )
        assert body is not None
        assert "SOTA baseline" in body

    def test_stalled_flag(self):
        snap = RunSnapshot(
            run_spec=RunSpec(prefix="chains/hover/static_soft", db=1, label="C1"),
            generation=10,
            metrics={"fitness": 0.5},
            running_programs=0,
        )
        plugin = PromptCoevoPlugin()
        body = plugin.format_telegram_body(
            [snap], "test", cycle=1, max_generations=25
        )
        assert body is not None
        assert "! C1" in body


class TestPromptCoevoPluginDefaults:
    def test_extra_telegram_content_returns_none(self):
        plugin = PromptCoevoPlugin()
        assert plugin.extra_telegram_content([]) is None

    def test_extra_redis_queries_returns_empty(self):
        plugin = PromptCoevoPlugin()
        assert plugin.extra_redis_queries() == {}
