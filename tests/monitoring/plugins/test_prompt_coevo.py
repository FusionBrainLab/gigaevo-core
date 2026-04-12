"""Tests for PromptCoevoPlugin -- prompt co-evolution experiment monitoring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestPromptCoevoPluginGeneratePlots:
    def test_generates_plots_per_group(self, tmp_path):
        plugin = PromptCoevoPlugin()
        snapshots = [
            _make_code_snapshot("C1", 9),
            _make_prompt_snapshot("P1", 11),
        ]

        def fake_run(cmd, **kwargs):
            out_dir = None
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

    def test_empty_snapshots(self, tmp_path):
        plugin = PromptCoevoPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []


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
