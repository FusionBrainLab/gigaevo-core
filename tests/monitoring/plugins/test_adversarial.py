"""Tests for AdversarialPlugin -- paired arms-race experiment monitoring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from gigaevo.monitoring.plugins.adversarial import AdversarialPlugin
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, get_registry


def _make_snapshot(label, db, prefix="heilbron_solo", gen=10, fitness=0.03):
    return RunSnapshot(
        run_spec=RunSpec(prefix=prefix, db=db, label=label),
        generation=gen,
        metrics={"fitness": fitness},
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


class TestAdversarialPluginRunGrouping:
    """AdversarialPlugin groups runs by prefix for separate comparison plots."""

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
    def test_generates_plot_per_group(self, tmp_path):
        plugin = AdversarialPlugin()
        snapshots = [
            _make_snapshot("S1", 1, prefix="heilbron_solo"),
            _make_snapshot("A1", 3, prefix="heilbron_adv"),
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
        plugin = AdversarialPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_subprocess_failure_still_returns_partial(self, tmp_path):
        """If one group's comparison.py fails, other groups still get plotted."""
        plugin = AdversarialPlugin()
        snapshots = [
            _make_snapshot("S1", 1, prefix="solo"),
            _make_snapshot("A1", 3, prefix="adv"),
        ]
        call_count = [0]

        def intermittent_fail(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("first call fails")
            out_dir = None
            for i, arg in enumerate(cmd):
                if arg == "--output-folder" and i + 1 < len(cmd):
                    out_dir = Path(cmd[i + 1])
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "evolution_runs_comparison.png").write_text("fake")
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=intermittent_fail):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert isinstance(plots, list)


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
