"""Tests for AdversarialPlugin -- CLI-delegating plot generation + Telegram formatting."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

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
    running_programs=5,
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
        running_programs=running_programs,
    )


def _make_g_snapshot(label="G1", db=1, gen=10, fitness=0.03):
    return _make_snapshot(
        label=label, db=db, prefix="heilbron_pop_a", gen=gen, fitness=fitness
    )


def _make_d_snapshot(label="D1", db=2, gen=10, fitness=0.02):
    return _make_snapshot(
        label=label, db=db, prefix="heilbron_pop_b", gen=gen, fitness=fitness
    )


def _mock_subprocess_success(output_dir, output_file="arms_race.png"):
    """Create a side_effect that writes a fake PNG after subprocess.run."""

    def side_effect(cmd, **kwargs):
        from pathlib import Path

        out_dir = None
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                out_dir = Path(cmd[i + 1])
                break
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / output_file).write_bytes(b"fake-png-data")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    return side_effect


class TestAdversarialPluginRegistration:
    def test_registered_as_adversarial(self):
        assert "adversarial" in get_registry()
        assert get_registry()["adversarial"] is AdversarialPlugin

    def test_is_watchdog_plugin_subclass(self):
        assert issubclass(AdversarialPlugin, WatchdogPlugin)


class TestAdversarialPluginUsesSubprocess:
    """Verify subprocess-based CLI delegation."""

    def test_has_subprocess_import(self):
        import inspect

        source = inspect.getsource(AdversarialPlugin)
        assert "subprocess" in source

    def test_no_ax_bar_calls(self):
        import inspect

        source = inspect.getsource(AdversarialPlugin)
        assert "ax.bar(" not in source


class TestAdversarialPluginInit:
    def test_default_plot_metrics(self):
        plugin = AdversarialPlugin()
        assert plugin._plot_metrics == ["fitness"]

    def test_custom_plot_metrics(self):
        plugin = AdversarialPlugin(
            plot_metrics=["fitness", "actual_fitness", "soft_fitness"]
        )
        assert plugin._plot_metrics == ["fitness", "actual_fitness", "soft_fitness"]

    def test_accepts_plot_commands(self):
        plugin = AdversarialPlugin(plot_commands=[MagicMock()])
        assert len(plugin._plot_commands) == 1


class TestAdversarialPluginRunGrouping:
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


class TestAdversarialPluginBuildRunArgs:
    def test_builds_run_args_from_snapshots(self):
        snapshots = [
            _make_g_snapshot("G1", db=1),
            _make_d_snapshot("D1", db=2),
        ]
        args = AdversarialPlugin._build_run_args(snapshots)
        assert args == [
            "-r",
            "heilbron_pop_a@1:G1",
            "-r",
            "heilbron_pop_b@2:D1",
        ]


class TestAdversarialPluginGeneratePlots:
    def test_generates_arms_race_and_comparison(self, tmp_path):
        """Calls subprocess for arms-race and comparison plots."""
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot("G1", 1), _make_d_snapshot("D1", 2)]

        call_count = [0]

        def multi_output(cmd, **kwargs):
            call_count[0] += 1
            from pathlib import Path

            out_dir = None
            for i, arg in enumerate(cmd):
                if arg == "-o" and i + 1 < len(cmd):
                    out_dir = Path(cmd[i + 1])
                    break
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                if "arms-race" in cmd:
                    (out_dir / "arms_race.png").write_bytes(b"fake")
                elif "comparison" in cmd:
                    (out_dir / "evolution_runs_comparison.png").write_bytes(b"fake")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            side_effect=multi_output,
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)

        assert len(plots) == 2
        assert call_count[0] == 2
        for p in plots:
            assert isinstance(p, PlotAttachment)
            assert p.path.exists()

    def test_empty_snapshots(self, tmp_path):
        plugin = AdversarialPlugin()
        plots = plugin.generate_plots([], tmp_path, cycle=1)
        assert plots == []

    def test_subprocess_failure_returns_empty(self, tmp_path):
        """If subprocess returns non-zero, returns empty list."""
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot(), _make_d_snapshot()]

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, b"", b"error"),
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert plots == []

    def test_subprocess_timeout_returns_empty(self, tmp_path):
        """If subprocess times out, returns empty list."""
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot(), _make_d_snapshot()]

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            side_effect=subprocess.TimeoutExpired([], 120),
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=1)
        assert plots == []

    def test_cycle_number_in_filename(self, tmp_path):
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot(), _make_d_snapshot()]

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            side_effect=_mock_subprocess_success(tmp_path, "arms_race.png"),
        ):
            plots = plugin.generate_plots(snapshots, tmp_path, cycle=42)
        assert any("0042" in p.path.name for p in plots)

    def test_arms_race_command_includes_paired_arg(self, tmp_path):
        """Subprocess call for arms-race includes --paired with G:D labels."""
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot("G1"), _make_d_snapshot("D1")]

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as mock_run:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)

        # First call should be arms-race
        first_call = mock_run.call_args_list[0]
        cmd = first_call[0][0]
        assert "arms-race" in cmd
        assert "--paired" in cmd
        paired_idx = cmd.index("--paired")
        assert cmd[paired_idx + 1] == "G1:D1"

    def test_comparison_command_includes_no_frontier_for(self, tmp_path):
        """Subprocess call for comparison includes --no-frontier-for D labels."""
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot("G1"), _make_d_snapshot("D1")]

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as mock_run:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)

        # Second call should be comparison
        second_call = mock_run.call_args_list[1]
        cmd = second_call[0][0]
        assert "comparison" in cmd
        assert "--no-frontier-for" in cmd
        nf_idx = cmd.index("--no-frontier-for")
        assert "D1" in cmd[nf_idx + 1]

    def test_missing_g_runs_skips_arms_race(self, tmp_path):
        """If no pop_a runs, arms-race is skipped but comparison still runs."""
        plugin = AdversarialPlugin()
        snapshots = [_make_d_snapshot("D1")]

        with patch(
            "gigaevo.monitoring.plugins.adversarial.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as mock_run:
            plugin.generate_plots(snapshots, tmp_path, cycle=1)

        # Only comparison should be called (not arms-race)
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert "comparison" in cmd


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


class TestAdversarialPluginFormatTelegramBody:
    def test_contains_gd_sections(self):
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot("G1"), _make_d_snapshot("D1")]
        body = plugin.format_telegram_body(
            snapshots, "test/exp", cycle=1, max_generations=50
        )
        assert body is not None
        assert "Constructor (G)" in body
        assert "Improver (D)" in body

    def test_contains_sota_comparison_when_baseline_set(self):
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot("G1", fitness=0.03)]
        body = plugin.format_telegram_body(
            snapshots, "test/exp", cycle=1, max_generations=50, baseline=0.034
        )
        assert body is not None
        assert "SOTA baseline" in body
        assert "of SOTA" in body

    def test_no_sota_without_baseline(self):
        plugin = AdversarialPlugin()
        snapshots = [_make_g_snapshot("G1")]
        body = plugin.format_telegram_body(
            snapshots, "test/exp", cycle=1, max_generations=50, baseline=None
        )
        assert body is not None
        assert "SOTA" not in body

    def test_max_gd_section_with_baseline(self):
        plugin = AdversarialPlugin()
        snapshots = [
            _make_g_snapshot("G1", fitness=0.03),
            _make_d_snapshot("D1", fitness=0.025),
        ]
        body = plugin.format_telegram_body(
            snapshots, "test/exp", cycle=1, max_generations=50, baseline=0.034
        )
        assert body is not None
        assert "max(G,D)" in body

    def test_completion_message_when_all_done(self):
        plugin = AdversarialPlugin()
        snapshots = [
            _make_g_snapshot("G1", gen=50),
            _make_d_snapshot("D1", gen=50),
        ]
        body = plugin.format_telegram_body(
            snapshots, "test/exp", cycle=1, max_generations=50
        )
        assert body is not None
        assert "ALL RUNS COMPLETE" in body

    def test_stalled_flag_when_no_running_programs(self):
        plugin = AdversarialPlugin()
        snap = RunSnapshot(
            run_spec=RunSpec(prefix="heilbron_pop_a", db=1, label="G1"),
            generation=10,
            metrics={"fitness": 0.03},
            running_programs=0,
        )
        body = plugin.format_telegram_body(
            [snap], "test/exp", cycle=1, max_generations=50
        )
        assert body is not None
        assert "! G1" in body


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
