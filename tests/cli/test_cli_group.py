"""Tests for CLI group: rich-click, global flags, lazy imports."""

from __future__ import annotations

import sys

from click.testing import CliRunner
import pytest


class TestHelpOutput:
    def test_help_exits_zero(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_help_contains_gigaevo(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "GigaEvo" in result.output or "gigaevo" in result.output.lower()

    def test_help_lists_subcommands(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "status" in result.output
        assert "plot" in result.output


class TestGlobalFlags:
    def test_format_choices_in_help(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "format" in result.output.lower()

    def test_experiment_flag_in_help(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "experiment" in result.output.lower()

    def test_run_flag_in_help(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "run" in result.output.lower()


class TestLazyImports:
    def test_no_matplotlib_at_import(self):
        mods_before = set(sys.modules.keys())
        if "gigaevo.cli" in sys.modules:
            return
        import gigaevo.cli  # noqa: F401

        new_mods = set(sys.modules.keys()) - mods_before
        matplotlib_mods = [m for m in new_mods if "matplotlib" in m]
        assert matplotlib_mods == [], (
            f"matplotlib imported at CLI load: {matplotlib_mods}"
        )


@pytest.mark.skip(reason="analyze and collect commands not yet registered in CLI")
class TestAnalyzeAndCollectRegistered:
    def test_analyze_in_command_list(self):
        """analyze appears in the CLI command listing."""
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "analyze" in result.output

    def test_collect_in_command_list(self):
        """collect appears in the CLI command listing."""
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "collect" in result.output

    def test_analyze_resolves_to_click_command(self):
        """analyze resolves to a Click command (not None)."""
        import click

        from gigaevo.cli import main

        ctx = click.Context(main)
        cmd = main.get_command(ctx, "analyze")
        assert cmd is not None
        assert isinstance(cmd, click.BaseCommand)

    def test_collect_resolves_to_click_command(self):
        """collect resolves to a Click command (not None)."""
        import click

        from gigaevo.cli import main

        ctx = click.Context(main)
        cmd = main.get_command(ctx, "collect")
        assert cmd is not None
        assert isinstance(cmd, click.BaseCommand)


class TestContextObject:
    def test_context_has_formatter(self):
        from gigaevo.cli import main

        runner = CliRunner()
        captured_ctx = {}

        @main.command("_test_ctx")
        @__import__("click").pass_context
        def _test_ctx(ctx):
            captured_ctx.update(ctx.obj)

        result = runner.invoke(main, ["_test_ctx"])
        assert result.exit_code == 0
        assert "formatter" in captured_ctx
