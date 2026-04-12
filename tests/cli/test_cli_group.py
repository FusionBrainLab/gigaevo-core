"""Tests for CLI group: rich-click, global flags, lazy imports."""

from __future__ import annotations

import sys

from click.testing import CliRunner


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
