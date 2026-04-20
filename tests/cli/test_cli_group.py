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


class TestMissingExperimentIsClickException:
    """Missing experiment.yaml must exit 1 with a one-line error, no traceback."""

    def test_status_missing_experiment(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["-e", "nonexistent/does-not-exist", "status"])
        assert result.exit_code == 1
        assert "experiment.yaml" in result.output
        assert "Traceback" not in result.output

    def test_logs_missing_experiment(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["-e", "nonexistent/xxx", "logs"])
        assert result.exit_code == 1
        assert "experiment.yaml" in result.output
        assert "Traceback" not in result.output

    def test_manifest_gate_missing_experiment(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main, ["-e", "nonexistent/xxx", "manifest", "gate", "running"]
        )
        assert result.exit_code == 1
        assert "experiment.yaml" in result.output
        assert "Traceback" not in result.output


class TestMalformedRunSpec:
    """Malformed -r values must raise BadParameter, not raw ValueError."""

    def test_garbage_run_spec_is_clean_error(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["-r", "wrong_syntax", "status"])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
        assert "--run" in result.output or "-r" in result.output


class TestManifestGateEnum:
    """`manifest gate <garbage>` must reject non-canonical status values."""

    def test_invalid_status_rejected_before_manifest_load(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main, ["-e", "nonexistent/x", "manifest", "gate", "bogus_state"]
        )
        # Click.Choice fires before command body → exit 2, no Traceback,
        # and the error must name the valid choices.
        assert result.exit_code == 2
        assert "Traceback" not in result.output
        assert "preregistered" in result.output
        assert "running" in result.output


class TestTopRangeValidation:
    """-n/--top-n must reject <= 0 with BadParameter."""

    def test_top_n_zero_rejected(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main, ["-r", "p@1:L", "top", "-n", "0"], catch_exceptions=False
        )
        assert result.exit_code == 2
        assert "must be >= 1" in result.output

    def test_top_n_negative_rejected(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main, ["-r", "p@1:L", "top", "-n", "-3"], catch_exceptions=False
        )
        assert result.exit_code == 2
        assert "must be >= 1" in result.output
