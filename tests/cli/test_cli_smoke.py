"""CLI smoke tests: verify every --help is non-empty and well-documented.

Parametrized over all lazy-loaded subcommands and their group children.
Catches regressions like missing @click.option help= strings and empty docstrings.
"""

from __future__ import annotations

from click.testing import CliRunner
import pytest

from gigaevo.cli import _LAZY_SUBCOMMANDS, main


class TestCliSmoke:
    """Smoke tests for CLI help output."""

    @pytest.mark.parametrize("cmd_name", sorted(_LAZY_SUBCOMMANDS.keys()))
    def test_subcommand_help_is_nonempty(self, cmd_name: str) -> None:
        """Every registered subcommand --help output is non-empty and well-formed."""
        runner = CliRunner()
        result = runner.invoke(main, [cmd_name, "--help"], catch_exceptions=False)

        assert result.exit_code == 0, (
            f"{cmd_name} --help exited {result.exit_code}: {result.output}"
        )
        assert len(result.output) > 100, (
            f"{cmd_name} --help output too short ({len(result.output)} chars)"
        )
        assert "No help" not in result.output, (
            f"{cmd_name} --help contains 'No help' — missing option help="
        )

    def test_group_commands_have_children(self) -> None:
        """Group commands (plot, manifest, flush) list their children."""
        groups = {
            "plot": ["comparison", "trajectory", "arms-race"],
            "manifest": ["get", "update", "gate", "pr-description"],
            "flush": ["dry-run"],
        }  # flush dry-run is a group itself
        for group_name, expected_children in groups.items():
            runner = CliRunner()
            result = runner.invoke(main, [group_name, "--help"], catch_exceptions=False)
            assert result.exit_code == 0, f"{group_name} --help failed"
            for child in expected_children:
                assert child in result.output, (
                    f"{child} not listed in {group_name} --help"
                )

    @pytest.mark.parametrize(
        "group,children",
        [
            ("plot", ["comparison", "trajectory", "arms-race"]),
            ("manifest", ["get", "update", "gate", "pr-description"]),
        ],
    )
    def test_group_leaf_help_is_nonempty(self, group: str, children: list[str]) -> None:
        """Every group child's --help is non-empty."""
        for child in children:
            runner = CliRunner()
            result = runner.invoke(
                main, [group, child, "--help"], catch_exceptions=False
            )
            assert result.exit_code == 0, (
                f"{group} {child} --help exited {result.exit_code}: {result.output}"
            )
            assert len(result.output) > 50, (
                f"{group} {child} --help too short ({len(result.output)} chars)"
            )
            assert "No help" not in result.output, (
                f"{group} {child} --help contains 'No help'"
            )

    def test_root_help_shows_examples_and_flag_ordering(self) -> None:
        """Root gigaevo --help includes examples and global flag ordering note."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Examples" in result.output, "Root --help should mention Examples"
        assert "Argument order" in result.output, (
            "Root --help should mention flag ordering"
        )
        assert "-e" in result.output and "-r" in result.output, (
            "Global flags should be documented"
        )

    def test_no_help_lines_anywhere(self) -> None:
        """Scan all help outputs for the "No help" string (indicates missing help=)."""
        runner = CliRunner()
        all_helps = []

        # Root help
        result = runner.invoke(main, ["--help"], catch_exceptions=False)
        all_helps.append(("root", result.output))

        # Every subcommand
        for cmd_name in sorted(_LAZY_SUBCOMMANDS.keys()):
            result = runner.invoke(main, [cmd_name, "--help"], catch_exceptions=False)
            all_helps.append((cmd_name, result.output))

        # Every group child (plot, manifest)
        for group in ["plot", "manifest"]:
            result = runner.invoke(main, [group, "--help"], catch_exceptions=False)
            for line in result.output.split("\n"):
                # Extract child command names from help output (heuristic: lines with "  " indent)
                if line.startswith("  ") and not line.startswith("    "):
                    parts = line.strip().split()
                    if parts:
                        child_name = parts[0]
                        child_result = runner.invoke(
                            main, [group, child_name, "--help"], catch_exceptions=False
                        )
                        all_helps.append((f"{group} {child_name}", child_result.output))

        # Assert no "No help" in any output
        for name, output in all_helps:
            assert "No help" not in output, f"Found 'No help' in {name} --help"
