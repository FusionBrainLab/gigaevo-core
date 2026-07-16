"""Tests for the logs CLI subcommand.

The logs command tails per-run log files at experiments/<name>/run_<label>.log.

Resolution priority (highest to lowest):
  1. Explicit --file path (bypasses manifest entirely).
  2. Positional run labels (resolved via manifest).
  3. No args + --experiment: list mode shows all candidate run logs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
import pytest

from gigaevo.cli import main


@pytest.fixture(autouse=True)
def _project_root(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("gigaevo.experiment.manifest.PROJ", tmp_path)


def _mock_manifest(labels: list[str]):
    manifest = MagicMock()
    manifest.contract.runs = []
    for label in labels:
        run = MagicMock()
        run.label = label
        run.db = 1
        run.prefix = f"pfx_{label}"
        run.log_path = None
        manifest.contract.runs.append(run)
    return manifest


class TestLogsExplicitFile:
    def test_reads_explicit_log_file(self, tmp_path: Path):
        """--file reads the specified log file (no experiment required)."""
        log = tmp_path / "test.log"
        log.write_text("line1\nline2\nline3\nline4\nline5\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["logs", "--file", str(log), "-n", "3"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "line3" in result.output
        assert "line5" in result.output

    def test_missing_file_shows_error(self):
        """Missing log file exits with error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["logs", "--file", "/nonexistent/path.log"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0


class TestLogsByLabel:
    def test_manifest_log_paths_do_not_depend_on_cwd(self, tmp_path: Path, monkeypatch):
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_A.log").write_text("project-root log\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        with patch(
            "gigaevo.cli.logs._load_manifest", return_value=_mock_manifest(["A"])
        ):
            result = CliRunner().invoke(
                main,
                ["-e", "task/exp1", "logs", "A"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert "project-root log" in result.output

    def test_uses_manifest_log_path(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "custom-a.log").write_text("configured path\n")
        manifest = _mock_manifest(["A"])
        manifest.contract.runs[0].log_path = "custom-a.log"

        with patch("gigaevo.cli.logs._load_manifest", return_value=manifest):
            result = CliRunner().invoke(
                main,
                ["-e", "task/exp1", "logs", "A"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert "configured path" in result.output

    def test_tails_run_log_by_label(self, tmp_path: Path, monkeypatch):
        """Positional label resolves to experiments/<exp>/run_<label>.log."""
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_A3_G.log").write_text("A3_G line1\nA3_G line2\n")
        (exp_dir / "run_B5_D.log").write_text("B5_D should not appear\n")

        with patch(
            "gigaevo.cli.logs._load_manifest",
            return_value=_mock_manifest(["A3_G", "B5_D"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "logs", "A3_G"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert "A3_G line1" in result.output
        assert "B5_D should not appear" not in result.output

    def test_unknown_label_errors(self, tmp_path: Path, monkeypatch):
        """Label not present in manifest exits with a helpful error."""
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)

        with patch(
            "gigaevo.cli.logs._load_manifest",
            return_value=_mock_manifest(["A3_G", "B5_D"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "logs", "BOGUS"],
                catch_exceptions=False,
            )

        assert result.exit_code != 0
        assert "BOGUS" in result.output
        assert "A3_G" in result.output  # lists known labels

    def test_multiple_labels_tails_all(self, tmp_path: Path, monkeypatch):
        """Multiple labels tail multiple files via a single tail invocation."""
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_A3_G.log").write_text("a-content\n")
        (exp_dir / "run_B5_G.log").write_text("b-content\n")

        with (
            patch(
                "gigaevo.cli.logs._load_manifest",
                return_value=_mock_manifest(["A3_G", "B5_G"]),
            ),
            patch("gigaevo.cli.logs.subprocess") as mock_sub,
        ):
            mock_sub.run.return_value = MagicMock(stdout="", returncode=0)
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "logs", "A3_G", "B5_G"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        call_args = mock_sub.run.call_args[0][0]
        assert "tail" in call_args
        assert str(exp_dir / "run_A3_G.log") in call_args
        assert str(exp_dir / "run_B5_G.log") in call_args

    def test_follow_flag_with_label(self, tmp_path: Path, monkeypatch):
        """-f with a label calls tail -f on the resolved run_<label>.log."""
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_A3_G.log").write_text("content\n")

        with (
            patch(
                "gigaevo.cli.logs._load_manifest",
                return_value=_mock_manifest(["A3_G"]),
            ),
            patch("gigaevo.cli.logs.subprocess") as mock_sub,
        ):
            mock_sub.run.return_value = MagicMock(returncode=0)
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "logs", "A3_G", "-f"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        call_args = mock_sub.run.call_args[0][0]
        assert "tail" in call_args
        assert "-f" in call_args
        assert str(exp_dir / "run_A3_G.log") in call_args

    def test_missing_log_file_errors(self, tmp_path: Path, monkeypatch):
        """Known label but log file absent exits with error mentioning path."""
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        # No run_A3_G.log created.

        with patch(
            "gigaevo.cli.logs._load_manifest",
            return_value=_mock_manifest(["A3_G"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "logs", "A3_G"],
                catch_exceptions=False,
            )

        assert result.exit_code != 0
        assert "run_A3_G.log" in result.output


class TestLogsListMode:
    def test_no_labels_lists_candidate_logs(self, tmp_path: Path, monkeypatch):
        """No labels + --experiment prints a list of run logs with sizes."""
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_A3_G.log").write_text("x" * 100)
        (exp_dir / "run_B5_D.log").write_text("y" * 50)

        with patch(
            "gigaevo.cli.logs._load_manifest",
            return_value=_mock_manifest(["A3_G", "B5_D"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "-f", "json", "logs"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert "A3_G" in result.output
        assert "B5_D" in result.output

    def test_missing_experiment_and_no_args_errors(self):
        """No --file, no --experiment, no labels → error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["logs"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0


class TestLogsTailErrors:
    def test_tail_stderr_is_reported(self, tmp_path: Path):
        log = tmp_path / "test.log"
        log.write_text("content\n")
        completed = MagicMock(stdout="", stderr="tail failed\n", returncode=2)
        with patch("gigaevo.cli.logs.subprocess.run", return_value=completed):
            result = CliRunner().invoke(
                main,
                ["logs", "--file", str(log)],
                catch_exceptions=False,
            )
        assert result.exit_code == 2
        assert "tail failed" in result.output

    def test_negative_tail_is_rejected(self, tmp_path: Path):
        log = tmp_path / "test.log"
        log.write_text("content\n")
        result = CliRunner().invoke(
            main,
            ["logs", "--file", str(log), "--tail", "-1"],
            catch_exceptions=False,
        )
        assert result.exit_code == 2
