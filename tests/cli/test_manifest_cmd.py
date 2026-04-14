"""Tests for manifest CLI subcommand group (get, set, update, gate, pr-description)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from gigaevo.cli import main

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_run(
    label: str = "A",
    db: int = 4,
    prefix: str = "chains/hover/static",
    pipeline: str = "standard",
    pid: int | None = 1234,
) -> MagicMock:
    run = MagicMock()
    run.label = label
    run.db = db
    run.prefix = prefix
    run.pipeline = pipeline
    run.problem_name = f"{prefix}"
    run.condition = "treatment"
    run.pid = pid
    return run


def _make_manifest(
    status: str = "preregistered",
    runs: list | None = None,
    max_generations: int = 50,
    pr_number: int | None = None,
    raw: dict[str, Any] | None = None,
) -> MagicMock:
    m = MagicMock()
    m.experiment.name = "hover/test"
    m.experiment.task = "hover"
    m.experiment.status = status
    m.experiment.max_generations = max_generations
    m.experiment.branch = "exp/hover/test"
    m.experiment.pr_number = pr_number
    m.experiment.tracking_issue = 42
    m.experiment.prereg_commit = "abc1234"
    m.runs = runs if runs is not None else [_make_run("A", 4), _make_run("B", 5)]
    m.servers = ["server1.example.com"]
    _raw = (
        raw
        if raw is not None
        else {
            "experiment": {"status": status, "name": "hover/test"},
            "launch": {"watchdog_pid": 9999, "time": "2026-01-01T00:00:00Z"},
            "config": {"stopping_rule": "stagnation_10"},
        }
    )
    m.model_dump.return_value = _raw
    m.config = _raw.get("config", {})
    m.launch = MagicMock()
    m.launch.watchdog_pid = 9999
    return m


_MANIFEST_MOD = "gigaevo.monitoring.manifest"


# ---------------------------------------------------------------------------
# get subcommand
# ---------------------------------------------------------------------------


class TestManifestGetStatus:
    def test_get_status_prints_value(self):
        """get status prints the experiment status string."""
        manifest = _make_manifest(status="preregistered")
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "get", "status"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert "preregistered" in result.output


class TestManifestGetRuns:
    def test_get_runs_shows_table(self):
        """get runs prints a table with run details."""
        manifest = _make_manifest()
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "get", "runs"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert "A" in result.output
            assert "B" in result.output

    def test_get_runs_json_format(self):
        """get runs --format json prints JSON array."""
        manifest = _make_manifest()
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "-f", "json", "manifest", "get", "runs"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert '"Label"' in result.output or '"label"' in result.output.lower()


class TestManifestGetStoppingRule:
    def test_get_stopping_rule_prints_value(self):
        """get stopping_rule prints the config stopping rule."""
        manifest = _make_manifest()
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "get", "stopping_rule"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert "stagnation_10" in result.output


class TestManifestGetMaxGenerations:
    def test_get_max_generations_prints_integer(self):
        """get max_generations prints the integer value."""
        manifest = _make_manifest(max_generations=75)
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "get", "max_generations"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert "75" in result.output


class TestManifestGetDottedPath:
    def test_get_nested_field_via_dotted_path(self):
        """get launch.watchdog_pid traverses dotted path to nested field."""
        manifest = _make_manifest()
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "get", "launch.watchdog_pid"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert "9999" in result.output

    def test_get_nonexistent_field_exits_1(self):
        """get nonexistent_field exits 1 with error message."""
        manifest = _make_manifest()
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "get", "nonexistent.field"],
                catch_exceptions=False,
            )
            assert result.exit_code == 1
            assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# gate subcommand
# ---------------------------------------------------------------------------


class TestManifestGate:
    def test_gate_passes_when_status_matches(self):
        """gate implemented exits 0 when status == implemented."""
        manifest = _make_manifest(status="implemented")
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "gate", "implemented"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert "GATE PASSED" in result.output

    def test_gate_fails_when_status_mismatches(self):
        """gate implemented exits 1 when status != implemented."""
        manifest = _make_manifest(status="preregistered")
        with patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "gate", "implemented"],
                catch_exceptions=False,
            )
            assert result.exit_code == 1
            assert "BLOCKED" in result.output


# ---------------------------------------------------------------------------
# set subcommand
# ---------------------------------------------------------------------------


class TestManifestSet:
    def test_set_status_calls_set_status(self):
        """set status running calls set_status with correct args."""
        updated_manifest = _make_manifest(status="running")
        with patch(
            f"{_MANIFEST_MOD}.set_status", return_value=updated_manifest
        ) as mock_set:
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "set", "status", "running"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_set.assert_called_once_with("hover/test", "running")

    def test_set_non_status_field_exits_1(self):
        """set non-status field exits 1 with guidance message."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-e", "hover/test", "manifest", "set", "branch", "new-branch"],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "update" in result.output.lower()


# ---------------------------------------------------------------------------
# update subcommand
# ---------------------------------------------------------------------------


class TestManifestUpdate:
    def test_update_nested_field(self):
        """update launch.watchdog_pid 12345 calls update_manifest."""
        updated_manifest = _make_manifest()
        with patch(
            f"{_MANIFEST_MOD}.update_manifest", return_value=updated_manifest
        ) as mock_update:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "-e",
                    "hover/test",
                    "manifest",
                    "update",
                    "launch.watchdog_pid",
                    "12345",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_update.assert_called_once()
            experiment_arg = mock_update.call_args[0][0]
            assert experiment_arg == "hover/test"

    def test_update_string_value(self):
        """update launch.time sets string value via updater."""
        updated_manifest = _make_manifest()
        with patch(
            f"{_MANIFEST_MOD}.update_manifest", return_value=updated_manifest
        ) as mock_update:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "-e",
                    "hover/test",
                    "manifest",
                    "update",
                    "launch.time",
                    "2026-01-01T00:00:00Z",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_update.assert_called_once()

    def test_update_int_conversion(self):
        """update converts integer-looking values to int."""
        updated_manifest = _make_manifest()
        captured_updater = {}

        def capture_updater(experiment, updater):
            raw = {"launch": {"watchdog_pid": 0}}
            updater(raw)
            captured_updater["raw"] = raw
            return updated_manifest

        with patch(f"{_MANIFEST_MOD}.update_manifest", side_effect=capture_updater):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "-e",
                    "hover/test",
                    "manifest",
                    "update",
                    "launch.watchdog_pid",
                    "12345",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert captured_updater["raw"]["launch"]["watchdog_pid"] == 12345

    def test_update_bool_conversion(self):
        """update converts true/false strings to bool."""
        updated_manifest = _make_manifest()
        captured_updater = {}

        def capture_updater(experiment, updater):
            raw = {"smoke_test": {"completed": False}}
            updater(raw)
            captured_updater["raw"] = raw
            return updated_manifest

        with patch(f"{_MANIFEST_MOD}.update_manifest", side_effect=capture_updater):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "-e",
                    "hover/test",
                    "manifest",
                    "update",
                    "smoke_test.completed",
                    "true",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert captured_updater["raw"]["smoke_test"]["completed"] is True


# ---------------------------------------------------------------------------
# pr-description subcommand
# ---------------------------------------------------------------------------


class TestManifestPrDescription:
    def test_pr_description_prints_output(self):
        """pr-description generates and prints PR description."""
        with patch(
            f"{_MANIFEST_MOD}.generate_pr_description",
            return_value="# exp: hover/test\n**Status**: preregistered\n",
        ) as mock_gen:
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "pr-description"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_gen.assert_called_once_with("hover/test")
            assert "hover/test" in result.output

    def test_pr_description_push_calls_gh(self):
        """pr-description --push calls gh pr edit."""
        manifest = _make_manifest(pr_number=42)
        with (
            patch(
                f"{_MANIFEST_MOD}.generate_pr_description",
                return_value="# desc\n",
            ),
            patch(f"{_MANIFEST_MOD}.load_manifest", return_value=manifest),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "hover/test", "manifest", "pr-description", "--push"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            assert "gh" in cmd_args
            assert "42" in [str(a) for a in cmd_args]


# ---------------------------------------------------------------------------
# Missing experiment flag
# ---------------------------------------------------------------------------


class TestManifestRequiresExperiment:
    def test_get_without_experiment_exits_1(self):
        """get without --experiment flag exits 1 with error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["manifest", "get", "status"],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "experiment" in result.output.lower()

    def test_gate_without_experiment_exits_1(self):
        """gate without --experiment flag exits 1 with error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["manifest", "gate", "implemented"],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "experiment" in result.output.lower()
