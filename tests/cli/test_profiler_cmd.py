"""Tests for the `gigaevo profiler` CLI subcommand.

The profiler parses an evolution runner log and emits two artifacts per run:

* ``<out_dir>/profile_<label>.txt`` -- plain-text pipeline summary
* ``<out_dir>/profile_<label>.html`` -- interactive Plotly dashboard

Resolution priority mirrors ``logs``:

  1. Explicit ``--file <path>`` -- bypass manifest, profile that one file.
  2. Positional run labels resolved via the manifest under ``-e/--experiment``.
  3. No labels + ``-e`` -- profile every run listed in the manifest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from gigaevo.cli import main

LOG_FIXTURE = """\
2026-05-13 00:00:00.000 INFO [mutation] Task 1: ['aaaaaaaa'] -> bbbbbbbb
2026-05-13 00:00:00.100 INFO [DAG][bbbbbbbb] Run started
2026-05-13 00:00:01.000 INFO [DagScheduler] DAG completed for bbbbbbbb
2026-05-13 00:00:01.100 INFO MultiIsland: adding program bbbbbbbb
2026-05-13 00:00:01.110 INFO bbbbbbbb successfully added to island
2026-05-13 00:00:01.500 INFO ParentRefresher: flipped 1 parents DONE->QUEUED
2026-05-13 00:00:02.000 INFO [mutation] Task 2: ['bbbbbbbb'] -> cccccccc
2026-05-13 00:00:02.100 INFO [DAG][cccccccc] Run started
2026-05-13 00:00:03.000 INFO [DagScheduler] DAG completed for cccccccc
2026-05-13 00:00:03.100 INFO [ingestor] cccccccc REJECTED
"""


def _mock_manifest(labels: list[str]):
    manifest = MagicMock()
    manifest.contract.runs = []
    for label in labels:
        run = MagicMock()
        run.label = label
        run.db = 1
        run.prefix = f"pfx_{label}"
        manifest.contract.runs.append(run)
    return manifest


class TestProfilerExplicitFile:
    def test_profiles_explicit_file_emits_txt_and_html(self, tmp_path: Path):
        log = tmp_path / "run.log"
        log.write_text(LOG_FIXTURE)
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "profiler",
                "--file",
                str(log),
                "--out-dir",
                str(out_dir),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        txt = out_dir / "profile_run.txt"
        html = out_dir / "profile_run.html"
        assert txt.exists(), f"txt not found: {sorted(out_dir.iterdir())}"
        assert html.exists(), f"html not found: {sorted(out_dir.iterdir())}"

        body = txt.read_text()
        assert "Flow Profile" in body
        assert "Programs:" in body
        assert "Refresh:" in body

        html_body = html.read_text()
        assert "<!doctype html>" in html_body.lower()
        assert "plotly" in html_body.lower()

    def test_missing_file_errors(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["profiler", "--file", str(tmp_path / "nope.log")],
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_text_only_skips_html(self, tmp_path: Path):
        log = tmp_path / "run.log"
        log.write_text(LOG_FIXTURE)
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "profiler",
                "--file",
                str(log),
                "--out-dir",
                str(out_dir),
                "--text-only",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "profile_run.txt").exists()
        assert not (out_dir / "profile_run.html").exists()

    def test_html_only_skips_txt(self, tmp_path: Path):
        log = tmp_path / "run.log"
        log.write_text(LOG_FIXTURE)
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "profiler",
                "--file",
                str(log),
                "--out-dir",
                str(out_dir),
                "--html-only",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "profile_run.html").exists()
        assert not (out_dir / "profile_run.txt").exists()


class TestProfilerByLabel:
    def test_profiles_specific_label(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_A3_G.log").write_text(LOG_FIXTURE)
        (exp_dir / "run_B5_D.log").write_text(LOG_FIXTURE)

        with patch(
            "gigaevo.cli.profiler_cmd._load_manifest",
            return_value=_mock_manifest(["A3_G", "B5_D"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "profiler", "A3_G"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        out = exp_dir / "profiler"
        assert (out / "profile_A3_G.txt").exists()
        assert (out / "profile_A3_G.html").exists()
        assert not (out / "profile_B5_D.txt").exists()

    def test_unknown_label_errors(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments" / "task" / "exp1").mkdir(parents=True)

        with patch(
            "gigaevo.cli.profiler_cmd._load_manifest",
            return_value=_mock_manifest(["A3_G"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "profiler", "BOGUS"],
                catch_exceptions=False,
            )

        assert result.exit_code != 0
        assert "BOGUS" in result.output
        assert "A3_G" in result.output

    def test_no_labels_profiles_all_runs(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "task" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_A3_G.log").write_text(LOG_FIXTURE)
        (exp_dir / "run_B5_D.log").write_text(LOG_FIXTURE)

        with patch(
            "gigaevo.cli.profiler_cmd._load_manifest",
            return_value=_mock_manifest(["A3_G", "B5_D"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "profiler"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        out = exp_dir / "profiler"
        assert (out / "profile_A3_G.txt").exists()
        assert (out / "profile_A3_G.html").exists()
        assert (out / "profile_B5_D.txt").exists()
        assert (out / "profile_B5_D.html").exists()

    def test_missing_log_file_errors(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments" / "task" / "exp1").mkdir(parents=True)
        # No run_A3_G.log on disk.

        with patch(
            "gigaevo.cli.profiler_cmd._load_manifest",
            return_value=_mock_manifest(["A3_G"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "task/exp1", "profiler", "A3_G"],
                catch_exceptions=False,
            )
        assert result.exit_code != 0
        assert "run_A3_G.log" in result.output


class TestProfilerArgValidation:
    def test_no_experiment_no_file_errors(self):
        runner = CliRunner()
        result = runner.invoke(main, ["profiler"], catch_exceptions=False)
        assert result.exit_code != 0

    def test_text_and_html_only_conflict(self, tmp_path: Path):
        log = tmp_path / "run.log"
        log.write_text(LOG_FIXTURE)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "profiler",
                "--file",
                str(log),
                "--text-only",
                "--html-only",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert (
            "text-only" in result.output.lower() or "html-only" in result.output.lower()
        )
