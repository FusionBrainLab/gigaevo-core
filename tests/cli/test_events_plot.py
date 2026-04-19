"""Integration tests for the `gigaevo events plot` CLI.

The command parses a canonical-event log, buckets events by run_label, and
emits per-run + (optional) grouped plots + a summary.md that surfaces the
registry's health_question and missing-after-gen violations in plain English.

Key design points the tests pin down:
- `--group-by` takes a regex with a named capture group; runs whose label
  does not match are listed as "ungrouped" in summary.md.
- Without `--group-by`, no grouped plot is emitted.
- Event discovery is automatic — the CLI reads CANONICAL_EVENTS, not a
  hardcoded list.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from gigaevo.cli.events_cmd import events


def _event_line(name: str, **payload: object) -> str:
    payload["event"] = name
    return f"[{name}] {json.dumps(payload)}"


def _synthetic_log() -> str:
    """Three runs: A (no role), B_G (role=G), B_D (role=D)."""
    lines: list[str] = []
    for run_label in ("A", "B_G", "B_D"):
        lines.append(_event_line("GENERATION_BOUNDARY", gen=1, run_label=run_label))
        lines.append(
            _event_line(
                "STAGE_EXEC",
                stage="mutation",
                program_id="p1",
                decision="miss",
                duration_ms=42.0,
                run_label=run_label,
            )
        )
        lines.append(
            _event_line(
                "LLM_CALL",
                stage="mutation",
                endpoint="chat",
                model="gpt-oss",
                ok=True,
                latency_ms=120.0,
                run_label=run_label,
            )
        )
    return "\n".join(lines) + "\n"


class TestEventsPlotPerRun:
    def test_per_run_outputs_created(self, tmp_path: Path) -> None:
        log = tmp_path / "smoke.log"
        log.write_text(_synthetic_log())
        out = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            events,
            ["plot", "--log", str(log), "--out", str(out)],
        )
        assert result.exit_code == 0, result.output

        # Default per-run plots exist.
        assert (out / "counts_per_run.png").exists()
        assert (out / "events_over_time.png").exists()
        # Summary exists and references all 3 run labels.
        summary = (out / "summary.md").read_text()
        for label in ("A", "B_G", "B_D"):
            assert label in summary
        # No role-grouped plot when --group-by is absent.
        assert not (out / "role_totals.png").exists()

    def test_summary_includes_health_questions(self, tmp_path: Path) -> None:
        log = tmp_path / "smoke.log"
        log.write_text(_synthetic_log())
        out = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(events, ["plot", "--log", str(log), "--out", str(out)])
        assert result.exit_code == 0, result.output
        summary = (out / "summary.md").read_text()
        # At least one canonical event's health_question must appear verbatim.
        # GENERATION_BOUNDARY's is "Where are we in the run?"
        assert "Where are we in the run?" in summary


class TestEventsPlotGroupBy:
    def test_group_by_regex_emits_grouped_plot(self, tmp_path: Path) -> None:
        log = tmp_path / "smoke.log"
        log.write_text(_synthetic_log())
        out = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            events,
            [
                "plot",
                "--log",
                str(log),
                "--out",
                str(out),
                "--group-by",
                r".*_(?P<role>[GD])$",
            ],
        )
        assert result.exit_code == 0, result.output

        # With --group-by, the grouped totals plot must render.
        assert (out / "role_totals.png").exists()
        summary = (out / "summary.md").read_text()
        # Summary lists ungrouped runs (A does not match the regex).
        assert "Ungrouped" in summary or "ungrouped" in summary
        assert "A" in summary

    def test_no_group_flag_skips_grouped_plot(self, tmp_path: Path) -> None:
        log = tmp_path / "smoke.log"
        log.write_text(_synthetic_log())
        out = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            events,
            [
                "plot",
                "--log",
                str(log),
                "--out",
                str(out),
                "--group-by",
                r".*_(?P<role>[GD])$",
                "--no-group",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not (out / "role_totals.png").exists()


class TestEventsPlotFiltering:
    def test_runs_filter_limits_output(self, tmp_path: Path) -> None:
        log = tmp_path / "smoke.log"
        log.write_text(_synthetic_log())
        out = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            events,
            [
                "plot",
                "--log",
                str(log),
                "--out",
                str(out),
                "--runs",
                "A,B_G",
            ],
        )
        assert result.exit_code == 0, result.output
        summary = (out / "summary.md").read_text()
        # B_D must be excluded from the filtered summary.
        assert "B_D" not in summary

    def test_events_filter_limits_output(self, tmp_path: Path) -> None:
        log = tmp_path / "smoke.log"
        log.write_text(_synthetic_log())
        out = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            events,
            [
                "plot",
                "--log",
                str(log),
                "--out",
                str(out),
                "--events",
                "LLM_CALL",
            ],
        )
        assert result.exit_code == 0, result.output
        summary = (out / "summary.md").read_text()
        assert "LLM_CALL" in summary
        # Other events filtered out of the main breakdown.
        assert "STAGE_EXEC" not in summary
