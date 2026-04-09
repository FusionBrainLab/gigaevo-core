from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from gigaevo.memory.ideas_tracker.cli import main as cli_main
from gigaevo.memory.ideas_tracker.csv_loader import load_programs_from_csv
import gigaevo.memory.ideas_tracker.run_ideas_tracker_from_csv as run_from_csv_mod
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_row(
    *,
    program_id: str = "11111111-1111-1111-1111-111111111111",
    code: str = "def solve(): return 42",
    parent_ids: list[str] | None = None,
    generation: int = 2,
    fitness: float = 0.75,
    is_valid: float = 1.0,
    mutation_output: dict | None = None,
) -> dict:
    return {
        "program_id": program_id,
        "code": code,
        "parent_ids": json.dumps(
            ["22222222-2222-2222-2222-222222222222"]
            if parent_ids is None
            else parent_ids
        ),
        "lineage_generation": generation,
        "metric_fitness": fitness,
        "metric_is_valid": is_valid,
        "metadata_mutation_output": json.dumps(
            mutation_output
            or {"archetype": "exploitation", "changes": [], "insights_used": []}
        ),
    }


class TestLoadProgramsFromCsv:
    def test_single_program_loaded(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "evolution_data.csv"
        _write_csv(csv_path, [_make_row()])
        programs = load_programs_from_csv(csv_path)
        assert len(programs) == 1
        assert isinstance(programs[0], Program)

    def test_program_id_and_code_mapped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(
            csv_path,
            [
                _make_row(
                    program_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    code="def f(): pass",
                )
            ],
        )
        prog = load_programs_from_csv(csv_path)[0]
        assert prog.id == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        assert prog.code == "def f(): pass"

    def test_lineage_parents_and_generation_mapped(self, tmp_path: Path) -> None:
        pid = "22222222-2222-2222-2222-222222222222"
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row(parent_ids=[pid], generation=5)])
        prog = load_programs_from_csv(csv_path)[0]
        assert prog.lineage.parents == [pid]
        assert prog.lineage.generation == 5

    def test_metrics_mapped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row(fitness=0.88, is_valid=1.0)])
        prog = load_programs_from_csv(csv_path)[0]
        assert prog.metrics["fitness"] == pytest.approx(0.88)
        assert prog.metrics["is_valid"] == pytest.approx(1.0)

    def test_mutation_output_in_metadata(self, tmp_path: Path) -> None:
        mo = {
            "archetype": "exploration",
            "changes": [{"description": "Used BFS"}],
            "insights_used": ["hint1"],
        }
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row(mutation_output=mo)])
        prog = load_programs_from_csv(csv_path)[0]
        assert prog.metadata["mutation_output"]["archetype"] == "exploration"
        assert prog.metadata["mutation_output"]["insights_used"] == ["hint1"]

    def test_root_program_has_no_parents(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row(parent_ids=[])])
        prog = load_programs_from_csv(csv_path)[0]
        assert prog.lineage.parents == []

    def test_multiple_programs_loaded(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        rows = []
        for i in range(5):
            row = _make_row(fitness=float(i))
            row["program_id"] = f"0000000{i}-0000-0000-0000-000000000000"
            rows.append(row)
        _write_csv(csv_path, rows)
        programs = load_programs_from_csv(csv_path)
        assert len(programs) == 5

    def test_empty_csv_returns_empty_list(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text(
            "program_id,code,parent_ids,lineage_generation\n", encoding="utf-8"
        )
        assert load_programs_from_csv(csv_path) == []

    def test_missing_is_valid_column_skipped_gracefully(self, tmp_path: Path) -> None:
        """CSV from old runs may not have metric_is_valid — should not crash."""
        row = _make_row()
        del row["metric_is_valid"]
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [row])
        programs = load_programs_from_csv(csv_path)
        assert len(programs) == 1
        assert "is_valid" not in programs[0].metrics

    def test_program_state_is_done(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row()])
        prog = load_programs_from_csv(csv_path)[0]
        assert prog.state == ProgramState.DONE

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row()])
        programs = load_programs_from_csv(str(csv_path))
        assert len(programs) == 1

    def test_negative_fitness_loaded_correctly(self, tmp_path: Path) -> None:
        """Negative fitness (sentinel value) should be preserved, not filtered."""
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row(fitness=-1e5, is_valid=0.0)])
        prog = load_programs_from_csv(csv_path)[0]
        assert prog.metrics["fitness"] == pytest.approx(-1e5)
        assert prog.metrics["is_valid"] == pytest.approx(0.0)


class TestCliCsvPath:
    def test_cli_runs_tracker_with_csv_path(self, tmp_path: Path) -> None:
        """main() with --csv-path should load programs and run the tracker."""
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row()])

        with (
            patch(
                "gigaevo.memory.ideas_tracker.ideas_tracker.IdeaTracker.run"
            ) as mock_run,
            patch(
                "gigaevo.memory.ideas_tracker.ideas_tracker._summarise_task_description",
                return_value="s",
            ),
            patch(
                "gigaevo.memory.ideas_tracker.llm._init_clients",
                return_value=(MagicMock(), MagicMock(), False),
            ),
        ):
            cli_main(["--csv-path", str(csv_path), "--no-memory-write"])

        mock_run.assert_called_once()
        programs_arg = mock_run.call_args[0][0]
        assert len(programs_arg) == 1

    def test_cli_without_csv_path_loads_from_redis(self) -> None:
        """Without --csv-path, programs are loaded from Redis and passed to tracker.run()."""
        fake_programs: list = []
        with (
            patch(
                "gigaevo.memory.ideas_tracker.ideas_tracker.IdeaTracker.run"
            ) as mock_run,
            patch(
                "gigaevo.memory.ideas_tracker.ideas_tracker._summarise_task_description",
                return_value="s",
            ),
            patch(
                "gigaevo.memory.ideas_tracker.llm._init_clients",
                return_value=(MagicMock(), MagicMock(), False),
            ),
            patch(
                "gigaevo.memory.ideas_tracker.cli.load_programs_from_redis",
                return_value=fake_programs,
            ) as mock_redis,
        ):
            cli_main(
                [
                    "--redis-db",
                    "5",
                    "--redis-prefix",
                    "chains/test",
                    "--no-memory-write",
                ]
            )

        mock_redis.assert_called_once_with(
            host="localhost", port=6379, db=5, prefix="chains/test"
        )
        mock_run.assert_called_once_with(fake_programs)


class TestRunFromCsvEntryPoint:
    def test_run_from_csv_help_shows_csv_path(self) -> None:
        """The entry point module should expose --csv-path via --help."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gigaevo.memory.ideas_tracker.run_ideas_tracker_from_csv",
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert "--csv-path" in result.stdout

    def test_run_from_csv_forwards_argv_to_main(self, tmp_path: Path) -> None:
        """run_ideas_tracker_from_csv.main() forwards args to cli.main."""
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row()])

        captured: list[str] = []

        def fake_main(argv: list[str]) -> int:
            captured.extend(argv)
            return 0

        with patch(
            "gigaevo.memory.ideas_tracker.run_ideas_tracker_from_csv._cli_main",
            fake_main,
        ):
            run_from_csv_mod.main(["--csv-path", str(csv_path)])

        assert "--csv-path" in captured
        assert str(csv_path) in captured
