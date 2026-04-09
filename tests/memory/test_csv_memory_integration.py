"""Integration tests: CSV export → load_programs_from_csv → IdeaTracker.run().

Verifies the complete post-hoc analysis flow: programs loaded from an
evolution_data.csv file are correctly filtered, analysed, and logged.
LLM calls are patched throughout — no network traffic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from gigaevo.memory.ideas_tracker.csv_loader import load_programs_from_csv
from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker
from gigaevo.memory.ideas_tracker.models import AnalysisResult
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NS = uuid.NAMESPACE_DNS


def _uid(name: str) -> str:
    """Return a deterministic UUID string from a short human-readable name."""
    return str(uuid.uuid5(_NS, name))


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
    program_id: str = "prog-001",
    code: str = "def solve(): return 42",
    parent_ids: list[str] | None = None,
    generation: int = 2,
    fitness: float = 0.75,
    is_valid: float = 1.0,
    mutation_output: dict | None = None,
    memory_ids: list[str] | None = None,
) -> dict:
    """Build a CSV row dict with UUID-format IDs (required by Program validation)."""
    mo = mutation_output or {
        "archetype": "exploitation",
        "changes": [],
        "insights_used": [],
    }
    default_parents = [_uid("seed-001")]
    resolved_parents = (
        [_uid(p) for p in parent_ids] if parent_ids is not None else default_parents
    )
    row: dict = {
        "program_id": _uid(program_id),
        "code": code,
        "parent_ids": json.dumps(resolved_parents),
        "lineage_generation": str(generation),
        "metric_fitness": str(fitness),
        "metric_is_valid": str(is_valid),
        "metadata_mutation_output": json.dumps(mo),
    }
    if memory_ids is not None:
        row["metadata_memory_selected_idea_ids"] = json.dumps(memory_ids)
    return row


_LLM_PATCH = patch(
    "gigaevo.memory.ideas_tracker.llm._init_clients",
    return_value=(MagicMock(), MagicMock(), False),
)
_TASK_PATCH = patch(
    "gigaevo.memory.ideas_tracker.ideas_tracker._summarise_task_description",
    return_value="test task summary",
)


def _make_tracker(tmp_path: Path, **kwargs) -> IdeaTracker:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with _LLM_PATCH:
        tracker = IdeaTracker(
            logs_dir=logs_dir,
            memory_write_enabled=False,
            memory_usage_tracking_enabled=False,
            **kwargs,
        )
    # Stub out the analyzer's async methods so no real LLM calls are made
    tracker._analyzer.analyze_async = AsyncMock(return_value=AnalysisResult())
    tracker._analyzer.call_async = AsyncMock(
        side_effect=lambda step, content="": json.dumps({"keywords": [], "summary": ""})
    )
    return tracker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCsvToIdeaTrackerFlow:
    def test_evolved_programs_produce_log_files(self, tmp_path: Path) -> None:
        """Full flow: CSV → load → tracker.run() → log files created."""
        csv_path = tmp_path / "evolution_data.csv"
        _write_csv(
            csv_path,
            [
                _make_row(
                    program_id="prog-001", parent_ids=[], generation=1
                ),  # root, skipped
                _make_row(program_id="prog-002", fitness=0.8),
                _make_row(program_id="prog-003", fitness=0.9),
            ],
        )

        programs = load_programs_from_csv(csv_path)
        assert len(programs) == 3

        tracker = _make_tracker(tmp_path)
        with _TASK_PATCH:
            tracker.run(programs)

        assert len(tracker._all_records) == 2

        session_dirs = list((tmp_path / "logs").glob("*/"))
        assert len(session_dirs) == 1
        session = session_dirs[0]
        assert (session / "log.txt").exists()
        assert (session / "banks.json").exists()
        assert (session / "programs.json").exists()

    def test_root_programs_are_excluded_from_analysis(self, tmp_path: Path) -> None:
        """Programs with no parents must not reach _eligible_records."""
        csv_path = tmp_path / "data.csv"
        _write_csv(
            csv_path,
            [
                _make_row(program_id="root", parent_ids=[], generation=1),
            ],
        )
        programs = load_programs_from_csv(csv_path)
        tracker = _make_tracker(tmp_path)
        with _TASK_PATCH:
            tracker.run(programs)
        assert tracker._all_records == []

    def test_invalid_programs_excluded(self, tmp_path: Path) -> None:
        """Programs with is_valid=0.0 must not be processed."""
        csv_path = tmp_path / "data.csv"
        _write_csv(
            csv_path,
            [
                _make_row(program_id="bad", is_valid=0.0, fitness=-1e5),
            ],
        )
        programs = load_programs_from_csv(csv_path)
        tracker = _make_tracker(tmp_path)
        with _TASK_PATCH:
            tracker.run(programs)
        assert tracker._all_records == []

    def test_negative_fitness_programs_are_valid(self, tmp_path: Path) -> None:
        """Negative fitness is valid — only is_valid=0 means invalid."""
        csv_path = tmp_path / "data.csv"
        _write_csv(
            csv_path,
            [
                _make_row(program_id="p1", fitness=-0.5, is_valid=1.0),
            ],
        )
        programs = load_programs_from_csv(csv_path)
        tracker = _make_tracker(tmp_path)
        with _TASK_PATCH:
            tracker.run(programs)
        assert len(tracker._all_records) == 1

    def test_program_state_is_done_after_csv_load(self, tmp_path: Path) -> None:
        """All CSV-loaded programs must have ProgramState.DONE."""
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row()])
        programs = load_programs_from_csv(csv_path)
        assert all(p.state == ProgramState.DONE for p in programs)

    def test_duplicate_programs_not_double_counted(self, tmp_path: Path) -> None:
        """Same program seen in a second run() call must not be re-counted."""
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row(program_id="dup-001")])
        programs = load_programs_from_csv(csv_path)
        tracker = _make_tracker(tmp_path)
        # First run: program is seen and added
        with _TASK_PATCH:
            tracker.run(programs)
        assert len(tracker._all_records) == 1
        # Second run with same program: seen_ids prevents re-processing
        with _TASK_PATCH:
            tracker.run(programs)
        assert len(tracker._all_records) == 1

    def test_programs_file_contains_records(self, tmp_path: Path) -> None:
        """programs.json must contain the eligible program record."""
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [_make_row(program_id="p1", fitness=0.88)])
        programs = load_programs_from_csv(csv_path)
        tracker = _make_tracker(tmp_path)
        with _TASK_PATCH:
            tracker.run(programs)

        session = list((tmp_path / "logs").glob("*/"))[0]
        data = json.loads((session / "programs.json").read_text())
        # programs.json is [{timestamp: ..., programs: [...]}]
        assert isinstance(data, list) and len(data) == 1
        records = data[0]["programs"]
        assert len(records) == 1
        assert records[0]["id"] == _uid("p1")
