"""Load Program objects from an evolution_data.csv produced by tools/redis2pd.py."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState


def _parse_cell(value: Any) -> Any:
    """JSON-decode strings that look like JSON objects or arrays."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped and stripped[0] in ("{", "["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def load_programs_from_csv(path: str | Path) -> list[Program]:
    """Read a CSV produced by redis2pd and return Program objects.

    Only columns needed by IdeaTracker are reconstructed:
    program_id, code, parent_ids, lineage_generation, metric_*, metadata_*.
    """
    path = Path(path)
    programs: list[Program] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            programs.append(_row_to_program(row))
    return programs


def _row_to_program(row: dict[str, Any]) -> Program:
    program_id = str(row.get("program_id", "")).strip()
    code = str(row.get("code", "")).strip() or " "

    raw_parents = _parse_cell(row.get("parent_ids", "[]"))
    parents = [str(p) for p in raw_parents] if isinstance(raw_parents, list) else []
    try:
        generation = max(int(row.get("lineage_generation", 1)), 1)
    except (TypeError, ValueError):
        generation = 1

    metrics: dict[str, float] = {}
    for key, val in row.items():
        if key.startswith("metric_"):
            f = _to_float(val)
            if f is not None:
                metrics[key[len("metric_") :]] = f

    metadata: dict[str, Any] = {}
    for key, val in row.items():
        if key.startswith("metadata_"):
            metadata[key[len("metadata_") :]] = _parse_cell(val)

    return Program(
        id=program_id,
        code=code,
        state=ProgramState.DONE,
        lineage=Lineage(parents=parents, generation=generation),
        metrics=metrics,
        metadata=metadata,
    )
