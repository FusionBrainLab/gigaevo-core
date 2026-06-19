from __future__ import annotations

import json

from gigaevo.memory.core.events import (
    DEFAULT_MEMORY_EVENTS_FILENAME,
    emit_memory_event,
    memory_event_context,
    resolve_memory_event_path,
)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_emit_memory_event_writes_canonical_jsonl_with_context(
    tmp_path,
) -> None:
    path = tmp_path / "memory_events.jsonl"

    with memory_event_context(
        decision_id="decision-1",
        program_id="program-1",
        parent_ids=["program-1", "program-2"],
        event_path=path,
    ):
        record = emit_memory_event(
            component="UnitTest",
            event_type="unit.test",
            payload={"ids": {"b", "a"}, "bad_float": float("nan")},
        )

    rows = _rows(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "memory_event.v1"
    assert row["event_id"] == record.event_id
    assert row["component"] == "UnitTest"
    assert row["event_type"] == "unit.test"
    assert row["decision_id"] == "decision-1"
    assert row["program_id"] == "program-1"
    assert row["parent_ids"] == ["program-1", "program-2"]
    assert sorted(row["payload"]["ids"]) == ["a", "b"]
    assert row["payload"]["bad_float"] is None


def test_resolve_memory_event_path_uses_checkpoint_dir(tmp_path) -> None:
    assert (
        resolve_memory_event_path(tmp_path / "memory")
        == tmp_path / "memory" / DEFAULT_MEMORY_EVENTS_FILENAME
    )


def test_emit_without_path_does_not_require_hydra(tmp_path) -> None:
    emit_memory_event(
        component="UnitTest",
        event_type="unit.no_path",
        payload={"x": 1},
    )

    assert not list(tmp_path.iterdir())
