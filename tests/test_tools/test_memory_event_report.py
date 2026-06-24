from __future__ import annotations

import json
import subprocess
import sys

from tools.memory_event_report import build_report, format_report


def _append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _event(event_type: str, payload: dict, *, decision_id: str = "d1") -> dict:
    return {
        "schema_version": "memory_event.v1",
        "event_id": f"event-{event_type}-{decision_id}",
        "timestamp_utc": "2026-06-19T00:00:00+00:00",
        "component": event_type.split(".", 1)[0],
        "event_type": event_type,
        "decision_id": decision_id,
        "program_id": "p1",
        "parent_ids": ["p1"],
        "payload": payload,
    }


def _write_fixture(run_dir):
    memory_dir = run_dir / "memory"
    _append_jsonl(
        memory_dir / "memory_events.jsonl",
        [
            _event(
                "read.request",
                {"max_cards": 1, "exclude_count": 0},
            ),
            _event(
                "read.retrieval",
                {"duration_ms": 12.5, "raw_memory": {"type": "dict"}},
            ),
            _event(
                "read.selection",
                {
                    "candidate_ids": ["idea-a", "program-x"],
                    "candidate_count": 2,
                    "fetched_ids": ["idea-a", "program-x"],
                    "missing_ids": [],
                    "auction_winner_ids": ["idea-a", "program-x"],
                    "budgeted_ids": ["idea-a"],
                    "render_dropped_ids": [],
                    "selected_ids": ["idea-a"],
                    "selected_count": 1,
                    "empty_reason": "",
                    "timing_ms": {"total": 20.0},
                    "slate": [
                        {"card_id": "idea-a", "selected": True},
                        {"card_id": "program-x", "selected": True},
                    ],
                },
            ),
            _event(
                "auction.run",
                {
                    "candidate_count": 2,
                    "winner_count": 2,
                    "winner_ids": ["idea-a", "program-x"],
                },
            ),
            _event(
                "budget.cap",
                {
                    "winner_count": 2,
                    "max_cards": 1,
                    "kept_ids": ["idea-a"],
                    "dropped_ids": ["program-x"],
                },
            ),
            _event(
                "read.selection",
                {
                    "candidate_ids": ["idea-b"],
                    "candidate_count": 1,
                    "fetched_ids": ["idea-b"],
                    "missing_ids": [],
                    "auction_winner_ids": [],
                    "budgeted_ids": [],
                    "render_dropped_ids": [],
                    "selected_ids": [],
                    "selected_count": 0,
                    "empty_reason": "auction_rejected",
                    "timing_ms": {"total": 30.0},
                    "slate": [{"card_id": "idea-b", "selected": False}],
                },
                decision_id="d2",
            ),
            _event(
                "write.ingest",
                {
                    "incoming_id": "idea-a",
                    "final_id": "idea-a",
                    "outcome": "added",
                    "category": "general",
                },
            ),
            _event(
                "write.sweep",
                {
                    "incoming_id": "idea-b",
                    "final_id": "idea-b",
                    "outcome": "evicted",
                    "category": "general",
                },
            ),
            _event(
                "store.research",
                {"mode": "gam", "duration_ms": 18.0},
            ),
            _event(
                "store.rebuild",
                {"outcome": "rebuilt", "duration_ms": 42.0},
            ),
            _event(
                "gam.plan",
                {
                    "outcome": "ok",
                    "filtered_tools": ["keyword"],
                    "duration_ms": 2.0,
                },
            ),
            _event(
                "gam.search.tool",
                {
                    "mode": "no_integrate",
                    "tool": "keyword",
                    "hit_count": 3,
                },
            ),
            _event(
                "gam.search",
                {
                    "outcome": "ideas",
                    "mode": "no_integrate",
                    "selected_tools": ["keyword"],
                    "idea_count": 2,
                    "duration_ms": 4.0,
                },
            ),
            _event(
                "gam.reflection",
                {
                    "outcome": "ok",
                    "mode": "final",
                    "top_idea_ids": ["idea-a"],
                    "duration_ms": 6.0,
                },
            ),
            _event(
                "injection_posterior.compute",
                {
                    "card_count": 2,
                    "scorable_child_count": 5,
                    "confident_count": 1,
                    "epsilon": 0.01,
                },
                decision_id="",
            ),
        ],
    )
    _append_jsonl(
        memory_dir / "write_ledger.jsonl",
        [
            {
                "incoming_id": "idea-a",
                "final_id": "idea-a",
                "outcome": "added",
                "category": "general",
            },
            {
                "incoming_id": "program-x",
                "final_id": "program-x",
                "outcome": "merged",
                "category": "program",
            },
            {
                "incoming_id": "idea-b",
                "final_id": "",
                "outcome": "rejected_harm",
                "category": "general",
            },
        ],
    )
    _append_jsonl(
        memory_dir / "amem_exports" / "amem_memories.jsonl",
        [
            {
                "id": "idea-a",
                "category": "general",
                "evolution_statistics": {
                    "ALL": {
                        "posterior_a": 4.0,
                        "posterior_b": 1.0,
                        "intro_events": 3,
                        "efficacy_confident": True,
                    }
                },
            },
            {
                "id": "program-x",
                "program_id": "x",
                "category": "program",
                "evolution_statistics": {
                    "ALL": {
                        "posterior_a": 2.0,
                        "posterior_b": 2.0,
                        "intro_events": 7,
                    }
                },
            },
        ],
    )
    return memory_dir


def test_build_report_summarizes_memory_events_and_artifacts(tmp_path) -> None:
    _write_fixture(tmp_path)

    summary = build_report(tmp_path, top_n=5)

    assert summary["files"]["checkpoint_dir"] == str(tmp_path / "memory")
    assert summary["events"]["by_component"]["write"] == 2
    assert summary["read"]["decisions"] == 2
    assert summary["read"]["request_events"] == 1
    assert summary["read"]["retrieval_events"] == 1
    assert summary["read"]["avg_retrieval_ms"] == 12.5
    assert summary["read"]["avg_total_ms"] == 25.0
    assert summary["read"]["selected_decisions"] == 1
    assert summary["read"]["empty_reasons"] == {"selected": 1, "auction_rejected": 1}
    assert summary["read"]["empty_after_candidates"] == 1
    assert summary["read"]["top_selected"] == [{"id": "idea-a", "count": 1}]
    assert summary["auction"]["slate_total"] == 3
    assert summary["auction"]["slate_rejected"] == 1
    assert summary["budget"]["top_dropped"] == [{"id": "program-x", "count": 1}]
    assert summary["card_types"]["candidate"] == {"idea": 2, "program": 1}
    assert summary["card_types"]["selected"] == {"idea": 1}
    assert summary["write_ledger"]["outcomes"] == {
        "added": 1,
        "merged": 1,
        "rejected_harm": 1,
    }
    assert summary["write_events"]["events"] == 2
    assert summary["write_events"]["outcomes"] == {"added": 1, "evicted": 1}
    assert summary["store_events"]["events"] == 2
    assert summary["store_events"]["by_type"] == {
        "store.research": 1,
        "store.rebuild": 1,
    }
    assert summary["store_events"]["modes"] == {"gam": 1}
    assert summary["store_events"]["outcomes"] == {"rebuilt": 1}
    assert summary["gam_events"]["events"] == 4
    assert summary["gam_events"]["by_type"] == {
        "gam.plan": 1,
        "gam.search.tool": 1,
        "gam.search": 1,
        "gam.reflection": 1,
    }
    assert summary["gam_events"]["outcomes"] == {"ok": 2, "ideas": 1}
    assert summary["gam_events"]["modes"] == {"no_integrate": 2, "final": 1}
    assert summary["gam_events"]["tools"] == {"keyword": 3}
    assert summary["gam_events"]["avg_duration_ms"] == 4.0
    assert summary["gam_events"]["max_duration_ms"] == 6.0
    assert summary["bank"]["cards"] == 2
    assert summary["bank"]["posterior_cards"] == 2
    assert summary["bank"]["confident_cards"] == 1
    assert summary["bank"]["intro_events_median"] == 5
    assert summary["posterior_bridge"]["last_card_count"] == 2


def test_format_report_contains_debug_sections(tmp_path) -> None:
    _write_fixture(tmp_path)
    text = format_report(build_report(tmp_path, top_n=5))

    assert "Read Decisions" in text
    assert "auction_rejected: 1" in text
    assert "Top Selected Cards" in text
    assert "program-x: 1" in text
    assert "Write Ledger" in text
    assert "Write Events" in text
    assert "Store Events" in text
    assert "store.rebuild: 1" in text
    assert "GAM Events" in text
    assert "gam.reflection: 1" in text
    assert "by component:" in text
    assert "Exported Bank" in text


def test_cli_json_output(tmp_path) -> None:
    _write_fixture(tmp_path)

    result = subprocess.run(
        [sys.executable, "tools/memory_event_report.py", str(tmp_path), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["read"]["decisions"] == 2
    assert payload["files"]["events_exists"] is True
