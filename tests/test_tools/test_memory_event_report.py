from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from tools.memory_event_report import build_report, format_report

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + "{truncated\n",
        encoding="utf-8",
    )


def _gain(value: float, *, invalid: bool = False) -> dict:
    return {
        "context": {"parent_metrics": {"fitness": 0.1}},
        "gain": value,
        "invalid": invalid,
    }


def _event(event: str, fields: dict, *, decision_id: str = "d1") -> dict:
    return {
        "event": event,
        "timestamp_utc": "2026-07-02T00:00:00+00:00",
        "decision_id": decision_id,
        "program_id": "p1",
        "parent_ids": ["p0"],
        **fields,
    }


def _bid(card_id: str, selected: bool) -> dict:
    return {
        "card_id": card_id,
        "posterior_a": 2.0,
        "posterior_b": 1.0,
        "theta": 0.7,
        "baseline_a": 3.0,
        "baseline_b": 3.0,
        "baseline_theta": 0.4,
        "selected": selected,
        "magnitude": 0.1,
        "bid": 0.07,
    }


def _write_fixture(run_dir):
    memory_dir = run_dir / "memory"
    _append_jsonl(
        memory_dir / "memory_events.jsonl",
        [
            _event(
                "MEMORY_RESEARCH_STEP",
                {
                    "step": 1,
                    "scopes": ["desc_expl"],
                    "query_count": 2,
                    "hit_ids": ["mem-a", "program-x", "mem-a"],
                    "decision": "final",
                    "duration_ms": 12.0,
                },
            ),
            _event(
                "MEMORY_RESEARCH",
                {
                    "outcome": "ok",
                    "iterations": 1,
                    "query_chars": 80,
                    "exclude_count": 0,
                    "candidate_ids": ["mem-a", "program-x"],
                    "duration_ms": 15.0,
                },
            ),
            _event(
                "MEMORY_AUCTION_RUN",
                {
                    "auction": "thompson_ev",
                    "candidate_count": 2,
                    "winner_count": 2,
                    "winner_ids": ["mem-a", "program-x"],
                    "baseline_prior": [3.0, 3.0],
                    "prior_magnitude": 0.1,
                    "ev_floor": 0.0,
                    "bids": [_bid("mem-a", True), _bid("program-x", True)],
                },
            ),
            _event(
                "MEMORY_BUDGET_CAP",
                {
                    "rank_key": "bid",
                    "winner_count": 2,
                    "max_cards": 1,
                    "kept_ids": ["mem-a"],
                    "dropped_ids": ["program-x"],
                    "rank_by_card_id": {"mem-a": 0.07, "program-x": 0.05},
                },
            ),
            _event(
                "MEMORY_READ_SELECTION",
                {
                    "mutation_mode": "rewrite",
                    "max_cards": 1,
                    "exclude_ids": [],
                    "research_iterations": 1,
                    "candidate_ids": ["mem-a", "program-x"],
                    "auction_winner_ids": ["mem-a", "program-x"],
                    "budgeted_ids": ["mem-a"],
                    "render_dropped_ids": [],
                    "selected_ids": ["mem-a"],
                    "slate": [_bid("mem-a", True), _bid("program-x", True)],
                    "empty_reason": "",
                    "timing_ms": {"research": 15.0, "auction": 1.0, "total": 20.0},
                    "error": "",
                },
            ),
            _event(
                "MEMORY_READ_SELECTION",
                {
                    "mutation_mode": "rewrite",
                    "max_cards": 1,
                    "exclude_ids": [],
                    "research_iterations": 1,
                    "candidate_ids": ["mem-b"],
                    "auction_winner_ids": [],
                    "budgeted_ids": [],
                    "render_dropped_ids": [],
                    "selected_ids": [],
                    "slate": [_bid("mem-b", False)],
                    "empty_reason": "auction_rejected",
                    "timing_ms": {"research": 10.0, "auction": 1.0, "total": 30.0},
                    "error": "",
                },
                decision_id="d2",
            ),
            _event(
                "MEMORY_STORE_WRITE",
                {"op": "save", "outcome": "ok", "card_ids": ["mem-a"], "bank_count": 3},
            ),
            _event(
                "MEMORY_STORE_WRITE",
                {
                    "op": "merge",
                    "outcome": "ok",
                    "card_ids": ["mem-a", "mem-z"],
                    "bank_count": 2,
                },
            ),
            _event(
                "MEMORY_STORE_SYNC",
                {"op": "refresh", "outcome": "ok", "card_count": 3, "duration_ms": 5.0},
            ),
            _event(
                "MEMORY_STORE_SYNC",
                {
                    "op": "rebuild",
                    "outcome": "ok",
                    "card_count": 3,
                    "duration_ms": 42.0,
                },
            ),
            _event(
                "MEMORY_GAIN_RESTAMP",
                {
                    "credited_card_count": 2,
                    "event_count_by_card_id": {"mem-a": 3, "program-x": 5},
                },
                decision_id="",
            ),
            _event(
                "MEMORY_EVICTION_SWEEP",
                {"bank_count": 3, "evicted_ids": ["mem-z"]},
            ),
            _event(
                "MEMORY_CONSOLIDATION_PASS",
                {"outcome": "ok", "merged": 1, "failures": 0, "error": ""},
            ),
        ],
    )
    _append_jsonl(
        memory_dir / "write_ledger.jsonl",
        [
            {
                "incoming_id": "mem-a",
                "final_id": "mem-a",
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
                "incoming_id": "mem-b",
                "final_id": "",
                "outcome": "rejected_harm",
                "category": "general",
            },
        ],
    )
    (memory_dir / "cards.json").write_text(
        json.dumps(
            {
                "cards": {
                    "mem-a": {
                        "id": "mem-a",
                        "kind": "insight",
                        "description": "a",
                        "gain_events": [_gain(0.01), _gain(0.02), _gain(0.015)],
                    },
                    "mem-b": {
                        "id": "mem-b",
                        "kind": "insight",
                        "description": "b",
                    },
                    "program-x": {
                        "id": "program-x",
                        "kind": "program",
                        "program_id": "x",
                        "description": "p",
                        "gain_events": [
                            _gain(0.02),
                            _gain(-0.03),
                            _gain(0.01),
                            _gain(-0.04),
                            _gain(0.005),
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return memory_dir


def test_build_report_summarizes_memory_events_and_artifacts(tmp_path) -> None:
    _write_fixture(tmp_path)

    summary = build_report(tmp_path, top_n=5)

    assert summary["files"]["checkpoint_dir"] == str(tmp_path / "memory")
    assert summary["events"]["total"] == 14
    assert summary["events"]["invalid_json"] == 1
    assert summary["events"]["by_event"]["MEMORY_READ_SELECTION"] == 2
    assert "None" not in summary["events"]["by_event"]
    assert summary["read"]["decisions"] == 2
    assert summary["read"]["selected_decisions"] == 1
    assert summary["read"]["empty_decisions"] == 1
    assert summary["read"]["empty_reasons"] == {"selected": 1, "auction_rejected": 1}
    assert summary["read"]["empty_after_candidates"] == 1
    assert summary["read"]["candidate_total"] == 3
    assert summary["read"]["top_selected"] == [{"id": "mem-a", "count": 1}]
    assert summary["read"]["avg_total_ms"] == 25.0
    assert summary["research"]["events"] == 1
    assert summary["research"]["steps"] == 1
    assert summary["research"]["outcomes"] == {"ok": 1}
    assert summary["research"]["avg_duration_ms"] == 15.0
    assert summary["research"]["step_decisions"] == {"final": 1}
    assert summary["research"]["step_scopes"] == {"desc_expl": 1}
    assert summary["auction"]["by_auction"] == {"thompson_ev": 1}
    assert summary["auction"]["slate_total"] == 3
    assert summary["auction"]["slate_selected"] == 2
    assert summary["auction"]["slate_rejected"] == 1
    assert summary["budget"]["cap_events"] == 1
    assert summary["budget"]["top_dropped"] == [{"id": "program-x", "count": 1}]
    assert summary["card_kinds"]["candidate"] == {"insight": 2, "program": 1}
    assert summary["card_kinds"]["selected"] == {"insight": 1}
    assert summary["card_kinds"]["bank"] == {"insight": 2, "program": 1}
    assert summary["write_ledger"]["invalid_json"] == 1
    assert summary["write_ledger"]["outcomes"] == {
        "added": 1,
        "merged": 1,
        "rejected_harm": 1,
    }
    assert summary["write_ledger"]["categories"] == {"general": 2, "program": 1}
    assert summary["store"]["write_events"] == 2
    assert summary["store"]["write_ops"] == {"save:ok": 1, "merge:ok": 1}
    assert summary["store"]["last_bank_count"] == 2
    assert summary["store"]["sync_events"] == 2
    assert summary["store"]["sync_ops"] == {"refresh:ok": 1, "rebuild:ok": 1}
    assert summary["store"]["max_sync_ms"] == 42.0
    assert summary["maintenance"]["eviction_sweeps"] == 1
    assert summary["maintenance"]["top_evicted"] == [{"id": "mem-z", "count": 1}]
    assert summary["maintenance"]["consolidation_passes"] == 1
    assert summary["maintenance"]["consolidation_outcomes"] == {"ok": 1}
    assert summary["maintenance"]["consolidation_merged"] == 1
    assert summary["bank"]["cards"] == 3
    assert summary["bank"]["posterior_cards"] == 2
    assert summary["bank"]["confident_cards"] == 1
    assert summary["bank"]["intro_events_median"] == 4
    assert summary["gain_restamp"]["last_credited_card_count"] == 2
    assert summary["gain_restamp"]["last_event_count"] == 8


def test_format_report_contains_debug_sections(tmp_path) -> None:
    _write_fixture(tmp_path)
    text = format_report(build_report(tmp_path, top_n=5))

    assert "Read Decisions" in text
    assert "auction_rejected: 1" in text
    assert "Top Selected Cards" in text
    assert "mem-a: 1" in text
    assert "Research" in text
    assert "thompson_ev: 1" in text
    assert "Write Ledger" in text
    assert "save:ok: 1" in text
    assert "Maintenance" in text
    assert "Card Bank" in text
    assert "Gain Restamp" in text
    assert "by event:" in text


def test_cli_json_output(tmp_path) -> None:
    _write_fixture(tmp_path)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools" / "memory_event_report.py"),
            str(tmp_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO_ROOT,
    )

    payload = json.loads(result.stdout)
    assert payload["read"]["decisions"] == 2
    assert payload["files"]["events_exists"] is True
