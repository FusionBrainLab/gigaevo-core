from __future__ import annotations

import json

import pytest

from gigaevo.memory.cards import AssignmentRecord, DecisionContext
from gigaevo.memory.events import MemoryOpeSummary
from gigaevo.memory.ope.reporter import MemoryOpeReporter


def _assignment_row(
    decision_id: str,
    *,
    probe_arm: str,
    propensity: float,
    q1: float,
    q0: float,
) -> dict:
    offered_id = f"offered-{decision_id}"
    assigned_ids = (offered_id,) if probe_arm == "treated" else ()
    assignment = AssignmentRecord(
        decision_id=decision_id,
        policy_version="TestPolicy:v1",
        task_key="hover",
        assigned_ids=assigned_ids,
        arm="injected" if assigned_ids else "none",
        probe_arm=probe_arm,
        randomized=True,
        propensity_kind="probe_bernoulli",
        propensities={offered_id: propensity},
        predicted_help={offered_id: 0.6},
        predicted_gain={offered_id: q1},
        predicted_no_card_gain={offered_id: q0},
        q_hat_treated=q1,
        q_hat_control=q0,
        context=DecisionContext(task_key="hover"),
    )
    return {
        "event": "MEMORY_ASSIGNMENT",
        "decision_id": decision_id,
        "assignment": assignment.model_dump(mode="json"),
    }


def _outcome_row(decision_id: str, value: float) -> dict:
    return {
        "event": "MEMORY_OUTCOME",
        "decision_id": decision_id,
        "outcome_value": value,
    }


def _probe_ledger(*, n_per_arm: int, tau: float) -> list[dict]:
    rows: list[dict] = []
    for arm in ("treated", "control"):
        for index in range(n_per_arm):
            decision_id = f"{arm}-{index}"
            base = float(index % 5) / 10.0
            rows.append(
                _assignment_row(
                    decision_id,
                    probe_arm=arm,
                    propensity=0.5,
                    q1=base + tau,
                    q0=base,
                )
            )
            outcome = base + tau if arm == "treated" else base
            rows.append(_outcome_row(decision_id, outcome))
    return rows


def _write_ledger(checkpoint_dir, rows: list[dict]):
    path = checkpoint_dir / "memory_events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_reporter_writes_summary_with_recovered_tau(tmp_path) -> None:
    _write_ledger(tmp_path, _probe_ledger(n_per_arm=40, tau=0.75))

    MemoryOpeReporter(checkpoint_dir=tmp_path).refresh()

    summary = json.loads((tmp_path / "ope_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["probe_dr_itt"]["n_treated"] == 40
    assert summary["probe_dr_itt"]["n_control"] == 40
    assert summary["probe_dr_itt"]["tau_dr"] == pytest.approx(0.75)
    assert summary["reconciliation"]["orphans"] == 0


def test_reporter_degrades_without_outcomes(tmp_path) -> None:
    _write_ledger(
        tmp_path,
        [
            _assignment_row(
                "orphan", probe_arm="treated", propensity=0.5, q1=0.8, q0=0.2
            )
        ],
    )

    MemoryOpeReporter(checkpoint_dir=tmp_path).refresh()

    summary = json.loads((tmp_path / "ope_summary.json").read_text())
    assert summary["status"] == "insufficient_data"
    assert summary["probe_dr_itt"] is None
    assert summary["reconciliation"]["orphans"] == 1


def test_reporter_tolerates_torn_trailing_line(tmp_path) -> None:
    rows = _probe_ledger(n_per_arm=40, tau=0.75)
    path = tmp_path / "memory_events.jsonl"
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    # A concurrent append can leave a half-written trailing line mid-run.
    text += '{"event": "MEMORY_OUTCOME", "decision_id": "torn", "outcome'
    path.write_text(text, encoding="utf-8")

    MemoryOpeReporter(checkpoint_dir=tmp_path).refresh()

    summary = json.loads((tmp_path / "ope_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["probe_dr_itt"]["tau_dr"] == pytest.approx(0.75)


def test_reporter_surfaces_reconciliation_health(tmp_path) -> None:
    rows = _probe_ledger(n_per_arm=40, tau=0.5)
    rows.append(
        _assignment_row("orphan-x", probe_arm="treated", propensity=0.5, q1=0.8, q0=0.2)
    )
    rows.append(
        _assignment_row("dup-x", probe_arm="control", propensity=0.5, q1=0.2, q0=0.2)
    )
    rows.append(_outcome_row("dup-x", 0.1))
    rows.append(_outcome_row("dup-x", 0.2))
    _write_ledger(tmp_path, rows)

    MemoryOpeReporter(checkpoint_dir=tmp_path).refresh()

    summary = json.loads((tmp_path / "ope_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["reconciliation"]["orphans"] == 1
    assert summary["reconciliation"]["dupes"] == 1


def test_reporter_noop_when_ledger_absent(tmp_path) -> None:
    MemoryOpeReporter(checkpoint_dir=tmp_path).refresh()

    assert not (tmp_path / "ope_summary.json").exists()


def test_reporter_swallows_malformed_probe_arm(tmp_path) -> None:
    row = _assignment_row("bad", probe_arm="treated", propensity=0.5, q1=0.8, q0=0.2)
    row["assignment"]["probe_arm"] = "bogus"
    _write_ledger(tmp_path, [row, _outcome_row("bad", 1.0)])

    MemoryOpeReporter(checkpoint_dir=tmp_path).refresh()

    summary = json.loads((tmp_path / "ope_summary.json").read_text())
    assert summary["status"] == "insufficient_data"


def test_reporter_cleans_up_temp_file_when_atomic_swap_fails(
    tmp_path, monkeypatch
) -> None:
    import gigaevo.memory.ope.reporter as reporter_mod

    _write_ledger(tmp_path, _probe_ledger(n_per_arm=40, tau=0.5))

    def _boom(src, dst):
        del src, dst
        raise OSError("swap failed")

    monkeypatch.setattr(reporter_mod.os, "replace", _boom)

    # refresh() must swallow the write failure (its read-only guarantee) ...
    MemoryOpeReporter(checkpoint_dir=tmp_path).refresh()

    # ... and leave neither a partial summary nor a temp residue behind.
    assert not (tmp_path / "ope_summary.json").exists()
    residue = [p.name for p in tmp_path.iterdir() if p.name != "memory_events.jsonl"]
    assert residue == []


def test_reporter_emits_ope_summary_event(tmp_path, monkeypatch) -> None:
    captured: list[MemoryOpeSummary] = []
    import gigaevo.memory.ope.reporter as reporter_mod

    monkeypatch.setattr(reporter_mod, "emit", captured.append)
    _write_ledger(tmp_path, _probe_ledger(n_per_arm=40, tau=0.5))

    MemoryOpeReporter(checkpoint_dir=tmp_path).refresh()

    assert len(captured) == 1
    event = captured[0]
    assert isinstance(event, MemoryOpeSummary)
    assert event.status == "ok"
    assert event.n_treated == 40
    assert event.n_control == 40
    assert event.tau_dr == pytest.approx(0.5)
