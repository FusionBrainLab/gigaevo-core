from __future__ import annotations

import logging

import pytest

from experiments.hover.diff_memory.reconcile_assignments import (
    assert_aa_balance,
    assert_probe_dr_baselines,
    assert_probe_itt_calibration,
    estimate_probe_itt_dr,
    neutral_arm,
    reconcile_rows,
)
from gigaevo.memory.cards import AssignmentRecord, DecisionContext


def _assignment_row(
    decision_id: str,
    *,
    probe_arm: str = "none",
    probe_propensity: float | None = None,
    include_dr_baseline: bool = False,
) -> dict:
    offered_id = f"offered-{decision_id}"
    propensities = {} if probe_propensity is None else {offered_id: probe_propensity}
    predicted_help = {offered_id: 0.6} if include_dr_baseline else {}
    predicted_gain = {offered_id: 0.8} if include_dr_baseline else {}
    predicted_no_card_gain = {offered_id: 0.2} if include_dr_baseline else {}
    q_hat_treated = 0.8 if include_dr_baseline else None
    q_hat_control = 0.2 if include_dr_baseline else None
    assigned_ids = (offered_id,) if probe_arm == "treated" else ()
    assignment = AssignmentRecord(
        decision_id=decision_id,
        policy_version="TestPolicy:v1",
        task_key="hover",
        assigned_ids=assigned_ids,
        arm="injected" if assigned_ids else "none",
        probe_arm=probe_arm,
        randomized=probe_arm != "none",
        propensity_kind=("probe_bernoulli" if probe_arm != "none" else "observational"),
        propensities=propensities,
        predicted_help=predicted_help,
        predicted_gain=predicted_gain,
        predicted_no_card_gain=predicted_no_card_gain,
        q_hat_treated=q_hat_treated,
        q_hat_control=q_hat_control,
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


def _set_q_hats(row: dict, q1: float, q0: float) -> None:
    decision_id = row["assignment"]["decision_id"]
    offered_id = f"offered-{decision_id}"
    row["assignment"]["predicted_gain"] = {offered_id: q1}
    row["assignment"]["predicted_no_card_gain"] = {offered_id: q0}
    row["assignment"]["q_hat_treated"] = q1
    row["assignment"]["q_hat_control"] = q0


def _probe_rows(
    *,
    n_per_arm: int,
    tau: float,
    q_hat: str = "true",
    propensity: float = 0.5,
) -> list[dict]:
    rows: list[dict] = []
    for arm in ("treated", "control"):
        for index in range(n_per_arm):
            decision_id = f"{arm}-{index}"
            base = float(index % 5) / 10.0
            row = _assignment_row(
                decision_id,
                probe_arm=arm,
                probe_propensity=propensity,
                include_dr_baseline=True,
            )
            if q_hat == "true":
                _set_q_hats(row, base + tau, base)
            elif q_hat == "wrong":
                _set_q_hats(row, 100.0 + base, -100.0 + base)
            else:
                raise ValueError(f"unsupported q_hat={q_hat!r}")
            outcome = base + tau if arm == "treated" else base
            rows.extend((row, _outcome_row(decision_id, outcome)))
    return rows


def test_reconciler_flags_orphan() -> None:
    result = reconcile_rows([_assignment_row("orphan")])

    assert result.orphans == ("orphan",)
    assert result.dupes == {}
    assert result.has_errors


def test_reconciler_flags_duplicate_terminals() -> None:
    result = reconcile_rows(
        [
            _assignment_row("dupe"),
            _outcome_row("dupe", 0.1),
            _outcome_row("dupe", 0.2),
        ]
    )

    assert result.orphans == ()
    assert len(result.dupes["dupe"]) == 2
    assert result.has_errors


def test_aa_split_uses_only_neutral_decision_hash() -> None:
    ids_by_arm: dict[str, list[str]] = {"a": [], "b": []}
    index = 0
    while not ids_by_arm["a"] or not ids_by_arm["b"]:
        decision_id = f"decision-{index}"
        ids_by_arm[neutral_arm(decision_id)].append(decision_id)
        index += 1
    rows = []
    for decision_id in (ids_by_arm["a"][0], ids_by_arm["b"][0]):
        rows.extend((_assignment_row(decision_id), _outcome_row(decision_id, 1.0)))

    summary = assert_aa_balance(reconcile_rows(rows), tolerance=0.0)

    assert summary.difference == 0.0
    assert (summary.n_a, summary.n_b) == (1, 1)


def test_reconciler_consumes_probe_arms_for_itt_calibration() -> None:
    configured_rate = 0.4
    rows = [
        _assignment_row(
            f"treated-{index}",
            probe_arm="treated",
            probe_propensity=configured_rate,
        )
        for index in range(40)
    ]
    rows.extend(
        _assignment_row(
            f"control-{index}",
            probe_arm="control",
            probe_propensity=configured_rate,
        )
        for index in range(60)
    )
    rows.extend(_assignment_row(f"observational-{index}") for index in range(7))

    summary = assert_probe_itt_calibration(reconcile_rows(rows), tolerance=0.001)

    assert (summary.n_treated, summary.n_control, summary.n_observational) == (
        40,
        60,
        7,
    )
    assert summary.realized_treated_fraction == configured_rate
    assert summary.mean_recorded_propensity == configured_rate
    assert summary.difference == 0.0


def test_reconciler_validates_probe_dr_baselines_and_computes_aipw() -> None:
    rows = [
        _assignment_row(
            "treated",
            probe_arm="treated",
            probe_propensity=0.5,
            include_dr_baseline=True,
        ),
        _outcome_row("treated", 1.0),
        _assignment_row(
            "control",
            probe_arm="control",
            probe_propensity=0.5,
            include_dr_baseline=True,
        ),
        _outcome_row("control", 0.0),
    ]

    summary = assert_probe_dr_baselines(reconcile_rows(rows))

    assert summary.n_randomized == 2
    assert summary.n_with_outcome == 2
    assert summary.n_aipw == 2
    assert summary.dr_probe_effect == pytest.approx(1.0)


def test_reconciler_rejects_probe_decision_missing_q_hat() -> None:
    row = _assignment_row(
        "missing",
        probe_arm="control",
        probe_propensity=0.5,
        include_dr_baseline=True,
    )
    row["assignment"]["predicted_help"] = {}

    with pytest.raises(AssertionError, match="predicted_help keys"):
        assert_probe_dr_baselines(reconcile_rows([row]))


def test_probe_dr_itt_recovers_planted_effect_with_ci() -> None:
    tau = 0.75
    summary = estimate_probe_itt_dr(reconcile_rows(_probe_rows(n_per_arm=80, tau=tau)))

    assert summary.n_treated == 80
    assert summary.n_control == 80
    assert summary.tau_dr == pytest.approx(tau)
    assert summary.ci[0] <= tau <= summary.ci[1]
    assert not summary.low_power


def test_probe_dr_itt_matches_ips_with_true_q_and_survives_wrong_q() -> None:
    tau = 2.0
    true_summary = estimate_probe_itt_dr(
        reconcile_rows(_probe_rows(n_per_arm=40, tau=tau, q_hat="true"))
    )
    wrong_summary = estimate_probe_itt_dr(
        reconcile_rows(_probe_rows(n_per_arm=40, tau=tau, q_hat="wrong"))
    )

    assert true_summary.tau_dr == pytest.approx(true_summary.tau_ips)
    assert true_summary.tau_dr == pytest.approx(tau)
    assert wrong_summary.tau_dr == pytest.approx(tau)


def test_probe_dr_itt_counts_missing_q_hat_ips_fallback() -> None:
    rows = _probe_rows(n_per_arm=30, tau=1.0)
    rows[0]["assignment"]["q_hat_treated"] = None

    summary = estimate_probe_itt_dr(reconcile_rows(rows))

    assert summary.n_ips_fallback == 1


def test_probe_dr_itt_logs_propensity_clipping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = _probe_rows(n_per_arm=1, tau=1.0)
    rows[0]["assignment"]["propensities"] = {"offered-treated-0": 0.0}

    with caplog.at_level(logging.WARNING):
        summary = estimate_probe_itt_dr(reconcile_rows(rows), eps=0.01)

    assert summary.clipped == 1
    assert "clipped 1 propensity denominator" in caplog.text


def test_probe_dr_itt_logs_low_power_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        summary = estimate_probe_itt_dr(
            reconcile_rows(_probe_rows(n_per_arm=1, tau=1.0))
        )

    assert summary.low_power
    assert "low power" in caplog.text


def test_control_arm_terminal_outcome_prefers_child_y_over_card_credit() -> None:
    control = _assignment_row(
        "control-y",
        probe_arm="control",
        probe_propensity=0.5,
        include_dr_baseline=True,
    )
    child_outcome = {
        "event": "PROGRAM_EVALUATED",
        "metadata": {"memory_assignment_decision_id": "control-y"},
        "fitness_delta": 1.25,
    }
    spurious_card_credit = {
        "event": "MEMORY_CARD_GAIN",
        "decision_id": "control-y",
        "card_id": "offered-control-y",
        "gain": 0.0,
        "unused": True,
    }

    result = reconcile_rows([control, spurious_card_credit, child_outcome])

    assert result.reconciled_ids == ("control-y",)
    (terminal,) = result.terminals["control-y"]
    assert terminal.outcome == 1.25
    assert terminal.source == "<memory>:?"
