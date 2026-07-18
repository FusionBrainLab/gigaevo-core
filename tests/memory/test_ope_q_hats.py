from __future__ import annotations

import pytest

from gigaevo.memory.cards import AssignmentRecord, DecisionContext
from gigaevo.memory.ope.reconcile import (
    estimate_probe_itt_dr,
    reconcile_rows,
)


def _dr_rows(*, tau: float, wrong_q: bool) -> list[dict]:
    rows = []
    for arm in ("treated", "control"):
        for index in range(40):
            decision_id = f"{arm}-{index}"
            base = float(index % 5) / 10.0
            offered = f"offered-{decision_id}"
            q0 = -100.0 + base if wrong_q else base
            q1 = 100.0 + base if wrong_q else base + tau
            assignment = AssignmentRecord(
                decision_id=decision_id,
                policy_version="test-policy",
                task_key="test",
                assigned_ids=(offered,) if arm == "treated" else (),
                delivered_ids=(offered,) if arm == "treated" else (),
                arm="injected" if arm == "treated" else "none",
                probe_arm=arm,
                randomized=True,
                propensity_kind="probe_bernoulli",
                propensities={offered: 0.5},
                ope_eligible=True,
                q_hat_control=q0,
                q_hat_treated=q1,
                # These intentionally conflict: the estimator must use action q-hats.
                predicted_gain={offered: -999.0},
                predicted_no_card_gain={offered: 999.0},
                context=DecisionContext(task_key="test"),
            )
            outcome = base + tau if arm == "treated" else base
            rows.extend(
                (
                    {
                        "event": "MEMORY_ASSIGNMENT",
                        "decision_id": decision_id,
                        "assignment": assignment.model_dump(mode="json"),
                    },
                    {
                        "event": "MEMORY_OUTCOME",
                        "decision_id": decision_id,
                        "fitness_delta": outcome,
                    },
                )
            )
    return rows


@pytest.mark.parametrize("wrong_q", [False, True])
def test_dr_recovers_planted_effect_with_exact_propensity(
    wrong_q: bool,
) -> None:
    tau = 1.75

    summary = estimate_probe_itt_dr(reconcile_rows(_dr_rows(tau=tau, wrong_q=wrong_q)))

    assert summary.tau_dr == pytest.approx(tau)
