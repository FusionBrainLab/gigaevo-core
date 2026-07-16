from __future__ import annotations

import pytest

from gigaevo.memory.cards import AssignmentRecord, DecisionContext
from gigaevo.memory.context import GlobalMemoryContext
from gigaevo.memory.ope.reconcile import (
    estimate_probe_itt_dr,
    reconcile_rows,
)
from gigaevo.memory.read.auction import AuctionBid
from gigaevo.memory.read.reader import MemoryReader


def _bid(
    card_id: str,
    *,
    magnitude: float,
    support_kind: str,
    control: bool,
    treated: bool,
    offered: bool = False,
    selected: bool = False,
    no_card_baseline: float = 0.2,
) -> AuctionBid:
    return AuctionBid(
        card_id=card_id,
        posterior_a=3.0,
        posterior_b=1.0,
        theta=0.75,
        baseline_a=3.0,
        baseline_b=3.0,
        baseline_theta=0.5,
        selected=selected,
        magnitude=magnitude,
        support_kind=support_kind,
        no_card_baseline=no_card_baseline,
        probe_offered=offered,
        probe_propensity=0.5 if offered else None,
        probe_selected=offered and selected,
        probe_control_selected=control,
        probe_treated_selected=treated,
    )


def _assignment(
    slate: tuple[AuctionBid, ...],
    assigned_ids: tuple[str, ...],
    *,
    renderable_ids: frozenset[str] | None = None,
) -> AssignmentRecord:
    reader = object.__new__(MemoryReader)
    reader._policy_version = "test-policy"  # type: ignore[attr-defined]
    reader._context_model = GlobalMemoryContext()  # type: ignore[attr-defined]
    return reader._assignment(  # type: ignore[attr-defined]
        decision_id="memsel-q",
        context=DecisionContext(task_key="test"),
        assigned_ids=assigned_ids,
        slate=slate,
        renderable_ids=renderable_ids,
        timestamp=None,
    )


def test_empty_probe_q_hat_is_no_card_level_plus_incremental_card_ev() -> None:
    assignment = _assignment(
        (
            _bid(
                "cold",
                magnitude=0.4,
                support_kind="cold_prior",
                control=False,
                treated=True,
                offered=True,
                selected=True,
            ),
        ),
        ("cold",),
    )

    assert assignment.schema_version == 2
    assert assignment.q_hat_control == pytest.approx(0.2)
    assert assignment.q_hat_treated == pytest.approx(0.2 + 0.75 * 0.4)


def test_warm_replacement_q_hats_describe_both_complete_slates() -> None:
    assignment = _assignment(
        (
            _bid(
                "retained-warm",
                magnitude=0.6,
                support_kind="ev_rewards",
                control=True,
                treated=True,
                selected=True,
            ),
            _bid(
                "replaced-warm",
                magnitude=0.4,
                support_kind="ev_rewards",
                control=True,
                treated=False,
            ),
            _bid(
                "cold",
                magnitude=0.1,
                support_kind="ev_rewards",
                control=False,
                treated=True,
                offered=True,
                selected=True,
            ),
        ),
        ("cold", "retained-warm"),
    )

    assert assignment.q_hat_control == pytest.approx(0.2 + 0.6 + 0.4)
    assert assignment.q_hat_treated == pytest.approx(0.2 + 0.6 + 0.1)


def test_q_hats_exclude_cards_dropped_by_the_renderer() -> None:
    assignment = _assignment(
        (
            _bid(
                "cold",
                magnitude=0.4,
                support_kind="cold_prior",
                control=False,
                treated=True,
                offered=True,
                selected=True,
            ),
        ),
        (),
        renderable_ids=frozenset(),
    )

    assert assignment.q_hat_control == pytest.approx(0.2)
    assert assignment.q_hat_treated == pytest.approx(0.2)


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
