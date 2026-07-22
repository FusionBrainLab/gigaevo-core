"""Additive-field backward-compat for persisted memory-v2 rows.

A new field must carry a default so a durable row whose stored JSON predates the
field still parses on re-load (the additive-field invariant; cf. the sibling
``lineage_observations`` default). The operative re-load consumer for rows that
predate this PR is the offline calibration/analysis path
(``calibration.load_calibration_trajectory`` -> ``_parse_decision``), which
parses stored decisions WITHOUT re-deriving the ``DecisionKey`` identity; the
live ``snapshot()`` path re-derives that identity and so fails a pre-PR decision
before it ever reaches its terminals. Either way an added field lacking a default
would break the parse — this guards that for the new observation counters and
``used_card_ids``.
"""

from __future__ import annotations

import json

from pydantic import ValidationError
import pytest

from gigaevo.memory_v2.models import (
    OutcomeMeasurement,
    PosteriorFitDiagnostics,
    TerminalOutcome,
    canonical_digest,
)


def _full_diagnostics_payload() -> dict:
    return PosteriorFitDiagnostics(
        evidence_count=3,
        offered_observations=5,
        used_observations=2,
        ignored_observations=3,
        reward_observations=2,
        safety_observations=1,
        reward_residual_sd=0.2,
        reward_card_effect_sd=0.25,
        reward_optimizer_method="fixed_prior",
        reward_optimizer_success=True,
        reward_optimizer_iterations=0,
        reward_hyperparameters_at_boundary=False,
        reward_residual_scale_ess=8.0,
        reward_quadrature_error=0.0,
        reward_residual_moment_error=0.0,
        reward_coefficient_mean_error=0.0,
        reward_residual_boundary_probability=0.0,
        reward_residual_upper_boundary_probability=0.0,
        safety_optimizer_method="prior",
        safety_optimizer_iterations=0,
        safety_objective=0.0,
        safety_gradient_inf=0.0,
        safety_hessian_condition=1.0,
        offer_probability_hash=canonical_digest({}),
    ).model_dump(mode="json")


def test_diagnostics_load_when_use_accounting_fields_absent():
    old = _full_diagnostics_payload()
    for field in ("offered_observations", "used_observations", "ignored_observations"):
        old.pop(field)

    loaded = PosteriorFitDiagnostics.model_validate(old)

    assert loaded.offered_observations == 0
    assert loaded.used_observations == 0
    assert (
        loaded.ignored_observations == 0
    )  # 0 == 0 + 0 satisfies the accounting contract


def test_diagnostics_are_still_strict_for_non_defaulted_fields():
    # A field this PR did NOT default must still be required — proving the row
    # above loads because of the added defaults, not because the model is lax.
    old = _full_diagnostics_payload()
    old.pop("reward_observations")

    with pytest.raises(ValidationError):
        PosteriorFitDiagnostics.model_validate(old)


def test_terminal_outcome_loads_when_used_card_ids_absent():
    payload = TerminalOutcome(
        decision_id="decision-1",
        child_id="child-1",
        base_id="parent-1",
        primary_metric="fitness",
        higher_is_better=True,
        ope_eligible=True,
        status="outcome",
        used_card_ids=("card-a",),
        measurement=OutcomeMeasurement(value=0.2, se=None, kind="scalar"),
        completion_ordinal=4,
    ).model_dump(mode="json")
    payload.pop("used_card_ids")

    # A row persisted before cited-use tracking must still parse from its stored
    # JSON; the offline calibration path re-loads such rows (the live snapshot
    # path is DecisionKey-gated and never reaches a pre-PR terminal).
    loaded = TerminalOutcome.model_validate_json(json.dumps(payload))

    assert loaded.used_card_ids == ()
    assert loaded.status == "outcome"


def test_scalar_outcome_accepts_reported_standard_error():
    measurement = OutcomeMeasurement(value=0.2, se=0.03, kind="scalar")

    assert measurement.se == 0.03
    assert measurement.n_pairs is None
    assert measurement.pairing_signature == ""


def test_scalar_outcome_still_rejects_paired_metadata():
    with pytest.raises(ValidationError, match="paired-cohort metadata"):
        OutcomeMeasurement(
            value=0.2,
            se=0.03,
            n_pairs=3,
            kind="scalar",
        )
