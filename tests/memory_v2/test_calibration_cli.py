from __future__ import annotations

import json

from click.testing import CliRunner
from pydantic import ValidationError
import pytest

from gigaevo.cli import main
from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator
from gigaevo.memory_v2.calibration import (
    _prepare_units,
    calibrate_safety_priors,
    load_calibration_trajectory,
)
from gigaevo.memory_v2.ledger import SqliteCausalLedger
from gigaevo.memory_v2.models import (
    CandidateActionProbability,
    CardSnapshot,
    DecisionRecord,
    EnvironmentFingerprint,
    EvolutionContext,
    OutcomeMeasurement,
    TerminalOutcome,
)

from .factories import decision_record, prediction


def test_environment_serializes_concrete_mutation_operator(
    environment: EnvironmentFingerprint,
) -> None:
    payload = json.loads(environment.model_dump_json(exclude_computed_fields=True))
    assert payload["mutation_operator"].endswith(".LLMMutationOperator")
    restored = EnvironmentFingerprint.model_validate(payload)
    assert restored.mutation_operator is LLMMutationOperator
    assert restored == environment

    payload["mutation_operator"] = "builtins.str"
    with pytest.raises(ValidationError, match="MutationOperator class"):
        EnvironmentFingerprint.model_validate(payload)


def _closed_calibration_ledger(
    path,
    environment: EnvironmentFingerprint,
    context: EvolutionContext,
    card: CardSnapshot,
) -> None:
    ledger = SqliteCausalLedger(path=path, environment=environment)
    ledger.activate()
    fitted: tuple[str, ...] = ()
    for ordinal, invalid in enumerate((False, True, False)):
        current_context = context.model_copy(
            update={
                "parent_id": f"parent-{ordinal}",
                "parent_iteration": ordinal + 1,
            }
        )
        record = decision_record(
            current_context,
            card,
            ordinal=ordinal,
            delivered=bool(ordinal % 2),
            attempt_id=f"attempt-{ordinal}",
            safety_gate_mode="exclude_confident_incremental_harm",
            max_treated_invalid_probability=None,
        ).model_copy(update={"fitted_observation_ids": fitted})
        ledger.record_decision(record)
        assert ledger.link_attempt_child(
            attempt_id=record.attempt_id,
            child_id=f"child-{ordinal}",
            completion_ordinal=ordinal,
        )
        ledger.record_terminal(
            TerminalOutcome(
                decision_id=record.decision_id,
                child_id=f"child-{ordinal}",
                base_id=current_context.parent_id,
                primary_metric="fitness",
                higher_is_better=True,
                ope_eligible=True,
                status="invalid" if invalid else "outcome",
                measurement=(
                    None
                    if invalid
                    else OutcomeMeasurement(value=0.1, se=None, kind="scalar")
                ),
                completion_ordinal=ordinal,
            )
        )
        fitted = (*fitted, record.decision_id)
    abstain_context = context.model_copy(
        update={"parent_id": "parent-abstain", "parent_iteration": 4}
    )
    abstention = decision_record(
        abstain_context,
        card,
        ordinal=3,
        delivered=False,
        attempt_id="attempt-abstain",
        safety_gate_mode="exclude_confident_incremental_harm",
        max_treated_invalid_probability=None,
    )
    unsafe_probability = CandidateActionProbability(
        treatment_id=card.treatment_id,
        bank_card_id=card.bank_card_id,
        proposal_probability=0.0,
        proposal_mc_se=0.0,
        offer_probability=None,
        joint_treated_probability=0.0,
        joint_control_probability=0.0,
        safe=False,
        prediction=prediction(card),
    )
    abstention_payload = abstention.model_dump(exclude_computed_fields=True)
    abstention_payload.update(
        {
            "fitted_observation_ids": fitted,
            "action_probabilities": (unsafe_probability,),
            "abstain_probability": 1.0,
            "proposed_treatment_id": None,
            "delivered": False,
            "offer_probability": None,
            "proposal_probability": None,
            "joint_action_probability": None,
            "reward_q_hat_control": None,
            "reward_q_hat_treated": None,
            "risk_q_hat_control": None,
            "risk_q_hat_treated": None,
        }
    )
    ledger.record_decision(DecisionRecord.model_validate(abstention_payload))
    ledger.close()


def test_calibrator_uses_frozen_histories_and_emits_cli_report(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    ledger_path = tmp_path / "memory_v2_selection_evidence.sqlite3"
    _closed_calibration_ledger(
        ledger_path, environment, evolution_context, revisions[0]
    )

    trajectory = load_calibration_trajectory(ledger_path)
    report = calibrate_safety_priors(
        (trajectory,),
        prior_probabilities=(0.05, 0.2),
        baseline_sds=(0.15,),
        shared_effect_means=(-0.4, 0.0),
        shared_effect_sd=0.31,
        card_effect_sd=0.72,
        min_observations=2,
    )
    group = report["groups"][0]
    assert group["eligible_proposal_outcomes"] == 3
    assert group["gate_replay_decisions"] == 4
    assert group["status"] == "provisional_single_trajectory"
    assert len(group["grid_ranking"]) == 4
    assert set(group["hydra_overrides"]) >= {
        "memory.posterior_config.safety_shared_effect_prior_sd=0.31",
        "memory.posterior_config.safety_card_effect_prior_sd=0.72",
    }
    assert {
        row["prior"]["safety_shared_effect_prior_mean"] for row in group["grid_ranking"]
    } == {-0.4, 0.0}
    assert "gate_replay" in group["best_calibrated_prior"]
    assert (
        group["best_calibrated_prior"]["gate_replay"]["all_candidates"]["candidates"]
        == 4
    )
    assert group["environment"]["mutation_operator"].endswith(".LLMMutationOperator")
    assert group["environment"]["llm"]["model_name"] == environment.llm.model_name

    result = CliRunner().invoke(
        main,
        [
            "-f",
            "json",
            "memory",
            "calibrate-safety",
            "--prior-probabilities",
            "0.05",
            "--baseline-sds",
            "0.15",
            "--min-observations",
            "2",
            str(ledger_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    cli_report = json.loads(result.output)
    assert cli_report["schema"] == "gigaevo.memory_v2.safety_calibration/v1"
    assert cli_report["groups"][0]["eligible_proposal_outcomes"] == 3
    assert {
        row["prior"]["safety_shared_effect_prior_mean"]
        for row in cli_report["groups"][0]["grid_ranking"]
    } == {-0.693147, 0.0, 0.693147}


def test_calibrator_replays_the_recorded_card_kind_contrast(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    ledger_path = tmp_path / "kind-contrast.sqlite3"
    ledger = SqliteCausalLedger(path=ledger_path, environment=environment)
    ledger.activate()
    ledger.record_decision(
        decision_record(
            evolution_context,
            revisions[0],
            card_kind_contrast=True,
        )
    )
    ledger.close()

    (unit,) = _prepare_units((load_calibration_trajectory(ledger_path),))

    assert unit.space.config.card_kind_contrast is True
    assert unit.space.kind_effect_dim == 1


def test_calibrator_does_not_pool_distinct_typed_environments(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    changed_environment = environment.model_copy(
        update={
            "llm": environment.llm.model_copy(
                update={"temperature": environment.llm.temperature + 0.1}
            )
        }
    )
    changed_context = evolution_context.model_copy(
        update={"environment": changed_environment}
    )
    first = tmp_path / "first.sqlite3"
    second = tmp_path / "second.sqlite3"
    _closed_calibration_ledger(first, environment, evolution_context, revisions[0])
    _closed_calibration_ledger(
        second, changed_environment, changed_context, revisions[0]
    )

    report = calibrate_safety_priors(
        (load_calibration_trajectory(first), load_calibration_trajectory(second)),
        prior_probabilities=(0.1,),
        baseline_sds=(0.15,),
        shared_effect_sd=0.2,
        card_effect_sd=0.6,
        min_observations=2,
    )

    assert len(report["groups"]) == 2
    assert {
        group["environment"]["llm"]["temperature"] for group in report["groups"]
    } == {environment.llm.temperature, changed_environment.llm.temperature}
