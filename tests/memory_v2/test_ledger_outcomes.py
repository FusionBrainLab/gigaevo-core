from __future__ import annotations

import os
import sqlite3
import stat
from unittest.mock import AsyncMock

import numpy as np
import pytest

from gigaevo.evolution.engine.mutation import MutationFailure, generate_one_mutation
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORES_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
)
from gigaevo.memory.cards import AssignmentRecord, DecisionContext
from gigaevo.memory.events import MemoryOutcome
from gigaevo.memory.outcomes import (
    record_memory_attempt_failure,
    record_program_memory_outcome,
)
from gigaevo.memory_v2.ledger import CausalLedgerConflict, SqliteCausalLedger
from gigaevo.memory_v2.models import (
    CardSnapshot,
    EnvironmentFingerprint,
    EvolutionContext,
    OutcomeMeasurement,
    TerminalOutcome,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.paired import (
    PER_SAMPLE_SCORES_KEY,
    PER_SAMPLE_SIGNATURE_KEY,
)
from gigaevo.programs.program import Program

from .factories import decision_record


def metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            )
        }
    )


def test_sqlite_ledger_counts_control_as_pending_and_freezes_terminal(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    ledger = SqliteCausalLedger(
        path=tmp_path / "causal.sqlite3", environment=environment
    )
    ledger.activate()
    record = decision_record(evolution_context, revisions[0], delivered=False)
    ledger.record_decision(record)
    ledger.record_decision(record)
    assert ledger.link_attempt_child(
        attempt_id=record.attempt_id, child_id="child", completion_ordinal=4
    )

    pending = ledger.snapshot()
    assert pending.pending_by_treatment == {revisions[0].treatment_id: 1}
    assert pending.pending_by_bank_card == {revisions[0].bank_card_id: 1}
    assert pending.observations == ()

    terminal = TerminalOutcome(
        decision_id=record.decision_id,
        child_id="child",
        base_id=evolution_context.parent_id,
        primary_metric=evolution_context.reward.primary_metric,
        higher_is_better=evolution_context.reward.higher_is_better,
        ope_eligible=True,
        status="outcome",
        measurement=OutcomeMeasurement(value=0.2, se=None, kind="scalar"),
        completion_ordinal=4,
    )
    ledger.record_terminal(terminal)
    ledger.record_terminal(terminal)
    snapshot = ledger.snapshot()
    assert snapshot.pending_by_treatment == {}
    assert len(snapshot.observations) == 1
    assert snapshot.reward_observations == snapshot.observations
    assert snapshot.lineage_outcomes[0].best_depth == 1
    assert snapshot.observations[0].treatment is False
    assert snapshot.observations[0].measurement == terminal.measurement

    with pytest.raises(CausalLedgerConflict, match="conflicting terminal"):
        ledger.record_terminal(
            terminal.model_copy(update={"status": "invalid", "measurement": None})
        )
    with pytest.raises(CausalLedgerConflict, match="conflicting decision"):
        ledger.record_decision(record.model_copy(update={"reward_q_hat_treated": 0.5}))


def test_sqlite_ledger_keeps_delayed_reward_debt_in_pending_limits(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    reward = evolution_context.reward.model_copy(
        update={
            "endpoint": "bounded_lineage_utility",
            "lineage_depth": 2,
            "lineage_opportunity_budget": 2,
        }
    )
    ledger = SqliteCausalLedger(
        path=tmp_path / "delayed-debt.sqlite3", environment=environment
    )
    ledger.activate()

    def context(parent_id: str, ordinal: int) -> EvolutionContext:
        return evolution_context.model_copy(
            update={
                "parent_id": parent_id,
                "parent_iteration": ordinal,
                "parent_generation": ordinal + 1,
                "reward": reward,
            }
        )

    def close(record, child_id: str) -> None:
        assert ledger.link_attempt_child(
            attempt_id=record.attempt_id,
            child_id=child_id,
            completion_ordinal=record.event_ordinal,
        )
        ledger.record_terminal(
            TerminalOutcome(
                decision_id=record.decision_id,
                child_id=child_id,
                base_id=record.context.parent_id,
                primary_metric="fitness",
                higher_is_better=True,
                ope_eligible=True,
                status="outcome",
                measurement=OutcomeMeasurement(value=0.0, se=None, kind="scalar"),
                completion_ordinal=record.event_ordinal,
            )
        )

    root = decision_record(
        context("parent", 0),
        revisions[0],
        ordinal=0,
        attempt_id="root",
    )
    ledger.record_decision(root)
    close(root, "root-child")

    pending = ledger.snapshot()
    assert pending.pending_by_treatment == {revisions[0].treatment_id: 1}
    assert pending.pending_by_bank_card == {revisions[0].bank_card_id: 1}
    assert pending.reward_observations == ()

    first = decision_record(
        context("other-1", 1),
        revisions[1],
        ordinal=1,
        attempt_id="first",
    )
    second = decision_record(
        context("other-2", 2),
        revisions[1],
        ordinal=2,
        attempt_id="second",
    )
    for record, child_id in ((first, "first-child"), (second, "second-child")):
        ledger.record_decision(record)
        close(record, child_id)

    matured = ledger.snapshot()
    assert revisions[0].treatment_id not in matured.pending_by_treatment
    assert revisions[0].bank_card_id not in matured.pending_by_bank_card
    assert any(
        row.decision_id == root.decision_id for row in matured.reward_observations
    )


@pytest.mark.parametrize(
    ("higher_is_better", "parent_value", "impossible_gain"),
    (
        (True, 0.9, 0.2),
        (True, 0.1, -0.2),
        (False, 0.1, 0.2),
        (False, 0.9, -0.2),
    ),
)
def test_ledger_rejects_gain_outside_parent_specific_opportunity(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
    higher_is_better: bool,
    parent_value: float,
    impossible_gain: float,
) -> None:
    context = evolution_context.model_copy(
        update={
            "parent_metrics": {
                **evolution_context.parent_metrics,
                "fitness": parent_value,
            },
            "reward": evolution_context.reward.model_copy(
                update={"higher_is_better": higher_is_better}
            ),
        }
    )
    ledger = SqliteCausalLedger(
        path=tmp_path / f"bounds-{higher_is_better}-{parent_value}.sqlite3",
        environment=environment,
    )
    ledger.activate()
    record = decision_record(context, revisions[0])
    ledger.record_decision(record)
    ledger.link_attempt_child(
        attempt_id=record.attempt_id,
        child_id="child",
        completion_ordinal=1,
    )

    with pytest.raises(CausalLedgerConflict, match="parent-specific metric bounds"):
        ledger.record_terminal(
            TerminalOutcome(
                decision_id=record.decision_id,
                child_id="child",
                base_id=context.parent_id,
                primary_metric="fitness",
                higher_is_better=higher_is_better,
                ope_eligible=True,
                status="outcome",
                measurement=OutcomeMeasurement(
                    value=impossible_gain,
                    se=None,
                    kind="scalar",
                ),
                completion_ordinal=1,
            )
        )

    assert ledger.terminals() == ()


def test_ledger_attempt_lifecycle_and_terminal_contracts(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    ledger = SqliteCausalLedger(
        path=tmp_path / "lifecycle.sqlite3", environment=environment
    )
    ledger.activate()
    unlinked = decision_record(
        evolution_context, revisions[0], ordinal=0, attempt_id="unlinked"
    )
    linked = decision_record(
        evolution_context, revisions[1], ordinal=1, attempt_id="linked"
    )
    ledger.record_decision(unlinked)
    ledger.record_decision(linked)
    assert ledger.link_attempt_child(
        attempt_id="linked", child_id="missing-child", completion_ordinal=3
    )

    assert ledger.reconcile_unlinked_attempts(completion_ordinal=4) == 1
    assert ledger.record_missing_child(
        "missing-child", failure_stage="startup_child_missing"
    )
    terminals = {row.decision_id: row for row in ledger.terminals()}
    assert terminals[unlinked.decision_id].failure_stage == (
        "startup_orphan_reconciliation"
    )
    assert terminals[linked.decision_id].child_id == "missing-child"
    assert ledger.snapshot().censored_count == 2

    with pytest.raises(CausalLedgerConflict, match="terminal base"):
        ledger.record_terminal(
            terminals[linked.decision_id].model_copy(
                update={
                    "decision_id": linked.decision_id,
                    "base_id": "wrong-parent",
                    "child_id": "different-child",
                }
            )
        )


def test_ledger_detects_payload_tampering_on_read(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    ledger = SqliteCausalLedger(
        path=tmp_path / "tamper.sqlite3", environment=environment
    )
    ledger.activate()
    record = decision_record(evolution_context, revisions[0])
    ledger.record_decision(record)
    with sqlite3.connect(ledger._database_path) as connection:
        connection.execute(
            "UPDATE decisions SET record_json = replace(record_json, '0.095', '0.099')"
        )
        connection.commit()
    with pytest.raises(CausalLedgerConflict, match="payload hash mismatch"):
        ledger.snapshot()


def test_network_ledger_uses_local_database_and_atomic_mirror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    local_root = tmp_path / "local"
    monkeypatch.setenv("TMPDIR", str(local_root))
    monkeypatch.setattr("gigaevo.memory_v2.ledger._filesystem_type", lambda _: "nfs4")
    artifact = tmp_path / "checkpoint" / "evidence.sqlite3"
    ledger = SqliteCausalLedger(path=artifact, environment=environment)
    assert not artifact.exists()
    ledger.activate()
    trajectory_id = ledger.trajectory_id
    record = decision_record(evolution_context, revisions[0])
    ledger.record_decision(record)

    assert ledger._database_path != artifact
    assert artifact.is_file()
    ledger.close()
    reopened = SqliteCausalLedger(path=artifact, environment=environment)
    reopened.activate()
    assert reopened.trajectory_id == trajectory_id
    assert reopened.decisions() == (record,)


def test_ledger_constructor_is_inert_and_activation_is_exclusive(
    tmp_path,
    environment: EnvironmentFingerprint,
) -> None:
    artifact = tmp_path / "exclusive.sqlite3"
    first = SqliteCausalLedger(path=artifact, environment=environment)
    second = SqliteCausalLedger(path=artifact, environment=environment)

    assert not artifact.exists()
    first.activate()
    published = artifact.read_bytes()
    assert second._database_path == artifact
    assert artifact.read_bytes() == published
    with pytest.raises(CausalLedgerConflict, match="already active"):
        second.activate()

    first.close()
    second.activate()
    second.close()


def test_prepublication_mirror_failure_rolls_back_unexposed_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr("gigaevo.memory_v2.ledger._filesystem_type", lambda _: "nfs4")
    artifact = tmp_path / "checkpoint" / "failure.sqlite3"
    ledger = SqliteCausalLedger(path=artifact, environment=environment)
    ledger.activate()
    record = decision_record(evolution_context, revisions[0])

    with monkeypatch.context() as failure:
        failure.setattr(
            "gigaevo.memory_v2.ledger.os.replace",
            lambda *_args: (_ for _ in ()).throw(OSError("publish failed")),
        )
        with pytest.raises(OSError, match="publish failed"):
            ledger.record_decision(record)

    assert ledger.decisions() == ()
    with sqlite3.connect(artifact) as connection:
        assert connection.execute("SELECT count(*) FROM decisions").fetchone() == (0,)
    ledger.close()


def test_directory_fsync_failure_after_replace_is_committed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr("gigaevo.memory_v2.ledger._filesystem_type", lambda _: "nfs4")
    artifact = tmp_path / "checkpoint" / "post-replace.sqlite3"
    ledger = SqliteCausalLedger(path=artifact, environment=environment)
    ledger.activate()
    record = decision_record(evolution_context, revisions[0])
    system_fsync = os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync unsupported")
        system_fsync(fd)

    with monkeypatch.context() as failure:
        failure.setattr("gigaevo.memory_v2.ledger.os.fsync", fail_directory_fsync)
        ledger.record_decision(record)

    assert ledger.decisions() == (record,)
    ledger.close()


@pytest.mark.asyncio
async def test_paired_outcome_requires_matching_ordered_cohort_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[MemoryOutcome] = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child_scores = [0.6, 0.8, 1.0, 0.8]
    base_scores = [0.4, 0.7, 0.7, 0.6]
    signature = "sample-order-v1"
    child = Program(code="def child(): return 1", iteration=9)
    child.metrics = {"is_valid": 1.0, "fitness": float(np.mean(child_scores))}
    child.set_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY, "decision-paired")
    child.set_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY, "base")
    child.set_metadata(
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
        {"is_valid": 1.0, "fitness": float(np.mean(base_scores))},
    )
    child.set_metadata(PER_SAMPLE_SCORES_KEY, child_scores)
    child.set_metadata(MUTATION_MEMORY_BASE_SCORES_METADATA_KEY, base_scores)
    child.set_metadata(PER_SAMPLE_SIGNATURE_KEY, signature)
    child.set_metadata(MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY, signature)

    await record_program_memory_outcome(
        child, storage=AsyncMock(), metrics_context=metrics_context()
    )
    paired = emitted[-1]
    expected_se = float(np.std(np.subtract(child_scores, base_scores), ddof=1) / 2)
    assert paired.measurement_kind == "paired"
    assert paired.fitness_delta_se == pytest.approx(expected_se)
    assert paired.n_pairs == 4
    assert paired.pairing_signature == signature
    assert paired.completion_ordinal == 9

    emitted.clear()
    child.metadata.pop("memory_outcome_terminal")
    child.set_metadata(PER_SAMPLE_SIGNATURE_KEY, "different-order")
    await record_program_memory_outcome(
        child, storage=AsyncMock(), metrics_context=metrics_context()
    )
    scalar = emitted[-1]
    assert scalar.measurement_kind == "scalar"
    assert scalar.fitness_delta_se is None
    assert scalar.n_pairs is None
    assert scalar.pairing_signature == ""


def test_generation_failure_closes_the_durable_decision(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    ledger = SqliteCausalLedger(
        path=tmp_path / "failure.sqlite3", environment=environment
    )
    ledger.activate()
    record = decision_record(evolution_context, revisions[0])
    ledger.record_decision(record)
    parent = Program(
        id=evolution_context.parent_id,
        code="def parent(): return 1",
        metrics=dict(evolution_context.parent_metrics),
    )
    parent.set_metadata(
        MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
        AssignmentRecord(
            decision_id=record.decision_id,
            policy_version="MemoryV2:test",
            task_key=environment.task_key,
            assigned_ids=(revisions[0].bank_card_id,),
            delivered_ids=(revisions[0].bank_card_id,),
            arm="injected",
            probe_arm="treated",
            randomized=True,
            propensity_kind="probe_bernoulli",
            propensities={revisions[0].bank_card_id: 0.6},
            context=DecisionContext(
                task_key=environment.task_key,
                parent_id=parent.id,
                parent_metrics=dict(parent.metrics),
            ),
        ).model_dump(mode="json"),
    )

    count = record_memory_attempt_failure(
        [parent],
        outcome_sink=ledger,
        metrics_context=metrics_context(),
        status="invalid",
        failure_stage="mutation_generation",
        completion_ordinal=7,
    )

    assert count == 1
    (terminal,) = ledger.terminals()
    assert terminal.status == "invalid"
    assert terminal.failure_stage == "mutation_generation"
    assert terminal.completion_ordinal == 7
    (observation,) = ledger.snapshot().observations
    assert observation.invalid


@pytest.mark.asyncio
async def test_mutation_failure_observer_distinguishes_action_and_infrastructure() -> (
    None
):
    parent = Program(code="def parent(): return 1")
    mutator = AsyncMock()
    mutator.mutate_single.return_value = None
    failures: list[MutationFailure] = []

    child_id = await generate_one_mutation(
        [parent],
        mutator=mutator,
        storage=AsyncMock(),
        state_manager=AsyncMock(),
        iteration=1,
        failure_observer=failures.append,
    )

    assert child_id is None
    assert failures == [MutationFailure(status="invalid", stage="mutation_generation")]

    failures.clear()
    mutator.mutate_single.return_value = MutationSpec(
        code="def child(): return 1",
        parents=[parent],
        name="child",
    )
    storage = AsyncMock()
    storage.add.side_effect = OSError("disk unavailable")
    child_id = await generate_one_mutation(
        [parent],
        mutator=mutator,
        storage=storage,
        state_manager=AsyncMock(),
        iteration=2,
        failure_observer=failures.append,
    )

    assert child_id is None
    assert failures == [
        MutationFailure(status="censored", stage="mutation_persistence")
    ]
