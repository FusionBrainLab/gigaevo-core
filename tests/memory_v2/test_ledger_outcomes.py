from __future__ import annotations

import json
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
    MUTATION_MEMORY_BASE_EVALUATION_MEASUREMENTS_METADATA_KEY,
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORES_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
)
from gigaevo.memory.cards import (
    AssignmentRecord,
    DecisionContext,
    MutationAssignmentRecord,
)
from gigaevo.memory.events import MemoryOutcome
from gigaevo.memory.outcomes import (
    MutationTopologyOutcome,
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
    canonical_digest,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.evaluation import EVALUATION_MEASUREMENTS_METADATA_KEY
from gigaevo.programs.metrics.paired import (
    PER_SAMPLE_SCORES_KEY,
    PER_SAMPLE_SIGNATURE_KEY,
)
from gigaevo.programs.program import Program

from .factories import decision_record


def _used_card_ids(record) -> tuple[str, ...]:
    if not record.delivered or record.proposed_treatment_id is None:
        return ()
    card = next(
        card
        for card in record.candidates
        if card.treatment_id == record.proposed_treatment_id
    )
    return (card.bank_card_id,)


def link_decision(
    ledger: SqliteCausalLedger,
    record,
    child_id: str,
    completion_ordinal: int,
) -> None:
    ledger.record_mutation_edge(
        parent_id=record.context.parent_id,
        child_id=child_id,
        island_id=record.context.map_elites.island_id,
        completion_ordinal=completion_ordinal,
    )
    assert ledger.link_attempt_child(
        attempt_id=record.attempt_id,
        child_id=child_id,
        completion_ordinal=completion_ordinal,
    )


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
    assert not ledger.has_attempt_decision(record.attempt_id)
    ledger.record_decision(record)
    assert ledger.has_attempt_decision(record.attempt_id)
    ledger.record_decision(record)
    link_decision(ledger, record, "child", 4)

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
        used_card_ids=_used_card_ids(record),
        measurement=OutcomeMeasurement(value=0.2, se=0.03, kind="scalar"),
        completion_ordinal=4,
    )
    ledger.record_terminal(terminal)
    ledger.record_mutation_outcome(
        MutationTopologyOutcome(
            child_id="child",
            status="outcome",
            fitness_delta=0.2,
            fitness_delta_se=0.03,
            n_pairs=None,
            measurement_kind="scalar",
            pairing_signature="",
            failure_stage="",
        )
    )
    ledger.record_archive_disposition("child", accepted=True)
    ledger.record_terminal(terminal)
    snapshot = ledger.snapshot()
    assert snapshot.pending_by_treatment == {}
    assert len(snapshot.observations) == 1
    assert snapshot.lineage_observations == ()
    assert snapshot.lineage_outcomes[0].best_depth == 1
    assert snapshot.observations[0].treatment is False
    assert snapshot.observations[0].measurement == terminal.measurement

    with pytest.raises(CausalLedgerConflict, match="conflicting terminal"):
        ledger.record_terminal(
            terminal.model_copy(update={"status": "invalid", "measurement": None})
        )
    with pytest.raises(CausalLedgerConflict, match="conflicting decision"):
        ledger.record_decision(record.model_copy(update={"reward_q_hat_treated": 0.5}))


def test_uncited_delivery_advances_model_version_and_counts_lineage(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    ledger = SqliteCausalLedger(
        path=tmp_path / "uncited.sqlite3",
        environment=environment,
    )
    ledger.activate()
    lineage_context = evolution_context.model_copy(
        update={
            "reward": evolution_context.reward.model_copy(
                update={
                    "endpoint": "bounded_lineage_utility",
                    "lineage_depth": 3,
                    "lineage_opportunity_budget": 2,
                }
            )
        }
    )
    record = decision_record(lineage_context, revisions[0], delivered=True)
    ledger.record_decision(record)
    link_decision(ledger, record, "ignored-child", 1)
    before = ledger.snapshot()

    ledger.record_terminal(
        TerminalOutcome(
            decision_id=record.decision_id,
            child_id="ignored-child",
            base_id=lineage_context.parent_id,
            primary_metric=lineage_context.reward.primary_metric,
            higher_is_better=lineage_context.reward.higher_is_better,
            ope_eligible=True,
            status="outcome",
            used_card_ids=(),
            measurement=OutcomeMeasurement(value=0.49, se=None, kind="scalar"),
            completion_ordinal=1,
        )
    )
    ignored = ledger.snapshot()

    assert ignored.version != before.version
    assert ignored.model_version != before.model_version
    assert len(ignored.observations) == 1
    assert ignored.observations[0].treatment is True
    assert ignored.observations[0].card_used is False
    assert ignored.observations[0].use_contrast == -0.5
    assert ignored.lineage_pending_by_bank_card == {revisions[0].bank_card_id: 1}

    cited_context = lineage_context.model_copy(
        update={"parent_id": "cited-parent", "parent_iteration": 2}
    )
    cited_record = decision_record(
        cited_context,
        revisions[0],
        ordinal=1,
        attempt_id="cited-attempt",
        delivered=True,
    )
    ledger.record_decision(cited_record)
    link_decision(ledger, cited_record, "cited-child", 2)
    ledger.record_terminal(
        TerminalOutcome(
            decision_id=cited_record.decision_id,
            child_id="cited-child",
            base_id=cited_context.parent_id,
            primary_metric=cited_context.reward.primary_metric,
            higher_is_better=cited_context.reward.higher_is_better,
            ope_eligible=True,
            status="outcome",
            used_card_ids=(revisions[0].bank_card_id,),
            measurement=OutcomeMeasurement(value=0.1, se=None, kind="scalar"),
            completion_ordinal=2,
        )
    )
    cited = ledger.snapshot()

    assert cited.model_version != ignored.model_version
    assert cited.observations[-1].card_used is True
    assert cited.observations[-1].use_contrast == 0.5
    assert cited.lineage_pending_by_bank_card == {revisions[0].bank_card_id: 2}


@pytest.mark.parametrize(
    ("delivered", "use_proposed_card"),
    [(False, True), (False, False), (True, False)],
)
def test_ledger_rejects_use_credit_outside_the_delivered_card(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
    delivered: bool,
    use_proposed_card: bool,
) -> None:
    ledger = SqliteCausalLedger(
        path=tmp_path / "withheld-cited.sqlite3",
        environment=environment,
    )
    ledger.activate()
    record = decision_record(evolution_context, revisions[0], delivered=delivered)
    ledger.record_decision(record)
    link_decision(ledger, record, "child", 1)

    with pytest.raises(CausalLedgerConflict, match="outside the delivered decision"):
        ledger.record_terminal(
            TerminalOutcome(
                decision_id=record.decision_id,
                child_id="child",
                base_id=evolution_context.parent_id,
                primary_metric=evolution_context.reward.primary_metric,
                higher_is_better=evolution_context.reward.higher_is_better,
                ope_eligible=True,
                status="outcome",
                used_card_ids=(
                    revisions[0].bank_card_id if use_proposed_card else "foreign-card",
                ),
                measurement=OutcomeMeasurement(value=0.1, se=None, kind="scalar"),
                completion_ordinal=1,
            )
        )


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
        link_decision(ledger, record, child_id, record.event_ordinal)
        ledger.record_terminal(
            TerminalOutcome(
                decision_id=record.decision_id,
                child_id=child_id,
                base_id=record.context.parent_id,
                primary_metric="fitness",
                higher_is_better=True,
                ope_eligible=True,
                status="outcome",
                used_card_ids=_used_card_ids(record),
                measurement=OutcomeMeasurement(value=0.0, se=None, kind="scalar"),
                completion_ordinal=record.event_ordinal,
            )
        )
        ledger.record_mutation_outcome(
            MutationTopologyOutcome(
                child_id=child_id,
                status="outcome",
                fitness_delta=0.0,
                fitness_delta_se=None,
                n_pairs=None,
                measurement_kind="scalar",
                pairing_signature="",
                failure_stage="",
            )
        )
        ledger.record_archive_disposition(child_id, accepted=True)

    root = decision_record(
        context("parent", 0),
        revisions[0],
        ordinal=0,
        attempt_id="root",
    )
    ledger.record_decision(root)
    close(root, "root-child")

    pending = ledger.snapshot()
    assert pending.pending_by_treatment == {}
    assert pending.pending_by_bank_card == {}
    assert pending.lineage_pending_by_treatment == {revisions[0].treatment_id: 1}
    assert pending.lineage_pending_by_bank_card == {revisions[0].bank_card_id: 1}
    assert pending.lineage_observations == ()

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
    assert revisions[0].treatment_id not in matured.lineage_pending_by_treatment
    assert revisions[0].bank_card_id not in matured.lineage_pending_by_bank_card
    assert any(
        row.decision_id == root.decision_id for row in matured.lineage_observations
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
    link_decision(ledger, record, "child", 1)

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
                used_card_ids=_used_card_ids(record),
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
    link_decision(ledger, linked, "missing-child", 3)

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


def test_memory_free_missing_child_is_reconcilable(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
) -> None:
    ledger = SqliteCausalLedger(
        path=tmp_path / "memory-free.sqlite3",
        environment=environment,
    )
    ledger.activate()
    ledger.record_mutation_edge(
        parent_id=evolution_context.parent_id,
        child_id="memory-free-missing",
        island_id=evolution_context.map_elites.island_id,
        completion_ordinal=3,
    )

    assert ledger.pending_child_ids() == ("memory-free-missing",)
    assert ledger.record_missing_child(
        "memory-free-missing",
        failure_stage="startup_child_missing",
    )
    assert ledger.pending_child_ids() == ()
    edge = ledger.mutation_edges()[0]
    assert edge.status == "censored"
    assert edge.failure_stage == "startup_child_missing"


def test_missing_child_closes_pending_decision_without_rewriting_terminal_topology(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    ledger = SqliteCausalLedger(
        path=tmp_path / "split-terminal-crash.sqlite3",
        environment=environment,
    )
    ledger.activate()
    record = decision_record(evolution_context, revisions[0])
    ledger.record_decision(record)
    link_decision(ledger, record, "closed-topology-child", 3)
    ledger.record_mutation_outcome(
        MutationTopologyOutcome(
            child_id="closed-topology-child",
            status="outcome",
            fitness_delta=0.1,
            fitness_delta_se=None,
            n_pairs=None,
            measurement_kind="scalar",
            pairing_signature="",
            failure_stage="",
        )
    )

    assert ledger.pending_child_ids() == ("closed-topology-child",)
    assert ledger.record_missing_child(
        "closed-topology-child",
        failure_stage="startup_child_missing",
    )

    assert ledger.pending_child_ids() == ()
    edge = ledger.mutation_edges()[0]
    assert edge.status == "outcome"
    assert edge.measurement == OutcomeMeasurement(
        value=0.1,
        se=None,
        kind="scalar",
    )
    terminal = ledger.terminals()[0]
    assert terminal.status == "censored"
    assert terminal.failure_stage == "startup_child_missing"


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


def test_ledger_reads_pre_citation_contrast_rows(
    tmp_path,
    environment: EnvironmentFingerprint,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    """Rows hashed before the citation_contrast field existed must still verify."""

    ledger = SqliteCausalLedger(
        path=tmp_path / "pre-contrast.sqlite3", environment=environment
    )
    ledger.activate()
    record = decision_record(evolution_context, revisions[0])
    ledger.record_decision(record)
    legacy_payload = record.model_dump(mode="json", exclude_computed_fields=True)
    del legacy_payload["citation_contrast"]
    with sqlite3.connect(ledger._database_path) as connection:
        connection.execute(
            "UPDATE decisions SET record_json = ?, record_hash = ?",
            (json.dumps(legacy_payload), canonical_digest(legacy_payload)),
        )
        connection.commit()

    loaded = ledger.decisions()[0]
    assert loaded.decision_id == record.decision_id
    assert loaded.citation_contrast is False

    ledger.record_decision(record)
    assert len(ledger.decisions()) == 1


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
    child.set_metadata(
        MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
        MutationAssignmentRecord(
            mutation_id=child.id,
            used_ids=(),
        ).model_dump(mode="json"),
    )
    child.set_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY, "base")
    child.set_metadata(
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
        {"is_valid": 1.0, "fitness": float(np.mean(base_scores))},
    )
    child.set_metadata(PER_SAMPLE_SCORES_KEY, child_scores)
    child.set_metadata(MUTATION_MEMORY_BASE_SCORES_METADATA_KEY, base_scores)
    child.set_metadata(PER_SAMPLE_SIGNATURE_KEY, signature)
    child.set_metadata(MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY, signature)
    child.set_metadata(
        EVALUATION_MEASUREMENTS_METADATA_KEY,
        {
            "fitness": {
                "value": child.metrics["fitness"],
                "se": 0.5,
                "method": "fallback",
            }
        },
    )
    child.set_metadata(
        MUTATION_MEMORY_BASE_EVALUATION_MEASUREMENTS_METADATA_KEY,
        {
            "fitness": {
                "value": float(np.mean(base_scores)),
                "se": 0.5,
                "method": "fallback",
            }
        },
    )

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
    assert scalar.fitness_delta_se == pytest.approx(np.hypot(0.5, 0.5))
    assert scalar.n_pairs is None
    assert scalar.pairing_signature == ""


@pytest.mark.asyncio
async def test_reported_evaluation_measurements_supply_scalar_gain_se(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[MemoryOutcome] = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child = Program(code="def child(): return 1", iteration=10)
    child.metrics = {"is_valid": 1.0, "fitness": 0.8}
    child.set_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY, "decision-reported")
    child.set_metadata(
        MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
        MutationAssignmentRecord(
            mutation_id=child.id,
            used_ids=(),
        ).model_dump(mode="json"),
    )
    child.set_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY, "base")
    child.set_metadata(
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
        {"is_valid": 1.0, "fitness": 0.6},
    )
    child.set_metadata(
        EVALUATION_MEASUREMENTS_METADATA_KEY,
        {
            "fitness": {
                "value": 0.8,
                "sample_sd": 0.12,
                "n": 4,
                "method": "cross_validation",
            }
        },
    )
    child.set_metadata(
        MUTATION_MEMORY_BASE_EVALUATION_MEASUREMENTS_METADATA_KEY,
        {
            "fitness": {
                "value": 0.6,
                "sample_sd": 0.08,
                "n": 4,
                "method": "cross_validation",
            }
        },
    )

    await record_program_memory_outcome(
        child, storage=AsyncMock(), metrics_context=metrics_context()
    )

    outcome = emitted[-1]
    assert outcome.measurement_kind == "scalar"
    assert outcome.fitness_delta == pytest.approx(0.2)
    assert outcome.fitness_delta_se == pytest.approx(
        np.hypot(0.12 / np.sqrt(4), 0.08 / np.sqrt(4))
    )
    assert outcome.n_pairs is None
    assert outcome.pairing_signature == ""


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
