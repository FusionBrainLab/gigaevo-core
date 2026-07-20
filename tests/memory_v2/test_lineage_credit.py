from __future__ import annotations

import pytest

from gigaevo.memory_v2.credit import LineageCreditResolver
from gigaevo.memory_v2.models import (
    ArchiveDisposition,
    CardSnapshot,
    CausalObservation,
    DecisionRecord,
    EvolutionContext,
    MutationEdge,
    OutcomeMeasurement,
    TerminalOutcome,
)
from gigaevo.memory_v2.posterior import HierarchicalTerminalUtilityPosterior

from .factories import decision_record


def _context(
    template: EvolutionContext,
    *,
    parent_id: str,
    ordinal: int,
    depth: int,
    opportunities: int,
) -> EvolutionContext:
    return template.model_copy(
        update={
            "parent_id": parent_id,
            "parent_iteration": ordinal,
            "parent_generation": ordinal + 1,
            "reward": template.reward.model_copy(
                update={
                    "endpoint": (
                        "bounded_proximal_utility"
                        if depth == 1
                        else "bounded_lineage_utility"
                    ),
                    "lineage_depth": depth,
                    "lineage_opportunity_budget": opportunities,
                }
            ),
        }
    )


def _record(
    context: EvolutionContext,
    card: CardSnapshot,
    *,
    ordinal: int,
) -> DecisionRecord:
    return decision_record(
        context,
        card,
        ordinal=ordinal,
        attempt_id=f"attempt-{ordinal}",
        delivered=ordinal % 2 == 0,
    )


def _terminal(
    record: DecisionRecord,
    *,
    child_id: str,
    gain: float | None,
    status: str = "outcome",
) -> TerminalOutcome:
    return TerminalOutcome(
        decision_id=record.decision_id,
        child_id=child_id,
        base_id=record.context.parent_id,
        primary_metric=record.context.reward.primary_metric,
        higher_is_better=record.context.reward.higher_is_better,
        ope_eligible=True,
        status=status,
        used_card_ids=(
            (
                next(
                    card.bank_card_id
                    for card in record.candidates
                    if card.treatment_id == record.proposed_treatment_id
                ),
            )
            if record.delivered
            else ()
        ),
        measurement=(
            OutcomeMeasurement(value=gain, se=None, kind="scalar")
            if gain is not None
            else None
        ),
        completion_ordinal=record.event_ordinal,
    )


def _observation(
    record: DecisionRecord,
    terminal: TerminalOutcome,
) -> CausalObservation:
    card = next(
        row
        for row in record.candidates
        if row.treatment_id == record.proposed_treatment_id
    )
    assert record.offer_probability is not None
    assert record.proposal_probability is not None
    assert record.joint_action_probability is not None
    assert record.reward_q_hat_control is not None
    assert record.reward_q_hat_treated is not None
    assert record.risk_q_hat_control is not None
    assert record.risk_q_hat_treated is not None
    return CausalObservation(
        decision_id=record.decision_id,
        event_ordinal=record.event_ordinal,
        card=card,
        context=record.context,
        treatment=record.delivered,
        card_used=(record.delivered and card.bank_card_id in terminal.used_card_ids),
        offer_propensity=record.offer_probability,
        proposal_propensity=record.proposal_probability,
        joint_action_propensity=record.joint_action_probability,
        status=terminal.status,
        measurement=terminal.measurement,
        reward_q_hat_control=record.reward_q_hat_control,
        reward_q_hat_treated=record.reward_q_hat_treated,
        risk_q_hat_control=record.risk_q_hat_control,
        risk_q_hat_treated=record.risk_q_hat_treated,
    )


def _edge(
    context: EvolutionContext,
    *,
    parent_id: str,
    child_id: str,
    ordinal: int,
    gain: float | None,
    status: str = "outcome",
    accepted: bool = True,
) -> MutationEdge:
    return MutationEdge(
        parent_id=parent_id,
        child_id=child_id,
        island_id=context.map_elites.island_id,
        completion_ordinal=ordinal,
        status=status,
        measurement=(
            OutcomeMeasurement(value=gain, se=None, kind="scalar")
            if status == "outcome" and gain is not None
            else None
        ),
        archive_disposition=(
            ArchiveDisposition.ACCEPTED if accepted else ArchiveDisposition.REJECTED
        ),
        failure_stage="" if status == "outcome" else "mutation",
    )


def _resolve(
    records: tuple[DecisionRecord, ...],
    terminals: tuple[TerminalOutcome, ...],
    edges: tuple[MutationEdge, ...],
):
    terminal_map = {row.decision_id: row for row in terminals}
    immediate = {
        record.decision_id: _observation(record, terminal_map[record.decision_id])
        for record in records
        if record.decision_id in terminal_map
        and terminal_map[record.decision_id].status != "censored"
    }
    return LineageCreditResolver().resolve(
        records,
        terminal_map,
        immediate,
        edges,
    )


def test_depth_one_keeps_direct_reward_out_of_lineage_head(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    context = _context(
        evolution_context,
        parent_id=evolution_context.parent_id,
        ordinal=0,
        depth=1,
        opportunities=1,
    )
    record = _record(context, revisions[0], ordinal=0)
    measurement = OutcomeMeasurement(
        value=-0.1,
        se=0.02,
        n_pairs=4,
        kind="paired",
        pairing_signature="ordered-cohort",
    )
    terminal = _terminal(record, child_id="child", gain=0.0).model_copy(
        update={"measurement": measurement}
    )

    outcomes, lineage_rows = _resolve((record,), (terminal,), ())

    assert lineage_rows == ()
    assert outcomes[0].measurement == measurement
    assert outcomes[0].best_depth == 1


def test_memory_free_descendant_adds_only_incremental_option_value(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    context = _context(
        evolution_context,
        parent_id="parent",
        ordinal=0,
        depth=2,
        opportunities=2,
    )
    root = _record(context, revisions[0], ordinal=0)
    terminal = _terminal(root, child_id="root-child", gain=0.1)
    edges = (
        _edge(
            context,
            parent_id="parent",
            child_id="root-child",
            ordinal=0,
            gain=0.1,
        ),
        _edge(
            context,
            parent_id="root-child",
            child_id="memory-free-child",
            ordinal=1,
            gain=0.15,
        ),
        _edge(
            context,
            parent_id="other",
            child_id="closing-opportunity",
            ordinal=2,
            gain=0.0,
        ),
    )

    outcomes, lineage_rows = _resolve((root,), (terminal,), edges)

    assert outcomes[0].measurement is not None
    assert outcomes[0].measurement.value == pytest.approx(0.25)
    assert outcomes[0].residual_measurement is not None
    assert outcomes[0].residual_measurement.value == pytest.approx(0.15)
    assert outcomes[0].best_descendant_id == "memory-free-child"
    assert outcomes[0].archive_survived
    assert lineage_rows[0].measurement == outcomes[0].residual_measurement


def test_lower_is_better_lineage_uses_oriented_root_bounds(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    context = _context(
        evolution_context.model_copy(
            update={
                "parent_metrics": {
                    **evolution_context.parent_metrics,
                    "fitness": 0.8,
                },
                "reward": evolution_context.reward.model_copy(
                    update={"higher_is_better": False}
                ),
            }
        ),
        parent_id="parent",
        ordinal=0,
        depth=2,
        opportunities=2,
    )
    root = _record(context, revisions[0], ordinal=0)
    terminal = _terminal(root, child_id="root-child", gain=0.1)
    edges = (
        _edge(
            context,
            parent_id="parent",
            child_id="root-child",
            ordinal=0,
            gain=0.1,
        ),
        _edge(
            context,
            parent_id="root-child",
            child_id="lower-fitness-child",
            ordinal=1,
            gain=0.2,
        ),
        _edge(
            context,
            parent_id="other",
            child_id="closing",
            ordinal=2,
            gain=0.0,
        ),
    )

    outcomes, _ = _resolve((root,), (terminal,), edges)

    assert outcomes[0].measurement is not None
    assert outcomes[0].measurement.value == pytest.approx(0.3)
    assert outcomes[0].residual_measurement is not None
    assert outcomes[0].residual_measurement.value == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("direct_gain", "descendant_gain"),
    ((0.6, 0.0), (0.4, 0.2)),
)
def test_lineage_rejects_gain_outside_root_metric_bounds(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
    direct_gain: float,
    descendant_gain: float,
) -> None:
    context = _context(
        evolution_context,
        parent_id="parent",
        ordinal=0,
        depth=2,
        opportunities=2,
    )
    root = _record(context, revisions[0], ordinal=0)
    terminal = _terminal(root, child_id="root-child", gain=direct_gain)
    edges = (
        _edge(
            context,
            parent_id="parent",
            child_id="root-child",
            ordinal=0,
            gain=direct_gain,
        ),
        _edge(
            context,
            parent_id="root-child",
            child_id="descendant",
            ordinal=1,
            gain=descendant_gain,
        ),
        _edge(
            context,
            parent_id="other",
            child_id="closing",
            ordinal=2,
            gain=0.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="lineage gain exceeds root-specific metric bounds",
    ):
        _resolve((root,), (terminal,), edges)


def test_lineage_waits_for_fixed_opportunity_budget_and_archive_result(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    context = _context(
        evolution_context,
        parent_id="parent",
        ordinal=0,
        depth=2,
        opportunities=2,
    )
    root = _record(context, revisions[0], ordinal=0)
    terminal = _terminal(root, child_id="root-child", gain=0.1)
    root_edge = _edge(
        context,
        parent_id="parent",
        child_id="root-child",
        ordinal=0,
        gain=0.1,
    )
    first = _edge(
        context,
        parent_id="other",
        child_id="first",
        ordinal=1,
        gain=0.0,
    )

    outcomes, rows = _resolve((root,), (terminal,), (root_edge, first))
    assert outcomes[0].status == "pending"
    assert rows == ()

    unresolved = MutationEdge(
        parent_id="other",
        child_id="second",
        island_id=context.map_elites.island_id,
        completion_ordinal=2,
    )
    outcomes, _ = _resolve(
        (root,),
        (terminal,),
        (root_edge, first, unresolved),
    )
    assert outcomes[0].status == "pending"


def test_invalid_root_closes_immediately_without_lineage_debt(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    context = _context(
        evolution_context,
        parent_id="parent",
        ordinal=0,
        depth=3,
        opportunities=32,
    )
    root = _record(context, revisions[0], ordinal=0)
    terminal = _terminal(root, child_id="invalid-child", gain=None, status="invalid")

    outcomes, rows = _resolve((root,), (terminal,), ())

    assert outcomes[0].status == "invalid"
    assert rows == ()


def test_rejected_descendant_does_not_create_option_value(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    context = _context(
        evolution_context,
        parent_id="parent",
        ordinal=0,
        depth=2,
        opportunities=2,
    )
    root = _record(context, revisions[0], ordinal=0)
    terminal = _terminal(root, child_id="root-child", gain=0.1)
    edges = (
        _edge(
            context,
            parent_id="parent",
            child_id="root-child",
            ordinal=0,
            gain=0.1,
        ),
        _edge(
            context,
            parent_id="root-child",
            child_id="rejected-child",
            ordinal=1,
            gain=0.4,
            accepted=False,
        ),
        _edge(
            context,
            parent_id="other",
            child_id="closing",
            ordinal=2,
            gain=0.0,
        ),
    )

    outcomes, rows = _resolve((root,), (terminal,), edges)

    assert outcomes[0].best_descendant_id == "root-child"
    assert outcomes[0].residual_measurement is not None
    assert outcomes[0].residual_measurement.value == 0.0
    assert rows[0].measurement == outcomes[0].residual_measurement


def test_shared_breakthrough_has_one_closest_credit_owner(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    first_context = _context(
        evolution_context,
        parent_id="parent",
        ordinal=0,
        depth=3,
        opportunities=3,
    )
    second_context = _context(
        evolution_context,
        parent_id="first-child",
        ordinal=1,
        depth=2,
        opportunities=2,
    )
    first = _record(first_context, revisions[0], ordinal=0)
    second = _record(second_context, revisions[1], ordinal=1)
    terminals = (
        _terminal(first, child_id="first-child", gain=0.1),
        _terminal(second, child_id="second-child", gain=0.05),
    )
    edges = (
        _edge(
            first_context,
            parent_id="parent",
            child_id="first-child",
            ordinal=0,
            gain=0.1,
        ),
        _edge(
            first_context,
            parent_id="first-child",
            child_id="second-child",
            ordinal=1,
            gain=0.05,
        ),
        _edge(
            first_context,
            parent_id="second-child",
            child_id="breakthrough",
            ordinal=2,
            gain=0.2,
        ),
        _edge(
            first_context,
            parent_id="other",
            child_id="closing",
            ordinal=3,
            gain=0.0,
        ),
    )

    outcomes, rows = _resolve((first, second), terminals, edges)
    by_id = {row.decision_id: row for row in outcomes}
    residual_by_id = {
        row.decision_id: row.measurement.value
        for row in rows
        if row.measurement is not None
    }

    assert by_id[first.decision_id].best_descendant_id == "breakthrough"
    assert by_id[second.decision_id].best_descendant_id == "breakthrough"
    assert by_id[first.decision_id].credit_owner_decision_id == second.decision_id
    assert residual_by_id[first.decision_id] == 0.0
    assert residual_by_id[second.decision_id] > 0.0


def test_uncited_root_still_owns_closest_breakthrough_credit(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    first_context = _context(
        evolution_context,
        parent_id="parent",
        ordinal=0,
        depth=3,
        opportunities=3,
    )
    second_context = _context(
        evolution_context,
        parent_id="first-child",
        ordinal=1,
        depth=2,
        opportunities=2,
    )
    first = _record(first_context, revisions[0], ordinal=0)
    second = decision_record(
        second_context,
        revisions[1],
        ordinal=1,
        attempt_id="ignored-attempt",
        delivered=True,
    )
    terminals = (
        _terminal(first, child_id="first-child", gain=0.1),
        _terminal(second, child_id="second-child", gain=0.05).model_copy(
            update={"used_card_ids": ()}
        ),
    )
    edges = (
        _edge(
            first_context,
            parent_id="parent",
            child_id="first-child",
            ordinal=0,
            gain=0.1,
        ),
        _edge(
            first_context,
            parent_id="first-child",
            child_id="second-child",
            ordinal=1,
            gain=0.05,
        ),
        _edge(
            first_context,
            parent_id="second-child",
            child_id="breakthrough",
            ordinal=2,
            gain=0.2,
        ),
        _edge(
            first_context,
            parent_id="other",
            child_id="closing",
            ordinal=3,
            gain=0.0,
        ),
    )

    outcomes, rows = _resolve((first, second), terminals, edges)
    by_id = {row.decision_id: row for row in outcomes}
    residual_by_id = {
        row.decision_id: row.measurement.value
        for row in rows
        if row.measurement is not None
    }
    use_contrast_by_id = {row.decision_id: row.use_contrast for row in rows}

    assert use_contrast_by_id == {
        first.decision_id: 0.5,
        second.decision_id: -0.5,
    }
    assert by_id[first.decision_id].credit_owner_decision_id == second.decision_id
    assert by_id[second.decision_id].credit_owner_decision_id == second.decision_id
    assert residual_by_id[first.decision_id] == 0.0
    assert residual_by_id[second.decision_id] > 0.0


def test_posterior_fits_direct_and_lineage_heads_separately(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    first = _record(evolution_context, revisions[0], ordinal=0)
    second = _record(evolution_context, revisions[0], ordinal=1)
    immediate = tuple(
        _observation(
            record, _terminal(record, child_id=f"child-{index}", gain=gain)
        ).model_copy(update={"offer_propensity": 0.5, "joint_action_propensity": 0.5})
        for index, (record, gain) in enumerate(((first, 0.1), (second, -0.1)))
    )
    lineage = immediate[0].model_copy(
        update={
            "measurement": OutcomeMeasurement(
                value=0.05,
                se=None,
                kind="scalar",
            )
        }
    )

    fitted = posterior_model.fit(
        immediate,
        revisions,
        lineage_observations=(lineage,),
    )

    assert fitted.safety.observations == 2
    assert fitted.reward.observations == 2
    assert fitted.lineage_reward.observations == 1


def test_posterior_rejects_misaligned_lineage_residuals(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    record = _record(evolution_context, revisions[0], ordinal=0)
    immediate = _observation(
        record,
        _terminal(record, child_id="child", gain=0.1),
    ).model_copy(update={"offer_propensity": 0.5, "joint_action_propensity": 0.5})

    with pytest.raises(ValueError, match="duplicate decision ids"):
        posterior_model.fit(
            (immediate,),
            revisions,
            lineage_observations=(immediate, immediate),
        )
    with pytest.raises(ValueError, match="subset of immediate"):
        posterior_model.fit(
            (immediate,),
            revisions,
            lineage_observations=(
                immediate.model_copy(update={"decision_id": "foreign"}),
            ),
        )
    with pytest.raises(ValueError, match="change only the measurement"):
        posterior_model.fit(
            (immediate,),
            revisions,
            lineage_observations=(
                immediate.model_copy(update={"treatment": not immediate.treatment}),
            ),
        )
