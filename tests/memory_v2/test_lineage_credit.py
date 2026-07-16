from __future__ import annotations

import pytest

from gigaevo.memory_v2.credit import LineageCreditResolver
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    DecisionRecord,
    EvolutionContext,
    OutcomeMeasurement,
    TerminalOutcome,
)
from gigaevo.memory_v2.posterior import HierarchicalTerminalUtilityPosterior

from .factories import decision_record


def _context(
    template: EvolutionContext,
    *,
    parent_id: str,
    parent_fitness: float,
    ordinal: int,
    island_id: str = "main",
    depth: int = 2,
    opportunities: int = 2,
) -> EvolutionContext:
    return template.model_copy(
        update={
            "parent_id": parent_id,
            "parent_iteration": ordinal,
            "parent_generation": ordinal + 1,
            "parent_metrics": {
                **template.parent_metrics,
                "fitness": parent_fitness,
            },
            "reward": template.reward.model_copy(
                update={
                    "endpoint": "bounded_lineage_utility",
                    "lineage_depth": depth,
                    "lineage_opportunity_budget": opportunities,
                }
            ),
            "map_elites": template.map_elites.model_copy(
                update={"island_id": island_id}
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
        measurement=(
            OutcomeMeasurement(value=gain, se=None, kind="scalar")
            if gain is not None
            else None
        ),
        completion_ordinal=record.event_ordinal,
    )


def _observation(
    record: DecisionRecord, terminal: TerminalOutcome
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


def _resolve(
    records: tuple[DecisionRecord, ...],
    terminals: tuple[TerminalOutcome, ...],
):
    terminal_map = {row.decision_id: row for row in terminals}
    immediate = {
        record.decision_id: _observation(record, terminal_map[record.decision_id])
        for record in records
        if record.decision_id in terminal_map
        and terminal_map[record.decision_id].status != "censored"
    }
    return LineageCreditResolver().resolve(records, terminal_map, immediate)


def test_depth_one_preserves_the_exact_immediate_measurement(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    record = _record(evolution_context, revisions[0], ordinal=0)
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
    immediate = _observation(record, terminal)

    outcomes, reward_rows = LineageCreditResolver().resolve(
        (record,),
        {record.decision_id: terminal},
        {record.decision_id: immediate},
    )

    assert reward_rows == (immediate,)
    assert outcomes[0].measurement == measurement
    assert outcomes[0].best_depth == 1
    assert outcomes[0].opportunities_observed == 0


def test_local_opportunity_clock_excludes_siblings_and_truncates_depth(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    root = _record(
        _context(
            evolution_context,
            parent_id="parent",
            parent_fitness=0.50,
            ordinal=0,
        ),
        card,
        ordinal=0,
    )
    foreign = _record(
        _context(
            evolution_context,
            parent_id="foreign-parent",
            parent_fitness=0.40,
            ordinal=1,
            island_id="other",
        ),
        card,
        ordinal=1,
    )
    descendant = _record(
        _context(
            evolution_context,
            parent_id="root-child",
            parent_fitness=0.60,
            ordinal=2,
        ),
        card,
        ordinal=2,
    )
    too_deep = _record(
        _context(
            evolution_context,
            parent_id="grandchild",
            parent_fitness=0.75,
            ordinal=3,
            island_id="other",
        ),
        card,
        ordinal=3,
    )
    sibling = _record(
        _context(
            evolution_context,
            parent_id="parent",
            parent_fitness=0.50,
            ordinal=4,
        ),
        card,
        ordinal=4,
    )
    records = (root, foreign, descendant, too_deep, sibling)
    terminals = (
        _terminal(root, child_id="root-child", gain=0.10),
        _terminal(descendant, child_id="grandchild", gain=0.15),
        _terminal(too_deep, child_id="great-grandchild", gain=0.20),
        _terminal(sibling, child_id="sibling-child", gain=0.49),
    )

    outcomes, reward_rows = _resolve(records, terminals)
    root_outcome = next(row for row in outcomes if row.decision_id == root.decision_id)
    root_reward = next(
        row for row in reward_rows if row.decision_id == root.decision_id
    )

    assert root_outcome.maturity_ordinal == 4
    assert root_outcome.opportunities_observed == 2
    assert root_outcome.descendant_count == 2
    assert root_outcome.best_descendant_id == "grandchild"
    assert root_outcome.best_depth == 2
    assert root_outcome.measurement is not None
    assert root_outcome.measurement.value == 0.25
    assert root_reward.measurement == root_outcome.measurement


def test_lineage_waits_for_local_budget_and_terminal_completion(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    root = _record(
        _context(
            evolution_context,
            parent_id="parent",
            parent_fitness=0.5,
            ordinal=0,
        ),
        card,
        ordinal=0,
    )
    child = _record(
        _context(
            evolution_context,
            parent_id="root-child",
            parent_fitness=0.6,
            ordinal=1,
        ),
        card,
        ordinal=1,
    )
    closing = _record(
        _context(
            evolution_context,
            parent_id="unrelated",
            parent_fitness=0.4,
            ordinal=2,
        ),
        card,
        ordinal=2,
    )
    records = (root, child, closing)
    root_terminal = _terminal(root, child_id="root-child", gain=0.1)
    child_terminal = _terminal(child, child_id="grandchild", gain=0.1)

    outcomes, reward_rows = _resolve(records, (root_terminal, child_terminal))
    root_pending = next(row for row in outcomes if row.decision_id == root.decision_id)
    assert root_pending.status == "pending"
    assert root_pending.opportunities_observed == 2
    assert root_pending.maturity_ordinal == 2
    assert all(row.decision_id != root.decision_id for row in reward_rows)

    closing_terminal = _terminal(closing, child_id="unrelated-child", gain=0.0)
    outcomes, reward_rows = _resolve(
        records, (root_terminal, child_terminal, closing_terminal)
    )
    root_outcome = next(row for row in outcomes if row.decision_id == root.decision_id)
    assert root_outcome.status == "outcome"
    assert any(row.decision_id == root.decision_id for row in reward_rows)


def test_invalid_root_waits_for_reward_but_updates_immediate_safety(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    root = _record(
        _context(
            evolution_context,
            parent_id="parent",
            parent_fitness=0.5,
            ordinal=0,
            opportunities=2,
        ),
        revisions[0],
        ordinal=0,
    )
    terminal = _terminal(
        root,
        child_id="no-child:invalid",
        gain=None,
        status="invalid",
    )

    outcomes, reward_rows = _resolve((root,), (terminal,))

    assert outcomes[0].status == "pending"
    assert reward_rows == ()

    first = _record(
        _context(
            evolution_context,
            parent_id="other-1",
            parent_fitness=0.5,
            ordinal=1,
            opportunities=2,
        ),
        revisions[0],
        ordinal=1,
    )
    second = _record(
        _context(
            evolution_context,
            parent_id="other-2",
            parent_fitness=0.5,
            ordinal=2,
            opportunities=2,
        ),
        revisions[0],
        ordinal=2,
    )
    outcomes, reward_rows = _resolve(
        (root, first, second),
        (
            terminal,
            _terminal(first, child_id="first-child", gain=0.0),
            _terminal(second, child_id="second-child", gain=0.0),
        ),
    )
    root_outcome = next(row for row in outcomes if row.decision_id == root.decision_id)
    assert root_outcome.status == "invalid"
    assert root_outcome.maturity_ordinal == 2
    assert next(
        row for row in reward_rows if row.decision_id == root.decision_id
    ).invalid


def test_unresolved_pre_root_and_foreign_decisions_do_not_block_maturity(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    pre_root = _record(
        _context(
            evolution_context,
            parent_id="old",
            parent_fitness=0.4,
            ordinal=0,
        ),
        card,
        ordinal=0,
    )
    foreign = _record(
        _context(
            evolution_context,
            parent_id="foreign",
            parent_fitness=0.4,
            ordinal=1,
            island_id="other",
        ),
        card,
        ordinal=1,
    )
    root = _record(
        _context(
            evolution_context,
            parent_id="parent",
            parent_fitness=0.5,
            ordinal=2,
        ),
        card,
        ordinal=2,
    )
    first = _record(
        _context(
            evolution_context,
            parent_id="root-child",
            parent_fitness=0.6,
            ordinal=3,
        ),
        card,
        ordinal=3,
    )
    second = _record(
        _context(
            evolution_context,
            parent_id="unrelated",
            parent_fitness=0.4,
            ordinal=4,
        ),
        card,
        ordinal=4,
    )
    records = (pre_root, foreign, root, first, second)
    terminals = (
        _terminal(root, child_id="root-child", gain=0.1),
        _terminal(first, child_id="grandchild", gain=0.1),
        _terminal(second, child_id="unrelated-child", gain=0.0),
    )

    outcomes, _ = _resolve(records, terminals)

    root_outcome = next(row for row in outcomes if row.decision_id == root.decision_id)
    assert root_outcome.status == "outcome"
    assert root_outcome.maturity_ordinal == 4


def test_reachable_censored_descendant_censors_lineage_reward(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    root = _record(
        _context(
            evolution_context,
            parent_id="parent",
            parent_fitness=0.5,
            ordinal=0,
        ),
        card,
        ordinal=0,
    )
    censored = _record(
        _context(
            evolution_context,
            parent_id="root-child",
            parent_fitness=0.6,
            ordinal=1,
        ),
        card,
        ordinal=1,
    )
    closing = _record(
        _context(
            evolution_context,
            parent_id="unrelated",
            parent_fitness=0.4,
            ordinal=2,
        ),
        card,
        ordinal=2,
    )
    censored_terminal = TerminalOutcome(
        decision_id=censored.decision_id,
        child_id="no-child:censored",
        base_id=censored.context.parent_id,
        primary_metric="fitness",
        higher_is_better=True,
        ope_eligible=True,
        status="censored",
        censor_reason="administrative_failure",
        completion_ordinal=1,
    )

    outcomes, reward_rows = _resolve(
        (root, censored, closing),
        (
            _terminal(root, child_id="root-child", gain=0.1),
            censored_terminal,
            _terminal(closing, child_id="closing-child", gain=0.0),
        ),
    )

    root_outcome = next(row for row in outcomes if row.decision_id == root.decision_id)
    assert root_outcome.status == "censored"
    assert "reachable descendant" in root_outcome.reason
    assert all(row.decision_id != root.decision_id for row in reward_rows)


def test_lower_is_better_lineage_gain_has_correct_orientation(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    base = _context(
        evolution_context,
        parent_id="parent",
        parent_fitness=0.8,
        ordinal=0,
    )
    base = base.model_copy(
        update={"reward": base.reward.model_copy(update={"higher_is_better": False})}
    )
    root = _record(base, card, ordinal=0)
    child_context = _context(
        evolution_context,
        parent_id="root-child",
        parent_fitness=0.7,
        ordinal=1,
    )
    child_context = child_context.model_copy(
        update={
            "reward": child_context.reward.model_copy(
                update={"higher_is_better": False}
            )
        }
    )
    child = _record(child_context, card, ordinal=1)
    closing_context = child_context.model_copy(
        update={
            "parent_id": "other",
            "parent_iteration": 2,
            "parent_generation": 3,
        }
    )
    closing = _record(closing_context, card, ordinal=2)

    outcomes, _ = _resolve(
        (root, child, closing),
        (
            _terminal(root, child_id="root-child", gain=0.1),
            _terminal(child, child_id="grandchild", gain=0.2),
            _terminal(closing, child_id="other-child", gain=0.0),
        ),
    )

    root_outcome = next(row for row in outcomes if row.decision_id == root.decision_id)
    assert root_outcome.measurement is not None
    assert root_outcome.measurement.value == pytest.approx(0.3)


def test_lineage_rejects_impossible_reconstructed_fitness(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    root = _record(
        _context(
            evolution_context,
            parent_id="parent",
            parent_fitness=0.9,
            ordinal=0,
            opportunities=1,
        ),
        revisions[0],
        ordinal=0,
    )
    closing = _record(
        _context(
            evolution_context,
            parent_id="other",
            parent_fitness=0.5,
            ordinal=1,
            opportunities=1,
        ),
        revisions[0],
        ordinal=1,
    )

    with pytest.raises(ValueError, match="descendant fitness"):
        _resolve(
            (root, closing),
            (
                _terminal(root, child_id="impossible-child", gain=0.2),
                _terminal(closing, child_id="closing-child", gain=0.0),
            ),
        )


def test_posterior_uses_delayed_rows_only_for_reward_head(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    first = _record(evolution_context, revisions[0], ordinal=0)
    second = _record(evolution_context, revisions[0], ordinal=1)
    first_terminal = _terminal(first, child_id="first-child", gain=0.1)
    second_terminal = _terminal(second, child_id="second-child", gain=-0.1)
    immediate = tuple(
        row.model_copy(update={"offer_propensity": 0.5, "joint_action_propensity": 0.5})
        for row in (
            _observation(first, first_terminal),
            _observation(second, second_terminal),
        )
    )

    fitted = posterior_model.fit(
        immediate,
        revisions,
        reward_observations=immediate[:1],
    )

    assert fitted.safety.observations == 2
    assert fitted.reward.observations == 1


def test_posterior_rejects_duplicate_or_misaligned_delayed_rows(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    record = _record(evolution_context, revisions[0], ordinal=0)
    terminal = _terminal(record, child_id="child", gain=0.1)
    immediate = _observation(record, terminal).model_copy(
        update={"offer_propensity": 0.5, "joint_action_propensity": 0.5}
    )

    with pytest.raises(ValueError, match="duplicate decision ids"):
        posterior_model.fit(
            (immediate,),
            revisions,
            reward_observations=(immediate, immediate),
        )
    with pytest.raises(ValueError, match="subset of immediate"):
        posterior_model.fit(
            (immediate,),
            revisions,
            reward_observations=(
                immediate.model_copy(update={"decision_id": "foreign"}),
            ),
        )
    with pytest.raises(ValueError, match="change only the outcome measurement"):
        posterior_model.fit(
            (immediate,),
            revisions,
            reward_observations=(
                immediate.model_copy(update={"treatment": not immediate.treatment}),
            ),
        )


def test_posterior_requires_explicit_long_horizon_reward_rows(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    context = _context(
        evolution_context,
        parent_id=evolution_context.parent_id,
        parent_fitness=0.5,
        ordinal=0,
        depth=2,
        opportunities=2,
    )
    record = _record(context, revisions[0], ordinal=0)
    immediate = _observation(
        record, _terminal(record, child_id="child", gain=0.1)
    ).model_copy(update={"offer_propensity": 0.5, "joint_action_propensity": 0.5})

    with pytest.raises(ValueError, match="explicit matured lineage observations"):
        posterior_model.fit((immediate,), revisions)
