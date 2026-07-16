"""Deterministic lineage reward credit derived from the causal ledger."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence

from gigaevo.memory_v2.models import (
    CausalObservation,
    DecisionRecord,
    LineageOutcome,
    OutcomeMeasurement,
    TerminalOutcome,
)


class _LineageCensored(RuntimeError):
    pass


class LineageCreditResolver:
    """Resolve one fixed-opportunity endpoint per randomized decision.

    Ledger edges are the selected parent in ``DecisionRecord.context`` and the
    resulting child in ``TerminalOutcome``. Later card decisions therefore stay
    inside the downstream policy instead of receiving duplicated reward rows.
    """

    def resolve(
        self,
        decisions: Sequence[DecisionRecord],
        terminals: Mapping[str, TerminalOutcome],
        immediate: Mapping[str, CausalObservation],
    ) -> tuple[tuple[LineageOutcome, ...], tuple[CausalObservation, ...]]:
        if not decisions:
            return (), ()
        ordered = tuple(sorted(decisions, key=lambda row: row.event_ordinal))
        observed_through = ordered[-1].event_ordinal

        children_by_parent: dict[str, list[tuple[DecisionRecord, TerminalOutcome]]] = (
            defaultdict(list)
        )
        for record in ordered:
            terminal = terminals.get(record.decision_id)
            if terminal is not None:
                children_by_parent[record.context.parent_id].append((record, terminal))

        outcomes: list[LineageOutcome] = []
        reward_rows: list[CausalObservation] = []
        for root in ordered:
            immediate_row = immediate.get(root.decision_id)
            if root.proposed_treatment_id is None:
                continue
            terminal = terminals.get(root.decision_id)
            if terminal is None:
                continue
            reward = root.context.reward
            if terminal.status == "censored" or not terminal.ope_eligible:
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="censored",
                        lineage_depth=reward.lineage_depth,
                        opportunity_budget=reward.lineage_opportunity_budget,
                        opportunities_observed=0,
                        maturity_ordinal=root.event_ordinal,
                        observed_through_ordinal=observed_through,
                        reason=(
                            terminal.censor_reason
                            or "root terminal is not eligible for causal inference"
                        ),
                    )
                )
                continue
            if immediate_row is None:
                raise ValueError(
                    f"eligible terminal {root.decision_id!r} lacks immediate evidence"
                )
            if reward.lineage_depth == 1:
                if terminal.status == "invalid":
                    outcomes.append(
                        LineageOutcome(
                            decision_id=root.decision_id,
                            root_child_id=terminal.child_id,
                            status="invalid",
                            lineage_depth=1,
                            opportunity_budget=reward.lineage_opportunity_budget,
                            opportunities_observed=0,
                            maturity_ordinal=root.event_ordinal,
                            observed_through_ordinal=observed_through,
                        )
                    )
                    reward_rows.append(immediate_row)
                    continue
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="outcome",
                        lineage_depth=1,
                        opportunity_budget=reward.lineage_opportunity_budget,
                        opportunities_observed=0,
                        maturity_ordinal=root.event_ordinal,
                        observed_through_ordinal=observed_through,
                        descendant_count=1,
                        best_descendant_id=terminal.child_id,
                        best_depth=1,
                        measurement=terminal.measurement,
                    )
                )
                reward_rows.append(immediate_row)
                continue

            stream = root.context.map_elites.island_id
            later_opportunities = tuple(
                record
                for record in ordered
                if record.event_ordinal > root.event_ordinal
                and record.context.map_elites.island_id == stream
            )
            has_budget = len(later_opportunities) >= reward.lineage_opportunity_budget
            maturity = (
                later_opportunities[reward.lineage_opportunity_budget - 1].event_ordinal
                if has_budget
                else None
            )
            opportunity_window_complete = has_budget and all(
                record.decision_id in terminals
                for record in later_opportunities[: reward.lineage_opportunity_budget]
            )
            if not opportunity_window_complete:
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="pending",
                        lineage_depth=reward.lineage_depth,
                        opportunity_budget=reward.lineage_opportunity_budget,
                        opportunities_observed=min(
                            len(later_opportunities),
                            reward.lineage_opportunity_budget,
                        ),
                        maturity_ordinal=maturity,
                        observed_through_ordinal=observed_through,
                    )
                )
                continue

            assert maturity is not None
            if terminal.status == "invalid":
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="invalid",
                        lineage_depth=reward.lineage_depth,
                        opportunity_budget=reward.lineage_opportunity_budget,
                        opportunities_observed=reward.lineage_opportunity_budget,
                        maturity_ordinal=maturity,
                        observed_through_ordinal=observed_through,
                    )
                )
                reward_rows.append(immediate_row)
                continue
            try:
                measurement, best_id, best_depth, descendant_count = self._best_gain(
                    root,
                    terminal,
                    children_by_parent,
                    maturity_ordinal=maturity,
                    stream=stream,
                )
            except _LineageCensored as exc:
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="censored",
                        lineage_depth=reward.lineage_depth,
                        opportunity_budget=reward.lineage_opportunity_budget,
                        opportunities_observed=reward.lineage_opportunity_budget,
                        maturity_ordinal=maturity,
                        observed_through_ordinal=observed_through,
                        reason=str(exc),
                    )
                )
                continue
            outcomes.append(
                LineageOutcome(
                    decision_id=root.decision_id,
                    root_child_id=terminal.child_id,
                    status="outcome",
                    lineage_depth=reward.lineage_depth,
                    opportunity_budget=reward.lineage_opportunity_budget,
                    opportunities_observed=reward.lineage_opportunity_budget,
                    maturity_ordinal=maturity,
                    observed_through_ordinal=observed_through,
                    descendant_count=descendant_count,
                    best_descendant_id=best_id,
                    best_depth=best_depth,
                    measurement=measurement,
                )
            )
            reward_rows.append(
                immediate_row.model_copy(update={"measurement": measurement})
            )
        return tuple(outcomes), tuple(reward_rows)

    @staticmethod
    def _best_gain(
        root: DecisionRecord,
        root_terminal: TerminalOutcome,
        children_by_parent: Mapping[
            str, Sequence[tuple[DecisionRecord, TerminalOutcome]]
        ],
        *,
        maturity_ordinal: int,
        stream: str,
    ) -> tuple[OutcomeMeasurement, str, int, int]:
        reward = root.context.reward
        root_parent_value = root.context.parent_metrics[reward.primary_metric]
        queue: deque[tuple[DecisionRecord, TerminalOutcome, int]] = deque(
            [(root, root_terminal, 1)]
        )
        seen_children: set[str] = set()
        best_gain: float | None = None
        best_id = ""
        best_depth = 1
        while queue:
            record, terminal, depth = queue.popleft()
            if terminal.child_id in seen_children:
                continue
            seen_children.add(terminal.child_id)
            if record.context.reward != reward:
                raise ValueError("one lineage mixes incompatible reward definitions")
            if terminal.measurement is None:
                raise ValueError("valid lineage descendant lacks a measurement")
            parent_value = record.context.parent_metrics[reward.primary_metric]
            child_value = (
                parent_value + terminal.measurement.value
                if reward.higher_is_better
                else parent_value - terminal.measurement.value
            )
            if not (
                reward.metric_lower_bound - 1e-9
                <= child_value
                <= reward.metric_upper_bound + 1e-9
            ):
                raise ValueError("lineage descendant fitness exceeds metric bounds")
            gain = (
                child_value - root_parent_value
                if reward.higher_is_better
                else root_parent_value - child_value
            )
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_id = terminal.child_id
                best_depth = depth
            if depth >= reward.lineage_depth:
                continue
            for child_record, child_terminal in children_by_parent.get(
                terminal.child_id, ()
            ):
                if not (
                    root.event_ordinal < child_record.event_ordinal <= maturity_ordinal
                    and child_record.context.map_elites.island_id == stream
                ):
                    continue
                if child_terminal.status == "censored":
                    raise _LineageCensored(
                        "reachable descendant decision "
                        f"{child_record.decision_id!r} is censored"
                    )
                if child_terminal.status == "outcome":
                    queue.append((child_record, child_terminal, depth + 1))

        if best_gain is None:
            raise ValueError("valid lineage contains no measured descendant")
        if reward.higher_is_better:
            lower = reward.metric_lower_bound - root_parent_value
            upper = reward.metric_upper_bound - root_parent_value
        else:
            lower = root_parent_value - reward.metric_upper_bound
            upper = root_parent_value - reward.metric_lower_bound
        if best_gain < lower - 1e-9 or best_gain > upper + 1e-9:
            raise ValueError("lineage gain exceeds root-specific metric bounds")
        best_gain = min(max(best_gain, lower), upper)
        return (
            OutcomeMeasurement(value=best_gain, se=None, kind="scalar"),
            best_id,
            best_depth,
            len(seen_children),
        )
