"""Deterministic direct-plus-lineage reward credit from mutation topology."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence

from gigaevo.memory_v2.models import (
    ArchiveDisposition,
    CausalObservation,
    DecisionRecord,
    LineageOutcome,
    MutationEdge,
    OutcomeMeasurement,
    TerminalOutcome,
)


class _LineageCensored(RuntimeError):
    pass


class LineageCreditResolver:
    """Resolve direct utility and additive, exclusive lineage option value.

    The closest eligible root owns a shared breakthrough. Overlapping non-owner
    roots remain in the lineage regression as structural zero residuals; dropping
    them would selectively censor on the realized outcome. This estimand is
    exclusive incremental enablement, not the total downstream value reachable
    from every ancestor. Direct descendant D1 utility and ancestor option value
    are intentionally different attributions and must not be summed across
    decisions as if they were realized search fitness. Structural zeros count as
    observations and therefore activate the delayed head.
    """

    _GAIN_BOUND_TOLERANCE = 1e-9

    def resolve(
        self,
        decisions: Sequence[DecisionRecord],
        terminals: Mapping[str, TerminalOutcome],
        immediate: Mapping[str, CausalObservation],
        mutation_edges: Sequence[MutationEdge],
    ) -> tuple[tuple[LineageOutcome, ...], tuple[CausalObservation, ...]]:
        if not decisions:
            return (), ()
        ordered = tuple(sorted(decisions, key=lambda row: row.event_ordinal))
        edges = tuple(
            sorted(
                mutation_edges,
                key=lambda row: (row.completion_ordinal, row.child_id),
            )
        )
        edge_by_child = {edge.child_id: edge for edge in edges}
        if len(edge_by_child) != len(edges):
            raise ValueError("mutation topology contains duplicate children")
        children_by_parent: dict[str, list[MutationEdge]] = defaultdict(list)
        for edge in edges:
            children_by_parent[edge.parent_id].append(edge)
        observed_through = max(
            ordered[-1].event_ordinal,
            max((edge.completion_ordinal for edge in edges), default=0),
        )

        outcomes: list[LineageOutcome] = []
        immediate_by_outcome: dict[str, CausalObservation] = {}
        root_ordinal: dict[str, int] = {}
        for root in ordered:
            if root.proposed_treatment_id is None:
                continue
            terminal = terminals.get(root.decision_id)
            if terminal is None:
                continue
            immediate_row = immediate.get(root.decision_id)
            reward = root.context.reward
            if terminal.status == "censored" or not terminal.ope_eligible:
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="censored",
                        lineage_depth=reward.lineage_depth,
                        opportunity_budget=reward.lineage_opportunity_budget,
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
            if terminal.status == "invalid":
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="invalid",
                        lineage_depth=reward.lineage_depth,
                        opportunity_budget=reward.lineage_opportunity_budget,
                        maturity_ordinal=root.event_ordinal,
                        observed_through_ordinal=observed_through,
                    )
                )
                continue
            if reward.lineage_depth == 1:
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="outcome",
                        lineage_depth=1,
                        opportunity_budget=reward.lineage_opportunity_budget,
                        maturity_ordinal=root.event_ordinal,
                        observed_through_ordinal=observed_through,
                        descendant_count=1,
                        valid_descendant_count=1,
                        best_descendant_id=terminal.child_id,
                        best_depth=1,
                        measurement=terminal.measurement,
                    )
                )
                continue

            root_edge = edge_by_child.get(terminal.child_id)
            if root_edge is None:
                outcomes.append(
                    LineageOutcome(
                        decision_id=root.decision_id,
                        root_child_id=terminal.child_id,
                        status="censored",
                        lineage_depth=reward.lineage_depth,
                        opportunity_budget=reward.lineage_opportunity_budget,
                        maturity_ordinal=root.event_ordinal,
                        observed_through_ordinal=observed_through,
                        reason="root mutation is absent from the topology table",
                    )
                )
                continue
            stream = root.context.map_elites.island_id
            later_opportunities = tuple(
                edge
                for edge in edges
                if edge.completion_ordinal > root_edge.completion_ordinal
                and edge.island_id == stream
            )
            has_budget = len(later_opportunities) >= reward.lineage_opportunity_budget
            maturity = (
                later_opportunities[
                    reward.lineage_opportunity_budget - 1
                ].completion_ordinal
                if has_budget
                else None
            )
            opportunity_window_complete = has_budget and all(
                edge.status != "pending"
                and edge.archive_disposition is not ArchiveDisposition.PENDING
                for edge in later_opportunities[: reward.lineage_opportunity_budget]
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
            try:
                (
                    total_measurement,
                    residual_measurement,
                    best_id,
                    best_depth,
                    descendant_count,
                    valid_count,
                    invalid_count,
                    archive_survived,
                ) = self._lineage_value(
                    root,
                    root_edge,
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
            outcome = LineageOutcome(
                decision_id=root.decision_id,
                root_child_id=terminal.child_id,
                status="outcome",
                lineage_depth=reward.lineage_depth,
                opportunity_budget=reward.lineage_opportunity_budget,
                opportunities_observed=reward.lineage_opportunity_budget,
                maturity_ordinal=maturity,
                observed_through_ordinal=observed_through,
                descendant_count=descendant_count,
                valid_descendant_count=valid_count,
                invalid_descendant_count=invalid_count,
                archive_survived=archive_survived,
                best_descendant_id=best_id,
                best_depth=best_depth,
                measurement=total_measurement,
                residual_measurement=residual_measurement,
            )
            outcomes.append(outcome)
            immediate_by_outcome[root.decision_id] = immediate_row
            root_ordinal[root.decision_id] = root.event_ordinal

        owned = self._assign_exclusive_credit(outcomes, root_ordinal)
        lineage_rows = tuple(
            immediate_by_outcome[outcome.decision_id].model_copy(
                update={"measurement": outcome.residual_measurement}
            )
            for outcome in owned
            if outcome.status == "outcome" and outcome.lineage_depth > 1
        )
        return tuple(owned), lineage_rows

    @staticmethod
    def _assign_exclusive_credit(
        outcomes: Sequence[LineageOutcome],
        root_ordinal: Mapping[str, int],
    ) -> list[LineageOutcome]:
        """Give a reused breakthrough to its closest eligible root exactly once."""

        by_descendant: dict[str, list[LineageOutcome]] = defaultdict(list)
        for outcome in outcomes:
            if (
                outcome.status == "outcome"
                and outcome.residual_measurement is not None
                and outcome.residual_measurement.value > 0.0
            ):
                by_descendant[outcome.best_descendant_id].append(outcome)
        owner_by_descendant = {
            descendant_id: min(
                rows,
                key=lambda row: (
                    row.best_depth or row.lineage_depth,
                    -root_ordinal[row.decision_id],
                    row.decision_id,
                ),
            ).decision_id
            for descendant_id, rows in by_descendant.items()
        }
        result: list[LineageOutcome] = []
        for outcome in outcomes:
            owner = owner_by_descendant.get(outcome.best_descendant_id, "")
            residual = outcome.residual_measurement
            if (
                owner
                and owner != outcome.decision_id
                and residual is not None
                and residual.value > 0.0
            ):
                residual = residual.model_copy(update={"value": 0.0})
            result.append(
                outcome.model_copy(
                    update={
                        "residual_measurement": residual,
                        "credit_owner_decision_id": owner,
                    }
                )
            )
        return result

    @staticmethod
    def _lineage_value(
        root: DecisionRecord,
        root_edge: MutationEdge,
        children_by_parent: Mapping[str, Sequence[MutationEdge]],
        *,
        maturity_ordinal: int,
        stream: str,
    ) -> tuple[
        OutcomeMeasurement,
        OutcomeMeasurement,
        str,
        int,
        int,
        int,
        int,
        bool,
    ]:
        reward = root.context.reward
        if root_edge.status != "outcome" or root_edge.measurement is None:
            raise ValueError("valid root terminal lacks a valid topology edge")
        parent_value = root.context.parent_metrics[reward.primary_metric]
        lower = reward.metric_lower_bound - parent_value
        upper = reward.metric_upper_bound - parent_value
        if not reward.higher_is_better:
            lower, upper = -upper, -lower
        direct_gain = LineageCreditResolver._validated_gain(
            root_edge.measurement.value,
            lower=lower,
            upper=upper,
            child_id=root_edge.child_id,
        )
        queue: deque[tuple[MutationEdge, int, float]] = deque(
            [(root_edge, 1, direct_gain)]
        )
        seen_children: set[str] = set()
        best_gain = direct_gain
        best_id = root_edge.child_id
        best_depth = 1
        valid_count = 0
        invalid_count = 0
        archive_survived = root_edge.archive_disposition is ArchiveDisposition.ACCEPTED
        while queue:
            edge, depth, cumulative_gain = queue.popleft()
            if edge.child_id in seen_children:
                continue
            seen_children.add(edge.child_id)
            if edge.status == "censored":
                raise _LineageCensored(
                    f"reachable mutation {edge.child_id!r} is censored"
                )
            if edge.status == "invalid":
                invalid_count += 1
                continue
            if edge.status != "outcome" or edge.measurement is None:
                raise _LineageCensored(
                    f"reachable mutation {edge.child_id!r} is incomplete"
                )
            cumulative_gain = LineageCreditResolver._validated_gain(
                cumulative_gain,
                lower=lower,
                upper=upper,
                child_id=edge.child_id,
            )
            valid_count += 1
            accepted = edge.archive_disposition is ArchiveDisposition.ACCEPTED
            archive_survived = archive_survived or accepted
            if depth > 1 and accepted and cumulative_gain > best_gain:
                best_gain = cumulative_gain
                best_id = edge.child_id
                best_depth = depth
            if depth >= reward.lineage_depth:
                continue
            for child_edge in children_by_parent.get(edge.child_id, ()):
                if not (
                    child_edge.completion_ordinal <= maturity_ordinal
                    and child_edge.island_id == stream
                ):
                    continue
                next_gain = cumulative_gain
                if child_edge.measurement is not None:
                    next_gain += child_edge.measurement.value
                queue.append((child_edge, depth + 1, next_gain))

        residual = max(0.0, best_gain - direct_gain)
        return (
            OutcomeMeasurement(value=best_gain, se=None, kind="scalar"),
            OutcomeMeasurement(value=residual, se=None, kind="scalar"),
            best_id,
            best_depth,
            len(seen_children),
            valid_count,
            invalid_count,
            archive_survived,
        )

    @staticmethod
    def _validated_gain(
        value: float,
        *,
        lower: float,
        upper: float,
        child_id: str,
    ) -> float:
        tolerance = LineageCreditResolver._GAIN_BOUND_TOLERANCE
        if value < lower - tolerance or value > upper + tolerance:
            raise ValueError(
                "lineage gain exceeds root-specific metric bounds for "
                f"child {child_id!r}: {value} not in [{lower}, {upper}]"
            )
        # Normalize only floating-point drift inside the accepted tolerance.
        return min(max(value, lower), upper)
