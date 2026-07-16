"""MemoryReader — the read-system facade.

Shortlist (agentic research over the store) → reputation → auction → budget →
render, every stage swappable behind a small Protocol. Fails to an empty
selection on every error path so a memory outage can never sink a mutation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from loguru import logger
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import AssignmentRecord, DecisionContext
from gigaevo.memory.context import GlobalMemoryContext, MemoryContextModel
from gigaevo.memory.events import (
    MemoryAssignment,
    MemoryReadSelection,
    emit_memory_event,
    memory_event_context,
    new_decision_id,
)
from gigaevo.memory.read.auction import (
    AuctionBid,
    PendingDiscountedBootstrapAuctioneer,
)
from gigaevo.memory.read.exclusion import is_card_excluded
from gigaevo.memory.read.interfaces import (
    Auctioneer,
    Budgeter,
    CandidateProjector,
    CardRenderer,
    ProbePolicy,
    ReputationModel,
    Shortlister,
)
from gigaevo.memory.read.probe import NoColdProbePolicy
from gigaevo.memory.read.projection import AuctionCandidateProjector

_MILLISECONDS_PER_SECOND = 1000.0
_TIMING_DECIMALS = 3


class MemorySelection(BaseModel):
    """Result of memory card selection for mutation guidance."""

    model_config = ConfigDict(frozen=True)

    cards: tuple[str, ...] = Field(
        default=(),
        description="Rendered mutator-facing text blocks, one per selected card.",
    )
    card_ids: tuple[str, ...] = Field(
        default=(),
        description="Bank ids of the selected cards, aligned with ``cards``.",
    )
    slate: tuple[AuctionBid, ...] = Field(
        default=(),
        description="Per-candidate auction audit records (winners and losers).",
    )
    decision_id: str = Field(
        default="", description="Correlation id of the read decision."
    )
    assignment: AssignmentRecord | None = Field(
        default=None, description="Durable assignment ledger payload."
    )
    preformatted: bool = Field(
        default=False,
        description="Cards already contain their complete mutator-facing wrapper.",
    )


def _elapsed_ms(started: float) -> float:
    return round(
        (perf_counter() - started) * _MILLISECONDS_PER_SECOND, _TIMING_DECIMALS
    )


def _ids(items: list[Any]) -> tuple[str, ...]:
    return tuple(str(item_id) for item in items if (item_id := getattr(item, "id", "")))


_POLICY_VOLATILE_NAMES = frozenset(
    {
        "bank",
        "cache",
        "events",
        "index",
        "lock",
        "rng",
        "semaphore",
        "state",
        "tracker",
    }
)
_POLICY_NESTED_NAMES = frozenset(
    {
        "agent",
        "config",
        "context_model",
        "inner",
        "llm",
        "policy_identifiers",
        "reputation",
        "store",
    }
)


def _policy_class(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _canonical_policy_value(value: Any, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_policy_value(item, seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical_policy_value(item, seen) for item in value]
        return (
            sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
            if isinstance(value, (set, frozenset))
            else items
        )

    seen = set() if seen is None else seen
    if id(value) in seen:
        return {"class": _policy_class(value)}
    seen.add(id(value))
    try:
        if isinstance(value, BaseModel):
            fields = type(value).model_fields
            model_config = {
                name: _canonical_policy_value(getattr(value, name), seen)
                for name in sorted(fields)
                if name not in {"path", "cache_dir"}
            }
            return {"class": _policy_class(value), "config": model_config}

        config: dict[str, Any] = {}
        for raw_name, item in sorted(vars(value).items()):
            name = raw_name.lstrip("_")
            if any(token in name for token in _POLICY_VOLATILE_NAMES):
                continue
            if name in _POLICY_NESTED_NAMES or isinstance(
                item, (bool, int, float, str, tuple, list, dict, BaseModel)
            ):
                config[name] = _canonical_policy_value(item, seen)
        return {"class": _policy_class(value), "config": config}
    finally:
        seen.remove(id(value))


def _policy_component(component: Any) -> dict[str, Any]:
    canonical = _canonical_policy_value(component)
    return canonical if isinstance(canonical, dict) else {"value": canonical}


def _policy_version(
    *,
    shortlister: Shortlister,
    reputation: ReputationModel,
    auctioneer: Auctioneer,
    budgeter: Budgeter,
    context_model: MemoryContextModel,
    candidate_projector: CandidateProjector,
    probe_policy: ProbePolicy,
    renderer: CardRenderer,
    exclusion_policy: Any = None,
    downstream_delivery: Mapping[str, Any] | None = None,
    max_cards: int,
) -> str:
    payload = {
        "retrieval": _policy_component(shortlister),
        "reputation": _policy_component(reputation),
        "auctioneer": _policy_component(auctioneer),
        "budgeter": _policy_component(budgeter),
        "context_model": _policy_component(context_model),
        "candidate_projector": _policy_component(candidate_projector),
        "probe_policy": _policy_component(probe_policy),
        "renderer": _policy_component(renderer),
        "exclusion_policy": _policy_component(exclusion_policy),
        "downstream_delivery": _canonical_policy_value(downstream_delivery or {}),
        "max_cards": max_cards,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = sha256(encoded).hexdigest()[:16]
    return f"MemoryPolicy:{digest}"


def extend_policy_version(
    policy_version: str,
    *,
    exclusion_policy: Any = None,
    downstream_delivery: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "read_policy_version": policy_version,
        "exclusion_policy": _policy_component(exclusion_policy),
        "downstream_delivery": _canonical_policy_value(downstream_delivery or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"MemoryPolicy:{sha256(encoded).hexdigest()[:16]}"


class MemoryReader:
    """Retrieve → auction → budget → render over a researched shortlist.

    The shortlister's recall width lives in the store's ``ResearchConfig``;
    ``max_cards`` here is the injection budget the budgeter caps to.
    """

    def __init__(
        self,
        *,
        shortlister: Shortlister,
        reputation: ReputationModel,
        auctioneer: Auctioneer,
        budgeter: Budgeter,
        renderer: CardRenderer,
        context_model: MemoryContextModel | None = None,
        candidate_projector: CandidateProjector | None = None,
        probe_policy: ProbePolicy | None = None,
        max_cards: int = 1,
        rng: Any = None,
    ) -> None:
        self._shortlister = shortlister
        self._reputation = reputation
        self._auctioneer = auctioneer
        self._budgeter = budgeter
        self._renderer = renderer
        self._context_model = (
            context_model if context_model is not None else GlobalMemoryContext()
        )
        self._projector = (
            candidate_projector
            if candidate_projector is not None
            else AuctionCandidateProjector(context_model=self._context_model)
        )
        self._probe_policy = (
            probe_policy if probe_policy is not None else NoColdProbePolicy()
        )
        self._max_cards = max_cards
        self._rng = rng if rng is not None else np.random.default_rng()
        self._lock = asyncio.Lock()
        self._policy_version = _policy_version(
            shortlister=shortlister,
            reputation=reputation,
            auctioneer=auctioneer,
            budgeter=budgeter,
            context_model=self._context_model,
            candidate_projector=self._projector,
            probe_policy=self._probe_policy,
            renderer=renderer,
            max_cards=max_cards,
        )

    @property
    def consumes_pending_counts(self) -> bool:
        return (
            isinstance(self._auctioneer, PendingDiscountedBootstrapAuctioneer)
            and self._auctioneer.pending_power > 0.0
        )

    def _record_context(
        self, context: DecisionContext | None, parents: list[Any]
    ) -> DecisionContext:
        parent = parents[0] if parents else None
        iteration = getattr(parent, "iteration", None)
        search_phase = f"iteration:{iteration}" if isinstance(iteration, int) else ""
        if context is not None:
            if context.search_phase or not search_phase:
                return context
            return context.model_copy(update={"search_phase": search_phase})
        return DecisionContext(
            task_key=str(getattr(self._context_model, "task_key", "") or ""),
            parent_metrics=dict(getattr(parent, "metrics", {}) or {}),
            parent_id=str(getattr(parent, "id", "") or ""),
            search_phase=search_phase,
        )

    def _bd_cell(self, context: DecisionContext) -> tuple[int, ...] | None:
        try:
            key = self._context_model.key_for(context)
            if key.kind != "bd_cell":
                return None
            return tuple(int(part) for part in key.parts)
        except Exception:
            return None

    def _probe_propensities(self, slate: tuple[AuctionBid, ...]) -> dict[str, float]:
        propensities: dict[str, float] = {}
        for bid in slate:
            if not bid.probe_offered or bid.probe_propensity is None:
                continue
            try:
                probability = float(bid.probe_propensity)
            except (TypeError, ValueError):
                continue
            if 0.0 <= probability <= 1.0:
                propensities[bid.card_id] = probability
        return propensities

    @staticmethod
    def _incremental_gain(bid: AuctionBid) -> float | None:
        denominator = bid.posterior_a + bid.posterior_b
        if (
            bid.magnitude is None
            or not math.isfinite(denominator)
            or denominator <= 0.0
        ):
            return None
        help_probability = bid.posterior_a / denominator
        if not math.isfinite(help_probability) or not 0.0 <= help_probability <= 1.0:
            return None
        gain = (
            bid.magnitude
            if bid.support_kind in {"ev_rewards", "zero_support"}
            else help_probability * bid.magnitude
        )
        return gain if math.isfinite(gain) else None

    @staticmethod
    def _assignment_predictions(
        slate: tuple[AuctionBid, ...],
        assigned_ids: tuple[str, ...],
        renderable_ids: frozenset[str] | None = None,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        float | None,
        float | None,
    ]:
        recorded_ids = set(assigned_ids)
        recorded_ids.update(bid.card_id for bid in slate if bid.probe_offered)
        predicted_help: dict[str, float] = {}
        predicted_gain: dict[str, float] = {}
        predicted_no_card_gain: dict[str, float] = {}
        for bid in slate:
            if bid.card_id not in recorded_ids:
                continue
            if bid.no_card_baseline is not None and math.isfinite(bid.no_card_baseline):
                predicted_no_card_gain[bid.card_id] = bid.no_card_baseline
            denominator = bid.posterior_a + bid.posterior_b
            if not math.isfinite(denominator) or denominator <= 0.0:
                continue
            help_probability = bid.posterior_a / denominator
            if (
                not math.isfinite(help_probability)
                or not 0.0 <= help_probability <= 1.0
            ):
                continue
            predicted_help[bid.card_id] = help_probability
            gain = MemoryReader._incremental_gain(bid)
            if gain is not None:
                predicted_gain[bid.card_id] = gain

        offered = next((bid for bid in slate if bid.probe_offered), None)
        q_hat_control: float | None = None
        q_hat_treated: float | None = None
        if (
            offered is not None
            and offered.no_card_baseline is not None
            and math.isfinite(offered.no_card_baseline)
        ):
            no_card_baseline = offered.no_card_baseline

            def action_level(*, treated: bool) -> float | None:
                selected = [
                    bid
                    for bid in slate
                    if (renderable_ids is None or bid.card_id in renderable_ids)
                    and (
                        bid.probe_treated_selected
                        if treated
                        else bid.probe_control_selected
                    )
                ]
                gains = [MemoryReader._incremental_gain(bid) for bid in selected]
                if any(gain is None for gain in gains):
                    return None
                return no_card_baseline + sum(
                    gain for gain in gains if gain is not None
                )

            q_hat_control = action_level(treated=False)
            q_hat_treated = action_level(treated=True)
        return (
            predicted_help,
            predicted_gain,
            predicted_no_card_gain,
            q_hat_control,
            q_hat_treated,
        )

    def _assignment(
        self,
        *,
        decision_id: str,
        context: DecisionContext,
        eligible_ids: tuple[str, ...] = (),
        assigned_ids: tuple[str, ...] = (),
        slate: tuple[AuctionBid, ...] = (),
        renderable_ids: frozenset[str] | None = None,
        policy_version: str | None = None,
        timestamp: datetime,
    ) -> AssignmentRecord:
        propensities = self._probe_propensities(slate)
        probe_offered_bids = [bid for bid in slate if bid.probe_offered]
        probe_arm: Literal["none", "treated", "control"] = (
            "treated"
            if any(bid.probe_selected for bid in probe_offered_bids)
            else "control"
            if probe_offered_bids
            else "none"
        )
        assigned = set(assigned_ids)
        (
            predicted_help,
            predicted_gain,
            predicted_no_card_gain,
            q_hat_control,
            q_hat_treated,
        ) = self._assignment_predictions(slate, assigned_ids, renderable_ids)
        pending_by_card = {
            bid.card_id: bid.pending_count for bid in slate if bid.card_id in assigned
        }
        pending_discount_by_card = {
            bid.card_id: bid.pending_discount
            for bid in slate
            if bid.card_id in assigned
        }
        return AssignmentRecord(
            decision_id=decision_id,
            policy_version=policy_version or self._policy_version,
            task_key=context.task_key,
            ordered_eligible_ids=eligible_ids,
            assigned_ids=tuple(sorted(assigned_ids)),
            delivered_ids=tuple(sorted(assigned_ids)),
            arm="injected" if assigned_ids else "none",
            probe_arm=probe_arm,
            randomized=bool(probe_offered_bids),
            propensity_kind=(
                "probe_bernoulli" if probe_offered_bids else "observational"
            ),
            propensities=propensities,
            ope_eligible=bool(probe_offered_bids),
            q_hat_control=q_hat_control,
            q_hat_treated=q_hat_treated,
            predicted_help=predicted_help,
            predicted_gain=predicted_gain,
            predicted_no_card_gain=predicted_no_card_gain,
            pending_by_card=pending_by_card,
            pending_discount_by_card=pending_discount_by_card,
            context=context,
            bd_cell=self._bd_cell(context),
            timestamp=timestamp,
        )

    @staticmethod
    def _emit_decision(
        event: MemoryReadSelection, assignment: AssignmentRecord
    ) -> None:
        for payload in (
            event,
            MemoryAssignment(decision_id=assignment.decision_id, assignment=assignment),
        ):
            try:
                emit_memory_event(payload)
            except Exception:
                logger.opt(exception=True).warning(
                    "[Memory][Reader] decision telemetry emit failed; continuing"
                )

    async def select(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str] = frozenset(),
        parent_contexts: list[str] | None = None,
        pending_counts: Mapping[str, int] | None = None,
        exclusion_policy: Any = None,
    ) -> MemorySelection:
        parent_ids = _ids(parents)
        decision_id = new_decision_id()
        decision_timestamp = datetime.now(UTC)
        decision_policy_version = (
            extend_policy_version(
                self._policy_version, exclusion_policy=exclusion_policy
            )
            if exclusion_policy is not None
            else self._policy_version
        )
        with memory_event_context(
            decision_id=decision_id,
            program_id=parent_ids[0] if parent_ids else "",
            parent_ids=parent_ids,
        ):
            base = MemoryReadSelection(
                decision_id=decision_id,
                mutation_mode=mutation_mode,
                max_cards=self._max_cards,
                exclude_ids=tuple(sorted(exclude_ids)),
            )
            if self._max_cards <= 0:
                assignment = self._assignment(
                    decision_id=decision_id,
                    context=self._record_context(None, parents),
                    policy_version=decision_policy_version,
                    timestamp=decision_timestamp,
                )
                self._emit_decision(
                    base.model_copy(update={"empty_reason": "max_cards_nonpositive"}),
                    assignment,
                )
                return MemorySelection(decision_id=decision_id, assignment=assignment)
            try:
                return await self._select(
                    parents=parents,
                    mutation_mode=mutation_mode,
                    task_description=task_description,
                    metrics_description=metrics_description,
                    exclude_ids=exclude_ids,
                    parent_contexts=parent_contexts,
                    pending_counts=pending_counts,
                    base=base,
                    decision_id=decision_id,
                    decision_timestamp=decision_timestamp,
                    policy_version=decision_policy_version,
                )
            except Exception as exc:
                logger.opt(exception=True).warning(
                    "[Memory][Reader] selection failed; returning empty: {}", exc
                )
                assignment = self._assignment(
                    decision_id=decision_id,
                    context=self._record_context(None, parents),
                    policy_version=decision_policy_version,
                    timestamp=decision_timestamp,
                )
                self._emit_decision(
                    base.model_copy(
                        update={
                            "empty_reason": "exception",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ),
                    assignment,
                )
                return MemorySelection(decision_id=decision_id, assignment=assignment)

    async def _select(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str],
        parent_contexts: list[str] | None,
        pending_counts: Mapping[str, int] | None,
        base: MemoryReadSelection,
        decision_id: str,
        decision_timestamp: datetime,
        policy_version: str,
    ) -> MemorySelection:
        started_total = perf_counter()
        started_research = perf_counter()
        async with self._lock:
            result = await self._shortlister.shortlist(
                parents=parents,
                mutation_mode=mutation_mode,
                task_description=task_description,
                metrics_description=metrics_description,
                exclude_ids=exclude_ids,
                parent_contexts=parent_contexts,
            )
        research_ms = _elapsed_ms(started_research)
        candidates = {
            card.id: card
            for card in result.cards
            if not is_card_excluded(card, exclude_ids)
        }

        started_reputation = perf_counter()
        decision_context = self._context_model.read_context(parents)
        baseline = self._projector.decision_baseline(decision_context)
        blocks = {
            card.id: self._reputation.card_stats(card, decision_context)
            for card in candidates.values()
        }
        auction_input = []
        for card_id, block in blocks.items():
            card = candidates[card_id]
            auction_input.append(
                self._projector.project(
                    card=card,
                    block=block,
                    reputation=self._reputation,
                    context=decision_context,
                    pending_counts=pending_counts,
                )
            )
        reputation_ms = _elapsed_ms(started_reputation)

        started_auction = perf_counter()
        auction_winner_ids, slate = self._auctioneer.run(
            auction_input, self._rng, baseline=baseline
        )
        auction_ms = _elapsed_ms(started_auction)
        started_budget = perf_counter()
        budgeted_ids = self._budgeter.cap(auction_winner_ids, slate, self._max_cards)
        budgeted_ids, slate = self._probe_policy.apply(
            budgeted_ids=budgeted_ids,
            slate=list(slate),
            max_cards=self._max_cards,
            rng=self._rng,
        )
        budget_ms = _elapsed_ms(started_budget)
        started_render = perf_counter()
        action_ids = set(budgeted_ids)
        action_ids.update(
            bid.card_id
            for bid in slate
            if (bid.probe_control_selected or bid.probe_treated_selected)
            and bid.card_id in candidates
        )
        rendered_by_id = {
            card_id: self._renderer.render(candidates[card_id], blocks.get(card_id))
            for card_id in action_ids
        }
        rendered = [
            (cid, text) for cid in budgeted_ids if (text := rendered_by_id[cid])
        ]
        renderable_ids = frozenset(
            card_id for card_id, text in rendered_by_id.items() if text
        )
        render_ms = _elapsed_ms(started_render)
        card_ids = tuple(cid for cid, _ in rendered)
        render_dropped_ids = tuple(cid for cid in budgeted_ids if cid not in card_ids)
        assignment_context = self._record_context(decision_context, parents)
        assignment = self._assignment(
            decision_id=decision_id,
            context=assignment_context,
            eligible_ids=tuple(candidates),
            assigned_ids=card_ids,
            slate=tuple(slate),
            renderable_ids=renderable_ids,
            policy_version=policy_version,
            timestamp=decision_timestamp,
        )

        empty_reason = ""
        if not card_ids:
            if not candidates:
                empty_reason = "research_empty"
            elif not auction_winner_ids:
                empty_reason = "auction_rejected"
            elif not budgeted_ids:
                empty_reason = "budget_empty"
            else:
                empty_reason = "render_empty"
        self._emit_decision(
            base.model_copy(
                update={
                    "research_iterations": result.iterations,
                    "candidate_ids": tuple(candidates),
                    "auction_winner_ids": tuple(auction_winner_ids),
                    "budgeted_ids": tuple(budgeted_ids),
                    "render_dropped_ids": render_dropped_ids,
                    "selected_ids": card_ids,
                    "slate": tuple(bid.model_dump(mode="json") for bid in slate),
                    "empty_reason": empty_reason,
                    "timing_ms": {
                        "research": research_ms,
                        "reputation": reputation_ms,
                        "auction": auction_ms,
                        "budget": budget_ms,
                        "render": render_ms,
                        "total": _elapsed_ms(started_total),
                    },
                }
            ),
            assignment,
        )
        logger.debug(
            "[Memory][Reader] Selected {}/{} card(s) after auction+budget (ids={})",
            len(card_ids),
            len(auction_input),
            list(card_ids),
        )
        return MemorySelection(
            cards=tuple(text for _, text in rendered),
            card_ids=card_ids,
            slate=tuple(slate),
            decision_id=decision_id,
            assignment=assignment,
        )
