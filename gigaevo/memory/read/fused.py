"""Fused shortlist ranking: semantic rank + reputation + recency novelty.

Wraps any :class:`~gigaevo.memory.read.reader.Shortlister` and reorders its
researched cards by ``w_sem * semantic + w_rep * reputation + w_nov *
novelty``. The semantic leg is rank-based (``ResearchResult`` carries no
distances across the seam); reputation resolves through the same
``card_stats`` authority the auction bids on; novelty decays with the card's
recent gain events. Two selectivity gates (an empty shortlist is a valid
outcome for both): ``rep_floor_quantile`` is self-normalizing — it drops
cards whose reputation leg falls below the q-quantile of the current bank's
own reputation distribution, so the same config transfers across tasks and
bank compositions; ``score_floor`` is an absolute fused-score cut, reserved
for experiment-calibrated pins (never shared config). When
``rep_floor_quantile`` is active the same pass also benches guaranteed
losers upstream — below-floor or non-positive-magnitude cards are merged
into ``exclude_ids`` before research, so they stop occupying digest and
shortlist slots they could never convert. With ``w_rep == w_nov == 0`` and
no gate the inner result passes through untouched — a pure seam until
A/B'd.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import math
from typing import Any

from gigaevo.memory.cards import Card, DecisionContext
from gigaevo.memory.context import GlobalMemoryContext, MemoryContextModel
from gigaevo.memory.read.interfaces import ReputationModel, Shortlister
from gigaevo.memory.storage.base import ResearchResult

_NEUTRAL_P_HELP = 0.5


class FusedRankingShortlister:
    """Reorders the inner shortlist by the fused score; inert at zero weights."""

    def __init__(
        self,
        inner: Shortlister,
        reputation: ReputationModel,
        *,
        w_sem: float = 1.0,
        w_rep: float = 0.0,
        w_nov: float = 0.0,
        novelty_window_hours: float = 24.0,
        score_floor: float | None = None,
        rep_floor_quantile: float | None = None,
        store: Any | None = None,
        context_model: MemoryContextModel | None = None,
    ) -> None:
        for name, weight in (("w_sem", w_sem), ("w_rep", w_rep), ("w_nov", w_nov)):
            if weight < 0:
                raise ValueError(f"{name} must be non-negative, got {weight}")
        if novelty_window_hours <= 0:
            raise ValueError(
                f"novelty_window_hours must be positive, got {novelty_window_hours}"
            )
        if rep_floor_quantile is not None:
            if not 0.0 <= rep_floor_quantile < 1.0:
                raise ValueError(
                    f"rep_floor_quantile must be in [0, 1), got {rep_floor_quantile}"
                )
            if store is None:
                raise ValueError(
                    "rep_floor_quantile requires a store to read the bank distribution"
                )
        self._inner = inner
        self._reputation = reputation
        self._w_sem = w_sem
        self._w_rep = w_rep
        self._w_nov = w_nov
        self._novelty_window = timedelta(hours=novelty_window_hours)
        self._score_floor = score_floor
        self._rep_floor_quantile = rep_floor_quantile
        self._store = store
        self._context_model = (
            context_model if context_model is not None else GlobalMemoryContext()
        )

    async def shortlist(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str] = frozenset(),
        parent_contexts: list[str] | None = None,
    ) -> ResearchResult:
        context = self._context_model.read_context(parents)
        rep_floor, benched_ids = self._bank_bench(context)
        result = await self._inner.shortlist(
            parents=parents,
            mutation_mode=mutation_mode,
            task_description=task_description,
            metrics_description=metrics_description,
            exclude_ids=exclude_ids | benched_ids,
            parent_contexts=parent_contexts,
        )
        if not result.cards or (
            self._w_rep == 0.0
            and self._w_nov == 0.0
            and self._score_floor is None
            and self._rep_floor_quantile is None
        ):
            return result
        cutoff = datetime.now(UTC) - self._novelty_window
        count = len(result.cards)
        scored = []
        for i, card in enumerate(result.cards):
            if card.id in benched_ids:
                continue
            semantic = 1.0 if count == 1 else 1.0 - i / (count - 1)
            p_help = self._p_help(card, context)
            score = (
                self._w_sem * semantic
                + self._w_rep * p_help
                + self._w_nov * self._novelty(card, cutoff)
            )
            if self._score_floor is not None and score < self._score_floor:
                continue
            if rep_floor is not None and not self._passes_rep_floor(
                card, context, p_help, rep_floor
            ):
                continue
            scored.append((-score, i, card))
        scored.sort(key=lambda item: (item[0], item[1]))
        return result.model_copy(update={"cards": tuple(card for _, _, card in scored)})

    def _bank_bench(
        self, context: DecisionContext | None
    ) -> tuple[float | None, frozenset[str]]:
        """One pass over the bank: the q-quantile reputation floor (empirical,
        lower-biased) plus the ids of cards that cannot survive downstream —
        reputation below the floor, or a non-positive magnitude (a signed bid
        can never clear a strictly positive EV reserve). Guaranteed losers are
        excluded upstream so they stop spending digest/research slots.
        ``(None, frozenset())`` when the gate is off or the bank is empty."""
        if self._rep_floor_quantile is None:
            return None, frozenset()
        assert self._store is not None
        bank = self._store.snapshot()
        if not bank:
            return None, frozenset()
        stats = []
        for card in bank:
            block = self._reputation.card_stats(card, context)
            magnitude = None if block is None else self._reputation.magnitude_of(block)
            stats.append((card, self._p_help_of(block), magnitude))
        dist = sorted(p_help for _, p_help, _ in stats)
        floor = dist[int(self._rep_floor_quantile * len(dist))]
        benched = frozenset(
            card.id
            for card, p_help, magnitude in stats
            if p_help < floor or (magnitude is not None and magnitude <= 0)
        )
        return floor, benched

    def _p_help(self, card: Card, context: DecisionContext | None) -> float:
        return self._p_help_of(self._reputation.card_stats(card, context))

    def _passes_rep_floor(
        self,
        card: Card,
        context: DecisionContext | None,
        p_help: float,
        rep_floor: float,
    ) -> bool:
        del card, context
        return p_help >= rep_floor

    @staticmethod
    def _p_help_of(block: Any) -> float:
        if block is None or block.p_help_lo20 is None:
            return _NEUTRAL_P_HELP
        p_help = float(block.p_help_lo20)
        return p_help if math.isfinite(p_help) else _NEUTRAL_P_HELP

    def _novelty(self, card: Card, cutoff: datetime) -> float:
        recent = 0
        for event in card.gain_events:
            stamp = event.context.timestamp
            if stamp is None:
                continue
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            if stamp >= cutoff:
                recent += 1
        return 1.0 / (1.0 + recent)


class BootstrapFusedRankingShortlister(FusedRankingShortlister):
    """Fused shortlister whose reputation axis and bench read explicit bootstrap
    EV fields. ``IntroGain_bootstrap_ev_lo20`` is the pessimistic EV used for
    ranking/flooring and ``IntroGain_bootstrap_ev_mean`` is the central EV. The
    inherited ``p_help_*`` fields stay probabilities. A card with no non-founding
    support stays on the cold path; severe founding failures are deleted by the
    write-side evictor. A card with only unused/invalid exposure has magnitude 0
    and is benched instead of receiving cold-start optimism."""

    def _bank_bench(
        self, context: DecisionContext | None
    ) -> tuple[float | None, frozenset[str]]:
        if self._rep_floor_quantile is None:
            return None, frozenset()
        assert self._store is not None
        bank = self._store.snapshot()
        if not bank:
            return None, frozenset()
        warm = []
        benched_ids: set[str] = set()
        for card in bank:
            block = self._reputation.card_stats(card, context)
            magnitude = None if block is None else self._reputation.magnitude_of(block)
            if not self._reputation.event_deltas(card, context):
                if magnitude is not None and magnitude <= 0.0:
                    benched_ids.add(card.id)
                continue
            warm.append((card, self._p_help_of(block), magnitude))
        if not warm:
            return None, frozenset(benched_ids)
        dist = sorted(ev_lo for _, ev_lo, _ in warm)
        floor = dist[int(self._rep_floor_quantile * len(dist))]
        benched_ids.update(
            card.id
            for card, ev_lo, ev_mean in warm
            if ev_lo < floor or (ev_mean is not None and ev_mean <= 0.0)
        )
        return floor, frozenset(benched_ids)

    def _passes_rep_floor(
        self,
        card: Card,
        context: DecisionContext | None,
        p_help: float,
        rep_floor: float,
    ) -> bool:
        if not self._reputation.event_deltas(card, context):
            block = self._reputation.card_stats(card, context)
            magnitude = None if block is None else self._reputation.magnitude_of(block)
            return magnitude is None or magnitude > 0.0
        return p_help >= rep_floor

    @staticmethod
    def _p_help_of(block: Any) -> float:
        if block is None or block.IntroGain_bootstrap_ev_lo20 is None:
            return 0.0
        ev = float(block.IntroGain_bootstrap_ev_lo20)
        return ev if math.isfinite(ev) else 0.0
