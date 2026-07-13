"""Staleness decay over a reputation model: evidence half-life = one bank cycle.

Without decay a benched card is frozen forever: it earns no gain events, so its
posterior never moves and it never re-enters selection. ``DecayingReputation``
decorates any threshold-carrying reputation (global Beta-Binomial or
BD-proximity) and discounts every event by its own bank rank: ``s_i`` is the
number of native bank stamps strictly newer than event ``i`` and
``w_i = 2**(-s_i / H)``. Each posterior contribution is scaled by causal credit
times ``w_i`` toward the card's cold prior ``(a0, b0)`` (Beta(1,1) when none is
configured). Effective counts therefore expire from the harm gate, and the
magnitude expires below one effective event, without a fresh event reviving
older history.
Self-normalizing throughout: rank counts and the bank's own size, no absolute
constants, no wall-clock arithmetic.
"""

from __future__ import annotations

import math

from scipy.stats import beta

from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.read.interfaces import DecayCompatibleReputation
from gigaevo.memory.read.staleness import bank_cycle_event_weights
from gigaevo.memory.storage.base import MemoryStore


class DecayingReputation:
    """Discounts the inner reputation's card block by bank-cycle staleness."""

    def __init__(
        self,
        inner: DecayCompatibleReputation,
        store: MemoryStore,
        *,
        half_life_cycles: float = 1.0,
    ) -> None:
        if half_life_cycles <= 0:
            raise ValueError(
                f"half_life_cycles must be positive, got {half_life_cycles}"
            )
        self._inner = inner
        self._store = store
        self._half_life_cycles = half_life_cycles

    @property
    def requires_decision_context(self) -> bool:
        return self._inner.requires_decision_context

    @property
    def policy_min_effective_events(self) -> float:
        return self._inner.policy_min_effective_events

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        task_key = context.task_key if context is not None else ""
        staleness = bank_cycle_event_weights(
            card.gain_events,
            self._store.snapshot(),
            self._half_life_cycles,
            task_key=task_key,
        )
        if all(weight == 1.0 for weight in staleness):
            return self._inner.card_stats(card, context)
        return self._inner.card_stats_with_staleness(
            card,
            context,
            staleness_weights=staleness,
        )

    def prior_base(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, float]:
        return self._inner.prior_base(card, context)

    def posterior_of(self, block: CardStatsBlock | None) -> tuple[float, float]:
        return self._inner.posterior_of(block)

    def magnitude_of(self, block: CardStatsBlock | None) -> float | None:
        return self._inner.magnitude_of(block)

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool:
        if block is None or block.posterior_a is None or block.posterior_b is None:
            return False
        a = float(block.posterior_a)
        b = float(block.posterior_b)
        if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
            return False
        effective_events = max(0.0, float(block.intro_events))
        if effective_events < self._inner.harm_min_events:
            return False
        return (
            float(beta.ppf(self._inner.harm_quantile, a, b))
            < self._inner.harm_threshold
        )

    def event_deltas(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        return self._inner.event_deltas(card, context)

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        return self._inner.event_weights(card, context)

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        return self._inner.evidence_events(card, context)

    def event_ses(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float | None, ...]:
        return self._inner.event_ses(card, context)

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        return self._inner.eviction_contexts(card)

    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        """Bank-cycle ages aligned with the delegated EV evidence subset."""
        task_key = context.task_key if context is not None else ""
        events = self.evidence_events(card, context)
        return bank_cycle_event_weights(
            events,
            self._store.snapshot(),
            self._half_life_cycles,
            task_key=task_key,
        )
