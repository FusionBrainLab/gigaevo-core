"""Staleness decay over a reputation model: evidence half-life = one bank cycle.

Without decay a benched card is frozen forever: it earns no gain events, so its
posterior never moves and it never re-enters selection. ``DecayingReputation``
decorates any threshold-carrying reputation (global Beta-Binomial or
BD-proximity) and discounts each card's evidence by how stale it is —
staleness measured in bank gain events, not wall clock: ``s`` = stamped events
anywhere in the bank strictly newer than the card's own latest event, and the
half-life is ``half_life_cycles`` bank sizes (one cycle ~= every card earning
one event). The discount ``w = 2**(-s / H)`` shrinks the posterior toward the
uniform prior (``a_eff = 1 + w(a-1)``), scales the event counts (so the harm
gate's ``harm_min_events`` requirement expires automatically), and drops the
EV magnitude once the card holds less than one effective event. Bootstrap EV
then decays known-card deltas toward neutral zero, while genuinely cold cards
remain explorable through the auction's cold scale.
Self-normalizing throughout: rank counts and the bank's own size, no absolute
constants, no wall-clock arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

from scipy.stats import beta

from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.context import NoCardBaselineOutcome
from gigaevo.memory.read.interfaces import DecayCompatibleReputation, NoCardBaseline
from gigaevo.memory.read.staleness import bank_cycle_weight
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
        block = self._inner.card_stats(card, context)
        if block is None:
            return None
        weight = self._weight(card, context)
        if weight >= 1.0:
            return block
        return self._discount(block, weight)

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
        effective_events = max(0.0, (a - 1.0) + (b - 1.0))
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

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        return self._inner.eviction_contexts(card)

    def staleness_weight(
        self, card: Card, context: DecisionContext | None = None
    ) -> float:
        """The bank-cycle discount as the bootstrap resample weight — the same
        ``w`` this decorator applies to the Beta posterior, one mechanism."""
        return self._weight(card, context)

    def fit_no_card_baseline(
        self, outcomes: Sequence[NoCardBaselineOutcome], *, higher_is_better: bool
    ) -> NoCardBaseline:
        return self._inner.fit_no_card_baseline(
            outcomes, higher_is_better=higher_is_better
        )

    def _weight(self, card: Card, context: DecisionContext | None) -> float:
        return bank_cycle_weight(
            card,
            self._store.snapshot(),
            self._half_life_cycles,
            reference_events=self.evidence_events(card, context),
        )

    def _discount(self, block: CardStatsBlock, weight: float) -> CardStatsBlock:
        events_eff = weight * block.intro_events
        updates: dict = {"intro_events": events_eff}
        if block.k_harm is not None:
            updates["k_harm"] = weight * block.k_harm
        magnitude = None if events_eff < 1.0 else block.IntroGain_best_median
        updates["IntroGain_best_median"] = magnitude
        lo: float | None = None
        if block.posterior_a is not None and block.posterior_b is not None:
            a_eff = 1.0 + weight * (float(block.posterior_a) - 1.0)
            b_eff = 1.0 + weight * (float(block.posterior_b) - 1.0)
            lo = float(beta.ppf(self._inner.confident_quantile, a_eff, b_eff))
            updates.update(
                posterior_a=a_eff,
                posterior_b=b_eff,
                p_help_mean=a_eff / (a_eff + b_eff),
                p_help_lo20=lo,
            )
        updates["efficacy_confident"] = bool(
            events_eff >= 1.0
            and lo is not None
            and lo > self._inner.confident_threshold
            and magnitude is not None
            and magnitude > 0
        )
        return block.model_copy(update=updates)
