"""Configured eviction policies over the card bank.

The write path must not import the read system, but eviction verdicts read the
same reputation/value view as prompt selection. ``CardScorer`` /
``CardValueScorer`` invert that dependency: this module declares the scoring
surface it needs, ``read/reputation.py``'s reputation models satisfy it
structurally, and the integration config wires one shared instance into both
sides.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol

from loguru import logger

from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.events import MemoryEvictionSweep, emit_memory_event
from gigaevo.programs.metrics.context import MetricsContext


class CardScorer(Protocol):
    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None: ...

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool: ...


class ContextualCardScorer(CardScorer, Protocol):
    @property
    def requires_decision_context(self) -> bool: ...

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]: ...


class CardValueScorer(ContextualCardScorer, Protocol):
    @property
    def policy_min_effective_events(self) -> float: ...

    def magnitude_of(self, block: CardStatsBlock | None) -> float | None: ...

    def event_deltas(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...

    def staleness_weight(
        self, card: Card, context: DecisionContext | None = None
    ) -> float: ...


class Evictor(Protocol):
    def should_evict(self, card: Card) -> bool: ...

    def eviction_reason(self, card: Card) -> str: ...

    def sweep(self, cards: Sequence[Card]) -> list[str]: ...


def _harm_evidence(card: Card) -> Card:
    """The card with founding events dropped, for the harm verdict only.

    Founding evidence is origin/admission evidence, not later-use evidence.
    ``HarmEvictor`` remains usage-based; catastrophic origin failures are owned
    by ``BirthFailureEvictor``.
    """
    if not any(event.founding for event in card.gain_events):
        return card
    return card.model_copy(
        update={"gain_events": tuple(e for e in card.gain_events if not e.founding)}
    )


def _event_weight(event: ContextualGain) -> float:
    attr = event.attribution
    if attr is not None and attr.credit_weight is not None:
        weight = float(attr.credit_weight)
    elif event.founding:
        weight = 0.0
    else:
        weight = 1.0
    return weight if math.isfinite(weight) and weight > 0.0 else 0.0


def _has_positive_direct_evidence(card: Card, neutral_gain: float) -> bool:
    for event in card.gain_events:
        if (
            event.founding
            or event.invalid
            or event.unused
            or _event_weight(event) <= 0.0
            or event.gain is None
            or not math.isfinite(float(event.gain))
        ):
            continue
        if float(event.gain) > neutral_gain:
            return True
    return False


class HarmEvictor:
    """Evicts cards whose injection posterior is confidently harmful."""

    def __init__(
        self,
        scorer: ContextualCardScorer,
        *,
        skip_contextual_without_context: bool = True,
    ) -> None:
        self._scorer = scorer
        self._skip_contextual_without_context = bool(skip_contextual_without_context)

    def should_evict(self, card: Card) -> bool:
        evidence = _harm_evidence(card)
        contexts = self._eviction_contexts(evidence)
        return bool(contexts) and all(
            self._is_harmful_in_context(evidence, context) for context in contexts
        )

    def eviction_reason(self, card: Card) -> str:
        del card
        return "injection posterior confidently harmful"

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        evicted = [card.id for card in cards if self.should_evict(card)]
        if evicted:
            emit_memory_event(
                MemoryEvictionSweep(bank_count=len(cards), evicted_ids=tuple(evicted))
            )
            logger.info(
                "[Memory][Evictor] Sweep evicting {}/{} card(s) as confidently harmful: {}",
                len(evicted),
                len(cards),
                evicted,
            )
        return evicted

    def _eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        contexts = self._scorer.eviction_contexts(card)
        if contexts:
            return contexts
        if (
            self._skip_contextual_without_context
            and self._scorer.requires_decision_context
        ):
            return ()
        return (None,)

    def _is_harmful_in_context(
        self, card: Card, context: DecisionContext | None
    ) -> bool:
        return self._scorer.is_confidently_harmful(
            self._scorer.card_stats(card, context)
        )


class PolicyNonViableEvictor:
    """Evicts cards the active value policy has made non-viable.

    This is not a statistical harm verdict. It is bank hygiene for cards with
    real non-founding evidence that the configured reputation/EV stack prices at
    or below the neutral no-card baseline, while no direct baseline-adjusted use
    has ever beaten that neutral point. Mixed-sign cards are deliberately left to
    the normal harm/confidence path.
    """

    def __init__(
        self,
        scorer: CardValueScorer,
        *,
        neutral_gain: float,
        min_effective_events: float | None = None,
        skip_contextual_without_context: bool = True,
    ) -> None:
        if not math.isfinite(neutral_gain):
            raise ValueError(f"neutral_gain must be finite, got {neutral_gain}")
        event_floor = (
            scorer.policy_min_effective_events
            if min_effective_events is None
            else min_effective_events
        )
        if event_floor < 0.0 or not math.isfinite(event_floor):
            raise ValueError(
                "min_effective_events must be finite and non-negative, "
                f"got {event_floor}"
            )
        self._scorer = scorer
        self._neutral_gain = float(neutral_gain)
        self._min_effective_events = float(event_floor)
        self._skip_contextual_without_context = bool(skip_contextual_without_context)

    def should_evict(self, card: Card) -> bool:
        evidence = _harm_evidence(card)
        if _has_positive_direct_evidence(evidence, self._neutral_gain):
            return False
        contexts = self._eviction_contexts(evidence)
        return bool(contexts) and all(
            self._context_is_nonviable(evidence, context) for context in contexts
        )

    def eviction_reason(self, card: Card) -> str:
        del card
        return (
            "policy non-viable: enough effective evidence, non-positive EV, "
            "and no positive direct evidence"
        )

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        evicted = [card.id for card in cards if self.should_evict(card)]
        if evicted:
            emit_memory_event(
                MemoryEvictionSweep(bank_count=len(cards), evicted_ids=tuple(evicted))
            )
            logger.info(
                "[Memory][PolicyNonViableEvictor] Sweep evicting {}/{} card(s) "
                "with non-positive EV and no positive direct evidence: {}",
                len(evicted),
                len(cards),
                evicted,
            )
        return evicted

    def _eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        contexts = self._scorer.eviction_contexts(card)
        if contexts:
            return contexts
        if (
            self._skip_contextual_without_context
            and self._scorer.requires_decision_context
        ):
            return ()
        return (None,)

    def _context_is_nonviable(
        self, card: Card, context: DecisionContext | None
    ) -> bool:
        deltas = self._scorer.event_deltas(card, context)
        if not deltas:
            return False
        if self._effective_support(card, deltas, context) < self._min_effective_events:
            return False
        if any(delta > self._neutral_gain for delta in deltas):
            return False
        block = self._scorer.card_stats(card, context)
        ev = self._scorer.magnitude_of(block)
        return (
            ev is not None
            and math.isfinite(float(ev))
            and float(ev) <= self._neutral_gain
        )

    def _effective_support(
        self,
        card: Card,
        deltas: Sequence[float],
        context: DecisionContext | None,
    ) -> float:
        weights = self._scorer.event_weights(card, context)
        if len(weights) != len(deltas):
            raise ValueError(
                "event_weights must align with event_deltas: "
                f"{len(weights)} weights for {len(deltas)} deltas"
            )
        event_support = sum(
            max(0.0, float(weight))
            for weight in weights
            if math.isfinite(float(weight))
        )
        factor = float(self._scorer.staleness_weight(card, context))
        if math.isfinite(factor) and factor >= 0.0:
            event_support *= factor
        return event_support


class BirthFailureEvictor:
    """Deletes cards whose only birth evidence is catastrophically bad.

    This is intentionally separate from ``HarmEvictor``. Harm eviction is about
    later card use. Birth-failure eviction is an admission/sweep guard: if the
    source child regressed by a task-scaled catastrophic margin and later direct
    evidence has not rescued the card, do not let it remain as cold advice.
    """

    def __init__(
        self,
        *,
        scorer: CardScorer | None = None,
        metrics_context: MetricsContext | None = None,
        scale: float | None = None,
        scale_multiplier: float = 2.0,
        rescue_min_events: float = 3.0,
        rescue_p_help_threshold: float = 0.5,
        rescue_ev_threshold: float = 0.0,
    ) -> None:
        if scale is not None and (not math.isfinite(scale) or scale <= 0.0):
            raise ValueError(f"scale must be finite and positive, got {scale}")
        if scale_multiplier <= 0.0 or not math.isfinite(scale_multiplier):
            raise ValueError(
                f"scale_multiplier must be finite and positive, got {scale_multiplier}"
            )
        self._scorer = scorer
        self._metrics_context = metrics_context
        self._scale = scale
        self._scale_multiplier = scale_multiplier
        self._rescue_min_events = rescue_min_events
        self._rescue_p_help_threshold = rescue_p_help_threshold
        self._rescue_ev_threshold = rescue_ev_threshold

    def should_evict(self, card: Card) -> bool:
        scale = self._resolved_scale()
        if scale is None:
            return False
        losses = [
            float(event.gain)
            for event in card.gain_events
            if event.founding
            and event.gain is not None
            and math.isfinite(float(event.gain))
        ]
        if not losses:
            return False
        if min(losses) > -(self._scale_multiplier * scale):
            return False
        return not self._has_rescue_evidence(card)

    def eviction_reason(self, card: Card) -> str:
        scale = self._resolved_scale()
        min_loss = min(
            (
                float(event.gain)
                for event in card.gain_events
                if event.founding
                and event.gain is not None
                and math.isfinite(float(event.gain))
            ),
            default=float("nan"),
        )
        threshold = (
            -(self._scale_multiplier * scale) if scale is not None else float("nan")
        )
        return (
            "catastrophic founding loss "
            f"{min_loss:.6g} <= {threshold:.6g} without later rescue evidence"
        )

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        evicted = [card.id for card in cards if self.should_evict(card)]
        if evicted:
            emit_memory_event(
                MemoryEvictionSweep(bank_count=len(cards), evicted_ids=tuple(evicted))
            )
            logger.info(
                "[Memory][BirthEvictor] Sweep evicting {}/{} card(s) for catastrophic birth loss: {}",
                len(evicted),
                len(cards),
                evicted,
            )
        return evicted

    def _resolved_scale(self) -> float | None:
        if self._scale is not None:
            return self._scale
        if self._metrics_context is None:
            return None
        sig = self._metrics_context.get_primary_spec().significant_change
        if sig is not None and math.isfinite(float(sig)) and float(sig) > 0.0:
            return float(sig)
        return None

    def _has_rescue_evidence(self, card: Card) -> bool:
        if self._scorer is None:
            return False
        block = self._scorer.card_stats(_harm_evidence(card))
        if block is None or block.intro_events < self._rescue_min_events:
            return False
        p_help = block.p_help_lo20
        if p_help is None or not math.isfinite(float(p_help)):
            return False
        ev = (
            block.IntroGain_bootstrap_ev_lo20
            if block.IntroGain_bootstrap_ev_lo20 is not None
            else block.IntroGain_best_median
        )
        return (
            float(p_help) > self._rescue_p_help_threshold
            and ev is not None
            and math.isfinite(float(ev))
            and float(ev) > self._rescue_ev_threshold
        )


class CompositeEvictor:
    """Runs several eviction policies as one write-path evictor."""

    def __init__(self, evictors: Sequence[Evictor]) -> None:
        self._evictors = tuple(evictors)

    def should_evict(self, card: Card) -> bool:
        return any(evictor.should_evict(card) for evictor in self._evictors)

    def eviction_reason(self, card: Card) -> str:
        for evictor in self._evictors:
            if evictor.should_evict(card):
                return evictor.eviction_reason(card)
        return ""

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        evicted = [card.id for card in cards if self.should_evict(card)]
        if evicted:
            emit_memory_event(
                MemoryEvictionSweep(bank_count=len(cards), evicted_ids=tuple(evicted))
            )
            logger.info(
                "[Memory][Evictor] Sweep evicting {}/{} card(s): {}",
                len(evicted),
                len(cards),
                evicted,
            )
        return evicted


class NullEvictor:
    """No-op evictor: never evicts. Runs the write path with eviction sweeps
    disabled, the bank-maintenance twin of ``memory=none`` on the read side."""

    def should_evict(self, card: Card) -> bool:
        return False

    def eviction_reason(self, card: Card) -> str:
        del card
        return ""

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        return []
