"""Harm eviction over the card bank.

The write path must not import the read system, but the harm verdict IS the
read side's injection posterior. ``CardScorer`` inverts that dependency: this
module declares the scoring surface it needs, ``read/reputation.py``'s
reputation models satisfy it structurally, and the integration config wires
one shared instance into both sides.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol

from loguru import logger

from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.events import MemoryEvictionSweep, emit_memory_event
from gigaevo.programs.metrics.context import MetricsContext


class CardScorer(Protocol):
    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None: ...

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool: ...


class Evictor(Protocol):
    def should_evict(self, card: Card) -> bool: ...

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


class HarmEvictor:
    """Evicts cards whose injection posterior is confidently harmful."""

    def __init__(self, scorer: CardScorer) -> None:
        self._scorer = scorer

    def should_evict(self, card: Card) -> bool:
        evidence = _harm_evidence(card)
        return self._scorer.is_confidently_harmful(self._scorer.card_stats(evidence))

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
                reason = getattr(evictor, "eviction_reason", None)
                if callable(reason):
                    return str(reason(card))
                return type(evictor).__name__
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
    """No-op evictor: never evicts. Runs the write path with the harm sweep
    disabled, the bank-maintenance twin of ``memory=none`` on the read side."""

    def should_evict(self, card: Card) -> bool:
        return False

    def eviction_reason(self, card: Card) -> str:
        del card
        return ""

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        return []
