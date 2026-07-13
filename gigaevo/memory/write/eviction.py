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
from typing import Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.context.evidence import (
    effective_support,
    event_weight,
    sign_help_counts,
    split_events_by_task,
)
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


@runtime_checkable
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

    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...


@runtime_checkable
class Evictor(Protocol):
    def should_evict(self, card: Card) -> bool: ...

    def eviction_reason(self, card: Card) -> str: ...

    def sweep(self, cards: Sequence[Card]) -> list[str]: ...


def foreign_retention_veto(
    card: Card, *, task_key: str, min_effective_events: float
) -> str | None:
    """Return why foreign hard-sign evidence vetoes deletion, if it does."""
    if not task_key or min_effective_events <= 0.0:
        return None
    by_task: dict[str, list[ContextualGain]] = {}
    for event in card.gain_events:
        foreign_task = event.context.task_key
        if foreign_task and foreign_task != task_key:
            by_task.setdefault(foreign_task, []).append(event)
    for foreign_task in sorted(by_task):
        help_mass, total_mass = sign_help_counts(by_task[foreign_task])
        if total_mass >= min_effective_events and help_mass > 0.5 * total_mass:
            return (
                f"deletion vetoed by foreign task {foreign_task} "
                f"help {help_mass:.6g}/{total_mass:.6g}"
            )
    return None


class CrossTaskRetentionGuard(BaseModel):
    """Veto global deletion when another task has enough net-help evidence."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
    )

    inner: Evictor
    task_key: str = ""
    min_effective_events: float = Field(ge=0.0)

    def should_evict(self, card: Card) -> bool:
        if not self.inner.should_evict(card):
            return False
        return self._foreign_veto(card, log=True) is None

    def eviction_reason(self, card: Card) -> str:
        reason = self.inner.eviction_reason(card)
        if not self.inner.should_evict(card):
            return reason
        veto = self._foreign_veto(card, log=False)
        if veto is None:
            return reason
        return f"{reason}; {veto}"

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        candidates = self.inner.sweep(cards)
        by_id = {card.id: card for card in cards}
        evictable: list[str] = []
        for card_id in candidates:
            card = by_id.get(card_id)
            if card is None or self._foreign_veto(card, log=True) is None:
                evictable.append(card_id)
        return evictable

    def _foreign_veto(self, card: Card, *, log: bool) -> str | None:
        veto = foreign_retention_veto(
            card,
            task_key=self.task_key,
            min_effective_events=self.min_effective_events,
        )
        if veto is not None and log:
            logger.debug(
                "[Memory][CrossTaskRetentionGuard] vetoed deletion of card {}: {}",
                card.id,
                veto,
            )
        return veto


def _harm_evidence(card: Card, task_key: str) -> Card:
    """The card with founding events dropped, for the harm verdict only.

    Founding evidence is origin/admission evidence, not later-use evidence.
    ``HarmEvictor`` remains usage-based; catastrophic origin failures are owned
    by ``BirthFailureEvictor``.
    """
    native, _ = split_events_by_task(card.gain_events, task_key)
    proof = tuple(event for event in native if not event.founding)
    if proof == card.gain_events:
        return card
    return card.model_copy(update={"gain_events": proof})


def _has_negative_direct_evidence(card: Card, task_key: str) -> bool:
    """True when the card has at least one genuinely negative outcome.

    A crash (``invalid``) or a baseline-adjusted loss on a real use counts;
    being ignored by the mutator (``unused``) never does. Stored gains are
    already no-card-baseline-relative, so the neutral point is 0.
    """
    native, _ = split_events_by_task(card.gain_events, task_key)
    for event in native:
        if event.founding or event_weight(event) <= 0.0:
            continue
        if event.invalid:
            return True
        if (
            not event.unused
            and event.gain is not None
            and math.isfinite(float(event.gain))
            and float(event.gain) < 0.0
        ):
            return True
    return False


def _has_positive_direct_evidence(
    card: Card, neutral_gain: float, task_key: str
) -> bool:
    native, _ = split_events_by_task(card.gain_events, task_key)
    for event in native:
        if (
            event.founding
            or event.invalid
            or event.unused
            or event_weight(event) <= 0.0
            or event.gain is None
            or not math.isfinite(float(event.gain))
        ):
            continue
        if float(event.gain) > neutral_gain:
            return True
    return False


class HarmEvictor:
    """Evicts cards whose injection posterior is confidently harmful.

    Harm eviction tombstones the card for the whole run, so the verdict is
    deliberately conservative on three counts. It requires at least one
    genuinely negative outcome (a loss or a crash): exposure-only evidence —
    however much of it — is never enough. It requires the same staleness-aged
    ``effective_support >= min_effective_events`` floor the read/probe lane uses,
    so a card the probe still treats as cold cannot be harm-tombstoned by the
    write lane. And it spares a card whose optimistic bootstrap EV band still
    clears ``neutral_gain``: the sign gate is magnitude-blind, so a fat-tailed
    winner (rare large gains, frequent small losses) can read confidently harmful
    yet carry positive expected value.

    The scorer only needs the ``ContextualCardScorer`` surface to be accepted.
    The aged-support partition and the EV veto require the ``CardValueScorer``
    surface; a scorer without it fails closed (never harm-tombstones) rather than
    raising, keeping the seam usable by minimal contextual scorers.
    """

    def __init__(
        self,
        scorer: ContextualCardScorer,
        *,
        neutral_gain: float = 0.0,
        min_effective_events: float | None = None,
        skip_contextual_without_context: bool = True,
        task_key: str = "",
    ) -> None:
        if not math.isfinite(neutral_gain):
            raise ValueError(f"neutral_gain must be finite, got {neutral_gain}")
        if min_effective_events is None:
            event_floor = (
                scorer.policy_min_effective_events
                if isinstance(scorer, CardValueScorer)
                else 0.0
            )
        else:
            event_floor = min_effective_events
        if event_floor < 0.0 or not math.isfinite(event_floor):
            raise ValueError(
                "min_effective_events must be finite and non-negative, "
                f"got {event_floor}"
            )
        self._scorer = scorer
        self._neutral_gain = float(neutral_gain)
        self._min_effective_events = float(event_floor)
        self._skip_contextual_without_context = bool(skip_contextual_without_context)
        self._task_key = task_key

    def should_evict(self, card: Card) -> bool:
        evidence = _harm_evidence(card, self._task_key)
        if not _has_negative_direct_evidence(evidence, self._task_key):
            return False
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
        context = _writer_context(context, self._task_key)
        if not isinstance(self._scorer, CardValueScorer):
            return False
        deltas = self._scorer.event_deltas(card, context)
        if not deltas:
            return False
        if effective_support(self._scorer, card, deltas, context) < (
            self._min_effective_events
        ):
            return False
        block = self._scorer.card_stats(card, context)
        if not self._scorer.is_confidently_harmful(block):
            return False
        ev_hi = None if block is None else block.IntroGain_bootstrap_ev_hi80
        if ev_hi is not None and math.isfinite(float(ev_hi)):
            return float(ev_hi) <= self._neutral_gain
        return True


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
        task_key: str = "",
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
        self._task_key = task_key

    def should_evict(self, card: Card) -> bool:
        evidence = _harm_evidence(card, self._task_key)
        if _has_positive_direct_evidence(evidence, self._neutral_gain, self._task_key):
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
        context = _writer_context(context, self._task_key)
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
        return effective_support(self._scorer, card, deltas, context)


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
        task_key: str = "",
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
        self._task_key = task_key

    def should_evict(self, card: Card) -> bool:
        scale = self._resolved_scale()
        if scale is None:
            return False
        native, _ = split_events_by_task(card.gain_events, self._task_key)
        losses = [
            float(event.gain)
            for event in native
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
        native, _ = split_events_by_task(card.gain_events, self._task_key)
        min_loss = min(
            (
                float(event.gain)
                for event in native
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
        evidence = _harm_evidence(card, self._task_key)
        block = self._scorer.card_stats(
            evidence, DecisionContext(task_key=self._task_key)
        )
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


def _writer_context(context: DecisionContext | None, task_key: str) -> DecisionContext:
    if context is None:
        return DecisionContext(task_key=task_key)
    return context.model_copy(update={"task_key": task_key})


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
