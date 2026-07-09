"""HarmEvictor delegates to the CardScorer; NullEvictor never evicts."""

from __future__ import annotations

import pytest

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.events import MemoryEvictionSweep
from gigaevo.memory.read.reputation import (
    BDProximityReputation,
    BetaBinomialReputation,
)
from gigaevo.memory.write.eviction import (
    BirthFailureEvictor,
    CompositeEvictor,
    HarmEvictor,
    NullEvictor,
    PolicyNonViableEvictor,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec


class MarkedScorer:
    """Flags cards whose id is in ``harmful`` as confidently harmful."""

    def __init__(self, harmful: set[str]) -> None:
        self._harmful = harmful
        self.scored: list[str] = []

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        self.scored.append(card.id)
        if card.id in self._harmful:
            return CardStatsBlock(efficacy_confident=True)
        return None

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool:
        return block is not None and bool(block.efficacy_confident)

    @property
    def requires_decision_context(self) -> bool:
        return False

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        del card
        return (None,)


class ContextOnlyScorer:
    """Raises if an evictor asks for a contextless score."""

    @property
    def requires_decision_context(self) -> bool:
        return True

    @property
    def policy_min_effective_events(self) -> float:
        return 1.0

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        del card
        return ()

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        del card
        if context is None:
            raise AssertionError("contextless card_stats call")
        return None

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool:
        del block
        return False

    def magnitude_of(self, block: CardStatsBlock | None) -> float | None:
        del block
        return None

    def event_deltas(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        del card
        if context is None:
            raise AssertionError("contextless event_deltas call")
        return ()

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        del card, context
        return ()

    def staleness_weight(
        self, card: Card, context: DecisionContext | None = None
    ) -> float:
        del card, context
        return 1.0


class MismatchedWeightReputation(BetaBinomialReputation):
    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        del card, context
        return ()


@pytest.fixture
def captured_events(monkeypatch):
    events: list = []
    monkeypatch.setattr(
        "gigaevo.memory.write.eviction.emit_memory_event", events.append
    )
    return events


def test_should_evict_delegates_through_scorer(make_card):
    bad = make_card()
    good = make_card()
    evictor = HarmEvictor(MarkedScorer({bad.id}))
    assert evictor.should_evict(bad) is True
    assert evictor.should_evict(good) is False


def test_sweep_returns_harmful_ids_and_emits_event(make_card, captured_events):
    bad = make_card()
    good = make_card()
    evictor = HarmEvictor(MarkedScorer({bad.id}))
    assert evictor.sweep([good, bad]) == [bad.id]
    assert len(captured_events) == 1
    event = captured_events[0]
    assert isinstance(event, MemoryEvictionSweep)
    assert event.bank_count == 2
    assert event.evicted_ids == (bad.id,)


def test_sweep_without_evictions_emits_nothing(make_card, captured_events):
    evictor = HarmEvictor(MarkedScorer(set()))
    assert evictor.sweep([make_card(), make_card()]) == []
    assert captured_events == []


def test_null_evictor_never_evicts(make_card):
    evictor = NullEvictor()
    card = make_card()
    assert evictor.should_evict(card) is False
    assert evictor.sweep([card]) == []


def test_contextual_evictors_skip_without_explicit_context(make_card):
    scorer = ContextOnlyScorer()
    card = make_card()

    assert HarmEvictor(scorer).should_evict(card) is False
    assert PolicyNonViableEvictor(scorer, neutral_gain=0.0).should_evict(card) is False


def _policy_evictor(*, min_effective_events: float = 3.0):
    return PolicyNonViableEvictor(
        BetaBinomialReputation(),
        neutral_gain=0.0,
        min_effective_events=min_effective_events,
    )


def test_policy_nonviable_keeps_single_negative_until_min_support(
    make_card, make_event
):
    evictor = _policy_evictor()
    card = make_card(gain_events=(make_event(-0.1),))

    assert evictor.should_evict(card) is False


def test_policy_nonviable_rejects_misaligned_scorer_weights(make_card, make_event):
    evictor = PolicyNonViableEvictor(
        MismatchedWeightReputation(),
        neutral_gain=0.0,
        min_effective_events=1.0,
    )
    card = make_card(gain_events=(make_event(-0.1),))

    with pytest.raises(ValueError, match="event_weights must align"):
        evictor.should_evict(card)


def test_policy_nonviable_derives_min_support_from_scorer(make_card, make_event):
    evictor = PolicyNonViableEvictor(
        BetaBinomialReputation(harm_min_events=4),
        neutral_gain=0.0,
    )
    below_floor = make_card(gain_events=tuple(make_event(-0.1) for _ in range(3)))
    at_floor = make_card(gain_events=tuple(make_event(-0.1) for _ in range(4)))

    assert evictor.should_evict(below_floor) is False
    assert evictor.should_evict(at_floor) is True


def test_policy_nonviable_evicts_repeated_negative_use_zombie(make_card, make_event):
    evictor = _policy_evictor()
    card = make_card(gain_events=tuple(make_event(-0.1) for _ in range(3)))

    assert evictor.should_evict(card) is True
    assert "policy non-viable" in evictor.eviction_reason(card)


def test_policy_nonviable_evicts_repeated_unused_only_zombie(make_card, make_event):
    evictor = _policy_evictor()
    card = make_card(gain_events=tuple(make_event(0.0, unused=True) for _ in range(3)))

    assert evictor.should_evict(card) is True


def test_policy_nonviable_skips_contextual_scorer_without_valid_event_context(
    make_card, make_event
):
    space = BehaviorSpace(
        bins={"b": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    evictor = PolicyNonViableEvictor(
        BDProximityReputation(behavior_space=space),
        neutral_gain=0.0,
        min_effective_events=3.0,
    )
    card = make_card(gain_events=tuple(make_event(-0.1) for _ in range(3)))

    assert evictor.should_evict(card) is False


def test_harm_evictor_keeps_bd_card_with_positive_evidence_elsewhere(
    make_card, make_event
):
    space = BehaviorSpace(
        bins={"b": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    evictor = HarmEvictor(BDProximityReputation(behavior_space=space))
    card = make_card(
        gain_events=(
            make_event(-0.5, metrics={"b": 0.2}),
            make_event(-0.5, metrics={"b": 0.2}),
            make_event(-0.5, metrics={"b": 0.2}),
            make_event(0.5, metrics={"b": 0.8}),
        )
    )

    assert evictor.should_evict(card) is False


def test_policy_nonviable_evicts_repeated_negative_bd_local_zombie(
    make_card, make_event
):
    space = BehaviorSpace(
        bins={"b": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    evictor = PolicyNonViableEvictor(
        BDProximityReputation(behavior_space=space),
        neutral_gain=0.0,
        min_effective_events=3.0,
    )
    card = make_card(
        gain_events=tuple(make_event(-0.1, metrics={"b": 0.2}) for _ in range(3))
    )

    assert evictor.should_evict(card) is True


def test_policy_nonviable_keeps_bd_card_with_positive_evidence_elsewhere(
    make_card, make_event
):
    space = BehaviorSpace(
        bins={"b": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    evictor = PolicyNonViableEvictor(
        BDProximityReputation(behavior_space=space),
        neutral_gain=0.0,
        min_effective_events=3.0,
    )
    card = make_card(
        gain_events=(
            make_event(-0.1, metrics={"b": 0.2}),
            make_event(-0.1, metrics={"b": 0.2}),
            make_event(-0.1, metrics={"b": 0.2}),
            make_event(0.2, metrics={"b": 0.8}),
        )
    )

    assert evictor.should_evict(card) is False


def test_policy_nonviable_keeps_baseline_positive_card(make_card, make_event):
    evictor = _policy_evictor()
    # Stored gain is already child-parent minus no-card baseline. A raw child
    # regression that beats an even-worse no-card baseline is positive here.
    card = make_card(gain_events=(make_event(0.1),))

    assert evictor.should_evict(card) is False


def test_policy_nonviable_keeps_mixed_sign_card(make_card, make_event):
    evictor = _policy_evictor()
    card = make_card(gain_events=(make_event(-0.3), make_event(0.1)))

    assert evictor.should_evict(card) is False


def test_policy_nonviable_ignores_founding_only_cards(make_card, make_event):
    evictor = _policy_evictor()
    card = make_card(gain_events=(make_event(-0.3, founding=True),))

    assert evictor.should_evict(card) is False


def _metrics(significant_change: float | None) -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness",
                higher_is_better=True,
                is_primary=True,
                significant_change=significant_change,
            )
        }
    )


def test_birth_failure_evicts_catastrophic_founding_loss(make_card, make_event):
    evictor = BirthFailureEvictor(metrics_context=_metrics(0.1))
    card = make_card(gain_events=(make_event(-0.21, founding=True),))
    assert evictor.should_evict(card) is True
    assert "catastrophic founding loss" in evictor.eviction_reason(card)


def test_birth_failure_uses_metric_scale_not_raw_sign(make_card, make_event):
    evictor = BirthFailureEvictor(metrics_context=_metrics(0.1))
    mild = make_card(gain_events=(make_event(-0.19, founding=True),))
    no_scale = BirthFailureEvictor(metrics_context=_metrics(None))
    severe_without_scale = make_card(gain_events=(make_event(-10.0, founding=True),))
    assert evictor.should_evict(mild) is False
    assert no_scale.should_evict(severe_without_scale) is False


def test_birth_failure_allows_later_rescue_evidence(make_card, make_event):
    evictor = BirthFailureEvictor(
        scorer=BetaBinomialReputation(), metrics_context=_metrics(0.1)
    )
    rescued = make_card(
        gain_events=(
            make_event(-0.5, founding=True),
            make_event(0.2),
            make_event(0.2),
            make_event(0.2),
        )
    )
    assert evictor.should_evict(rescued) is False


def test_composite_evictor_runs_birth_failure_before_harm(make_card, make_event):
    birth = BirthFailureEvictor(metrics_context=_metrics(0.1))
    harm = HarmEvictor(BetaBinomialReputation())
    evictor = CompositeEvictor((birth, harm))
    birth_bad = make_card(gain_events=(make_event(-0.21, founding=True),))
    use_bad = make_card(gain_events=tuple(make_event(-0.5) for _ in range(3)))

    assert evictor.should_evict(birth_bad) is True
    assert "catastrophic founding loss" in evictor.eviction_reason(birth_bad)
    assert evictor.should_evict(use_bad) is True
    assert "confidently harmful" in evictor.eviction_reason(use_bad)


def test_founding_events_never_trigger_eviction(make_card, make_event):
    """Pure HarmEvictor remains later-use-only: founding origin evidence is not
    enough to trip usage-harm deletion. The recommended composite evictor adds
    a separate catastrophic-birth policy for that case.
    """
    evictor = HarmEvictor(BetaBinomialReputation())
    founding_only = make_card(
        gain_events=tuple(make_event(-0.5, founding=True) for _ in range(3))
    )
    use_only = make_card(gain_events=tuple(make_event(-0.5) for _ in range(3)))
    assert evictor.should_evict(founding_only) is False
    assert evictor.should_evict(use_only) is True


def test_founding_events_do_not_lower_the_harm_bar(make_card, make_event):
    """Founding events must not count toward ``harm_min_events``: a card with two
    losing use events and one losing founding event is judged on the two use
    events (below the bar), not evicted as if it had three."""
    evictor = HarmEvictor(BetaBinomialReputation())
    mixed = make_card(
        gain_events=(
            make_event(-0.5),
            make_event(-0.5),
            make_event(-0.5, founding=True),
        )
    )
    assert evictor.should_evict(mixed) is False


def test_founding_strip_holds_under_bd_proximity_scorer(make_card, make_event):
    """The strip must hold under BD-local reputation too: founding-only has no
    eviction context, while real use evidence in a valid BD cell can evict."""
    space = BehaviorSpace(
        bins={"b": LinearBinning(min_val=0.0, max_val=1.0, num_bins=4)}
    )
    evictor = HarmEvictor(BDProximityReputation(behavior_space=space))
    founding_only = make_card(
        gain_events=tuple(
            make_event(-0.5, founding=True, metrics={"b": 0.25}) for _ in range(3)
        )
    )
    use_only = make_card(
        gain_events=tuple(make_event(-0.5, metrics={"b": 0.25}) for _ in range(3))
    )
    assert evictor.should_evict(founding_only) is False
    assert evictor.should_evict(use_only) is True
