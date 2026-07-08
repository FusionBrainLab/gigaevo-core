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
    """The strip must hold under BD-local reputation too: context-less harm
    scoring falls back to global evidence over the already-stripped events, so
    founding-only survives pure HarmEvictor while use-only evicts."""
    space = BehaviorSpace(
        bins={"b": LinearBinning(min_val=0.0, max_val=1.0, num_bins=4)}
    )
    evictor = HarmEvictor(BDProximityReputation(behavior_space=space))
    founding_only = make_card(
        gain_events=tuple(make_event(-0.5, founding=True) for _ in range(3))
    )
    use_only = make_card(gain_events=tuple(make_event(-0.5) for _ in range(3)))
    assert evictor.should_evict(founding_only) is False
    assert evictor.should_evict(use_only) is True
