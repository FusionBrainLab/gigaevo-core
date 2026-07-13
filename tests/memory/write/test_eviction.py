"""HarmEvictor delegates to the CardScorer; NullEvictor never evicts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.context.evidence import sign_help_counts
from gigaevo.memory.events import MemoryEvictionSweep
from gigaevo.memory.read.auction import BootstrapThompsonAuctioneer
from gigaevo.memory.read.probe import ColdProbePolicy
from gigaevo.memory.read.projection import AuctionCandidateProjector
from gigaevo.memory.read.reputation import (
    BDProximityReputation,
    BetaBinomialReputation,
    BootstrapReputation,
)
from gigaevo.memory.write.eviction import (
    BirthFailureEvictor,
    CompositeEvictor,
    CrossTaskRetentionGuard,
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

    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        return tuple(1.0 for _ in self.event_deltas(card, context))


class MismatchedWeightReputation(BetaBinomialReputation):
    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        del card, context
        return ()


class MismatchedStalenessReputation(BetaBinomialReputation):
    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        del card, context
        return ()


class NonFiniteStalenessReputation(BetaBinomialReputation):
    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        del card, context
        return (float("nan"), 0.5, -1.0)


@pytest.fixture
def captured_events(monkeypatch):
    events: list = []
    monkeypatch.setattr(
        "gigaevo.memory.write.eviction.emit_memory_event", events.append
    )
    return events


def test_should_evict_delegates_through_scorer(make_card, make_event):
    bad = make_card(gain_events=(make_event(-0.1),))
    good = make_card(gain_events=(make_event(-0.1),))
    evictor = HarmEvictor(MarkedScorer({bad.id}))
    assert evictor.should_evict(bad) is True
    assert evictor.should_evict(good) is False


def test_sweep_returns_harmful_ids_and_emits_event(
    make_card, make_event, captured_events
):
    bad = make_card(gain_events=(make_event(-0.1),))
    good = make_card(gain_events=(make_event(-0.1),))
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


def test_harm_evictor_never_evicts_on_ignores_alone(make_card, make_event):
    # Being ignored by the mutator is weak exposure evidence, not proof of
    # harm; harm-eviction tombstones for the whole run, so it must require at
    # least one genuinely negative outcome (a loss or a crash).
    evictor = HarmEvictor(BetaBinomialReputation())
    ignored = make_card(
        gain_events=tuple(make_event(0.0, unused=True) for _ in range(4))
    )
    assert evictor.should_evict(ignored) is False


def test_harm_evictor_keeps_card_with_only_wins_and_ignores(make_card, make_event):
    evictor = HarmEvictor(BetaBinomialReputation())
    card = make_card(
        gain_events=(
            make_event(0.02),
            make_event(0.0, unused=True),
            make_event(0.0, unused=True),
            make_event(0.0, unused=True),
        )
    )
    assert evictor.should_evict(card) is False


def test_harm_evictor_still_evicts_genuine_losses(make_card, make_event):
    evictor = HarmEvictor(BetaBinomialReputation())
    loser = make_card(gain_events=tuple(make_event(-0.1) for _ in range(4)))
    assert evictor.should_evict(loser) is True


def test_harm_evictor_only_consumes_writer_task_events(make_card, make_event):
    task_a_loser = make_card(
        gain_events=tuple(make_event(-0.1, task_key="task-a") for _ in range(4))
    )

    assert (
        HarmEvictor(BetaBinomialReputation(), task_key="task-b").should_evict(
            task_a_loser
        )
        is False
    )
    assert (
        HarmEvictor(BetaBinomialReputation(), task_key="task-a").should_evict(
            task_a_loser
        )
        is True
    )


def test_cross_task_guard_is_vacuous_for_single_task_bank(make_card, make_event):
    cards = [
        make_card(
            gain_events=tuple(make_event(-0.1, task_key="task-a") for _ in range(count))
        )
        for count in (4, 2)
    ]
    cards.append(
        make_card(
            gain_events=tuple(make_event(0.1, task_key="task-a") for _ in range(4))
        )
    )
    inner = HarmEvictor(BetaBinomialReputation(), task_key="task-a")
    guard = CrossTaskRetentionGuard(
        inner=inner, task_key="task-a", min_effective_events=3
    )

    assert [guard.should_evict(card) for card in cards] == [
        inner.should_evict(card) for card in cards
    ]
    assert guard.sweep(cards) == inner.sweep(cards)
    assert [guard.eviction_reason(card) for card in cards] == [
        inner.eviction_reason(card) for card in cards
    ]


def test_cross_task_guard_vetoes_only_foreign_majority_help(make_card, make_event):
    native_losses = tuple(make_event(-0.1, task_key="task-a") for _ in range(4))
    majority_help = make_card(
        gain_events=(
            *native_losses,
            make_event(0.2, task_key="task-b"),
            make_event(0.1, task_key="task-b"),
            make_event(-0.1, task_key="task-b"),
        )
    )
    majority_fail = make_card(
        gain_events=(
            *native_losses,
            make_event(0.2, task_key="task-b"),
            make_event(-0.1, task_key="task-b"),
            make_event(0.0, invalid=True, task_key="task-b"),
        )
    )
    guard = CrossTaskRetentionGuard(
        inner=HarmEvictor(BetaBinomialReputation(), task_key="task-a"),
        task_key="task-a",
        min_effective_events=3,
    )

    assert guard.should_evict(majority_help) is False
    assert "deletion vetoed by foreign task task-b help 2/3" in guard.eviction_reason(
        majority_help
    )
    assert guard.should_evict(majority_fail) is True
    assert guard.sweep([majority_help, majority_fail]) == [majority_fail.id]


def test_cross_task_guard_does_not_veto_below_support_floor(make_card, make_event):
    card = make_card(
        gain_events=(
            *(make_event(-0.1, task_key="task-a") for _ in range(4)),
            make_event(0.2, task_key="task-b"),
            make_event(0.1, task_key="task-b"),
        )
    )
    guard = CrossTaskRetentionGuard(
        inner=HarmEvictor(BetaBinomialReputation(), task_key="task-a"),
        task_key="task-a",
        min_effective_events=3,
    )

    assert guard.should_evict(card) is True
    assert guard.sweep([card]) == [card.id]


def test_cross_task_guard_and_reputation_share_foreign_sign_fold(make_card, make_event):
    foreign = (
        make_event(100.0, founding=True, task_key="task-b"),
        make_event(100.0, unused=True, task_key="task-b"),
        make_event(0.2, task_key="task-b"),
        make_event(0.1, task_key="task-b"),
        make_event(0.0, invalid=True, task_key="task-b"),
    )
    card = make_card(
        gain_events=(
            *(make_event(-0.1, task_key="task-a") for _ in range(4)),
            *foreign,
        )
    )
    guard = CrossTaskRetentionGuard(
        inner=HarmEvictor(BetaBinomialReputation(), task_key="task-a"),
        task_key="task-a",
        min_effective_events=3,
    )

    veto = guard._foreign_veto(card, log=False)
    block = BetaBinomialReputation().card_stats(
        card, DecisionContext(task_key="task-a")
    )

    assert veto == "deletion vetoed by foreign task task-b help 2/3"
    assert sign_help_counts(foreign) == (2.0, 3.0)
    assert block is not None
    assert (block.foreign_help_events, block.foreign_total_events) == (2.0, 3.0)


def test_cross_task_veto_is_invariant_to_same_sign_magnitude_and_se(
    make_card, make_event
):
    native_losses = tuple(make_event(-0.1, task_key="task-a") for _ in range(4))
    foreign_variants = (
        tuple(
            make_event(gain, task_key="task-b").model_copy(update={"gain_se": 1e300})
            for gain in (1e-300, 2e-300, -1e-300)
        ),
        tuple(
            make_event(gain, task_key="task-b").model_copy(update={"gain_se": 0.0})
            for gain in (1e300, 2e300, -1e300)
        ),
    )
    guard = CrossTaskRetentionGuard(
        inner=HarmEvictor(BetaBinomialReputation(), task_key="task-a"),
        task_key="task-a",
        min_effective_events=3,
    )
    cards = [
        make_card(gain_events=(*native_losses, *foreign))
        for foreign in foreign_variants
    ]

    assert [sign_help_counts(events) for events in foreign_variants] == [
        (2.0, 3.0),
        (2.0, 3.0),
    ]
    assert [guard.should_evict(card) for card in cards] == [False, False]


def test_legacy_empty_task_evictor_is_identical(make_card, make_event):
    loser = make_card(gain_events=tuple(make_event(-0.1) for _ in range(4)))

    assert HarmEvictor(BetaBinomialReputation()).should_evict(loser) is True
    assert (
        HarmEvictor(BetaBinomialReputation(), task_key="").should_evict(loser) is True
    )


def test_harm_evictor_still_evicts_crash_only_card(make_card, make_event):
    evictor = HarmEvictor(BetaBinomialReputation())
    crasher = make_card(
        gain_events=tuple(make_event(0.0, invalid=True) for _ in range(4))
    )
    assert evictor.should_evict(crasher) is True


def test_null_evictor_never_evicts(make_card):
    evictor = NullEvictor()
    card = make_card()
    assert evictor.should_evict(card) is False
    assert evictor.sweep([card]) == []


def test_contextual_evictors_skip_without_explicit_context(make_card, make_event):
    scorer = ContextOnlyScorer()
    card = make_card(gain_events=(make_event(-0.1),))

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


def test_policy_nonviable_rejects_misaligned_staleness_weights(make_card, make_event):
    card = make_card(gain_events=tuple(make_event(-0.1) for _ in range(3)))
    evictor = PolicyNonViableEvictor(MismatchedStalenessReputation(), neutral_gain=0.0)

    with pytest.raises(ValueError, match="staleness_weights must align"):
        evictor.should_evict(card)


def test_effective_support_sums_only_finite_nonnegative_products(make_card, make_event):
    card = make_card(gain_events=tuple(make_event(-0.1) for _ in range(3)))
    scorer = NonFiniteStalenessReputation()
    evictor = PolicyNonViableEvictor(scorer, neutral_gain=0.0)

    assert evictor._effective_support(
        card, scorer.event_deltas(card), None
    ) == pytest.approx(0.5)


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


def test_birth_failure_only_consumes_writer_task_events(make_card, make_event):
    card = make_card(gain_events=(make_event(-0.21, founding=True, task_key="task-a"),))

    assert (
        BirthFailureEvictor(
            metrics_context=_metrics(0.1), task_key="task-b"
        ).should_evict(card)
        is False
    )
    assert (
        BirthFailureEvictor(
            metrics_context=_metrics(0.1), task_key="task-a"
        ).should_evict(card)
        is True
    )


def test_probe_and_eviction_effective_support_are_in_lockstep(make_card, make_event):
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def stamped(gain, task_key, hours):
        event = make_event(gain, task_key=task_key)
        return event.model_copy(
            update={
                "context": event.context.model_copy(
                    update={"timestamp": start + timedelta(hours=hours)}
                )
            }
        )

    card = make_card(
        gain_events=(
            stamped(-0.1, "task-a", 0),
            stamped(-0.2, "task-a", 1),
            stamped(1000.0, "task-b", 100),
        )
    )
    newer = make_card(gain_events=(stamped(0.1, "task-a", 2),))

    class Store:
        def snapshot(self):
            return (card, newer)

    context = DecisionContext(task_key="task-a")
    scorer = BootstrapReputation(BetaBinomialReputation(), Store(), n_bootstrap=32)
    block = scorer.card_stats(card, context)
    candidate = AuctionCandidateProjector().project(
        card=card, block=block, reputation=scorer, context=context
    )
    _, slate = BootstrapThompsonAuctioneer().run([candidate], np.random.default_rng(7))
    evictor = PolicyNonViableEvictor(
        scorer, neutral_gain=0.0, min_effective_events=3.0, task_key="task-a"
    )
    deltas = scorer.event_deltas(card, context)
    write_support = evictor._effective_support(card, deltas, context)
    _, marked = ColdProbePolicy(enabled=False).apply(
        budgeted_ids=[], slate=slate, max_cards=1, rng=np.random.default_rng(9)
    )

    expected_support = 2.0 ** (-2 / 2) + 2.0 ** (-1 / 2)
    assert write_support == pytest.approx(expected_support)
    assert slate[0].support_n == write_support
    assert marked[0].support_kind == "ev_rewards"
    assert marked[0].probe_eligible is (write_support < 3.0)
    assert evictor.should_evict(card) is False


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
