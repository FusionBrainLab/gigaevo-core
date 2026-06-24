"""Read-time BD-cell partitioned reputation.

``BDProximityReputation`` re-buckets each card's stored ``gain_events`` into the
query parent's *current* MAP-Elites cell via the live ``behavior_space.get_cell``
and bids ``theta x magnitude`` over the in-cell subset only:

- magnitude = median in-cell ``child - base`` gain (the per-cell analogue of
  arm D's absolute median),
- theta ~ Beta(1 + n_help, 1 + n_harm) with harm = ``gain < 0`` or ``invalid``.

A parent cell with no in-cell event delegates byte-for-byte to a ``fallback``
reputation (today's numbers, no regression). The context arg is additive: every
existing reputation called with ``context=None`` behaves exactly as before.
"""

from __future__ import annotations

from math import inf, nan

import pytest

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.bd_proximity import BDProximityReputation
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.reward import AbsoluteMedianReward, RewardWeightedReputation
from gigaevo.memory.shared_memory.models import MemoryCard, MemoryCardExplanation


def _bs(num_bins: int = 10, max_val: float = 1.0) -> BehaviorSpace:
    return BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=max_val, num_bins=num_bins)}
    )


def _ctx(x: float) -> DecisionContext:
    return DecisionContext(parent_metrics={"x": x})


def _event(x: float, gain: float, *, invalid: bool = False) -> ContextualGain:
    return ContextualGain(context=_ctx(x), gain=gain, invalid=invalid)


def _card(
    *,
    events: list[ContextualGain] | None = None,
    all_block: dict | None = None,
) -> MemoryCard:
    es: dict = {}
    if events is not None:
        es["gain_events"] = events
    if all_block is not None:
        es["ALL"] = all_block
    return MemoryCard(
        id="m1",
        description="d",
        keywords=[],
        evolution_statistics=es,
        explanation=MemoryCardExplanation(summary=""),
    )


def _rep(fallback: object | None = None) -> BDProximityReputation:
    return BDProximityReputation(
        behavior_space=_bs(),
        fallback=fallback or BetaBinomialReputation(),
    )


class TestInCellAggregation:
    def test_magnitude_is_in_cell_median(self) -> None:
        # parent at x=0.05 -> cell 0; only the two cell-0 events count.
        card = _card(
            events=[
                _event(0.05, 0.2),
                _event(0.06, 0.4),
                _event(0.95, 1.0),  # cell 9, excluded
            ]
        )
        assert _rep().card_magnitude(card, _ctx(0.05)) == pytest.approx(0.3)

    def test_posterior_counts_in_cell_help_and_harm(self) -> None:
        card = _card(
            events=[
                _event(0.05, 0.2),  # help
                _event(0.06, -0.1),  # harm (gain < 0)
                _event(0.04, 0.0, invalid=True),  # harm (invalid)
                _event(0.95, -5.0),  # out of cell, ignored
            ]
        )
        assert _rep().card_posterior(card, _ctx(0.05)) == (2.0, 3.0)

    def test_invalid_event_is_harm_even_with_nonnegative_gain(self) -> None:
        card = _card(events=[_event(0.05, 0.0, invalid=True)])
        assert _rep().card_posterior(card, _ctx(0.05)) == (1.0, 2.0)

    def test_magnitude_excludes_invalid_events(self) -> None:
        # Invalid children are harm, not zero-gain successes: their stamped 0.0
        # must not drag a proven-harmful card's median up to a non-negative value.
        card = _card(
            events=[
                _event(0.05, -0.9),  # real harm
                _event(0.06, 0.0, invalid=True),
                _event(0.04, 0.0, invalid=True),
                _event(0.05, 0.0, invalid=True),
            ]
        )
        assert _rep().card_magnitude(card, _ctx(0.05)) == pytest.approx(-0.9)

    def test_magnitude_all_invalid_in_cell_is_zero(self) -> None:
        # In-cell evidence is entirely invalid: no positive value signal, so the
        # magnitude is a non-positive 0.0 (bid theta x 0 = 0 abstains under the
        # default ev_floor) rather than the optimistic cold prior.
        card = _card(
            events=[
                _event(0.05, 0.0, invalid=True),
                _event(0.06, 0.0, invalid=True),
            ]
        )
        assert _rep().card_magnitude(card, _ctx(0.05)) == 0.0

    def test_magnitude_drops_nonfinite_gains(self) -> None:
        # A corrupt/legacy persisted +inf gain must not reach the median: the
        # auction has no inf defense and median([inf, ...]) would win every slot.
        card = _card(
            events=[
                _event(0.05, inf),
                _event(0.06, -0.5),
            ]
        )
        assert _rep().card_magnitude(card, _ctx(0.05)) == pytest.approx(-0.5)

    def test_magnitude_all_nonfinite_in_cell_is_zero(self) -> None:
        card = _card(events=[_event(0.05, inf), _event(0.06, inf)])
        assert _rep().card_magnitude(card, _ctx(0.05)) == 0.0


class TestEmptyCellFallback:
    def test_posterior_delegates_to_fallback_when_no_in_cell_events(self) -> None:
        # events live in cell 0, parent is in cell 9 -> empty -> fallback.
        card = _card(
            events=[_event(0.05, 0.9)],
            all_block={"posterior_a": 4.0, "posterior_b": 2.0},
        )
        fallback = BetaBinomialReputation()
        rep = _rep(fallback)
        assert rep.card_posterior(card, _ctx(0.95)) == fallback.card_posterior(card)
        assert rep.card_posterior(card, _ctx(0.95)) == (4.0, 2.0)

    def test_no_gain_events_delegates_to_fallback(self) -> None:
        card = _card(all_block={"posterior_a": 4.0, "posterior_b": 2.0})
        fallback = BetaBinomialReputation()
        rep = _rep(fallback)
        assert rep.card_posterior(card, _ctx(0.05)) == fallback.card_posterior(card)

    def test_none_context_delegates_to_fallback(self) -> None:
        card = _card(
            events=[_event(0.05, 0.9)],
            all_block={"posterior_a": 4.0, "posterior_b": 2.0},
        )
        fallback = BetaBinomialReputation()
        rep = _rep(fallback)
        assert rep.card_posterior(card, None) == fallback.card_posterior(card)

    def test_magnitude_cold_cell_delegates_to_fallback(self) -> None:
        card = _card(
            events=[_event(0.05, 0.9)],  # cell 0
            all_block={"IntroGain_best_median": -0.05},
        )
        fallback = RewardWeightedReputation(reward=AbsoluteMedianReward())
        rep = _rep(fallback)
        # parent in cell 9 -> empty -> absolute-median fallback reads the raw block.
        assert rep.card_magnitude(card, _ctx(0.95)) == -0.05

    def test_nonfinite_query_coord_delegates_to_fallback(self) -> None:
        # A NaN behavior coord has no well-defined cell (LinearBinning would
        # silently clamp it to bin 0); abstain to fallback instead of crediting
        # the card to a spurious low-end cell.
        card = _card(
            events=[_event(0.05, 0.9)],
            all_block={"IntroGain_best_median": -0.05},
        )
        fallback = RewardWeightedReputation(reward=AbsoluteMedianReward())
        rep = _rep(fallback)
        assert rep.card_magnitude(card, _ctx(nan)) == -0.05
        assert rep.card_posterior(card, _ctx(nan)) == fallback.card_posterior(card)


class TestDynamicBinsParanoia:
    def test_in_cell_set_tracks_live_behavior_space_bounds(self) -> None:
        card = _card(events=[_event(0.45, 1.0), _event(0.55, 3.0)])
        rep = _rep()
        # bounds [0,1], 10 bins: 0.45 -> cell 4, 0.55 -> cell 5; only 0.45 in cell.
        assert rep.card_magnitude(card, _ctx(0.45)) == 1.0
        # Widen the SAME behavior_space object: [0,10], 10 bins -> both in cell 0.
        rep.behavior_space.bins["x"].max_val = 10.0
        assert rep.card_magnitude(card, _ctx(0.45)) == 2.0


class TestSeamBackCompat:
    def test_base_reputation_ignores_context_kwarg(self) -> None:
        card = _card(all_block={"posterior_a": 3.0, "posterior_b": 5.0})
        rep = BetaBinomialReputation()
        assert rep.card_posterior(card, _ctx(0.5)) == rep.card_posterior(card)

    def test_base_magnitude_ignores_context_kwarg(self) -> None:
        card = _card(all_block={"IntroGain_best_adj_median": 0.0123})
        rep = BetaBinomialReputation()
        assert rep.card_magnitude(card, _ctx(0.5)) == rep.card_magnitude(card)
