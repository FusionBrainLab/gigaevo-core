"""BD reputation's ``card_stats`` is the single in-cell statistics source.

``card_stats(card, context)`` returns the block the auction bids on: the in-cell
subset under a parent context, the ``fallback`` block when the cell is cold or no
context is given. ``card_posterior`` / ``card_magnitude`` are views over it, so a
renderer reading ``card_stats`` sees the same locality the selector used.
"""

from __future__ import annotations

import pytest

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.bd_proximity import BDProximityReputation
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.shared_memory.models import MemoryCard


def _bs(num_bins: int = 10, max_val: float = 1.0) -> BehaviorSpace:
    return BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=max_val, num_bins=num_bins)}
    )


def _ctx(x: float) -> DecisionContext:
    return DecisionContext(parent_metrics={"x": x})


def _event(x: float, gain: float, *, invalid: bool = False) -> ContextualGain:
    return ContextualGain(context=_ctx(x), gain=gain, invalid=invalid)


def _card(events: list[ContextualGain] | None = None) -> MemoryCard:
    return MemoryCard(
        id="m1",
        description="d",
        keywords=[],
        gain_events=events,
    )


def _rep(fallback: object | None = None) -> BDProximityReputation:
    return BDProximityReputation(
        behavior_space=_bs(),
        fallback=fallback or BetaBinomialReputation(),
    )


class TestCardStatsInCellSource:
    def test_card_stats_reflects_in_cell_events_not_global_all(self) -> None:
        # cell-0 events (median 0.3, both help); a cell-9 event would change the
        # global block. card_stats under a cell-0 parent must reflect ONLY the
        # cell-0 events.
        card = _card(
            [
                _event(0.05, 0.2),
                _event(0.06, 0.4),
                _event(0.95, 9.0),  # cell 9, excluded
            ]
        )
        block = _rep().card_stats(card, _ctx(0.05))
        assert block is not None
        assert block.intro_events == 2
        assert (block.posterior_a, block.posterior_b) == (3.0, 1.0)
        assert block.IntroGain_best_median == pytest.approx(0.3)

    def test_card_stats_is_the_source_for_posterior_and_magnitude(self) -> None:
        # The auction's posterior/magnitude reads are views over this one block.
        card = _card([_event(0.05, 0.2), _event(0.06, 0.4)])
        rep = _rep()
        ctx = _ctx(0.05)
        block = rep.card_stats(card, ctx)
        assert rep.card_posterior(card, ctx) == (block.posterior_a, block.posterior_b)
        assert rep.card_magnitude(card, ctx) == pytest.approx(
            block.IntroGain_best_median
        )

    def test_card_stats_cold_cell_delegates_to_fallback_block(self) -> None:
        # parent in empty cell 9 -> the block is the fallback's (over all events).
        card = _card([_event(0.05, 0.9)])
        fallback = BetaBinomialReputation()
        rep = _rep(fallback)
        assert rep.card_stats(card, _ctx(0.95)) == fallback.card_stats(card, None)

    def test_card_stats_none_context_delegates_to_fallback_block(self) -> None:
        card = _card([_event(0.05, 0.9)])
        fallback = BetaBinomialReputation()
        rep = _rep(fallback)
        assert rep.card_stats(card, None) == fallback.card_stats(card, None)
