"""Staleness decay over reputation: evidence half-life = one bank cycle."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy.stats import beta

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import ContextualGain, DecisionContext
from gigaevo.memory.read.auction import BootstrapThompsonAuctioneer
from gigaevo.memory.read.decay import DecayingReputation
from gigaevo.memory.read.projection import AuctionCandidateProjector
from gigaevo.memory.read.reputation import (
    BDProximityReputation,
    BetaBinomialReputation,
    BootstrapReputation,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_T0 = datetime(2026, 1, 1, tzinfo=UTC)


class StubStore:
    def __init__(self, cards: tuple) -> None:
        self._cards = cards

    def snapshot(self) -> tuple:
        return self._cards


class _StubBetaPrior:
    source = "stub_cohort"

    def __init__(self, alpha: float, beta: float) -> None:
        self._parameters = (alpha, beta)

    def as_tuple(self) -> tuple[float, float]:
        return self._parameters


class _StubCohortPrior:
    def __init__(self, alpha: float, beta: float) -> None:
        self._prior = _StubBetaPrior(alpha, beta)

    def cold_card_prior(self, card, context=None):
        del card, context
        return self._prior


def _event(
    gain: float,
    *,
    hours: float | None,
    metrics: dict[str, float] | None = None,
    task_key: str = "",
) -> ContextualGain:
    stamp = None if hours is None else _T0 + timedelta(hours=hours)
    return ContextualGain(
        context=DecisionContext(
            task_key=task_key, timestamp=stamp, parent_metrics=metrics or {}
        ),
        gain=gain,
    )


def _bank_with(card, *, newer_events: int, size: int, make_card):
    """A bank of ``size`` cards holding ``card`` plus fillers carrying
    ``newer_events`` stamped events newer than any of ``card``'s."""
    holder = make_card(
        gain_events=tuple(_event(0.5, hours=1000 + k) for k in range(newer_events))
    )
    fillers = tuple(make_card() for _ in range(size - 2))
    return (card, holder, *fillers)


def test_fresh_event_does_not_revive_older_history(make_card):
    card = make_card(gain_events=(_event(1.0, hours=10), _event(-1.0, hours=11)))
    bank = (card, make_card(gain_events=(_event(0.5, hours=5),)), make_card())
    inner = BetaBinomialReputation()
    decayed = DecayingReputation(inner=inner, store=StubStore(bank))
    block = decayed.card_stats(card)

    old_win_weight = 2.0 ** (-1 / 3)
    assert block is not None
    assert block.posterior_a == pytest.approx(1.0 + old_win_weight)
    assert block.posterior_b == pytest.approx(2.0)
    assert block.intro_events == pytest.approx(1.0 + old_win_weight)
    assert block.p_help_mean < 0.5


def test_one_bank_cycle_halves_the_evidence(make_card):
    card = make_card(gain_events=tuple(_event(1.0, hours=h) for h in range(4)))
    bank = _bank_with(card, newer_events=8, size=8, make_card=make_card)
    inner = BetaBinomialReputation()
    decayed = DecayingReputation(inner=inner, store=StubStore(bank))
    block = decayed.card_stats(card)
    effective_events = sum(2.0 ** (-s / 8) for s in (11, 10, 9, 8))
    a_eff = 1.0 + effective_events
    b_eff = 1.0
    assert block.posterior_a == pytest.approx(a_eff)
    assert block.posterior_b == pytest.approx(b_eff)
    assert block.p_help_mean == pytest.approx(a_eff / (a_eff + b_eff))
    assert block.p_help_lo20 == pytest.approx(float(beta.ppf(0.2, a_eff, b_eff)))
    assert block.intro_events == pytest.approx(1.764502935581178)


def test_fully_stale_posterior_shrinks_to_the_card_cold_prior(make_card):
    prior_parameters = (8.0, 2.0)
    gains = (1.0, 1.0, 1.0, -1.0)
    card = make_card(
        gain_events=tuple(_event(gain, hours=h) for h, gain in enumerate(gains))
    )
    bank = _bank_with(card, newer_events=128, size=2, make_card=make_card)
    inner = BetaBinomialReputation(prior=_StubCohortPrior(*prior_parameters))
    decayed = DecayingReputation(inner=inner, store=StubStore(bank))

    fresh = inner.card_stats(card)
    block = decayed.card_stats(card)

    assert fresh is not None and block is not None
    assert fresh.posterior_a > prior_parameters[0]
    assert fresh.posterior_b > prior_parameters[1]
    assert block.posterior_a == pytest.approx(prior_parameters[0], abs=1e-12)
    assert block.posterior_b == pytest.approx(prior_parameters[1], abs=1e-12)


def test_magnitude_expires_below_one_effective_event(make_card):
    card = make_card(gain_events=(_event(0.9, hours=0),))
    bank = _bank_with(card, newer_events=8, size=8, make_card=make_card)
    inner = BetaBinomialReputation()
    decayed = DecayingReputation(inner=inner, store=StubStore(bank))
    assert inner.magnitude_of(inner.card_stats(card)) == pytest.approx(0.9)
    block = decayed.card_stats(card)
    assert block.IntroGain_best_median is None
    assert decayed.magnitude_of(block) is None
    assert block.efficacy_confident is False


def test_harm_gate_expires_with_staleness(make_card):
    card = make_card(gain_events=tuple(_event(-1.0, hours=h) for h in range(4)))
    inner = BetaBinomialReputation()
    assert inner.is_confidently_harmful(inner.card_stats(card))
    bank = _bank_with(card, newer_events=16, size=8, make_card=make_card)
    decayed = DecayingReputation(inner=inner, store=StubStore(bank))
    block = decayed.card_stats(card)
    assert block.intro_events < inner.harm_min_events
    assert not decayed.is_confidently_harmful(block)


def test_foreign_posterior_mass_does_not_satisfy_native_harm_floor(make_card):
    inner = BetaBinomialReputation()
    card = make_card(
        gain_events=(
            _event(-1.0, hours=0, task_key="task-a"),
            *tuple(_event(-1.0, hours=h, task_key="task-b") for h in range(1, 9)),
        )
    )
    context = DecisionContext(task_key="task-a")
    decayed = DecayingReputation(inner=inner, store=StubStore((card,)))
    block = decayed.card_stats(card, context)

    assert block is not None
    assert block.intro_events == 1
    assert block.foreign_total_events == 8
    assert not decayed.is_confidently_harmful(block)


def test_unstamped_events_never_decay(make_card, make_event):
    card = make_card(gain_events=(make_event(1.0), make_event(1.0)))
    bank = _bank_with(card, newer_events=20, size=4, make_card=make_card)
    inner = BetaBinomialReputation()
    decayed = DecayingReputation(inner=inner, store=StubStore(bank))
    assert decayed.card_stats(card) == inner.card_stats(card)


def test_bd_decay_ages_the_contextual_evidence_subset(make_card):
    space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    card = make_card(
        gain_events=(
            _event(1.0, hours=1, metrics={"x": 0.1}),
            _event(1.0, hours=100, metrics={"x": 0.9}),
        )
    )
    context = DecisionContext(parent_metrics={"x": 0.1})
    inner = BDProximityReputation(behavior_space=space)
    decayed = DecayingReputation(inner=inner, store=StubStore((card,)))

    block = decayed.card_stats(card, context)

    assert block is not None
    assert block.intro_events == pytest.approx(0.5)
    assert decayed.staleness_weights(card, context) == pytest.approx((0.5,))


def test_bootstrap_staleness_uses_the_contextual_evidence_subset(make_card):
    space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    card = make_card(
        gain_events=(
            _event(1.0, hours=1, metrics={"x": 0.1}),
            _event(1.0, hours=100, metrics={"x": 0.9}),
        )
    )
    context = DecisionContext(parent_metrics={"x": 0.1})
    rep = BootstrapReputation(
        inner=BDProximityReputation(behavior_space=space),
        store=StubStore((card,)),
    )

    assert rep.staleness_weights(card, context) == pytest.approx((0.5,))


def test_decay_and_bootstrap_share_the_same_task_partition(make_card):
    card = make_card(
        gain_events=(
            _event(0.1, hours=0, task_key="task-a"),
            _event(100.0, hours=100, task_key="task-b"),
        )
    )
    newer_a = make_card(gain_events=(_event(0.2, hours=2, task_key="task-a"),))
    store = StubStore((card, newer_a))
    context = DecisionContext(task_key="task-a")
    inner = BetaBinomialReputation()
    decay = DecayingReputation(inner=inner, store=store)
    bootstrap = BootstrapReputation(inner=inner, store=store)

    assert decay.staleness_weights(card, context) == pytest.approx(
        bootstrap.staleness_weights(card, context)
    )


def test_stale_wins_plus_fresh_loss_age_posterior_ev_and_support_per_event(
    make_card,
):
    card = make_card(
        id="revival-regression",
        gain_events=(
            *tuple(_event(1.0, hours=h) for h in range(100)),
            _event(-1.0, hours=100),
        ),
    )
    store = StubStore((card,))
    decay = DecayingReputation(
        inner=BetaBinomialReputation(),
        store=store,
        half_life_cycles=0.5,
    )
    rep = BootstrapReputation(
        inner=decay,
        store=store,
        half_life_cycles=0.5,
        n_bootstrap=512,
    )

    block = rep.card_stats(card)
    candidate = AuctionCandidateProjector().project(
        card=card, block=block, reputation=rep, context=None
    )
    winners, slate = BootstrapThompsonAuctioneer(
        n_bootstrap=512, ev_floor_quantile=0.0
    ).run([candidate], np.random.default_rng(7))

    assert block is not None
    assert block.posterior_a == pytest.approx(4.0 / 3.0)
    assert block.posterior_b == pytest.approx(2.0)
    assert block.p_help_mean == pytest.approx(0.4)
    # EV mean/bid are Kish effective-N realizations (~3 draws/replicate, not all
    # 102): the skewed staleness weights shrink the resample count; the estimand
    # is unchanged (analytic EV -0.2857, grand mean -0.2855 over 200 seeds).
    assert block.IntroGain_bootstrap_ev_mean == pytest.approx(-0.294921875)
    assert candidate.delta_weights is not None
    assert sum(candidate.delta_weights) == pytest.approx(4.0 / 3.0)
    assert candidate.staleness_weight == 1.0
    assert winners == []
    assert slate[0].bid == pytest.approx(0.0, abs=1e-12)
    assert slate[0].support_n == pytest.approx(4.0 / 3.0)

    # The pre-fix latest-event scalar is one because the loss is fresh; it
    # revives every old win and produces the misleading near-certain posterior.
    assert 101.0 / 103.0 > 0.98
    assert (block.posterior_a, block.posterior_b) != (101.0, 2.0)


def test_bootstrap_decay_composition_ages_each_owned_quantity_once(make_card):
    card = make_card(id="single-stale", gain_events=(_event(1.0, hours=0),))
    newer = make_card(gain_events=(_event(0.1, hours=1),))
    store = StubStore((card, newer))
    decay = DecayingReputation(
        inner=BetaBinomialReputation(),
        store=store,
        half_life_cycles=0.5,
    )
    rep = BootstrapReputation(
        inner=decay,
        store=store,
        half_life_cycles=0.5,
        n_bootstrap=512,
    )

    block = rep.card_stats(card)

    assert block is not None
    assert decay.staleness_weights(card) == pytest.approx((0.5,))
    assert rep.staleness_weights(card) == pytest.approx((0.5,))
    assert rep.event_weights(card) == (1.0,)
    assert block.posterior_a == pytest.approx(1.5)
    assert block.posterior_a != pytest.approx(1.25)
    assert block.posterior_b == pytest.approx(1.0)
    assert block.IntroGain_bootstrap_ev_mean == pytest.approx(0.30859375)


def test_non_decay_bootstrap_keeps_posterior_credit_only(make_card):
    card = make_card(
        id="non-decay-stale",
        gain_events=(_event(1.0, hours=0), _event(-1.0, hours=1)),
    )
    inner = BetaBinomialReputation()
    rep = BootstrapReputation(
        inner=inner,
        store=StubStore((card,)),
        half_life_cycles=0.5,
        n_bootstrap=512,
    )

    block = rep.card_stats(card)
    unaged = inner.card_stats(card)

    assert block is not None and unaged is not None
    assert (
        (block.posterior_a, block.posterior_b)
        == (
            unaged.posterior_a,
            unaged.posterior_b,
        )
        == (2.0, 2.0)
    )
    assert rep.staleness_weights(card) == pytest.approx((0.25, 1.0))
    # Kish effective-N (~2 draws) realization of the skewed [0.25, 1.0] weights;
    # estimand unchanged (analytic EV -0.3333, grand mean -0.3337 over 300 seeds).
    assert block.IntroGain_bootstrap_ev_mean == pytest.approx(-0.361328125)


def test_decay_ages_foreign_sign_counts_by_each_foreign_event_stamp(make_card):
    card = make_card(
        gain_events=(
            _event(1.0, hours=0, task_key="task-b"),
            _event(-1.0, hours=1, task_key="task-a"),
        )
    )
    context = DecisionContext(task_key="task-a")
    decayed = DecayingReputation(
        inner=BetaBinomialReputation(), store=StubStore((card,))
    )

    block = decayed.card_stats(card, context)

    assert block is not None
    assert block.intro_events == pytest.approx(1.0)
    assert block.k_harm == pytest.approx(1.0)
    assert block.foreign_help_events == pytest.approx(0.5)
    assert block.foreign_total_events == pytest.approx(0.5)
    assert block.posterior_a == pytest.approx(1.5)
    assert block.posterior_b == pytest.approx(2.0)


def test_cold_card_stays_cold(make_card):
    inner = BetaBinomialReputation()
    decayed = DecayingReputation(inner=inner, store=StubStore((make_card(),)))
    assert decayed.card_stats(make_card()) is None
    assert decayed.posterior_of(None) == inner.cold_prior
    assert decayed.magnitude_of(None) is None


def test_half_life_cycles_must_be_positive():
    inner = BetaBinomialReputation()
    with pytest.raises(ValueError):
        DecayingReputation(inner=inner, store=StubStore(()), half_life_cycles=0.0)
    with pytest.raises(ValueError):
        DecayingReputation(inner=inner, store=StubStore(()), half_life_cycles=-1.0)


def test_decay_yaml_wraps_bd_proximity_and_default_stays_undecayed():
    decay = OmegaConf.load(
        _REPO_ROOT / "config" / "memory" / "reputation" / "bd_proximity_decay.yaml"
    )
    assert decay._target_ == "gigaevo.memory.read.decay.DecayingReputation"
    assert (
        decay.inner._target_ == "gigaevo.memory.read.reputation.BDProximityReputation"
    )
    raw = OmegaConf.to_container(decay, resolve=False)
    assert raw["store"] == "${ref:memory.store}"
    assert decay.half_life_cycles == 1.0
    full = OmegaConf.load(_REPO_ROOT / "config" / "memory" / "full.yaml")
    full_text = Path(_REPO_ROOT / "config" / "memory" / "full.yaml").read_text()
    assert "read_policy: adaptive" in full_text
    assert "reputation: bd_proximity_decay" not in full_text
    assert full is not None
