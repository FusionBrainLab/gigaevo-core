"""Staleness decay over reputation: evidence half-life = one bank cycle."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from omegaconf import OmegaConf
import pytest
from scipy.stats import beta

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.read.decay import DecayingReputation
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


def _event(
    gain: float, *, hours: float | None, metrics: dict[str, float] | None = None
) -> ContextualGain:
    stamp = None if hours is None else _T0 + timedelta(hours=hours)
    return ContextualGain(
        context=DecisionContext(timestamp=stamp, parent_metrics=metrics or {}),
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


def test_fresh_card_block_unchanged(make_card):
    card = make_card(gain_events=(_event(1.0, hours=10), _event(-1.0, hours=11)))
    bank = (card, make_card(gain_events=(_event(0.5, hours=5),)), make_card())
    inner = BetaBinomialReputation()
    decayed = DecayingReputation(inner=inner, store=StubStore(bank))
    assert decayed.card_stats(card) == inner.card_stats(card)


def test_one_bank_cycle_halves_the_evidence(make_card):
    card = make_card(gain_events=tuple(_event(1.0, hours=h) for h in range(4)))
    bank = _bank_with(card, newer_events=8, size=8, make_card=make_card)
    inner = BetaBinomialReputation()
    decayed = DecayingReputation(inner=inner, store=StubStore(bank))
    fresh = inner.card_stats(card)
    block = decayed.card_stats(card)
    a_eff = 1.0 + 0.5 * (fresh.posterior_a - 1.0)
    b_eff = 1.0 + 0.5 * (fresh.posterior_b - 1.0)
    assert block.posterior_a == pytest.approx(a_eff)
    assert block.posterior_b == pytest.approx(b_eff)
    assert block.p_help_mean == pytest.approx(a_eff / (a_eff + b_eff))
    assert block.p_help_lo20 == pytest.approx(float(beta.ppf(0.2, a_eff, b_eff)))
    assert block.intro_events == 2


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


def test_harm_gate_uses_fractional_posterior_evidence_not_display_count(make_card):
    inner = BetaBinomialReputation()
    decayed = DecayingReputation(inner=inner, store=StubStore((make_card(),)))
    block = CardStatsBlock(
        intro_events=2,
        posterior_a=1.0,
        posterior_b=4.0,
        p_help_lo20=0.0,
    )

    assert block.intro_events < inner.harm_min_events
    assert decayed.is_confidently_harmful(block)


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
    assert decayed.staleness_weight(card, context) == pytest.approx(0.5)


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

    assert rep.staleness_weight(card, context) == pytest.approx(0.5)


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
