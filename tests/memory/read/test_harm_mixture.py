"""Latent-sign mixture posterior for uncertain downside evidence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import product
import math
from pathlib import Path

from omegaconf import OmegaConf
import pytest
from scipy.stats import beta

from gigaevo.memory.cards import ContextualGain, DecisionContext
from gigaevo.memory.context.evidence import harm_mass
from gigaevo.memory.read.decay import DecayingReputation
from gigaevo.memory.read.reputation import (
    BetaBinomialReputation,
    _block_from_partition,
    beta_binomial_posterior,
    block_from_events,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_T0 = datetime(2026, 1, 1, tzinfo=UTC)


class _StaticStore:
    def __init__(self, cards: tuple) -> None:
        self._cards = cards

    def snapshot(self) -> tuple:
        return self._cards


def test_default_soft_count_is_exactly_the_legacy_posterior():
    gains = (0.25, -0.10, 0.0)
    weights = (1.0, 0.5, 0.25)
    ses = (0.20, 0.05, None)
    prior = (2.0, 3.0)
    invalid_events = 0.75
    unused_events = 0.25
    forced_failures = invalid_events + unused_events
    n = sum(weights) + forced_failures
    k_harm = (
        sum(
            weight * harm_mass(gain, se, 0.0)
            for gain, weight, se in zip(gains, weights, ses)
        )
        + forced_failures
    )
    legacy_a = prior[0] + (n - k_harm)
    legacy_b = prior[1] + k_harm

    block = beta_binomial_posterior(
        gains,
        prior=prior,
        weights=weights,
        event_ses=ses,
        invalid_events=invalid_events,
        unused_events=unused_events,
    )

    assert block.posterior_a == legacy_a
    assert block.posterior_b == legacy_b


@pytest.mark.parametrize("event_ses", [None, (0.0, 0.0, 0.0)])
def test_mixture_point_crediting_is_bit_identical_to_soft_count(event_ses):
    kwargs = {} if event_ses is None else {"event_ses": event_ses}
    common = {
        "gains": (0.4, -0.2, 0.0),
        "prior": (2.0, 3.0),
        "weights": (0.5, 1.0, 0.25),
        "invalid_events": 0.75,
        "unused_events": 0.25,
        **kwargs,
    }

    soft = beta_binomial_posterior(**common, harm_model="soft_count")
    mixture = beta_binomial_posterior(**common, harm_model="mixture")

    assert (mixture.posterior_a, mixture.posterior_b) == (
        soft.posterior_a,
        soft.posterior_b,
    )


def test_mixture_preserves_mean_but_strictly_widens_uncertain_posterior():
    common = {
        "gains": (-0.15, 0.05, 0.30),
        "prior": (2.0, 3.0),
        "event_ses": (0.30, 0.20, 0.40),
        "invalid_events": 1.0,
    }

    soft = beta_binomial_posterior(**common, harm_model="soft_count")
    mixture = beta_binomial_posterior(**common, harm_model="mixture")
    soft_total = soft.posterior_a + soft.posterior_b
    mixture_total = mixture.posterior_a + mixture.posterior_b

    assert (mixture.posterior_a, mixture.posterior_b) != (
        soft.posterior_a,
        soft.posterior_b,
    )
    assert mixture.posterior_a / mixture_total == pytest.approx(
        soft.posterior_a / soft_total, rel=1e-14, abs=1e-14
    )
    assert mixture_total < soft_total


def test_mixture_beta_matches_brute_force_latent_sign_moments():
    gains = (-0.20, 0.10, 0.35)
    ses = (0.30, 0.25, 0.50)
    prior = (2.3, 1.7)
    forced_failures = 1.0
    probabilities = tuple(harm_mass(gain, se, 0.0) for gain, se in zip(gains, ses))
    n = len(gains) + forced_failures
    mixture_mean = 0.0
    mixture_second_moment = 0.0
    for outcomes in product((0, 1), repeat=len(gains)):
        combo_probability = math.prod(
            probability if outcome else 1.0 - probability
            for probability, outcome in zip(probabilities, outcomes)
        )
        k_harm = sum(outcomes) + forced_failures
        a = prior[0] + n - k_harm
        b = prior[1] + k_harm
        total = a + b
        mixture_mean += combo_probability * a / total
        mixture_second_moment += (
            combo_probability * a * (a + 1.0) / (total * (total + 1.0))
        )
    mixture_variance = mixture_second_moment - mixture_mean**2

    block = beta_binomial_posterior(
        gains,
        prior=prior,
        event_ses=ses,
        invalid_events=forced_failures,
        harm_model="mixture",
    )
    a = block.posterior_a
    b = block.posterior_b
    matched_total = a + b
    matched_mean = a / matched_total
    matched_variance = a * b / (matched_total**2 * (matched_total + 1.0))

    assert abs(matched_mean - mixture_mean) < 1e-9
    assert abs(matched_variance - mixture_variance) < 1e-9


def _native(gain: float, se: float) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(task_key="native-task"), gain=gain, gain_se=se
    )


def _foreign(gain: float) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(task_key="other-task"), gain=gain, gain_se=0.0
    )


def test_mixture_foreign_fold_matches_brute_force_combined_moments():
    # C1: foreign hard-sign counts must fold into the native mixture's sample
    # total N, not onto the variance-shrunk matched total S = a*+b* (S <= N).
    # Brute force over the 2^m native latent-harm configs with the foreign
    # fails as deterministic harms fixes the true combined mean/var; the fold
    # must reproduce both.
    native_gains = (-0.2, 0.1, 0.35)
    native_ses = (0.3, 0.25, 0.5)
    prior = (1.0, 1.0)
    foreign_gains = (1.0, 1.0, -1.0)  # exact sign: 2 help, 1 fail
    foreign_help = sum(1 for g in foreign_gains if g >= 0.0)
    foreign_fail = len(foreign_gains) - foreign_help
    probabilities = tuple(
        harm_mass(gain, se, 0.0) for gain, se in zip(native_gains, native_ses)
    )
    n = len(native_gains) + len(foreign_gains)
    mean = 0.0
    second_moment = 0.0
    for outcomes in product((0, 1), repeat=len(native_gains)):
        weight = math.prod(
            probability if outcome else 1.0 - probability
            for probability, outcome in zip(probabilities, outcomes)
        )
        k_harm = sum(outcomes) + foreign_fail
        a = prior[0] + n - k_harm
        b = prior[1] + k_harm
        total = a + b
        mean += weight * a / total
        second_moment += weight * a * (a + 1.0) / (total * (total + 1.0))
    variance = second_moment - mean**2

    block = _block_from_partition(
        [_native(g, s) for g, s in zip(native_gains, native_ses)],
        [_foreign(g) for g in foreign_gains],
        prior=prior,
        confident_quantile=0.2,
        confident_threshold=0.5,
        harm_model="mixture",
    )
    matched_total = block.posterior_a + block.posterior_b
    matched_mean = block.posterior_a / matched_total
    matched_variance = (
        block.posterior_a
        * block.posterior_b
        / (matched_total**2 * (matched_total + 1.0))
    )

    assert abs(matched_mean - mean) < 1e-9
    assert abs(matched_variance - variance) < 1e-9
    assert block.foreign_help_events == foreign_help
    assert block.foreign_total_events == len(foreign_gains)


def test_foreign_fold_soft_count_is_native_posterior_plus_hard_counts():
    # The shipped soft-count path keeps the plain conjugate fold: foreign
    # help/fail counts add directly to the native Beta pseudo-counts (S == N,
    # so the mixture reconstruction collapses to the legacy update).
    native = [_native(g, s) for g, s in ((-0.2, 0.3), (0.1, 0.25), (0.35, 0.5))]
    foreign = [_foreign(g) for g in (1.0, 1.0, -1.0)]
    prior = (1.0, 1.0)
    folded = _block_from_partition(
        native,
        foreign,
        prior=prior,
        confident_quantile=0.2,
        confident_threshold=0.5,
        harm_model="soft_count",
    )
    base = block_from_events(
        native,
        prior=prior,
        confident_quantile=0.2,
        confident_threshold=0.5,
        harm_model="soft_count",
    )
    assert folded.posterior_a == base.posterior_a + 2.0  # 2 foreign helps
    assert folded.posterior_b == base.posterior_b + 1.0  # 1 foreign fail


def test_empty_mixture_is_the_legacy_nan_block():
    soft = beta_binomial_posterior(())
    mixture = beta_binomial_posterior((), harm_model="mixture")

    assert (mixture.posterior_a, mixture.posterior_b) == (
        soft.posterior_a,
        soft.posterior_b,
    )
    assert mixture.intro_events == soft.intro_events == 0
    assert math.isnan(mixture.p_help_lo20)


def test_mixture_block_remains_callable_by_the_harm_gate(make_card, make_event):
    card = make_card(
        gain_events=tuple(make_event(-0.1, gain_se=0.25) for _ in range(4))
    )
    reputation = BetaBinomialReputation(harm_model="mixture")
    block = reputation.card_stats(card)

    assert block is not None
    assert math.isfinite(float(beta.ppf(0.80, block.posterior_a, block.posterior_b)))
    assert isinstance(reputation.is_confidently_harmful(block), bool)


def test_harm_model_defaults_when_config_omits_it_and_presets_stay_unchanged():
    assert BetaBinomialReputation().harm_model == "soft_count"
    preset_style = {
        "harm_min_events": 3,
        "harm_quantile": 0.80,
        "harm_threshold": 0.5,
        "confident_quantile": 0.20,
        "confident_threshold": 0.5,
        "cold_prior": (3.0, 3.0),
    }
    assert (
        BetaBinomialReputation.model_validate(preset_style).harm_model == "soft_count"
    )

    path = _REPO_ROOT / "config" / "memory" / "reputation" / "beta_binomial.yaml"
    before = path.read_bytes()
    config = OmegaConf.load(path)
    assert "harm_model" not in config
    assert path.read_bytes() == before


def test_decaying_decorator_preserves_inner_mixture_model(make_card, make_event):
    def stamped(gain: float, se: float, hours: int):
        event = make_event(gain, gain_se=se)
        return event.model_copy(
            update={"context": DecisionContext(timestamp=_T0 + timedelta(hours=hours))}
        )

    card = make_card(gain_events=(stamped(-0.1, 0.30, 0), stamped(0.2, 0.40, 1)))
    newer = make_card(gain_events=(stamped(0.5, 0.0, 2),))
    store = _StaticStore((card, newer))
    mixture_rep = DecayingReputation(
        inner=BetaBinomialReputation(harm_model="mixture"), store=store
    )
    soft_rep = DecayingReputation(
        inner=BetaBinomialReputation(harm_model="soft_count"), store=store
    )

    mixture = mixture_rep.card_stats(card)
    soft = soft_rep.card_stats(card)

    assert mixture is not None and soft is not None
    assert (mixture.posterior_a, mixture.posterior_b) != (
        soft.posterior_a,
        soft.posterior_b,
    )
    assert mixture.posterior_a + mixture.posterior_b < (
        soft.posterior_a + soft.posterior_b
    )
