"""DRos optimistic shrinkage (Su & Wang 2020) on the conditional-offer DR estimator.

The shrinkage weight tames high importance weights at the cost of a little bias;
``shrinkage=None`` must stay byte-identical to the unshrunk DR estimate, and the
data-driven ``"auto"`` selection must never shrink unless it strictly lowers the
estimated MSE against the unshrunk DR as the low-bias reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvolutionContext,
    OutcomeMeasurement,
)
from gigaevo.memory_v2.ope import (
    ConditionalOfferDREvaluator,
    _optimistic_shrinkage,
    _select_shrinkage,
)


def _observation(
    context: EvolutionContext,
    card: CardSnapshot,
    *,
    decision_id: str,
    treatment: bool,
    value: float,
    offer_propensity: float = 0.5,
    run_id: str = "run-a",
) -> CausalObservation:
    return CausalObservation(
        decision_id=decision_id,
        event_ordinal=0,
        card=card,
        context=context.model_copy(update={"run_id": run_id}),
        treatment=treatment,
        card_used=treatment,
        offer_propensity=offer_propensity,
        proposal_propensity=1.0,
        joint_action_propensity=offer_propensity
        if treatment
        else 1.0 - offer_propensity,
        status="outcome",
        measurement=OutcomeMeasurement(value=value, se=None, kind="scalar"),
        reward_q_hat_control=0.0,
        reward_q_hat_treated=0.0,
        risk_q_hat_control=0.1,
        risk_q_hat_treated=0.2,
    )


def test_optimistic_shrinkage_weight_properties() -> None:
    lam = 4.0
    # bounded by the raw weight, zero at zero, and never negative
    for w in (0.0, 0.5, 1.0, 3.0, 25.0):
        shrunk = _optimistic_shrinkage(w, lam)
        assert 0.0 <= shrunk <= w if w > 0 else shrunk == 0.0
    # DRos shrinkage is hump-shaped, not monotone: it peaks at w = sqrt(lambda)
    # and drives very large weights back toward zero (the "optimistic" regime).
    grid = np.linspace(0.0, 30.0, 601)
    shrunk = _optimistic_shrinkage(grid, lam)
    assert np.all(shrunk <= grid + 1e-12)
    assert np.all(shrunk >= -1e-12)
    peak = grid[int(np.argmax(shrunk))]
    assert peak == pytest.approx(math.sqrt(lam), abs=grid[1] - grid[0])
    assert _optimistic_shrinkage(1e6, lam) < _optimistic_shrinkage(math.sqrt(lam), lam)
    # larger lambda shrinks less; infinite lambda is the identity (no shrinkage)
    assert _optimistic_shrinkage(10.0, 100.0) > _optimistic_shrinkage(10.0, 1.0)
    assert _optimistic_shrinkage(10.0, math.inf) == 10.0
    # the no-shrinkage path returns the very same array object (byte-identical)
    passthrough = np.array([1.5, 2.0, 3.0])
    assert _optimistic_shrinkage(passthrough, math.inf) is passthrough


def test_shrinkage_none_is_byte_identical(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    rows = (
        _observation(
            evolution_context,
            card,
            decision_id="t",
            treatment=True,
            value=1.0,
            run_id="run-a",
        ),
        _observation(
            evolution_context,
            card,
            decision_id="c",
            treatment=False,
            value=0.2,
            run_id="run-b",
        ),
    )
    report = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.75, shrinkage=None
    )
    # identical to the documented unshrunk baseline: estimate 0.8, ESS 1.6
    assert report.estimate == pytest.approx(0.8)
    assert report.effective_sample_size == pytest.approx(1.6)
    assert report.maximum_importance_weight == pytest.approx(1.5)


def test_shrinkage_none_matches_a_plain_dr_reimplementation(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    target = 0.5
    rows = tuple(
        _observation(
            evolution_context,
            card,
            decision_id=f"r{i}",
            treatment=bool(i % 2),
            value=0.1 * i,
            offer_propensity=0.2 + 0.05 * i,
            run_id=f"run-{i % 3}",
        )
        for i in range(9)
    )
    report = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: target, shrinkage=None
    )
    # plain DR (no shrinkage) reconstructed directly from the rows -- q-hats are 0,
    # so the baseline vanishes and the correction is the realized value.
    scores: list[float] = []
    weights: list[float] = []
    for row in rows:
        behavior = row.offer_propensity
        if row.treatment:
            weight = target / behavior
        else:
            weight = (1.0 - target) / (1.0 - behavior)
        scores.append(weight * row.measurement.value)
        weights.append(weight)
    expected_estimate = float(np.mean(scores))
    expected_ess = float(np.sum(weights) ** 2 / np.sum(np.square(weights)))
    assert report.estimate == pytest.approx(expected_estimate, rel=1e-12, abs=1e-15)
    assert report.effective_sample_size == pytest.approx(expected_ess, rel=1e-12)
    assert report.maximum_importance_weight == pytest.approx(max(weights), rel=1e-12)


def test_finite_shrinkage_pulls_estimate_toward_the_direct_method(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    # one heavy-tailed weight (behavior offer 0.02 -> weight 0.5/0.02 = 25) with a
    # large residual dominates the unshrunk estimate; the direct-method baseline is 0.
    rows = tuple(
        _observation(
            evolution_context, card, decision_id=f"n{i}", treatment=True, value=0.0
        )
        for i in range(5)
    ) + (
        _observation(
            evolution_context,
            card,
            decision_id="tail",
            treatment=True,
            value=1.0,
            offer_propensity=0.02,
        ),
    )
    plain = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=None
    )
    shrunk = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=1.0
    )
    # shrinkage pulls the estimate toward the direct-method baseline (0.0)
    assert abs(shrunk.estimate) < abs(plain.estimate)
    # and tames the dominating weight, so the effective sample size rises
    assert shrunk.effective_sample_size > plain.effective_sample_size
    assert shrunk.maximum_importance_weight < plain.maximum_importance_weight


def test_huge_lambda_recovers_the_unshrunk_estimate(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    rows = (
        _observation(
            evolution_context, card, decision_id="t", treatment=True, value=1.0
        ),
        _observation(
            evolution_context,
            card,
            decision_id="tail",
            treatment=True,
            value=0.5,
            offer_propensity=0.02,
        ),
    )
    plain = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=None
    )
    nearly_plain = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=1e12
    )
    assert nearly_plain.estimate == pytest.approx(plain.estimate, rel=1e-4)


def test_auto_shrinkage_shrinks_a_heavy_tail(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    rows = tuple(
        _observation(
            evolution_context, card, decision_id=f"n{i}", treatment=True, value=0.0
        )
        for i in range(5)
    ) + (
        _observation(
            evolution_context,
            card,
            decision_id="tail",
            treatment=True,
            value=1.0,
            offer_propensity=0.02,
        ),
    )
    plain = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=None
    )
    auto = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage="auto"
    )
    assert auto.effective_sample_size > plain.effective_sample_size
    assert abs(auto.estimate) < abs(plain.estimate)


def test_auto_shrinkage_leaves_a_pure_signal_unshrunk(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    # identical weights and identical corrections -> zero variance at no-shrinkage;
    # any shrinkage only adds bias, so auto must keep the unshrunk estimate.
    rows = tuple(
        _observation(
            evolution_context, card, decision_id=f"s{i}", treatment=True, value=0.5
        )
        for i in range(6)
    )
    plain = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=None
    )
    auto = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage="auto"
    )
    assert auto.estimate == pytest.approx(plain.estimate)
    assert auto.effective_sample_size == pytest.approx(plain.effective_sample_size)


@pytest.mark.parametrize("bad_lambda", [0.0, -1.0, -0.0001, float("nan")])
def test_nonpositive_or_nan_shrinkage_lambda_is_rejected(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
    bad_lambda: float,
) -> None:
    card = revisions[0]
    rows = (
        _observation(
            evolution_context, card, decision_id="t", treatment=True, value=1.0
        ),
    )
    with pytest.raises(ValueError, match="shrinkage lambda must be positive"):
        ConditionalOfferDREvaluator().evaluate_reward(
            rows, target_offer_probability=lambda _: 0.5, shrinkage=bad_lambda
        )


@pytest.mark.parametrize("mode", ["AUTO", "", "shrink"])
def test_unknown_shrinkage_mode_is_rejected(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
    mode: str,
) -> None:
    card = revisions[0]
    rows = (
        _observation(
            evolution_context, card, decision_id="t", treatment=True, value=1.0
        ),
    )
    with pytest.raises(ValueError, match="unknown shrinkage mode"):
        ConditionalOfferDREvaluator().evaluate_reward(
            rows, target_offer_probability=lambda _: 0.5, shrinkage=mode
        )


def test_select_shrinkage_searches_beyond_the_largest_candidate() -> None:
    # three weight scales where the MSE-minimizing lambda is the middle observed
    # square, not the largest -- a mutant that returns max(candidates) would fail.
    raw = np.array([1.0, 1.0, 1.0, 1.0, 9.896588844232632, 29.342843845879017])
    baselines = np.zeros(6)
    corrections = np.array(
        [0.0, 0.0, 0.0, 0.0, 2.6213748830152044, 0.11042126594561627]
    )
    run_ids = ["r"] * 6  # single run -> row-IID fallback variance
    chosen = _select_shrinkage(raw, baselines, corrections, run_ids)
    candidates = sorted(float(c) for c in np.unique(raw[raw > 0.0] ** 2))
    assert math.isfinite(chosen)
    assert chosen == pytest.approx(candidates[1])
    assert chosen != pytest.approx(candidates[-1])


def test_select_shrinkage_uses_run_clustered_variance() -> None:
    # every run-cluster of the unshrunk scores sums to the same value, so the
    # run-clustered variance is zero and no shrinkage is optimal; a row-IID
    # objective (the pre-fix behavior) would shrink here instead.
    raw = np.array([1.0, 0.5, 2.0, 4.0, 2.0, 1.0, 1.0, 1.0])
    baselines = np.zeros(8)
    corrections = np.array([-1.0, 0.0, -0.5, 0.0, -1.0, 1.0, -1.0, 0.0])
    run_ids = ["0", "0", "1", "1", "2", "2", "3", "3"]
    assert _select_shrinkage(raw, baselines, corrections, run_ids) == math.inf


def test_auto_shrinkage_pins_the_selected_lambda(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    # five clean zero-residual rows plus one weight-25 tail (offer 0.02); a single
    # run means the row-IID variance drives auto to lambda = 25**2 = 625, which
    # halves the tail weight to its DRos peak value sqrt(625)/2 = 12.5.
    rows = tuple(
        _observation(
            evolution_context, card, decision_id=f"n{i}", treatment=True, value=0.0
        )
        for i in range(5)
    ) + (
        _observation(
            evolution_context,
            card,
            decision_id="tail",
            treatment=True,
            value=1.0,
            offer_propensity=0.02,
        ),
    )
    auto = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage="auto"
    )
    assert auto.maximum_importance_weight == pytest.approx(12.5)


def test_cluster_robust_se_reflects_shrinkage(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    rows = (
        _observation(
            evolution_context,
            card,
            decision_id="a1",
            treatment=True,
            value=0.0,
            run_id="run-a",
        ),
        _observation(
            evolution_context,
            card,
            decision_id="a2",
            treatment=True,
            value=0.0,
            run_id="run-a",
        ),
        _observation(
            evolution_context,
            card,
            decision_id="b1",
            treatment=True,
            value=0.0,
            run_id="run-b",
        ),
        _observation(
            evolution_context,
            card,
            decision_id="b-tail",
            treatment=True,
            value=1.0,
            offer_propensity=0.02,
            run_id="run-b",
        ),
    )
    evaluator = ConditionalOfferDREvaluator()
    plain = evaluator.evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=None
    )
    shrunk = evaluator.evaluate_reward(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=1.0
    )
    assert plain.clusters == 2
    assert plain.cluster_robust_se is not None
    assert shrunk.cluster_robust_se is not None
    # shrinking the heavy tail changes the run-clustered SE, so it is recomputed
    # on the shrunk scores rather than copied from the unshrunk estimate.
    assert shrunk.cluster_robust_se != pytest.approx(plain.cluster_robust_se)


def test_evaluate_invalidity_honors_shrinkage(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    rows = tuple(
        _observation(
            evolution_context, card, decision_id=f"n{i}", treatment=True, value=0.0
        )
        for i in range(5)
    ) + (
        _observation(
            evolution_context,
            card,
            decision_id="tail",
            treatment=True,
            value=1.0,
            offer_propensity=0.02,
        ),
    )
    evaluator = ConditionalOfferDREvaluator()
    plain = evaluator.evaluate_invalidity(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=None
    )
    shrunk = evaluator.evaluate_invalidity(
        rows, target_offer_probability=lambda _: 0.5, shrinkage=1.0
    )
    # the invalidity head shares the shrinkage path, so the dominating weight is
    # tamed and the effective sample size rises there too.
    assert shrunk.maximum_importance_weight < plain.maximum_importance_weight
    assert shrunk.effective_sample_size > plain.effective_sample_size
