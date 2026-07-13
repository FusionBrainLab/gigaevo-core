"""Effect estimators: exact point crediting and paired per-sample se."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gigaevo.memory.write.crediting import (
    EffectEstimator,
    PairedEffectEstimator,
    PointEffectEstimator,
)
from gigaevo.memory.write.stats import InjectionOutcome
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.metrics.paired import PairedBootstrap

BASE_VEC = tuple([0.4, 0.6] * 150)  # mean 0.5, n=300 — a fixed shared eval set
CHILD_VEC = tuple([0.55, 0.65] * 150)  # mean 0.6, non-uniform paired diff


def outcome(**overrides) -> InjectionOutcome:
    params = {
        "id": "child-1",
        "fitness": 0.6,
        "base_selected_ids": ("card-a",),
        "base_metrics": {VALIDITY_KEY: 1.0, "fitness": 0.5},
        "base_id": "parent-1",
        "base_fitness": 0.5,
        "base_scores": BASE_VEC,
        "child_scores": CHILD_VEC,
        "card_ids_used": ("card-a",),
    }
    params.update(overrides)
    return InjectionOutcome(**params)


class TestPointEffectEstimator:
    def test_conforms_to_protocol(self):
        assert isinstance(PointEffectEstimator(), EffectEstimator)

    def test_oriented_delta_is_exact(self):
        measurement = PointEffectEstimator().estimate(outcome(), higher_is_better=True)
        assert measurement.value == pytest.approx(0.1)
        assert measurement.se == 0.0

    def test_orientation_flips_for_minimization(self):
        measurement = PointEffectEstimator().estimate(outcome(), higher_is_better=False)
        assert measurement.value == pytest.approx(-0.1)

    def test_missing_fitness_raises(self):
        with pytest.raises(ValueError, match="base_fitness"):
            PointEffectEstimator().estimate(
                outcome(fitness=None), higher_is_better=True
            )
        with pytest.raises(ValueError, match="base_fitness"):
            PointEffectEstimator().estimate(
                outcome(base_fitness=None), higher_is_better=True
            )


class TestPairedEffectEstimator:
    def test_conforms_to_protocol(self):
        assert isinstance(PairedEffectEstimator(), EffectEstimator)

    def test_value_is_point_delta_with_positive_se(self):
        estimator = PairedEffectEstimator()
        measurement = estimator.estimate(outcome(), higher_is_better=True)
        point = PointEffectEstimator().estimate(outcome(), higher_is_better=True)
        assert measurement.value == point.value
        assert measurement.se > 0.0
        assert not estimator.degraded

    def test_value_antisymmetric_se_orientation_invariant(self):
        up = PairedEffectEstimator().estimate(outcome(), higher_is_better=True)
        down = PairedEffectEstimator().estimate(outcome(), higher_is_better=False)
        assert up.value == -down.value
        assert up.se == down.se

    def test_identical_vectors_zero_se_without_degradation(self):
        estimator = PairedEffectEstimator()
        measurement = estimator.estimate(
            outcome(fitness=0.5, child_scores=BASE_VEC), higher_is_better=True
        )
        assert measurement.value == 0.0
        assert measurement.se == 0.0
        assert not estimator.degraded

    def test_missing_vector_degrades_to_unknown(self):
        estimator = PairedEffectEstimator()
        measurement = estimator.estimate(
            outcome(child_scores=None), higher_is_better=True
        )
        assert measurement.value == pytest.approx(0.1)
        assert measurement.se is None
        assert estimator.degraded == {"missing_vector": 1}

    def test_unusable_vector_degrades_to_unknown(self):
        estimator = PairedEffectEstimator()
        bad = (float("nan"),) * len(BASE_VEC)
        measurement = estimator.estimate(
            outcome(child_scores=bad), higher_is_better=True
        )
        assert measurement.se is None
        assert estimator.degraded == {"unusable_vector": 1}

    def test_length_mismatch_degrades_to_unknown(self):
        estimator = PairedEffectEstimator()
        measurement = estimator.estimate(
            outcome(child_scores=CHILD_VEC[:100]), higher_is_better=True
        )
        assert measurement.se is None
        assert estimator.degraded == {"length_mismatch": 1}

    def test_incoherent_vector_degrades_but_keeps_scalar_delta(self):
        estimator = PairedEffectEstimator()
        measurement = estimator.estimate(outcome(fitness=0.9), higher_is_better=True)
        assert measurement.value == pytest.approx(0.4)
        assert measurement.se is None
        assert estimator.degraded == {"incoherent_vector": 1}

    def test_degenerate_comparison_se_degrades_to_unknown(self):
        class NanSeComparison:
            def probability_better(self, challenger, incumbent):
                return 0.5

            def estimate(self, challenger, incumbent):
                return SimpleNamespace(value=0.1, se=float("nan"))

        estimator = PairedEffectEstimator(comparison=NanSeComparison())
        measurement = estimator.estimate(outcome(), higher_is_better=True)
        assert measurement.se is None
        assert estimator.degraded == {"degenerate_se": 1}

    def test_raising_comparison_degrades_to_unknown(self):
        class RaisingComparison:
            def probability_better(self, challenger, incumbent):
                return 0.5

            def estimate(self, challenger, incumbent):
                raise RuntimeError("comparison failed")

        estimator = PairedEffectEstimator(comparison=RaisingComparison())
        measurement = estimator.estimate(outcome(), higher_is_better=True)
        assert measurement.value == pytest.approx(0.1)
        assert measurement.se is None
        assert estimator.degraded == {"comparison_error": 1}

    def test_uncoercible_comparison_se_degrades_to_unknown(self):
        class InvalidSeComparison:
            def probability_better(self, challenger, incumbent):
                return 0.5

            def estimate(self, challenger, incumbent):
                return SimpleNamespace(value=0.1, se="not-a-number")

        estimator = PairedEffectEstimator(comparison=InvalidSeComparison())
        measurement = estimator.estimate(outcome(), higher_is_better=True)
        assert measurement.value == pytest.approx(0.1)
        assert measurement.se is None
        assert estimator.degraded == {"comparison_error": 1}

    def test_degradations_accumulate_per_reason(self):
        estimator = PairedEffectEstimator()
        estimator.estimate(outcome(child_scores=None), higher_is_better=True)
        estimator.estimate(outcome(base_scores=None), higher_is_better=True)
        estimator.estimate(outcome(child_scores=CHILD_VEC[:2]), higher_is_better=True)
        assert estimator.degraded == {"missing_vector": 2, "length_mismatch": 1}

    def test_missing_fitness_raises(self):
        with pytest.raises(ValueError, match="base_fitness"):
            PairedEffectEstimator().estimate(
                outcome(base_fitness=None), higher_is_better=True
            )


@pytest.mark.parametrize("n_resamples", [0, -1])
def test_paired_bootstrap_rejects_nonpositive_resamples(n_resamples):
    with pytest.raises(ValueError, match="n_resamples"):
        PairedBootstrap(n_resamples=n_resamples)
