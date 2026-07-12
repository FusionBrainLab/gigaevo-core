"""Tests for the paired per-sample comparison statistic and Program accessors."""

from __future__ import annotations

import numpy as np
import pytest

from gigaevo.programs.metrics.paired import (
    PER_SAMPLE_SCORES_KEY,
    PairedBootstrap,
    PairedComparison,
    get_paired_scores,
    get_per_sample_scores,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(scores=None, metrics=None) -> Program:
    metadata = {} if scores is None else {PER_SAMPLE_SCORES_KEY: scores}
    return Program(
        code="def solve(): return 42",
        state=ProgramState.RUNNING,
        metrics=metrics or {},
        metadata=metadata,
    )


BASE = np.tile([0.4, 0.6], 150)  # mean 0.5, n=300 — a fixed shared eval set


# ---------------------------------------------------------------------------
# PairedBootstrap
# ---------------------------------------------------------------------------


class TestPairedBootstrap:
    def test_conforms_to_protocol(self):
        assert isinstance(PairedBootstrap(), PairedComparison)

    def test_identical_vectors_probability_is_half(self):
        p = PairedBootstrap().probability_better(BASE, BASE.copy())
        assert p == 0.5

    def test_uniform_gain_probability_is_one(self):
        p = PairedBootstrap().probability_better(BASE + 0.02, BASE)
        assert p == 1.0

    def test_uniform_loss_probability_is_zero(self):
        p = PairedBootstrap().probability_better(BASE - 0.02, BASE)
        assert p == 0.0

    def test_antisymmetry_exact(self):
        challenger = BASE + np.tile([0.01, -0.0098], 150)
        p_ab = PairedBootstrap().probability_better(challenger, BASE)
        p_ba = PairedBootstrap().probability_better(BASE, challenger)
        assert p_ab + p_ba == pytest.approx(1.0, abs=1e-12)

    def test_deterministic_across_instances_and_calls(self):
        challenger = BASE + np.tile([0.01, -0.0098], 150)
        first = PairedBootstrap().probability_better(challenger, BASE)
        second = PairedBootstrap().probability_better(challenger, BASE)
        assert first == second

    def test_order_of_prior_calls_does_not_change_verdict(self):
        challenger = BASE + np.tile([0.01, -0.0098], 150)
        comparison = PairedBootstrap()
        first = comparison.probability_better(challenger, BASE)
        comparison.probability_better(BASE, challenger)
        third = comparison.probability_better(challenger, BASE)
        assert first == third

    def test_noise_dominated_gain_stays_uncertain(self):
        # mean diff 0.0001 vs sd ~0.0099: statistically indistinguishable
        challenger = BASE + np.tile([0.01, -0.0098], 150)
        p = PairedBootstrap().probability_better(challenger, BASE)
        assert 0.4 < p < 0.75

    def test_clear_small_gain_is_confident(self):
        # mean diff 0.005 vs sd ~0.005: clearly better
        challenger = BASE + np.tile([0.01, 0.0], 150)
        p = PairedBootstrap().probability_better(challenger, BASE)
        assert p > 0.9


class TestPairedBootstrapEstimate:
    def test_value_is_the_exact_paired_mean(self):
        challenger = BASE + np.tile([0.15, 0.05], 150)
        measurement = PairedBootstrap().estimate(challenger, BASE)
        assert measurement.value == pytest.approx(0.1)
        assert measurement.se > 0.0

    def test_identical_vectors_are_exactly_zero(self):
        measurement = PairedBootstrap().estimate(BASE, BASE.copy())
        assert measurement.value == 0.0
        assert measurement.se == 0.0

    def test_value_antisymmetric_se_order_invariant(self):
        challenger = BASE + np.tile([0.01, -0.0098], 150)
        ab = PairedBootstrap().estimate(challenger, BASE)
        ba = PairedBootstrap().estimate(BASE, challenger)
        assert ab.value == -ba.value
        assert ab.se == ba.se

    def test_se_tracks_the_paired_analytic_scale(self):
        rng = np.random.default_rng(0)
        challenger = BASE + rng.normal(0.001, 0.05, size=BASE.size)
        diff = challenger - BASE
        analytic_se = diff.std(ddof=1) / np.sqrt(diff.size)
        measurement = PairedBootstrap(n_resamples=4000).estimate(challenger, BASE)
        assert measurement.se == pytest.approx(analytic_se, rel=0.15)

    def test_deterministic_across_instances_and_calls(self):
        challenger = BASE + np.tile([0.01, -0.0098], 150)
        assert PairedBootstrap().estimate(
            challenger, BASE
        ) == PairedBootstrap().estimate(challenger, BASE)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


class TestGetPerSampleScores:
    def test_missing_key_returns_none(self):
        assert get_per_sample_scores(_prog()) is None

    def test_valid_list_returns_array(self):
        scores = get_per_sample_scores(_prog(scores=[0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(scores, [0.1, 0.2, 0.3])

    def test_non_numeric_returns_none(self):
        assert get_per_sample_scores(_prog(scores=["a", "b"])) is None

    def test_nan_returns_none(self):
        assert get_per_sample_scores(_prog(scores=[0.1, float("nan")])) is None

    def test_empty_returns_none(self):
        assert get_per_sample_scores(_prog(scores=[])) is None

    def test_non_1d_returns_none(self):
        assert get_per_sample_scores(_prog(scores=[[0.1], [0.2]])) is None

    def test_coherent_metric_returns_array(self):
        prog = _prog(scores=[0.4, 0.6], metrics={"fitness": 0.5})
        assert get_per_sample_scores(prog, metric_key="fitness") is not None

    def test_incoherent_metric_returns_none(self):
        prog = _prog(scores=[0.4, 0.6], metrics={"fitness": 0.6})
        assert get_per_sample_scores(prog, metric_key="fitness") is None

    def test_metric_key_absent_returns_none(self):
        prog = _prog(scores=[0.4, 0.6], metrics={})
        assert get_per_sample_scores(prog, metric_key="fitness") is None


class TestGetPairedScores:
    def test_both_present_equal_length(self):
        pair = get_paired_scores(_prog(scores=[0.1, 0.2]), _prog(scores=[0.3, 0.4]))
        assert pair is not None
        np.testing.assert_array_equal(pair[0], [0.1, 0.2])
        np.testing.assert_array_equal(pair[1], [0.3, 0.4])

    def test_length_mismatch_returns_none(self):
        assert get_paired_scores(_prog(scores=[0.1]), _prog(scores=[0.1, 0.2])) is None

    def test_either_missing_returns_none(self):
        assert get_paired_scores(_prog(scores=[0.1]), _prog()) is None
        assert get_paired_scores(_prog(), _prog(scores=[0.1])) is None

    def test_coherence_applies_to_both_sides(self):
        good = _prog(scores=[0.4, 0.6], metrics={"fitness": 0.5})
        drifted = _prog(scores=[0.4, 0.6], metrics={"fitness": 0.9})
        assert get_paired_scores(good, drifted, metric_key="fitness") is None
