"""Bootstrap EV substrate: exact-path rng identity and priced-noise jitter."""

from __future__ import annotations

import numpy as np
import pytest

from gigaevo.memory.read.bootstrap import bootstrap_ev_samples

DELTAS = [0.3, -0.1, 0.2]


def _samples(*, seed: int = 7, n: int = 256, **kwargs) -> np.ndarray:
    return bootstrap_ev_samples(
        DELTAS, 0.0, 1.0, n, np.random.default_rng(seed), **kwargs
    )


class TestPricedNoiseJitter:
    def test_zero_or_absent_ses_bit_exact_including_rng_position(self):
        # The exact path must not consume jitter draws: a point run replayed
        # with explicit zero ses has to stay seed-exact past the call.
        rngs = [np.random.default_rng(7) for _ in range(3)]
        absent = bootstrap_ev_samples(DELTAS, 0.0, 1.0, 64, rngs[0])
        none = bootstrap_ev_samples(DELTAS, 0.0, 1.0, 64, rngs[1], ses=None)
        zeros = bootstrap_ev_samples(DELTAS, 0.0, 1.0, 64, rngs[2], ses=[0.0] * 3)
        np.testing.assert_array_equal(absent, none)
        np.testing.assert_array_equal(absent, zeros)
        follow = [rng.random() for rng in rngs]
        assert follow[0] == follow[1] == follow[2]

    def test_positive_ses_widen_without_moving_the_center(self):
        point = _samples(n=8192)
        noisy = _samples(n=8192, ses=[0.2] * 3)
        assert noisy.std() > point.std() * 1.2
        assert noisy.mean() == pytest.approx(point.mean(), abs=0.02)

    def test_non_finite_or_negative_ses_degrade_to_exact(self):
        exact = _samples()
        degraded = _samples(ses=[float("nan"), -1.0, float("inf")])
        np.testing.assert_array_equal(exact, degraded)

    def test_ses_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="ses length"):
            _samples(ses=[0.1])

    def test_cold_atom_stays_exact(self):
        cold = bootstrap_ev_samples([], 0.5, 1.0, 16, np.random.default_rng(3))
        cold_with_ses = bootstrap_ev_samples(
            [], 0.5, 1.0, 16, np.random.default_rng(3), ses=[]
        )
        np.testing.assert_array_equal(cold, cold_with_ses)

    def test_jitter_composes_with_delta_weights(self):
        weighted = _samples(n=4096, delta_weights=[1.0, 0.5, 1.0])
        jittered = _samples(n=4096, delta_weights=[1.0, 0.5, 1.0], ses=[0.2] * 3)
        assert jittered.std() > weighted.std()
        assert jittered.mean() == pytest.approx(weighted.mean(), abs=0.02)
