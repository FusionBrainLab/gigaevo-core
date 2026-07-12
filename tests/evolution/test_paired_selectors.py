"""Tests for PairedBootstrapArchiveSelector.

The selector must be bit-identical to SumArchiveSelector whenever the paired
path cannot run (p_accept=0.5 OFF position, missing/mismatched/incoherent
vectors) and must gate on P(paired mean diff > 0) >= p_accept otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from gigaevo.evolution.strategies.paired_selectors import PairedBootstrapArchiveSelector
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.metrics.paired import PER_SAMPLE_SCORES_KEY
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = np.tile([0.4, 0.6], 150)  # mean 0.5, n=300


def _prog(fitness: float, scores=None) -> Program:
    metadata = {} if scores is None else {PER_SAMPLE_SCORES_KEY: list(scores)}
    return Program(
        code="def solve(): return 42",
        state=ProgramState.RUNNING,
        metrics={"fitness": fitness},
        metadata=metadata,
    )


def _selector(**kwargs) -> PairedBootstrapArchiveSelector:
    return PairedBootstrapArchiveSelector(["fitness"], **kwargs)


class _StubComparison:
    def __init__(self, p: float):
        self._p = p

    def probability_better(self, challenger, incumbent) -> float:
        return self._p


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_multiple_fitness_keys(self):
        with pytest.raises(ValueError, match="single-key"):
            PairedBootstrapArchiveSelector(["fitness", "aux"])

    def test_rejects_p_accept_below_half(self):
        with pytest.raises(ValueError, match="p_accept"):
            _selector(p_accept=0.49)

    def test_rejects_p_accept_of_one(self):
        with pytest.raises(ValueError, match="p_accept"):
            _selector(p_accept=1.0)

    def test_rejects_injected_comparison_in_off_position(self):
        # OFF never consults a comparison; silently discarding an explicit
        # injection would hand the caller the point rule they didn't ask for.
        with pytest.raises(ValueError, match="OFF"):
            _selector(p_accept=0.5, comparison=_StubComparison(0.9))


# ---------------------------------------------------------------------------
# OFF position and fallbacks — must match SumArchiveSelector exactly
# ---------------------------------------------------------------------------


class TestPointComparisonFallback:
    def test_off_position_matches_sum_selector_on_grid(self):
        off = _selector(p_accept=0.5)
        reference = SumArchiveSelector(["fitness"])
        grid = [0.4, 0.5, 0.5000001, 0.6]
        for new_fit in grid:
            for cur_fit in grid:
                new, cur = _prog(new_fit), _prog(cur_fit)
                assert off(new, cur) == reference(new, cur)

    def test_off_position_ignores_vectors(self):
        # Vectors say "clearly worse" but OFF must not consult them.
        off = _selector(p_accept=0.5)
        new = _prog(0.5001, scores=BASE - 0.02 + 0.0201)
        cur = _prog(0.5, scores=BASE)
        assert off(new, cur) is True

    def test_missing_vector_falls_back_to_point(self):
        sel = _selector(p_accept=0.75)
        assert sel(_prog(0.6, scores=BASE + 0.1), _prog(0.5)) is True
        assert sel(_prog(0.4, scores=BASE - 0.1), _prog(0.5)) is False

    def test_length_mismatch_falls_back_to_point(self):
        sel = _selector(p_accept=0.75)
        new = _prog(0.6, scores=[0.6, 0.6])
        cur = _prog(0.5, scores=BASE)
        assert sel(new, cur) is True

    def test_incoherent_vector_falls_back_to_point(self):
        sel = _selector(p_accept=0.75)
        # Vector mean 0.5 but stored fitness 0.6: contract drift → point rule.
        new = _prog(0.6, scores=BASE)
        cur = _prog(0.5, scores=BASE)
        assert sel(new, cur) is True


# ---------------------------------------------------------------------------
# Paired path
# ---------------------------------------------------------------------------


class TestPairedGate:
    def test_rejects_noise_sized_point_gain(self):
        # Point comparison accepts (0.5001 > 0.5); paired test sees noise.
        diff = np.tile([0.01, -0.0098], 150)  # mean 0.0001, sd ~0.0099
        new = _prog(float((BASE + diff).mean()), scores=BASE + diff)
        cur = _prog(float(BASE.mean()), scores=BASE)
        assert SumArchiveSelector(["fitness"])(new, cur) is True
        assert _selector(p_accept=0.75)(new, cur) is False

    def test_accepts_clear_gain(self):
        new = _prog(0.52, scores=BASE + 0.02)
        cur = _prog(0.5, scores=BASE)
        assert _selector(p_accept=0.75)(new, cur) is True

    def test_rejects_equal_programs(self):
        new = _prog(0.5, scores=BASE)
        cur = _prog(0.5, scores=BASE)
        assert _selector(p_accept=0.75)(new, cur) is False

    def test_verdict_antisymmetric(self):
        diff = np.tile([0.01, -0.0098], 150)
        a = _prog(float((BASE + diff).mean()), scores=BASE + diff)
        b = _prog(float(BASE.mean()), scores=BASE)
        sel = _selector(p_accept=0.75)
        assert not (sel(a, b) and sel(b, a))

    def test_lower_is_better_orientation(self):
        sel = PairedBootstrapArchiveSelector(
            ["fitness"], fitness_key_higher_is_better=[False], p_accept=0.75
        )
        lower = _prog(0.48, scores=BASE - 0.02)
        higher = _prog(0.5, scores=BASE)
        assert sel(lower, higher) is True
        assert sel(higher, lower) is False

    def test_threshold_is_inclusive_via_injected_comparison(self):
        new = _prog(0.5, scores=BASE)
        cur = _prog(0.5, scores=BASE)
        at = _selector(p_accept=0.75, comparison=_StubComparison(0.75))
        below = _selector(p_accept=0.75, comparison=_StubComparison(0.7499))
        assert at(new, cur) is True
        assert below(new, cur) is False
