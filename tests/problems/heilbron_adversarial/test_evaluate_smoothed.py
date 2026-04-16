"""Smoothed-fitness unit tests for heilbron_adversarial pop_a / pop_b.

Validates the redesign in experiments/heilbron/k5-budget-loose/REDESIGN.md:
  - Pop A resistance uses tanh smoothing; cold-start matches main branch formula.
  - Pop B fitness uses tanh smoothing; D-did-nothing maps to neutral 0.5.

These tests cover the 5 cases listed in Phase 1 Step 7 of the implementation plan.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POP_A_DIR = PROJECT_ROOT / "problems" / "heilbron_adversarial" / "pop_a"
POP_B_DIR = PROJECT_ROOT / "problems" / "heilbron_adversarial" / "pop_b"

# Both pop_a and pop_b modules import a top-level `helper` module that lives in
# their own directory. Import each evaluate.py via importlib with a per-call
# sys.path patch so the loader finds the right `helper`.


def _load_module(name: str, file: Path, extra_path: Path):
    sys.path.insert(0, str(extra_path))
    try:
        spec = importlib.util.spec_from_file_location(name, file)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(extra_path))


pop_a = _load_module("pop_a_eval", POP_A_DIR / "evaluate.py", POP_A_DIR)
pop_b = _load_module("pop_b_eval", POP_B_DIR / "evaluate.py", POP_B_DIR)
pop_a_helper = _load_module("pop_a_helper", POP_A_DIR / "helper.py", POP_A_DIR)


def _seed_grid(n: int = 11, seed: int = 42) -> np.ndarray:
    """A deterministic 11-point config strictly inside the unit triangle.

    Uses the grid layout from problems/heilbron_adversarial/pop_a/initial_programs/grid.py
    plus a small deterministic perturbation so the smallest-triangle area is > 0
    (a perfectly regular grid produces collinear triples → area=0 → INVALID).
    """
    rng = np.random.default_rng(seed)
    A, B, C = pop_a_helper.get_unit_triangle()
    pts = []
    rows = 5
    count = 0
    for row in range(rows):
        num = rows - row
        v = (row + 0.5) / rows
        for i in range(num):
            if count >= n:
                break
            u = (i + 0.5) / num * (1.0 - v)
            P = (1 - u - v) * A + u * B + v * C
            P = P + rng.uniform(-0.001, 0.001, size=2)
            pts.append(P)
            count += 1
        if count >= n:
            break
    return np.array(pts)


# ---------------------------------------------------------------------------
# Pop A — Constructor (resistance smoothing + cold-start consistency)
# ---------------------------------------------------------------------------


class TestPopASmoothedResistance:
    def test_cold_start_returns_consistent_fitness(self):
        """Cold start (no opponents): resistance=1.0, fitness=ALPHA*q + (1-ALPHA)*1.0."""
        pts = _seed_grid()
        out, artifact = pop_a.evaluate([], pts)
        q = out["quality"]
        assert out["resistance"] == pytest.approx(1.0)
        assert out["fitness"] == pytest.approx(
            pop_a.ALPHA * q + (1.0 - pop_a.ALPHA) * 1.0
        )
        assert out["is_valid"] == 1.0
        # Cold-start artifact: zero-length per-opp arrays, role="constructor".
        assert artifact["role"] == "constructor"
        assert artifact["n_opponents"] == 0
        assert artifact["per_opp_delta"] == []

    def test_all_opponents_fail_returns_max_resistance(self):
        """All opponents raise → resistance=1.0 (full upper bound)."""
        pts = _seed_grid()

        def boom(_pts):
            raise RuntimeError("boom")

        out, artifact = pop_a.evaluate([boom, boom, boom], pts)
        q = out["quality"]
        assert out["resistance"] == pytest.approx(1.0)
        # Same formula as cold-start branch.
        assert out["fitness"] == pytest.approx(
            pop_a.ALPHA * q + (1.0 - pop_a.ALPHA) * 1.0
        )
        # Failed opponents → NaN sentinels (no measurement).
        assert artifact["n_opponents"] == 3
        assert len(artifact["per_opp_delta"]) == 3
        assert all(math.isnan(d) for d in artifact["per_opp_delta"])

    def test_mixed_outcomes_lands_above_neutral(self):
        """Mix of failure (1.0) and success (delta>0 → tanh<0) → resistance ∈ (0.5, 1)."""
        pts = _seed_grid()

        def winner(p):
            # Move one point inward so the smallest triangle area grows.
            p = p.copy()
            centroid = p.mean(axis=0)
            p[0] = 0.7 * p[0] + 0.3 * centroid
            return p

        def loser(_p):
            raise RuntimeError("nope")

        out, artifact = pop_a.evaluate([winner, loser, loser], pts)
        # winner pulls resistance below 1.0 a little; two losers anchor it high.
        assert 0.5 < out["resistance"] < 1.0
        # Index alignment: slot 0 (winner) has a real delta, slots 1,2 (losers) NaN.
        assert math.isfinite(artifact["per_opp_delta"][0])
        assert math.isnan(artifact["per_opp_delta"][1])
        assert math.isnan(artifact["per_opp_delta"][2])

    def test_strong_success_drives_resistance_low(self):
        """All opponents succeed strongly (delta ≈ Q_MAX) → resistance ≈ (tanh(-1)+1)/2 ≈ 0.12."""
        pts = _seed_grid()
        Q_MAX = pop_a.Q_MAX
        raw_q = float(pop_a_helper.get_smallest_triangle_area(pts))

        # Hand-craft a fake "improver" that returns a config with min-area ≈ raw + Q_MAX.
        # Easiest mock: replace the module-level get_smallest_triangle_area used by
        # evaluate so that improved configs report raw+Q_MAX. Less invasive: write
        # an improver that scales the input slightly so post-area is larger.
        def strong(p):
            p = p.copy()
            # Scale all points slightly toward the centroid by a tiny amount; this
            # generally INCREASES the smallest triangle area for a near-degenerate
            # config, but to guarantee a target delta we monkey-patch the helper
            # below instead.
            return p

        # Monkey-patch the helper used inside pop_a.evaluate so that the
        # improver's output reports area = raw_q + Q_MAX (delta = Q_MAX exactly).
        original = pop_a.get_smallest_triangle_area
        call_idx = {"i": 0}

        def fake_area(arr):
            call_idx["i"] += 1
            # First call evaluates the constructor's points themselves → real value.
            if call_idx["i"] == 1:
                return original(arr)
            # All subsequent calls (one per opponent) return raw_q + Q_MAX.
            return raw_q + Q_MAX

        pop_a.get_smallest_triangle_area = fake_area
        try:
            out, artifact = pop_a.evaluate([strong, strong, strong], pts)
        finally:
            pop_a.get_smallest_triangle_area = original

        # tanh(-1) ≈ -0.7616 → rescale to ~0.119
        expected = (math.tanh(-1.0) + 1.0) / 2.0
        assert out["resistance"] == pytest.approx(expected, abs=1e-3)
        # All three D succeeded → all per-opp deltas finite and ≈ Q_MAX.
        assert all(
            d == pytest.approx(Q_MAX, abs=1e-9) for d in artifact["per_opp_delta"]
        )

    def test_negative_delta_increases_resistance(self):
        """When D makes config WORSE (negative delta), resistance > 0.5."""
        pts = _seed_grid()
        Q_MAX = pop_a.Q_MAX
        raw_q = float(pop_a_helper.get_smallest_triangle_area(pts))

        def regress(p):
            return p.copy()

        original = pop_a.get_smallest_triangle_area
        call_idx = {"i": 0}

        def fake_area(arr):
            call_idx["i"] += 1
            if call_idx["i"] == 1:
                return original(arr)
            return raw_q - 0.5 * Q_MAX  # delta = -0.5*Q_MAX

        pop_a.get_smallest_triangle_area = fake_area
        try:
            out, artifact = pop_a.evaluate([regress, regress, regress], pts)
        finally:
            pop_a.get_smallest_triangle_area = original

        # tanh(0.5)≈0.462 → rescale to (0.462+1)/2 = 0.731
        expected = (math.tanh(0.5) + 1.0) / 2.0
        assert out["resistance"] == pytest.approx(expected, abs=1e-3)
        assert out["resistance"] > 0.5
        # Negative delta (D made it worse) is preserved as a finite negative number.
        assert all(
            d == pytest.approx(-0.5 * Q_MAX, abs=1e-9)
            for d in artifact["per_opp_delta"]
        )


# ---------------------------------------------------------------------------
# Pop B — Improver (smoothed fitness rescaled to (0, 1))
# ---------------------------------------------------------------------------


class TestPopBSmoothedFitness:
    def test_noop_improver_yields_neutral_fitness(self):
        """delta=0 (improver returns input untouched) → fitness=0.5."""
        cfg = _seed_grid()

        def noop(p):
            return p.copy()

        out, artifact = pop_b.evaluate([cfg], noop)
        assert out["fitness"] == pytest.approx(0.5)
        assert out["is_valid"] == 1.0
        # noop → real measurement, delta=0 (post == pre), role="improver".
        assert artifact["role"] == "improver"
        assert artifact["n_opponents"] == 1
        assert artifact["per_opp_delta"][0] == pytest.approx(0.0, abs=1e-9)

    def test_strong_improvement_drives_fitness_high(self):
        """All opponents improved by ~Q_MAX → fitness ≈ (tanh(1)+1)/2 ≈ 0.881."""
        cfg = _seed_grid()
        Q_MAX = pop_b.Q_MAX

        # Same monkey-patch trick: first call (pre-area) is real; second (post-area)
        # returns pre + Q_MAX so delta = Q_MAX exactly.
        original = pop_b.get_smallest_triangle_area
        cycle = {"i": 0}

        def fake_area(arr):
            cycle["i"] += 1
            # Calls per opponent: pre, then post. Even-indexed calls (2,4,...) are post.
            if cycle["i"] % 2 == 1:
                return original(arr)
            real_pre = original(arr)  # arr is the IMPROVED config
            return real_pre + Q_MAX

        def passthrough(p):
            return p.copy()

        pop_b.get_smallest_triangle_area = fake_area
        try:
            out, artifact = pop_b.evaluate([cfg, cfg, cfg], passthrough)
        finally:
            pop_b.get_smallest_triangle_area = original

        expected = (math.tanh(1.0) + 1.0) / 2.0
        assert out["fitness"] == pytest.approx(expected, abs=1e-3)
        # All 3 slots: delta = +Q_MAX exactly.
        assert all(
            d == pytest.approx(Q_MAX, abs=1e-9) for d in artifact["per_opp_delta"]
        )

    def test_invalid_improver_output_yields_neutral(self):
        """Improver returns garbage → score=0 → fitness=0.5 (D-did-nothing)."""
        cfg = _seed_grid()

        def garbage(_p):
            return "not an array"

        out, artifact = pop_b.evaluate([cfg], garbage)
        assert out["fitness"] == pytest.approx(0.5)
        # D-did-nothing: delta=0 (post == pre by construction).
        assert artifact["per_opp_delta"][0] == pytest.approx(0.0, abs=1e-9)
