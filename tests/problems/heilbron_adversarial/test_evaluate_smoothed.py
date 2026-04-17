"""Unit tests for heilbron_adversarial pop_a / pop_b under v3 semantics.

Pop A (Constructor) — fitness is normalized min_area in [0, 1] (ALPHA/quality/
resistance are gone; fitness IS quality by another name). The artifact still
carries per-opponent fitness deltas for the DGTracker.

Pop B (Improver) — unchanged: tanh-smoothed mean Δ (D-did-nothing → 0.5).
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
# Pop A — Constructor (v3: fitness = normalized min_area; per-opp delta artifact)
# ---------------------------------------------------------------------------


class TestPopAv3Fitness:
    def test_cold_start_returns_normalized_quality(self):
        """No opponents → fitness = min(raw_quality / Q_MAX, 1.0); artifact carries no deltas."""
        pts = _seed_grid()
        out, artifact = pop_a.evaluate([], pts)
        raw_q = out["actual_fitness"]
        assert out["fitness"] == pytest.approx(min(raw_q / pop_a.Q_MAX, 1.0))
        assert out["is_valid"] == 1.0
        assert out["n_opponents"] == 0.0
        assert out["mean_improvement"] == pytest.approx(0.0)
        # Cold-start artifact per v3 evaluate: NaN-filled per-opp arrays, role="constructor".
        assert artifact["role"] == "constructor"
        assert artifact["n_opponents"] == 0

    def test_all_opponents_fail_keeps_fitness_equal_to_baseline(self):
        """All opponents raise → fitness unchanged (depends only on raw_quality)."""
        pts = _seed_grid()

        def boom(_pts):
            raise RuntimeError("boom")

        out, artifact = pop_a.evaluate([boom, boom, boom], pts)
        raw_q = out["actual_fitness"]
        # v3: fitness = normalized quality — opponent outcomes don't enter the scalar.
        assert out["fitness"] == pytest.approx(min(raw_q / pop_a.Q_MAX, 1.0))
        # No successful improver → mean_improvement is 0 (empty deltas).
        assert out["mean_improvement"] == pytest.approx(0.0)
        # Failed opponents → NaN sentinels in artifact (no measurement).
        assert artifact["n_opponents"] == 3
        assert len(artifact["per_opp_delta"]) == 3
        assert all(math.isnan(d) for d in artifact["per_opp_delta"])

    def test_mixed_outcomes_record_per_opp_delta(self):
        """Mix of failure and real improver → artifact[0] finite, [1:] NaN."""
        pts = _seed_grid()

        def winner(p):
            p = p.copy()
            centroid = p.mean(axis=0)
            p[0] = 0.7 * p[0] + 0.3 * centroid
            return p

        def loser(_p):
            raise RuntimeError("nope")

        out, artifact = pop_a.evaluate([winner, loser, loser], pts)
        # Fitness is normalized quality; unaffected by opponent outcomes in v3.
        raw_q = out["actual_fitness"]
        assert out["fitness"] == pytest.approx(min(raw_q / pop_a.Q_MAX, 1.0))
        # Index alignment in the artifact: slot 0 finite, 1 & 2 NaN.
        assert math.isfinite(artifact["per_opp_delta"][0])
        assert math.isnan(artifact["per_opp_delta"][1])
        assert math.isnan(artifact["per_opp_delta"][2])

    def test_strong_success_records_positive_delta_in_artifact(self):
        """Monkey-patched helper yields delta=+Q_MAX for every opponent."""
        pts = _seed_grid()
        Q_MAX = pop_a.Q_MAX
        raw_q = float(pop_a_helper.get_smallest_triangle_area(pts))

        def strong(p):
            return p.copy()

        original = pop_a.get_smallest_triangle_area
        call_idx = {"i": 0}

        def fake_area(arr):
            call_idx["i"] += 1
            if call_idx["i"] == 1:
                return original(arr)
            return raw_q + Q_MAX

        pop_a.get_smallest_triangle_area = fake_area
        try:
            out, artifact = pop_a.evaluate([strong, strong, strong], pts)
        finally:
            pop_a.get_smallest_triangle_area = original

        # All three D succeeded → all per-opp deltas finite and ≈ +Q_MAX.
        assert all(
            d == pytest.approx(Q_MAX, abs=1e-9) for d in artifact["per_opp_delta"]
        )
        # mean_improvement is the raw mean delta, not tanh-smoothed.
        assert out["mean_improvement"] == pytest.approx(Q_MAX, abs=1e-9)

    def test_negative_delta_recorded_as_negative(self):
        """When the improver makes things WORSE, the artifact preserves the negative sign."""
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
            return raw_q - 0.5 * Q_MAX

        pop_a.get_smallest_triangle_area = fake_area
        try:
            out, artifact = pop_a.evaluate([regress, regress, regress], pts)
        finally:
            pop_a.get_smallest_triangle_area = original

        # Negative delta preserved as-is in the artifact (DGTracker routes it to dg_g_resisted).
        assert all(
            d == pytest.approx(-0.5 * Q_MAX, abs=1e-9)
            for d in artifact["per_opp_delta"]
        )
        assert out["mean_improvement"] == pytest.approx(-0.5 * Q_MAX, abs=1e-9)


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
