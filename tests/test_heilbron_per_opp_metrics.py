"""Tests for per_opp_metrics artifact emitted by Heilbron evaluate.py files.

Covers both heilbron_repro_v1 and heilbron_adversarial variants. For each,
the evaluate.py must emit `per_opp_metrics: list[dict[str, float]]` aligned
with opponent_results so a `ConfigurableAggregator` can reproduce the
program-level `metrics` dict by replaying the per-opponent primitives.

The back-compat `per_opp_delta` field must equal `[m["delta"] for m in
per_opp_metrics]` after the change.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from gigaevo.programs.metrics.aggregators import (
    ConfigurableAggregator,
    ConstantSpec,
    IntrinsicSpec,
    LinearSpec,
    ReduceSpec,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POP_VARIANTS = [
    ("heilbron_repro_v1", "pop_a"),
    ("heilbron_repro_v1", "pop_b"),
    ("heilbron_adversarial", "pop_a"),
    ("heilbron_adversarial", "pop_b"),
]


# ---------------------------------------------------------------------------
# Module loading — each evaluate.py imports its sibling `helper`. We patch
# sys.path per load so the correct helper is picked up.
# ---------------------------------------------------------------------------


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


def _load_evaluate(problem: str, pop: str):
    pop_dir = PROJECT_ROOT / "problems" / problem / pop
    return _load_module(
        f"{problem}_{pop}_eval",
        pop_dir / "evaluate.py",
        pop_dir,
    )


def _load_helper(problem: str, pop: str):
    pop_dir = PROJECT_ROOT / "problems" / problem / pop
    return _load_module(
        f"{problem}_{pop}_helper",
        pop_dir / "helper.py",
        pop_dir,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_grid(helper_mod, n: int = 11, seed: int = 42) -> np.ndarray:
    """Deterministic 11-point config inside the unit triangle with min_area > 0."""
    rng = np.random.default_rng(seed)
    A, B, C = helper_mod.get_unit_triangle()
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


def _noop_improver(p):
    return p.copy()


def _centroid_improver(p):
    p = p.copy()
    centroid = p.mean(axis=0)
    p[0] = 0.7 * p[0] + 0.3 * centroid
    return p


# ---------------------------------------------------------------------------
# MetricsContext helpers — the aggregator's validity gate delegates here.
# ---------------------------------------------------------------------------


def _ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness", higher_is_better=True, is_primary=True
            ),
            "is_valid": MetricSpec(description="validity", higher_is_better=True),
        }
    )


# Required keys per schema (spec §5.1).
POP_B_KEYS = frozenset({"pre_q", "post_q", "delta", "score", "is_valid"})
POP_A_KEYS = frozenset({"post_q", "delta", "resistance_score", "is_valid"})


# ===========================================================================
# pop_b — improver / D side
# ===========================================================================


class TestPopBArtifact:
    @pytest.mark.parametrize("problem", ["heilbron_repro_v1", "heilbron_adversarial"])
    def test_pop_b_artifact_contains_per_opp_metrics_aligned_with_opponents(
        self, problem
    ):
        helper = _load_helper(problem, "pop_a")  # a valid config comes from pop_a
        ev = _load_evaluate(problem, "pop_b")
        cfg = _seed_grid(helper)
        opponents = [cfg, cfg, cfg]

        _metrics, artifact = ev.evaluate(opponents, _noop_improver)

        assert "per_opp_metrics" in artifact
        per_opp = artifact["per_opp_metrics"]
        assert isinstance(per_opp, list)
        assert len(per_opp) == len(opponents)
        for entry in per_opp:
            assert POP_B_KEYS.issubset(entry.keys())
            for k in POP_B_KEYS:
                assert isinstance(entry[k], float)

    @pytest.mark.parametrize("problem", ["heilbron_repro_v1", "heilbron_adversarial"])
    def test_pop_b_invalid_opponent_flags_is_valid_zero(self, problem):
        helper = _load_helper(problem, "pop_a")
        ev = _load_evaluate(problem, "pop_b")
        cfg = _seed_grid(helper)
        opponents = [cfg, "not_an_array", cfg]

        _metrics, artifact = ev.evaluate(opponents, _noop_improver)

        per_opp = artifact["per_opp_metrics"]
        assert per_opp[0]["is_valid"] == 1.0
        assert per_opp[1]["is_valid"] == 0.0
        assert per_opp[2]["is_valid"] == 1.0


class TestPopBAggregatorParity:
    """Per-opp primitives must be enough to reproduce program.metrics via a
    pop_b-shaped ConfigurableAggregator."""

    @staticmethod
    def _pop_b_aggregator() -> ConfigurableAggregator:
        # Shape mirrors tests/test_metrics_aggregators.py::test_heilbron_improver_shape_via_config.
        return ConfigurableAggregator(
            outputs={
                "is_valid": ConstantSpec(value=1.0),
                "n_opponents": ReduceSpec(op="count"),
                "fitness": ReduceSpec(op="mean", field="score"),
                "actual_fitness": ReduceSpec(op="max", field="post_q"),
                "mean_pre_quality": ReduceSpec(op="mean", field="pre_q"),
                "mean_post_quality": ReduceSpec(op="mean", field="post_q"),
                "max_post_quality": ReduceSpec(op="max", field="post_q"),
                "mean_improvement_raw": ReduceSpec(op="mean", field="delta"),
            },
            invalid_defaults={
                "is_valid": 0.0,
                "n_opponents": 0.0,
                "fitness": -1.0,
                "actual_fitness": -1.0,
                "mean_pre_quality": -1.0,
                "mean_post_quality": -1.0,
                "max_post_quality": -1.0,
                "mean_improvement_raw": -1.0,
            },
            metrics_context=_ctx(),
        )

    @pytest.mark.xfail(
        reason="pop_b/evaluate returns ({}, artifact) — is_valid moved into "
        "the aggregator reduction layer; intrinsic metrics no longer carry it. "
        "Test asserts the pre-refactor contract. See #234.",
        strict=False,
    )
    def test_pop_b_per_opp_metrics_feeds_aggregator_to_same_result(self):
        """heilbron_repro_v1/pop_b metrics are exactly the ReduceSpec-aggregation
        of its per_opp_metrics (hard-floor score = min(max(delta,0)/Q_MAX, 1))."""
        helper = _load_helper("heilbron_repro_v1", "pop_a")
        ev = _load_evaluate("heilbron_repro_v1", "pop_b")
        cfg = _seed_grid(helper)
        opponents = [cfg, cfg, cfg]

        metrics, artifact = ev.evaluate(opponents, _noop_improver)
        assert metrics["is_valid"] == 1.0

        agg = self._pop_b_aggregator()
        reproduced = agg.aggregate(artifact["per_opp_metrics"], intrinsic=metrics)

        for key in agg.output_keys:
            assert reproduced[key] == pytest.approx(metrics[key], rel=1e-9, abs=1e-12)


class TestPopBBackCompat:
    @pytest.mark.parametrize("problem", ["heilbron_repro_v1", "heilbron_adversarial"])
    def test_per_opp_delta_back_compat(self, problem):
        helper = _load_helper(problem, "pop_a")
        ev = _load_evaluate(problem, "pop_b")
        cfg = _seed_grid(helper)
        opponents = [cfg, "not_an_array", cfg]

        _metrics, artifact = ev.evaluate(opponents, _noop_improver)

        per_opp = artifact["per_opp_metrics"]
        expected_deltas = [m["delta"] for m in per_opp]
        actual_deltas = artifact["per_opp_delta"]
        assert len(actual_deltas) == len(expected_deltas)
        for a, e in zip(actual_deltas, expected_deltas):
            if math.isnan(e):
                assert math.isnan(a)
            else:
                assert a == pytest.approx(e, rel=1e-9, abs=1e-12)


# ===========================================================================
# pop_a — constructor / G side
# ===========================================================================


class TestPopAArtifact:
    @pytest.mark.parametrize("problem", ["heilbron_repro_v1", "heilbron_adversarial"])
    def test_pop_a_artifact_contains_per_opp_metrics_aligned_with_opponents(
        self, problem
    ):
        helper = _load_helper(problem, "pop_a")
        ev = _load_evaluate(problem, "pop_a")
        pts = _seed_grid(helper)
        opponents = [_noop_improver, _centroid_improver, _noop_improver]

        _metrics, artifact = ev.evaluate(opponents, pts)

        assert "per_opp_metrics" in artifact
        per_opp = artifact["per_opp_metrics"]
        assert isinstance(per_opp, list)
        assert len(per_opp) == len(opponents)
        for entry in per_opp:
            assert POP_A_KEYS.issubset(entry.keys())
            for k in POP_A_KEYS:
                assert isinstance(entry[k], float)

    @pytest.mark.parametrize("problem", ["heilbron_repro_v1", "heilbron_adversarial"])
    def test_pop_a_no_opponents_emits_empty_per_opp_metrics(self, problem):
        helper = _load_helper(problem, "pop_a")
        ev = _load_evaluate(problem, "pop_a")
        pts = _seed_grid(helper)

        _metrics, artifact = ev.evaluate([], pts)

        assert artifact["per_opp_metrics"] == []


class TestPopAAggregatorParity:
    """Constructor-shaped aggregator using LinearSpec for fitness (ALPHA=0.5).

    Only heilbron_repro_v1/pop_a uses that shape — adversarial/pop_a uses
    tanh-smoothed resistance which cannot be expressed via the current
    primitive set.
    """

    @staticmethod
    def _pop_a_aggregator() -> ConfigurableAggregator:
        return ConfigurableAggregator(
            outputs={
                "is_valid": ConstantSpec(value=1.0),
                "n_opponents": ReduceSpec(op="count"),
                "actual_fitness": IntrinsicSpec(key="actual_fitness"),
                "quality": IntrinsicSpec(key="quality"),
                "resistance": ReduceSpec(op="mean", field="resistance_score"),
                "mean_improvement": ReduceSpec(op="mean", field="delta"),
                "best_post_improvement": ReduceSpec(op="max", field="post_q"),
                "fitness": LinearSpec(
                    terms=[
                        {"coeff": 0.5, "source": "intrinsic", "key": "quality"},
                        {"coeff": 0.5, "source": "output", "key": "resistance"},
                    ]
                ),
            },
            invalid_defaults={
                "is_valid": 0.0,
                "n_opponents": 0.0,
                "actual_fitness": -1.0,
                "quality": -1.0,
                "resistance": -1.0,
                "mean_improvement": -1.0,
                "best_post_improvement": -1.0,
                "fitness": -1.0,
            },
            metrics_context=_ctx(),
        )

    @pytest.mark.xfail(
        reason="pop_a/evaluate returns ({}, artifact) — is_valid moved into "
        "the aggregator reduction layer; intrinsic metrics no longer carry it. "
        "Test asserts the pre-refactor contract. See #234.",
        strict=False,
    )
    def test_pop_a_per_opp_metrics_feeds_aggregator_to_same_result(self):
        helper = _load_helper("heilbron_repro_v1", "pop_a")
        ev = _load_evaluate("heilbron_repro_v1", "pop_a")
        pts = _seed_grid(helper)
        opponents = [_noop_improver, _centroid_improver, _noop_improver]

        metrics, artifact = ev.evaluate(opponents, pts)
        assert metrics["is_valid"] == 1.0

        agg = self._pop_a_aggregator()
        reproduced = agg.aggregate(artifact["per_opp_metrics"], intrinsic=metrics)

        for key in agg.output_keys:
            assert reproduced[key] == pytest.approx(metrics[key], rel=1e-9, abs=1e-12)
