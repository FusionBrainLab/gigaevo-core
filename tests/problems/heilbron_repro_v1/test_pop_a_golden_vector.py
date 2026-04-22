"""Golden-vector test — pop_a/evaluate.py + heilbron_constructor aggregator.

Pins expected metric values for a fixed input. Intrinsic metrics (quality,
actual_fitness) are candidate-level; per-opponent resistance is reduced by the
aggregator.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir

from gigaevo.programs.metrics.context import MetricSpec, MetricsContext

PROBLEM_DIR = Path(__file__).resolve().parents[3] / "problems/heilbron_repro_v1/pop_a"
CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "config")

EXPECTED = {
    "is_valid": 1.0,
    "quality": 0.5,  # intrinsic (from program output)
    "actual_fitness": 0.0365 * 0.5,  # raw_quality: intrinsic
    "fitness": 0.5,  # ALPHA * quality + (1 - ALPHA) * resistance = 0.5 * 0.5 + 0.5 * 0.5
    "resistance": 0.5,  # mean of [1.0, 0.0]
    "n_opponents": 2.0,
    "mean_improvement": 0.02,  # mean of [0.0, 0.04]
    "best_post_improvement": 0.0365 * 0.5 + 0.04,
}

INTRINSIC = {
    "quality": 0.5,
    "actual_fitness": 0.0365 * 0.5,
}

FIXTURE = {
    "per_opp_metrics": [
        {"post_q": 0.0365 * 0.5, "delta": 0.0, "resistance_score": 1.0, "is_valid": 1.0},
        {"post_q": 0.0365 * 0.5 + 0.04, "delta": 0.04, "resistance_score": 0.0, "is_valid": 1.0},
    ],
    "role": "constructor",
    "n_opponents": 2,
}


def test_pop_a_constructor_golden_vector():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "aggregator=heilbron_constructor",
                "problem.name=heilbron_repro_v1/pop_a",
                "redis.db=1",
            ],
        )
    metrics_ctx = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness", higher_is_better=True, is_primary=True
            ),
            "is_valid": MetricSpec(description="validity", higher_is_better=True),
        }
    )
    agg = hydra.utils.instantiate(cfg.aggregator, metrics_context=metrics_ctx)
    out = agg.aggregate(FIXTURE["per_opp_metrics"], intrinsic=INTRINSIC)
    for k, v in EXPECTED.items():
        assert out[k] == pytest.approx(v), f"{k}: got {out[k]} expected {v}"


def test_pop_a_schema_existence():
    """Every key emitted by the aggregator must be declared in metrics.yaml."""
    import yaml

    metrics_yaml = PROBLEM_DIR / "metrics.yaml"
    declared = set(yaml.safe_load(metrics_yaml.read_text())["specs"].keys())
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "aggregator=heilbron_constructor",
                "problem.name=heilbron_repro_v1/pop_a",
                "redis.db=1",
            ],
        )
    agg_keys = set(cfg.aggregator.outputs.keys())
    missing = agg_keys - declared
    assert not missing, f"aggregator emits keys not declared in metrics.yaml: {missing}"
