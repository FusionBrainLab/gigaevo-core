"""Golden-vector test — pop_b/evaluate.py + heilbron_improver aggregator.

Pins expected metric values for a fixed input. Any drift in heilbron_improver.yaml,
per_opp_metrics shape, or reduction logic will fail here.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
import pytest

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec

PROBLEM_DIR = Path(__file__).resolve().parents[3] / "problems/heilbron_repro_v1/pop_b"
CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "config")

EXPECTED = {
    "is_valid": 1.0,
    "n_opponents": 2.0,
    "fitness": 0.35,  # mean of [0.5, 0.2]
    "actual_fitness": 0.35,  # max of [0.35, 0.31]
    "mean_pre_quality": 0.30,
    "mean_post_quality": 0.33,  # mean of [0.35, 0.31]
    "max_post_quality": 0.35,
    "mean_improvement_raw": 0.03,  # mean of [0.05, 0.01]
}

FIXTURE = {
    "per_opp_metrics": [
        {"pre_q": 0.30, "post_q": 0.35, "delta": 0.05, "score": 0.5, "is_valid": 1.0},
        {"pre_q": 0.30, "post_q": 0.31, "delta": 0.01, "score": 0.2, "is_valid": 1.0},
    ],
    "role": "improver",
    "n_opponents": 2,
}


def test_pop_b_improver_golden_vector():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "aggregator=heilbron_improver",
                "problem.name=heilbron_repro_v1/pop_b",
                "redis.db=0",
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
    out = agg.aggregate(FIXTURE["per_opp_metrics"], intrinsic={})
    for k, v in EXPECTED.items():
        assert out[k] == pytest.approx(v), f"{k}: got {out[k]} expected {v}"


def test_pop_b_schema_existence():
    """Every key emitted by the aggregator must be declared in metrics.yaml.

    Lightweight sanity — fails loudly if we rename or drop an output without
    coordinating with the MetricsContext / metrics.yaml.
    """
    import yaml

    metrics_yaml = PROBLEM_DIR / "metrics.yaml"
    declared = set(yaml.safe_load(metrics_yaml.read_text())["specs"].keys())
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "aggregator=heilbron_improver",
                "problem.name=heilbron_repro_v1/pop_b",
                "redis.db=0",
            ],
        )
    agg_keys = set(cfg.aggregator.outputs.keys())
    missing = agg_keys - declared
    assert not missing, f"aggregator emits keys not declared in metrics.yaml: {missing}"
