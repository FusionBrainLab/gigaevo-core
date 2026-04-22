"""Hydra composition test — aggregator group with `none` default + per-role overrides.

Pins the Hydra wiring contract established in Task 4 of the aggregator-first
metrics plan:

  * `config/config.yaml` declares `aggregator: none` in its defaults list, so
    every run starts with a :class:`NullAggregator` sentinel.
  * `config/pipeline/heilbron_repro_v1.yaml` wires both
    ``pipeline_builder.aggregator`` and
    ``pipeline_builder.lineage_filter.aggregator`` to ``${ref:aggregator}`` —
    the SAME top-level singleton — so launch scripts flip both slots with a
    single ``aggregator=heilbron_improver`` override, no ``+`` prefix.
  * Non-Heilbron pipelines (e.g. adversarial_asymmetric) inherit the null
    default, preserving the legacy CallValidatorFunction → FetchMetrics DAG
    via the ``isinstance(aggregator, NullAggregator)`` gate in
    AsymmetricPipelineBuilder.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gigaevo.config.resolvers import register_resolvers

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")

register_resolvers()


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_default_aggregator_is_null():
    """Top-level default resolves to NullAggregator — no Heilbron pipeline needed."""
    cfg = _compose(
        [
            "pipeline=adversarial_asymmetric",
            "problem.name=heilbron_adversarial/pop_b",
            "redis.db=0",
            "opponent_redis_db=1",
            "opponent_redis_prefix=heilbron_adversarial/pop_a",
            "population_role=improver",
            "feedback_mode=composition",
        ]
    )
    assert cfg.aggregator._target_.endswith("NullAggregator")


def test_heilbron_repro_v1_d_uses_regular_override_not_plus_prefix():
    """aggregator= (no + prefix) — because `aggregator: none` is in defaults."""
    cfg = _compose(
        [
            "pipeline=heilbron_repro_v1",
            "aggregator=heilbron_improver",  # NO '+' prefix
            "problem.name=heilbron_repro_v1/pop_b",
            "redis.db=0",
            "opponent_redis_db=1",
            "opponent_redis_prefix=heilbron_repro_v1/pop_a",
            "population_role=improver",
            "feedback_mode=composition",
        ]
    )
    assert cfg.aggregator._target_.endswith("ConfigurableAggregator")
    assert "mean_improvement_raw" in cfg.aggregator.outputs
    # Both pipeline_builder.aggregator and lineage_filter.aggregator are set
    # to ${ref:aggregator} (string interpolation, not instantiation at compose time).
    # Use OmegaConf.to_container(resolve=False) to access raw interpolation strings.
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)
    assert raw_cfg["pipeline_builder"]["aggregator"] == "${ref:aggregator}"
    assert raw_cfg["pipeline_builder"]["lineage_filter"]["aggregator"] == "${ref:aggregator}"


def test_heilbron_repro_v1_g_uses_constructor_aggregator():
    cfg = _compose(
        [
            "pipeline=heilbron_repro_v1",
            "aggregator=heilbron_constructor",
            "problem.name=heilbron_repro_v1/pop_a",
            "redis.db=1",
            "opponent_redis_db=0",
            "opponent_redis_prefix=heilbron_repro_v1/pop_b",
            "population_role=constructor",
            "feedback_mode=composition",
        ]
    )
    assert "resistance" in cfg.aggregator.outputs
    # Both slots reference the same top-level singleton via ${ref:aggregator}.
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)
    assert raw_cfg["pipeline_builder"]["aggregator"] == "${ref:aggregator}"


def test_heilbron_adversarial_inherits_null_default():
    """heilbron_adversarial doesn't set aggregator — stays NullAggregator → legacy DAG."""
    cfg = _compose(
        [
            "pipeline=adversarial_asymmetric",
            "problem.name=heilbron_adversarial/pop_b",
            "redis.db=0",
            "opponent_redis_db=1",
            "opponent_redis_prefix=heilbron_adversarial/pop_a",
            "population_role=improver",
            "feedback_mode=composition",
        ]
    )
    assert cfg.aggregator._target_.endswith("NullAggregator")


def test_heilbron_repro_v1_without_aggregator_override_is_null():
    """heilbron_repro_v1 without `aggregator=...` override gets NullAggregator.

    The builder's isinstance(NullAggregator) check will skip ParseMetricsStage —
    a valid (if silent) configuration. Launch scripts MUST set aggregator=... to
    opt in. (Preflight contract checks in experiment.yaml catch missing overrides.)
    """
    cfg = _compose(
        [
            "pipeline=heilbron_repro_v1",
            "problem.name=heilbron_repro_v1/pop_b",
            "redis.db=0",
            "opponent_redis_db=1",
            "opponent_redis_prefix=heilbron_repro_v1/pop_a",
            "population_role=improver",
            "feedback_mode=composition",
        ]
    )
    assert cfg.aggregator._target_.endswith("NullAggregator")
