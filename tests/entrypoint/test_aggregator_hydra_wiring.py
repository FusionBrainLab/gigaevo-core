"""Hydra composition test — aggregator group default resolves to NullAggregator.

Pins the Hydra wiring contract:

  * ``config/config.yaml`` declares ``aggregator: none`` in its defaults list,
    so every run starts with a :class:`NullAggregator` sentinel.
  * Pipelines that don't opt in to an alternative aggregator inherit this null
    default, preserving the legacy CallValidatorFunction → FetchMetrics DAG via
    the ``isinstance(aggregator, NullAggregator)`` gate in builders.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from gigaevo.config.resolvers import register_resolvers

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")

register_resolvers()


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_default_aggregator_is_null():
    """Top-level default resolves to NullAggregator without any explicit override."""
    cfg = _compose(
        [
            "pipeline=adversarial_asymmetric",
            "problem.name=_test_",
            "redis.db=0",
            "opponent_redis_db=1",
            "opponent_redis_prefix=_test_opponent_",
            "population_role=improver",
            "feedback_mode=composition",
        ]
    )
    assert cfg.aggregator._target_.endswith("NullAggregator")
