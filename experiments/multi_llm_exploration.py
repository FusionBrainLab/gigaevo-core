"""Multi-LLM exploration experiment matching
``config/experiment/multi_llm_exploration.yaml``.

Single island with the heterogeneous OpenRouter bandit — bias toward
mutation diversity over MAP-Elites diversity. The bandit learns
which model produces the best mutations and concentrates the
sampling budget there over time.
"""

from __future__ import annotations

from gigaevo.config.algorithm_presets import build_single_island
from gigaevo.config.engine_presets import build_generational
from gigaevo.config.llm_presets import build_openrouter_bandit
from gigaevo.config.pipeline_presets import build_auto
from gigaevo.config.problem_presets import build_heilbron
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import (
    DataPlaneSettings,
    ExperimentConfig,
    RedisConfig,
)


_NAME = "heilbron_multi_llm_exploration"


def build() -> ExperimentConfig:
    redis = RedisConfig()
    return ExperimentConfig(
        name=_NAME,
        seed=42,
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{_NAME}"),
        problem=build_heilbron(name=_NAME),
        algorithm=build_single_island(),
        engine=build_generational(),
        pipeline=build_auto(),
        llm=build_openrouter_bandit(),
        runner=build_default_runner(),
    )
