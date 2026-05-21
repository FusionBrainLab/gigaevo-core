"""Single-island MAP-Elites driven by a 4-way LLM bandit on the Heilbron problem.

Showcases the UCB1 bandit router over a heterogeneous OpenRouter
pool, biasing toward mutation-source diversity instead of
behavior-space diversity. The bandit concentrates the sampling budget
on the model producing the largest fitness improvements over a
sliding window.
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
        seed=46,
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{_NAME}"),
        problem=build_heilbron(name=_NAME),
        algorithm=build_single_island(),
        engine=build_generational(),
        pipeline=build_auto(),
        llm=build_openrouter_bandit(),
        runner=build_default_runner(),
    )
