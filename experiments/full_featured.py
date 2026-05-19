"""Multi-island MAP-Elites + LLM bandit on the Heilbron triangle problem.

Combines the fitness x validity / fitness x complexity two-island
split with the 4-way OpenRouter bandit router, exercising both the
behavior-space and mutation-source diversity dimensions in one run.
Requires the ``complexity_score`` metric on the problem's evaluator.
"""

from __future__ import annotations

from gigaevo.config.algorithm_presets import build_multi_island_fitness_complexity
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


_NAME = "heilbron_full_featured"


def build() -> ExperimentConfig:
    redis = RedisConfig()
    return ExperimentConfig(
        name=_NAME,
        seed=43,
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{_NAME}"),
        problem=build_heilbron(name=_NAME),
        algorithm=build_multi_island_fitness_complexity(),
        engine=build_generational(),
        pipeline=build_auto(),
        llm=build_openrouter_bandit(),
        runner=build_default_runner(),
    )
