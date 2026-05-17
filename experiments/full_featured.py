"""Full-featured experiment matching ``config/experiment/full_featured.yaml``.

Multi-island MAP-Elites with the fitness × validity / fitness ×
complexity split, plus the four-way heterogeneous OpenRouter
ensemble. The shipped YAML requires the ``complexity_score`` metric;
the algorithm preset constructs the multi-island shape that consumes
it on the simplicity island.
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
        seed=42,
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{_NAME}"),
        problem=build_heilbron(name=_NAME),
        algorithm=build_multi_island_fitness_complexity(),
        engine=build_generational(),
        pipeline=build_auto(),
        llm=build_openrouter_bandit(),
        runner=build_default_runner(),
    )
