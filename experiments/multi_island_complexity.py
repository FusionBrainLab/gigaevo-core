"""Multi-island complexity experiment matching
``config/experiment/multi_island_complexity.yaml``.

Two islands (fitness + simplicity) with a single LLM endpoint —
narrower exploration than full_featured but with the complexity-axis
diversity preserved.
"""

from __future__ import annotations

from gigaevo.config.algorithm_presets import build_multi_island_fitness_complexity
from gigaevo.config.engine_presets import build_generational
from gigaevo.config.llm_presets import build_single
from gigaevo.config.pipeline_presets import build_auto
from gigaevo.config.problem_presets import build_heilbron
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import (
    DataPlaneSettings,
    ExperimentConfig,
    RedisConfig,
)


_NAME = "heilbron_multi_island_complexity"


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
        llm=build_single(
            "google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
        ),
        runner=build_default_runner(),
    )
