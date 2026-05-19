"""Steady-state experiment on the Heilbron triangle problem.

Single island combined with the steady-state engine variant for
continuous mutation / evaluation interleaving without a generational
barrier (~8-9x throughput vs the generational engine). LLM is a
single OpenRouter endpoint.
"""

from __future__ import annotations

from gigaevo.config.algorithm_presets import build_single_island
from gigaevo.config.engine_presets import build_steady_state
from gigaevo.config.llm_presets import build_single
from gigaevo.config.pipeline_presets import build_auto
from gigaevo.config.problem_presets import build_heilbron
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import (
    DataPlaneSettings,
    ExperimentConfig,
    RedisConfig,
)


_NAME = "heilbron_steady_state"


def build() -> ExperimentConfig:
    redis = RedisConfig()
    return ExperimentConfig(
        name=_NAME,
        seed=48,
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{_NAME}"),
        problem=build_heilbron(name=_NAME),
        algorithm=build_single_island(),
        engine=build_steady_state(),
        pipeline=build_auto(),
        llm=build_single(
            "google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
        ),
        runner=build_default_runner(),
    )
