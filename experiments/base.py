"""Single-island MAP-Elites on the Heilbron triangle problem.

Canonical baseline composition: single island, OpenRouter ensemble
LLM, auto pipeline, generational engine, default runner. Used as the
reference starting point most other experiments diverge from.

Override at the CLI to retarget another problem:

    python run.py experiments/base.py --problem.problem_dir=/path/to/other
"""

from __future__ import annotations

from gigaevo.config.algorithm_presets import build_single_island
from gigaevo.config.engine_presets import build_generational
from gigaevo.config.llm_presets import build_openrouter_ensemble
from gigaevo.config.pipeline_presets import build_auto
from gigaevo.config.problem_presets import build_heilbron
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import (
    DataPlaneSettings,
    ExperimentConfig,
    RedisConfig,
)


_NAME = "heilbron_base"


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
        llm=build_openrouter_ensemble(),
        runner=build_default_runner(),
    )
