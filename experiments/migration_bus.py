"""Migration bus experiment matching ``config/experiment/migration_bus.yaml``.

Single-island generational engine on each run, with the
fully-connected cross-run migration bus letting peers share
rejected-but-valid programs through a Redis stream on DB 15.

The shipped YAML uses ``${problem.name}@db${redis.db}`` as the run
identifier; the preset hardcodes ``heilbron@db0`` as the canonical
default. Override via tyro to run a second peer:

    python run.py experiments/migration_bus.py \\
        --engine.migration_bus.run_id heilbron@db1 \\
        --redis.db 1
"""

from __future__ import annotations

from gigaevo.config.algorithm_presets import build_single_island
from gigaevo.config.engine_presets import build_bus_engine
from gigaevo.config.pipeline_presets import build_auto
from gigaevo.config.problem_presets import build_heilbron
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import (
    BanditRouterConfig,
    ChatOpenAIConfig,
    DataPlaneSettings,
    ExperimentConfig,
    RedisConfig,
)


_NAME = "heilbron_migration_bus"
_PROBLEM_SHORT = "heilbron"
_DEFAULT_RUN_ID = f"{_PROBLEM_SHORT}@db0"


def build() -> ExperimentConfig:
    redis = RedisConfig()
    # The bus engine + bandit LLM cross-field invariant on
    # ExperimentConfig requires the LLM to be a BanditRouterConfig.
    # Wrap the single endpoint as a one-arm bandit so the validator
    # passes; the bandit's bookkeeping over a single arm is a no-op
    # but the type contract is satisfied.
    llm = BanditRouterConfig(
        models=[
            ChatOpenAIConfig(
                model="google/gemini-3-flash-preview",
                base_url="https://openrouter.ai/api/v1",
            )
        ],
    )
    return ExperimentConfig(
        name=_NAME,
        seed=42,
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{_NAME}"),
        problem=build_heilbron(name=_NAME),
        algorithm=build_single_island(),
        engine=build_bus_engine(
            run_id=_DEFAULT_RUN_ID,
            problem_name=_PROBLEM_SHORT,
        ),
        pipeline=build_auto(),
        llm=llm,
        runner=build_default_runner(),
    )
