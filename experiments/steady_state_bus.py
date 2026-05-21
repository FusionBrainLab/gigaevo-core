"""Bus migration experiment.

Composes the cross-run migration bus with a generational engine
backbone. The bus engine schema variant does not carry a
``max_in_flight`` knob; steady-state in-flight queueing lives on
``SteadyStateEngineConfig`` and is not composable with the bus
variant on the schema.
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


_NAME = "heilbron_steady_state_bus"
_PROBLEM_SHORT = "heilbron"
_DEFAULT_RUN_ID = f"{_PROBLEM_SHORT}@db0"


def build() -> ExperimentConfig:
    redis = RedisConfig()
    # Bus engine + bandit-LLM cross-field invariant on
    # ExperimentConfig requires a BanditRouterConfig. A one-arm
    # bandit satisfies the type contract with zero bookkeeping
    # overhead over the single endpoint.
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
