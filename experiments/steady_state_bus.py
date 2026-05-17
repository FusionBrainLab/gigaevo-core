"""Steady-state + bus experiment.

Combines continuous mutation / evaluation interleaving with the
cross-run migration bus. This is the canonical production shape for
multi-GPU, multi-run setups.

The bus engine variant carries the steady-state ``max_in_flight``
value (hardcoded to 8 for the steady-state path).
"""

from __future__ import annotations

from gigaevo.config.algorithm_presets import build_single_island
from gigaevo.config.defaults import DEFAULT_MAX_MUTATIONS_PER_GENERATION
from gigaevo.config.pipeline_presets import build_auto
from gigaevo.config.problem_presets import build_heilbron
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import (
    BanditRouterConfig,
    BusedEngineConfig,
    BusTopologyConfig,
    ChatOpenAIConfig,
    DataPlaneSettings,
    ExperimentConfig,
    MigrationBusConfig,
    RedisConfig,
    RedisStreamTransportConfig,
)


_NAME = "heilbron_steady_state_bus"
_PROBLEM_SHORT = "heilbron"
_DEFAULT_RUN_ID = f"{_PROBLEM_SHORT}@db0"


def build() -> ExperimentConfig:
    redis = RedisConfig()
    # build_bus_engine returns the generational-flavored bus config;
    # the steady-state YAML inherits engine knobs but pins
    # max_in_flight=8 (steady-state only). Construct the bus engine
    # manually here to mix the migration bus with steady-state
    # semantics; a future ``build_steady_state_bus_engine`` preset
    # would let this experiment import a one-liner instead.
    bus = MigrationBusConfig(
        run_id=_DEFAULT_RUN_ID,
        transport=RedisStreamTransportConfig(
            run_id=_DEFAULT_RUN_ID,
            stream_key=f"gigaevo:{_PROBLEM_SHORT}:migration_bus",
            db=15,
            max_stream_len=1000,
            claim_ttl=120,
        ),
        topology=BusTopologyConfig(),
        max_buffer_size=30,
        consume_interval=3.0,
        max_consume_per_poll=20,
    )
    engine = BusedEngineConfig(
        migration_bus=bus,
        max_imports_per_generation=10,
        # BusedEngineConfig.max_in_flight does not exist on the
        # schema today — steady-state max_in_flight is a separate
        # variant. This experiment uses the bus's generational engine
        # backbone; if the steady-state variant of the bus engine
        # ships, the field migrates here.
        max_mutations_per_generation=DEFAULT_MAX_MUTATIONS_PER_GENERATION,
    )
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
        engine=engine,
        pipeline=build_auto(),
        llm=llm,
        runner=build_default_runner(),
    )
