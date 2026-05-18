"""Bus migration experiment.

Composes the cross-run migration bus with a generational engine
backbone. The bus engine schema variant does not carry a
``max_in_flight`` knob — steady-state in-flight queueing is owned
by ``SteadyStateEngineConfig`` and is not composable with the bus
variant on the schema as it stands.
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
    # build_bus_engine returns the generational-flavored bus config.
    # The bus engine variant on the schema does not carry a
    # steady-state max_in_flight knob; the migration bus is composed
    # here against the generational engine backbone. A dedicated
    # steady-state-bus preset would let this experiment import a
    # one-liner instead.
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
        # BusedEngineConfig has no max_in_flight knob — steady-state
        # in-flight queueing lives on SteadyStateEngineConfig, which
        # is not composable with the migration bus on the schema as
        # it stands. The generational backbone is selected here.
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
