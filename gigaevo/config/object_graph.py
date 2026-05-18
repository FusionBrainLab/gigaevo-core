"""Typed-config to runtime-object adapter.

:func:`build_object_graph` takes a validated :class:`ExperimentConfig`
and constructs the runtime object tree: the Redis program storage, the
problem context, the LLM router with bandit parameters resolved from
the metrics context, the evolution strategy with the storage threaded
through, the runtime engine config with required behavior keys
resolved, the evolution context composing those, the pipeline builder,
the DAG blueprint, and the runtime DAG-runner config.

The function is intentionally explicit: every construction is a direct
class call with kwargs derived from typed schema fields. The mutation
operator, DAG runner instance, program loader, writer, metrics tracker
and evolution-engine compositor are left to the caller —
:func:`run_with_config` surfaces those gaps via a log warning before
returning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.config.schemas.algorithm import (
    MultiIslandConfig,
    SingleIslandConfig,
)
from gigaevo.config.schemas.experiment import ExperimentConfig
from gigaevo.config.schemas.llm import BanditRouterConfig

if TYPE_CHECKING:
    from gigaevo.problems.context import ProblemContext


def _resolve_bandit_keys(
    cfg: ExperimentConfig,
    problem_ctx: "ProblemContext | None" = None,
) -> tuple[str, bool]:
    """The bandit router needs a fitness key + direction; we resolve
    them from the problem's MetricsContext, falling back to the
    problem-side overrides on :class:`ProblemConfig` when set.

    ``problem_ctx`` may be passed by callers that already constructed
    one so the underlying ``metrics.yaml`` is read at most once per
    graph build.
    """
    if problem_ctx is None:
        problem_ctx = cfg.problem.build()
    metrics_ctx = problem_ctx.metrics_context
    primary_key = cfg.problem.primary_metric or metrics_ctx.get_primary_key()
    if cfg.problem.higher_is_better is not None:
        higher = cfg.problem.higher_is_better
    else:
        higher = metrics_ctx.is_higher_better(primary_key)
    return primary_key, higher


def _required_behavior_keys(cfg: ExperimentConfig) -> list[str]:
    """Collect the union of behavior keys across every island in the
    algorithm subtree. ``StandardEvolutionAcceptor`` requires every
    program have all of these present in its metrics before accepting."""
    if isinstance(cfg.algorithm, SingleIslandConfig):
        return list(cfg.algorithm.island.behavior_space.behavior_keys)
    if isinstance(cfg.algorithm, MultiIslandConfig):
        keys: set[str] = set()
        for island in cfg.algorithm.islands:
            keys.update(island.behavior_space.behavior_keys)
        return sorted(keys)
    raise NotImplementedError(
        f"behavior key extraction not implemented for "
        f"{type(cfg.algorithm).__name__}"
    )


def build_object_graph(cfg: ExperimentConfig) -> dict[str, Any]:
    """Construct the runtime object tree from a validated config.

    Returns a dict with the constructed objects. The keys form the
    canonical surface that consumers (CLI, parity harness, sweep
    utility) read against.

    Top-level keys:
        - ``redis_storage``: RedisProgramStorage
        - ``problem_context``: ProblemContext (lazy MetricsContext)
        - ``llm``: MultiModelRouter or BanditModelRouter
        - ``strategy``: MapElitesMultiIsland
        - ``runtime_engine_config``: EngineConfig / SteadyStateEngineConfig
        - ``evolution_context``: EvolutionContext
        - ``pipeline_builder``: PipelineBuilder subclass
        - ``dag_blueprint``: DAGBlueprint
        - ``runtime_runner_config``: DagRunnerConfig
        - ``primary_metric``: str (resolved from problem)
        - ``higher_is_better``: bool (resolved from problem)
        - ``required_behavior_keys``: list[str]

    The mutation operator, DAG-runner instance, writer, metrics
    tracker, program loader and the evolution-engine compositor are
    out of scope — :func:`run_with_config` surfaces the gap.
    """
    from gigaevo.database.redis_program_storage import RedisProgramStorage
    from gigaevo.entrypoint.evolution_context import EvolutionContext
    from gigaevo.memory.provider import NullMemoryProvider

    redis_storage_config = cfg.redis.to_storage_config(
        key_prefix=cfg.dataplane.key_prefix,
        max_connections=cfg.dataplane.max_connections,
    )
    redis_storage = RedisProgramStorage(redis_storage_config)

    problem_context = cfg.problem.build()
    primary_metric, higher_is_better = _resolve_bandit_keys(
        cfg, problem_ctx=problem_context
    )
    behavior_keys = _required_behavior_keys(cfg)

    if isinstance(cfg.llm, BanditRouterConfig):
        llm = cfg.llm.build(
            fitness_key=primary_metric,
            higher_is_better=higher_is_better,
        )
    else:
        llm = cfg.llm.build()

    strategy = cfg.algorithm.build(program_storage=redis_storage)
    runtime_engine_config = cfg.engine.build_runtime_config(
        required_behavior_keys=behavior_keys
    )

    prompt_fetcher_runtime = (
        cfg.prompt_fetcher.build() if cfg.prompt_fetcher is not None else None
    )
    evolution_context = EvolutionContext(
        problem_ctx=problem_context,
        llm_wrapper=llm,
        storage=redis_storage,
        prompts_dir=cfg.pipeline.prompts_dir,
        prompt_fetcher=prompt_fetcher_runtime,
        memory_provider=NullMemoryProvider(),
    )

    pipeline_builder = cfg.pipeline.builder.build(ctx=evolution_context)
    dag_blueprint = pipeline_builder.build_blueprint()

    runtime_runner_config = cfg.runner.build()

    return {
        "redis_storage": redis_storage,
        "problem_context": problem_context,
        "llm": llm,
        "strategy": strategy,
        "runtime_engine_config": runtime_engine_config,
        "evolution_context": evolution_context,
        "pipeline_builder": pipeline_builder,
        "dag_blueprint": dag_blueprint,
        "runtime_runner_config": runtime_runner_config,
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "required_behavior_keys": behavior_keys,
    }


async def run_with_config(cfg: ExperimentConfig) -> int:
    """End-to-end runner the CLI invokes when not in dry-run mode.

    Builds the object graph and surfaces the gaps that still block
    engine invocation. Returns zero on successful graph construction;
    the missing wiring is reported via a log warning.
    """
    graph = build_object_graph(cfg)
    logger.info(
        "Object graph constructed (experiment_id={}): {} top-level objects",
        cfg.experiment_id,
        len(graph),
    )
    logger.warning(
        "Engine invocation pending: mutation_operator, dag_runner, "
        "writer, metrics_tracker, program_loader, and evolution_engine "
        "are not yet constructed by build_object_graph. Use --dry-run "
        "for configuration validation only."
    )
    return 0
