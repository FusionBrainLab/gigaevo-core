import asyncio
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gigaevo.config.resolvers import register_resolvers
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.dataplane import (
    DataPlane,
    build_actor_identity,
    build_dataplane,
    build_engine_root,
    wire_archive_storage,
    wire_bandit_router,
    wire_dag_runner,
    wire_evolution_engine,
    wire_prompt_fetcher,
    wire_storage,
)
from gigaevo.evolution.engine import EvolutionEngine
from gigaevo.problems.initial_loaders import InitialProgramLoader
from gigaevo.programs.stages.python_executors.wrapper import (
    WorkerPool,
    default_exec_runner_pool,
    reset_ambient_exec_runner_pool,
    set_ambient_exec_runner_pool,
)
from gigaevo.runner.dag_runner import DagRunner
from gigaevo.utils.logger_setup import setup_logger
from gigaevo.utils.serve import serve_until_signal
from gigaevo.utils.trackers.base import LogWriter


async def run_experiment(cfg: DictConfig) -> None:
    start_time = time.time()
    logger.info("GigaEvo — Problem: {}", cfg.problem.name)

    redis_storage: RedisProgramStorage | None = None
    writer: LogWriter | None = None
    dataplane: DataPlane | None = None
    prompt_dataplane: DataPlane | None = None
    dag_runner: DagRunner | None = None
    evolution_engine: EvolutionEngine | None = None
    program_loader: InitialProgramLoader | None = None
    config_with_instances: Any | None = None

    # Ambient pool: ``run_exec_runner(pool=None)`` resolves to this pool
    # for the lifetime of the run, amortizing subprocess startup.
    exec_runner_pool: WorkerPool = default_exec_runner_pool()
    pool_token = set_ambient_exec_runner_pool(exec_runner_pool)
    try:
        try:
            config_with_instances = instantiate(cfg, recursive=True)
        except Exception:
            logger.exception("Hydra instantiation failed")
            raise

        redis_storage = config_with_instances.redis_storage
        program_loader = config_with_instances.program_loader
        dag_runner = config_with_instances.dag_runner
        evolution_engine = config_with_instances.evolution_engine
        writer = config_with_instances.writer

        logger.info(
            "Redis DB {db} at {host}:{port} | pipeline={pipeline}",
            db=cfg.redis.db,
            host=cfg.redis.host,
            port=cfg.redis.port,
            pipeline=cfg.get("pipeline_builder", {}).get("_target_", "(default)"),
        )

        # Build the coordinator from the already-instantiated storage's
        # connection info; ``build_dataplane`` opens the connection pool,
        # loads Lua scripts, and primes the FSM table.
        dataplane = await build_dataplane(
            str(redis_storage.config.redis_url),
            key_prefix=redis_storage.config.key_prefix,
        )
        # Single engine root: per-call FSM tokens derive by linear split
        # from this origin, so every per-program write is a child of the
        # engine's ProgramId subspace witness.
        engine_root = build_engine_root()
        wire_storage(redis_storage, dataplane, engine_root)
        wire_dag_runner(dag_runner, dataplane, engine_root)
        wire_evolution_engine(evolution_engine, dataplane, engine_root)
        # Wire archive cells for any strategy that exposes ``.islands``;
        # strategies without islands skip the loop cleanly.
        strategy = getattr(evolution_engine, "strategy", None)
        islands = getattr(strategy, "islands", None) if strategy is not None else None
        if islands is not None:
            for island in islands.values():
                archive = getattr(island, "archive_storage", None)
                if archive is not None:
                    wire_archive_storage(archive, dataplane, engine_root)
        actor = build_actor_identity(run_id=cfg.get("run_id"))
        llm_wrapper = getattr(evolution_engine.mutation_operator, "llm_wrapper", None)
        if llm_wrapper is not None:
            wire_bandit_router(llm_wrapper, dataplane, actor, engine_root)

        # Prompt-outcome counters share the engine's DataPlane; the
        # co-evolved prompt archive lives in a different Redis DB so it
        # gets its own DataPlane dialled at the fetcher's URL.
        prompt_fetcher = getattr(
            evolution_engine.mutation_operator, "_prompt_fetcher", None
        )
        from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

        if isinstance(prompt_fetcher, GigaEvoArchivePromptFetcher):
            prompt_url = (
                f"redis://{prompt_fetcher._host}:{prompt_fetcher._port}/"
                f"{prompt_fetcher._prompt_redis_db}"
            )
            prompt_dataplane = await build_dataplane(
                prompt_url,
                key_prefix=prompt_fetcher._prompt_prefix,
            )
            wire_prompt_fetcher(prompt_fetcher, dataplane, prompt_dataplane, actor)

        await redis_storage.acquire_instance_lock()

        has_data = await redis_storage.has_data()
        resume = cfg.redis.get("resume", False)

        if has_data and not resume:
            raise RuntimeError(
                f"Redis database {cfg.redis.db} is not empty. "
                f"Flush with: redis-cli -h {cfg.redis.host} -p {cfg.redis.port} "
                f"-n {cfg.redis.db} FLUSHDB  — or set redis.resume=true"
            )

        if has_data and resume:
            recovered = await redis_storage.recover_stranded_programs()
            if recovered:
                logger.info("Recovered {} stranded RUNNING program(s)", recovered)
            await evolution_engine.restore_state()
            await evolution_engine.strategy.restore_state()
            logger.info(
                "Resumed with {} existing programs",
                await redis_storage.size(),
            )
        else:
            programs = await program_loader.load(redis_storage)
            logger.info("Loaded {} initial programs", len(programs))

        try:
            dag_runner.start()
            evolution_engine.start()
            logger.info(
                "Evolution running (max_gen={})", cfg.max_generations or "unlimited"
            )

            await serve_until_signal(
                stop_coros=(evolution_engine.stop(), dag_runner.stop()),
                on_stop=(evolution_engine.task, dag_runner.task),
            )
        finally:
            # Idempotent stops: covers the path where something between
            # ``start()`` and ``serve_until_signal`` raises and leaves
            # the tasks alive. ``stop()`` on an already-stopped component
            # is a no-op.
            try:
                await evolution_engine.stop()
            except Exception:
                logger.exception("EvolutionEngine.stop failed")
            try:
                await dag_runner.stop()
            except Exception:
                logger.exception("DagRunner.stop failed")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Experiment failed")
        raise
    finally:
        # Drain pool workers before unbinding the contextvar so late
        # ``run_exec_runner`` calls during shutdown still resolve to
        # the shared pool.
        try:
            await exec_runner_pool.shutdown()
        except Exception:
            logger.exception("WorkerPool shutdown failed")
        reset_ambient_exec_runner_pool(pool_token)
        if redis_storage is not None:
            try:
                await redis_storage.close()
            except Exception:
                logger.exception("RedisProgramStorage close failed")
        # Shut the coordinator down after the storage so any tail writes
        # storage performs during ``close()`` still see a live pool.
        if dataplane is not None:
            try:
                await dataplane.shutdown()
            except Exception:
                logger.exception("DataPlane shutdown failed")
        # Prompt-archive coordinator uses its own pool; ordering vs
        # ``dataplane`` is independent.
        if prompt_dataplane is not None:
            try:
                await prompt_dataplane.shutdown()
            except Exception:
                logger.exception("Prompt DataPlane shutdown failed")
        if writer is not None:
            try:
                writer.close()
            except Exception:
                logger.exception("LogWriter close failed")
        duration = time.time() - start_time
        logger.info("Duration: {:.1f}s ({:.2f}h)", duration, duration / 3600)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    log_file_path = setup_logger(
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
    )
    hydra_config = hydra.core.hydra_config.HydraConfig.get().runtime
    logger.info(
        "Output dir: {} | Log: {}", Path(hydra_config.output_dir), log_file_path
    )
    asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    register_resolvers()
    main()
