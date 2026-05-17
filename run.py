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
    wire_bandit_router,
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

    # One pool per experiment. Bound as the ambient pool so every
    # ``run_exec_runner(pool=None)`` call site inside this asyncio.run
    # amortizes subprocess startup across the run instead of spinning a
    # fresh ``WorkerPool`` per call.
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

        # Construct the coordinator after Hydra has built the storage so
        # the redis_url / key_prefix are sourced from a single config
        # surface. ``build_dataplane`` starts the connection pool, loads
        # the seven Lua scripts, and primes the legacy ProgramState FSM
        # table; failures here surface as :class:`StartupError` before
        # the engine task starts. ``wire_*`` attach the coordinator to
        # the already-instantiated storage and bandit so production
        # routes through the dataplane without altering the Hydra schema.
        dataplane = await build_dataplane(
            str(redis_storage.config.redis_url),
            key_prefix=redis_storage.config.key_prefix,
        )
        # One engine, one root-token bundle. The engine root flows into
        # the storage so per-call FSM tokens derive by linear split from
        # this single origin — every per-program write is structurally
        # a child of the engine's single ProgramId subspace witness.
        engine_root = build_engine_root()
        wire_storage(redis_storage, dataplane, engine_root)
        # Hand the runner and engine the dataplane handle so their
        # freshness-pinned reads (currently: the timeout-discard
        # re-read in :class:`DagRunner._timeout_read_is_fresh`) route
        # through :meth:`DataPlane.read_program`. The engine_root is
        # forwarded for future writes; storage already owns the
        # canonical reference and rotates the program-subspace witness
        # on every transition.
        dag_runner._dataplane = dataplane
        dag_runner._engine_root = engine_root
        evolution_engine._dataplane = dataplane
        evolution_engine._engine_root = engine_root
        actor = build_actor_identity(run_id=cfg.get("run_id"))
        llm_wrapper = getattr(evolution_engine.mutation_operator, "llm_wrapper", None)
        if llm_wrapper is not None:
            wire_bandit_router(llm_wrapper, dataplane, actor)

        # Share the engine's main DataPlane with the prompt fetcher (so
        # prompt-outcome G-counters live under the same connection pool
        # the storage and bandit already own) while constructing a
        # separate :class:`DataPlane` for the co-evolved prompt archive
        # reads: that archive is in a different Redis DB, so it needs
        # its own connection pool dialled at a different URL. The
        # fetcher carries the prompt-archive DB / host / port / prefix
        # from its Hydra config; we read them directly off the instance
        # so the engine surface stays unaware of the fetcher's schema.
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
            # ``serve_until_signal`` already calls the two ``stop`` coros on
            # the happy path. If anything between ``start()`` and the
            # ``await serve_until_signal`` raises, the tasks are still alive
            # and would leak past the ``except`` block. ``stop()`` on both
            # components is idempotent: cancelling an already-cancelled task
            # and closing an already-closed storage / connection are no-ops,
            # so calling them twice on the happy path is safe.
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
        # Drain pool workers before unbinding the contextvar so any
        # late ``run_exec_runner`` calls during shutdown still amortize
        # through the shared pool.
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
        # Shutdown the coordinator after the storage so any tail
        # writes the storage performs during ``close()`` still go
        # through a live connection pool. The dataplane's own
        # ``shutdown`` is idempotent and safe to call after
        # ``startup`` failed mid-sequence.
        if dataplane is not None:
            try:
                await dataplane.shutdown()
            except Exception:
                logger.exception("DataPlane shutdown failed")
        # The prompt-archive coordinator is independent (separate DB
        # and key prefix) and therefore needs its own shutdown step;
        # ordering relative to ``dataplane`` does not matter because
        # the two pools share no resources.
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
