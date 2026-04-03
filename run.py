import asyncio
from pathlib import Path
import time

from tools.no_proxy import ensure_no_proxy

ensure_no_proxy()

from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gigaevo.config.resolvers import register_resolvers
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.engine import EvolutionEngine
from gigaevo.problems.initial_loaders import InitialProgramLoader
from gigaevo.programs.stages.python_executors.wrapper import default_exec_runner_pool
from gigaevo.runner.dag_runner import DagRunner
from gigaevo.utils.logger_setup import setup_logger
from gigaevo.utils.serve import serve_until_signal
from gigaevo.utils.trackers.base import LogWriter


async def run_experiment(cfg: DictConfig) -> None:
    start_time = time.time()
    logger.info("GigaEvo — Problem: {}", cfg.problem.name)

    redis_storage: RedisProgramStorage | None = None
    writer: LogWriter | None = None
    try:
        config_with_instances = instantiate(cfg, recursive=True)
        redis_storage: RedisProgramStorage = config_with_instances.redis_storage
        program_loader: InitialProgramLoader = config_with_instances.program_loader
        dag_runner: DagRunner = config_with_instances.dag_runner
        evolution_engine: EvolutionEngine = config_with_instances.evolution_engine
        writer: LogWriter = config_with_instances.writer

        logger.info(
            "Redis DB {db} at {host}:{port} | pipeline={pipeline}",
            db=cfg.redis.db,
            host=cfg.redis.host,
            port=cfg.redis.port,
            pipeline=cfg.get("pipeline_builder", {}).get("_target_", "(default)"),
        )

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

        dag_runner.start()
        evolution_engine.start()
        logger.info(
            "Evolution running (max_gen={})", cfg.max_generations or "unlimited"
        )

        await serve_until_signal(
            stop_coros=(evolution_engine.stop(), dag_runner.stop()),
            on_stop=(evolution_engine.task, dag_runner.task),
        )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Experiment failed")
        raise
    finally:
        await default_exec_runner_pool().shutdown()
        if redis_storage is not None:
            await redis_storage.close()
        if writer is not None:
            writer.close()
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
