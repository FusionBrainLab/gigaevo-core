import asyncio
from datetime import datetime, timezone
import time

from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gigaevo.config.resolvers import register_resolvers
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.engine import EvolutionEngine
from gigaevo.problems.initial_loaders import InitialProgramLoader
from gigaevo.runner.dag_runner import DagRunner
from gigaevo.utils.logger_setup import setup_logger
from gigaevo.utils.serve import serve_until_signal
from gigaevo.utils.trackers.base import LogWriter


async def run_experiment(cfg: DictConfig):
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("GigaEvo Evolution Experiment")
    logger.info("=" * 80)
    logger.info(f"Problem: {cfg.problem.name}")
    logger.info(f"Start time: {datetime.now(timezone.utc).isoformat()}")
    logger.info("")

    redis_storage: RedisProgramStorage | None = None
    writer: LogWriter | None = None
    try:
        logger.info("Step 1/5: Initializing components...")
        config_with_instances = instantiate(cfg, recursive=True)
        redis_storage: RedisProgramStorage = config_with_instances.redis_storage
        program_loader: InitialProgramLoader = config_with_instances.program_loader
        dag_runner: DagRunner = config_with_instances.dag_runner
        evolution_engine: EvolutionEngine = config_with_instances.evolution_engine
        writer: LogWriter = config_with_instances.writer
        logger.info("Step 1/5: Complete")
        logger.info("")

        logger.info("Step 2/5: Checking Redis database...")
        # Safety check: prevent accidental data loss
        if await redis_storage.has_data():
            db_num = cfg.redis.db
            redis_host = cfg.redis.host
            redis_port = cfg.redis.port
            error_msg = f"""
ERROR: Redis database is not empty!

  Database {db_num} at {redis_host}:{redis_port} contains existing programs.

To prevent accidental data loss, you must manually flush the database.

Run this command to flush:
  redis-cli -h {redis_host} -p {redis_port} -n {db_num} FLUSHDB

Or use a different database number:
  python run.py redis.db=<number> ...
"""
            logger.error(error_msg)
            raise RuntimeError(
                f"Redis database {db_num} is not empty. Flush manually to proceed."
            )
        logger.info("Step 2/5: Database is empty, ready to proceed")
        logger.info("")

        logger.info("Step 3/5: Loading initial programs...")
        programs = await program_loader.load(redis_storage)
        logger.info(f"Step 3/5: Loaded {len(programs)} initial programs")
        logger.info("")

        logger.info("Step 4/5: Starting evolution...")
        max_gens: int | None = cfg.max_generations
        logger.info(f"  Max generations: {max_gens if max_gens else 'unlimited'}")
        logger.info(f"  Population size: {len(programs)} programs")

        dag_runner.start()
        evolution_engine.start()
        logger.info("Step 4/5: Evolution running")
        logger.info("")

        logger.info("Step 5/5: Running until completion or signal...")
        await serve_until_signal(
            stop_coros=(evolution_engine.stop(), dag_runner.stop()),
            on_stop=(evolution_engine.task, dag_runner.task),
        )

    except KeyboardInterrupt:
        logger.info("Evolution experiment interrupted by user")
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Evolution experiment failed: {e}")
        raise
    finally:
        logger.info("")
        logger.info("Starting cleanup...")
        if redis_storage is not None:
            await redis_storage.close()
        if writer is not None:
            writer.close()
        duration = time.time() - start_time
        logger.info(
            f"Total experiment duration: {duration:.2f} seconds ({duration / 3600:.2f} hours)"
        )
        logger.info(f"End time: {datetime.now(timezone.utc).isoformat()}")
        logger.info("=" * 80)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entrypoint with Hydra configuration management."""
    load_dotenv()

    log_file_path = setup_logger(
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
    )
    logger.info(
        "Experiment working directory: {}.",
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    logger.info(f"Log file: {log_file_path}")
    asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    register_resolvers()
    main()
