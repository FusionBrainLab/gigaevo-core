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
from gigaevo.utils.logger import LogWriter
from gigaevo.utils.logger_setup import setup_logger
from gigaevo.utils.serve import serve_until_signal


async def run_experiment(cfg: DictConfig):
    start_time = time.time()

    logger.info("🔄 Starting GigaEvo Evolution Experiment")
    logger.info(f"📁 Problem: {cfg.problem.name}")
    logger.info(f"🕐 Start time: {datetime.now(timezone.utc).isoformat()}")

    try:
        config_with_instances = instantiate(cfg, recursive=True)
        redis_storage: RedisProgramStorage = config_with_instances.redis_storage
        program_loader: InitialProgramLoader = config_with_instances.program_loader
        dag_runner: DagRunner = config_with_instances.dag_runner
        evolution_engine: EvolutionEngine = config_with_instances.evolution_engine
        writer: LogWriter = config_with_instances.writer

        await redis_storage.flushdb()
        logger.info("✓ Redis database cleared")

        logger.info("🌱 Initializing database with initial programs...")
        programs = await program_loader.load(redis_storage)
        logger.info(f"✓ Loaded {len(programs)} initial programs")

        logger.info("🎯 Starting evolution run...")
        logger.info("Configuration:")
        logger.info(f"  - Problem directory: {cfg.problem.dir}")
        logger.info(f"  - Target DB: {cfg.redis.db}")
        logger.info(f"  - Initial population: {len(programs)} programs")
        max_gens: int | None = cfg.constants.max_generations
        logger.info(f"  - Max generations: {max_gens if max_gens else 'unlimited'}")

        dag_runner.start()
        evolution_engine.start()
        await serve_until_signal(
            stop_coros=(evolution_engine.stop(), dag_runner.stop()),
            on_stop=(evolution_engine.task, dag_runner.task),
        )

    except KeyboardInterrupt:
        logger.info("🛑 Evolution experiment interrupted by user")
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"❌ Evolution experiment failed: {e}")
        raise
    finally:
        logger.info("🧹 Starting cleanup...")
        await redis_storage.close()
        writer.close()
        duration = time.time() - start_time
        logger.info(
            f"Total experiment duration: {duration:.2f} seconds ({duration / 3600:.2f} hours)"
        )
        logger.info(f"End time: {datetime.now(timezone.utc).isoformat()}")


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
