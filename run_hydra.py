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
from gigaevo.runner.dag_blueprint import DAGBlueprint
from gigaevo.runner.runner import RunnerConfig, RunnerManager
from gigaevo.utils.logger_setup import setup_logger


async def run_experiment(cfg: DictConfig):
    start_time = time.time()

    logger.info("🔄 Starting GigaEvo Evolution Experiment")
    logger.info(f"📁 Problem: {cfg.problem.name}")
    logger.info(f"🕐 Start time: {datetime.now(timezone.utc).isoformat()}")

    try:
        redis_storage: RedisProgramStorage = instantiate(cfg.redis_storage)
        program_loader: InitialProgramLoader = instantiate(cfg.program_loader)

        dag_blueprint: DAGBlueprint = instantiate(cfg.dag_blueprint)
        evolution_engine: EvolutionEngine = instantiate(cfg.evolution_engine)
        runner_config: RunnerConfig = instantiate(cfg.runner_config)

        runner = RunnerManager(
            engine=evolution_engine,
            dag_blueprint=dag_blueprint,
            storage=redis_storage,
            config=runner_config,
        )

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
        max_gens: int | None = cfg.max_generations
        logger.info(f"  - Max generations: {max_gens if max_gens else 'unlimited'}")
        await runner.run()

    except KeyboardInterrupt:
        logger.info("🛑 Evolution experiment interrupted by user")
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"❌ Evolution experiment failed: {e}")
        raise
    finally:
        logger.info("🧹 Starting cleanup...")
        await redis_storage.close()
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
    logger.info(f"📁 Log file: {log_file_path}")

    asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    register_resolvers()
    main()
