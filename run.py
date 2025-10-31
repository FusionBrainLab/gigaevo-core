from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from loguru import logger

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from gigaevo.entrypoint.default_pipelines import (
    ContextPipelineBuilder,
    DefaultPipelineBuilder,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.evolution.engine import (
    EngineConfig,
    EvolutionEngine,
    MutationContextAndBehaviorKeysAcceptor,
)
from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator
from gigaevo.evolution.mutation.parent_selector import AllCombinationsParentSelector
from gigaevo.evolution.strategies.map_elites import (
    BehaviorSpace,
    BinningType,
    FitnessArchiveRemover,
    FitnessProportionalEliteSelector,
    IslandConfig,
    MapElitesMultiIsland,
    SumArchiveSelector,
    TopFitnessMigrantSelector,
)
from gigaevo.llm.models import MultiModelRouter, create_multi_model_router
from gigaevo.problems.context import ProblemContext
from gigaevo.problems.initial_loaders import (
    DirectoryProgramLoader,
    RedisTopProgramsLoader,
)
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.runner.runner import RunnerConfig, RunnerManager
from gigaevo.utils.logger_setup import setup_logger

DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GigaEvo: LLM-based Evolutionary Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use initial programs from directory
  %(prog)s --problem-dir problems/hexagon_pack
  %(prog)s --problem-dir problems/hexagon_pack --redis-db 1

  # Use top programs from existing Redis database (by main metric)
  %(prog)s --problem-dir problems/hexagon_pack --use-redis-selection --source-redis-db 0 --top-n 30
  %(prog)s --problem-dir problems/hexagon_pack --use-redis-selection --redis-host remote-host --redis-port 6379 --source-redis-db 2 --top-n 50
        """,
    )

    parser.add_argument(
        "--problem-dir",
        type=str,
        required=True,
        help="Directory containing problem files",
    )
    parser.add_argument(
        "--add-context",
        action="store_true",
        help="Add context to the problem (i.e., context.py will be run to produce an input to the main program)",
    )

    redis_group = parser.add_argument_group("Redis Configuration")
    redis_group.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis URL, e.g. redis://host:port/db (overrides host/port/db flags)",
    )
    redis_group.add_argument(
        "--redis-host",
        type=str,
        default=DEFAULT_REDIS_HOST,
        help=f"Redis host (default: {DEFAULT_REDIS_HOST})",
    )
    redis_group.add_argument(
        "--redis-port",
        type=int,
        default=DEFAULT_REDIS_PORT,
        help=f"Redis port (default: {DEFAULT_REDIS_PORT})",
    )
    redis_group.add_argument(
        "--redis-db",
        type=int,
        default=DEFAULT_REDIS_DB,
        help=f"Redis database number (default: {DEFAULT_REDIS_DB})",
    )

    evolution_group = parser.add_argument_group("Evolution Configuration")
    evolution_group.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Maximum number of generations (default: unlimited)",
    )
    redis_selection_group = parser.add_argument_group("Redis Selection Configuration")
    redis_selection_group.add_argument(
        "--use-redis-selection",
        action="store_true",
        help="Use Redis selection instead of initial programs directory",
    )
    redis_selection_group.add_argument(
        "--source-redis-url",
        type=str,
        default=None,
        help="Source Redis URL for program selection (overrides source host/port/db)",
    )
    redis_selection_group.add_argument(
        "--source-redis-db",
        type=int,
        default=0,
        help="Source Redis database number for program selection (default: 0)",
    )
    redis_selection_group.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top programs to select by fitness (default: 50)",
    )

    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    logging_group.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )

    performance_group = parser.add_argument_group("Performance Configuration")
    performance_group.add_argument(
        "--max-concurrent-dags",
        type=int,
        default=10,
        help="Maximum concurrent DAG executions (default: 10)",
    )

    return parser.parse_args()


def create_behavior_spaces(
    metrics_context: MetricsContext,
) -> list[BehaviorSpace]:
    """Create behavior spaces using bounds from MetricsContext."""

    primary_key = metrics_context.get_primary_key()
    primary_bounds = metrics_context.get_bounds(primary_key)
    valid_bounds = metrics_context.get_bounds(VALIDITY_KEY)

    if primary_bounds is None:
        raise ValueError(
            f"Primary metric '{primary_key}' must define lower_bound and upper_bound in metrics.yaml"
        )
    if valid_bounds is None:
        raise ValueError(
            f"'{VALIDITY_KEY}' must define lower_bound and upper_bound in metrics.yaml"
        )

    fitness_validity_space = BehaviorSpace(
        feature_bounds={primary_key: primary_bounds, VALIDITY_KEY: valid_bounds},
        resolution={primary_key: 150, VALIDITY_KEY: 2},
        binning_types={
            primary_key: BinningType.LINEAR,
            VALIDITY_KEY: BinningType.LINEAR,
        },
    )

    return [
        fitness_validity_space,
    ]


def create_island_configs(
    behavior_spaces: list[BehaviorSpace], metrics_context: MetricsContext
) -> list[IslandConfig]:
    """Create 1 island configurations with improved resolution balance and migration strategies."""

    primary_key = metrics_context.get_primary_key()
    hi_better = metrics_context.is_higher_better(primary_key)
    island_exploit = IslandConfig(
        island_id="fitness_island",
        max_size=75,
        behavior_space=behavior_spaces[0],
        archive_selector=SumArchiveSelector(
            [primary_key],
            fitness_key_higher_is_better={primary_key: hi_better},
        ),
        elite_selector=FitnessProportionalEliteSelector(primary_key, hi_better),
        archive_remover=FitnessArchiveRemover(primary_key, hi_better),
        migrant_selector=TopFitnessMigrantSelector(primary_key, hi_better),
        migration_rate=0.10,
    )

    return [
        island_exploit,
    ]


async def create_evolution_strategy(
    redis_storage: RedisProgramStorage,
    metrics_context: MetricsContext,
) -> MapElitesMultiIsland:
    behavior_spaces = create_behavior_spaces(metrics_context)
    island_configs = create_island_configs(behavior_spaces, metrics_context)

    strategy = MapElitesMultiIsland(
        island_configs=island_configs,
        program_storage=redis_storage,
        migration_interval=25,
        enable_migration=True,
        max_migrants_per_island=5,
    )

    return strategy


def setup_llm_wrapper() -> MultiModelRouter:
    """Setup LangChain chat models."""

    model_configs = [
        {
            "model": "Qwen3-235B-A22B-Thinking-2507",
            "temperature": 0.6,
            "max_tokens": 81920,
            "top_p": 0.95,
            "top_k": 20,
            "base_url": "http://localhost:8777/v1",
            "request_timeout": 1800,
        }
    ]
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    return create_multi_model_router(model_configs, [1.0], openai_api_key)


def _resolve_redis_url(host: str, port: int, db: int, url_override: str | None) -> str:
    """Build a Redis URL from host/port/db unless an override is provided."""
    if url_override:
        return url_override
    return f"redis://{host}:{port}/{db}"


def _parse_redis_url(url: str) -> tuple[str, int, int]:
    """Parse a redis://host:port/db URL into components."""
    parts = urlsplit(url)
    host = parts.hostname or "localhost"
    port = parts.port or 6379
    db: int
    try:
        db = int((parts.path or "/0").lstrip("/"))
    except ValueError:
        db = 0
    return host, port, db


async def run_evolution_experiment(cli_args: argparse.Namespace, log_file_path: str):
    """Run the complete evolution experiment with provided configuration."""

    start_time = time.time()
    problem_dir = Path(cli_args.problem_dir)

    logger.info("🔄 Starting GigaEvo Evolution Experiment")
    logger.info(f"📁 Problem directory: {problem_dir}")
    logger.info(f"📁 Log file: {log_file_path}")
    logger.info(f"🕐 Start time: {datetime.now(timezone.utc).isoformat()}")

    # Setup Redis storage
    target_redis_url = _resolve_redis_url(
        cli_args.redis_host,
        cli_args.redis_port,
        cli_args.redis_db,
        cli_args.redis_url,
    )
    redis_storage = RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=target_redis_url,
            key_prefix=f"{problem_dir.name}_evolution",
            max_connections=150,
            connection_pool_timeout=45.0,
            health_check_interval=120,
            max_retries=5,
            retry_delay=0.5,
        )
    )

    try:
        # Clear the target database to start fresh
        logger.info(f"🧹 Clearing Redis database {cli_args.redis_db} for restart...")
        await redis_storage.flushdb()
        logger.info(f"✓ Redis database {cli_args.redis_db} cleared")

        # Build problem context (centralized assets)
        problem_ctx = ProblemContext(problem_dir)
        problem_ctx.validate(add_context=cli_args.add_context)
        metrics_context = problem_ctx.metrics_context

        # Initialize new DB with initial programs
        if cli_args.use_redis_selection:
            logger.info("🔍 Initializing database with selected programs from Redis...")
            primary_key = metrics_context.get_primary_key()
            source_host = cli_args.redis_host
            source_port = cli_args.redis_port
            source_db = cli_args.source_redis_db
            if cli_args.source_redis_url:
                source_host, source_port, source_db = _parse_redis_url(
                    cli_args.source_redis_url
                )
            loader = RedisTopProgramsLoader(
                source_host=source_host,
                source_port=source_port,
                source_db=source_db,
                key_prefix=f"{problem_dir.name}_evolution",
                metric_key=primary_key,
                higher_is_better=metrics_context.is_higher_better(primary_key),
                top_n=cli_args.top_n,
            )
            programs = await loader.load(redis_storage)
        else:
            logger.info("🌱 Initializing database with initial programs...")
            programs = await DirectoryProgramLoader(problem_dir).load(redis_storage)

        logger.info("Setting up LLM wrapper...")
        llm_wrapper = setup_llm_wrapper()

        logger.info("Creating DAG pipeline...")
        pctx = EvolutionContext(
            problem_ctx=problem_ctx,
            llm_wrapper=llm_wrapper,
            storage=redis_storage,
        )
        if problem_ctx.is_contextual:
            logger.info(
                "Contextual problem detected. Using contextual pipeline builder..."
            )
            builder = ContextPipelineBuilder(pctx)
        else:
            logger.info(
                "Non-contextual problem detected. Using default pipeline builder..."
            )
            builder = DefaultPipelineBuilder(pctx)
        dag_blueprint = builder.build_blueprint()

        logger.info("Creating evolution strategy...")
        evolution_strategy = await create_evolution_strategy(
            redis_storage, metrics_context
        )

        logger.info("Creating LLM mutation operator...")

        mutation_operator = LLMMutationOperator(
            llm_wrapper=llm_wrapper,
            mutation_mode="rewrite",
            problem_context=problem_ctx,
        )
        required_behavior_keys = set()
        for island in evolution_strategy.islands.values():
            required_behavior_keys |= set(island.config.behavior_space.behavior_keys)

        logger.info("Creating evolution engine...")

        engine_config = EngineConfig(
            loop_interval=1.0,
            max_elites_per_generation=5,
            max_mutations_per_generation=8,
            max_generations=cli_args.max_generations,
            program_acceptor=MutationContextAndBehaviorKeysAcceptor(
                required_behavior_keys=required_behavior_keys
            ),
            parent_selector=AllCombinationsParentSelector(num_parents=2),
        )

        evolution_engine = EvolutionEngine(
            storage=redis_storage,
            strategy=evolution_strategy,
            mutation_operator=mutation_operator,
            config=engine_config,
        )

        logger.info("Creating runner...")
        runner_config = RunnerConfig(
            poll_interval=5.0,
            max_concurrent_dags=cli_args.max_concurrent_dags,
            log_interval=15,
            dag_timeout=1800,
        )

        runner = RunnerManager(
            engine=evolution_engine,
            dag_blueprint=dag_blueprint,
            storage=redis_storage,
            config=runner_config,
        )

        logger.info("🎯 Starting evolution run...")
        logger.info("Configuration:")
        logger.info(f"  - Problem directory: {problem_dir}")
        logger.info(f"  - Target DB: {cli_args.redis_db}")
        logger.info(f"  - Initial population: {len(programs)} programs")
        logger.info(
            f"  - Max generations: {cli_args.max_generations if cli_args.max_generations else 'unlimited'}"
        )
        logger.info(f"  - DAG stages: {list(dag_blueprint.nodes.keys())}")

        await runner.run()

    except KeyboardInterrupt:
        logger.info("🛑 Evolution experiment interrupted by user")
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"❌ Evolution experiment failed: {e}")
        raise
    finally:
        # Improved cleanup with connection pool closure
        logger.info("🧹 Starting cleanup...")
        try:
            await redis_storage.close()
            logger.info("✓ Redis connection closed")
        except Exception as cleanup_error:  # pylint: disable=broad-except
            logger.warning(f"⚠️ Redis cleanup warning: {cleanup_error}")
        logger.info("🧹 Cleanup completed")

        # Log experiment completion
        duration = time.time() - start_time
        logger.info(
            f"⏱️ Total experiment duration: {duration:.2f} seconds ({duration / 3600:.2f} hours)"
        )
        logger.info(f"🕐 End time: {datetime.now(timezone.utc).isoformat()}")


def main() -> int:
    cli_args = parse_arguments()

    log_file_path = setup_logger(
        log_dir=cli_args.log_dir,
        level=cli_args.log_level,
        rotation="50 MB",
        retention="30 days",
    )

    cli_problem_dir = Path(cli_args.problem_dir)
    if not cli_problem_dir.exists():
        logger.error(f"❌ Problem directory not found: {cli_problem_dir}")
        raise SystemExit(1)

    asyncio.run(run_evolution_experiment(cli_args, log_file_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
