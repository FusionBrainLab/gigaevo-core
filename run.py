#!/usr/bin/env python3
"""
MetaEvolve: LLM-based Evolutionary System for Optimization Problems

This script runs the MetaEvolve pipeline for optimization problems by:
1. Loading initial programs from a configurable problem directory
2. Creating diverse initial populations using problem-specific strategies
3. Running multi-island evolution with LLM-based mutation
4. Optimizing arrangements using geometric and structural diversity

Usage:
    python restart_llm_evolution_improved.py --problem-dir problems/hexagon_pack [OPTIONS]

"""

import argparse
import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

# Main imports
from loguru import logger

from src.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from src.evolution.engine import EngineConfig, EvolutionEngine, RequiredBehaviorKeysAcceptor
from src.evolution.mutation.llm import LLMMutationOperator
from src.evolution.mutation.parent_selector import (
    AllCombinationsParentSelector,
)
from src.evolution.strategies.map_elites import (
    BehaviorSpace,
    BinningType,
    FitnessArchiveRemover,
    FitnessProportionalEliteSelector,
    IslandConfig,
    MapElitesMultiIsland,
    SumArchiveSelector,
    TopFitnessMigrantSelector,
)
from src.llm.wrapper import LLMConfig, MultiModelLLMWrapper
from src.programs.metrics.context import MetricsContext, VALIDITY_KEY
from src.programs.metrics.formatter import MetricsFormatter
from src.problems.context import ProblemContext
from src.problems.initial_loaders import (
    DirectoryProgramLoader,
    RedisTopProgramsLoader,
)
from src.runner.manager import RunnerConfig, RunnerManager
from src.runner.pipeline_factory import (
    PipelineContext,
    DefaultPipelineBuilder,
    ContextPipelineBuilder,
)

# Setup logging first
from src.utils.logger_setup import setup_logger

# Global configuration
DEFAULT_PROBLEM_DIR = "problems/hexagon_pack"
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MetaEvolve: LLM-based Evolutionary Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use initial programs from directory
  %(prog)s --problem-dir problems/hexagon_pack
  %(prog)s --problem-dir problems/hexagon_pack --redis-db 1
  
  # Use top programs from existing Redis database (by fitness)
  %(prog)s --problem-dir problems/hexagon_pack --use-redis-selection --source-redis-db 0 --top-n 30
  %(prog)s --problem-dir problems/hexagon_pack --use-redis-selection --redis-host remote-host --redis-port 6379 --source-redis-db 2 --top-n 50
        """,
    )

    # Required arguments
    parser.add_argument(
        "--problem-dir",
        type=str,
        default=DEFAULT_PROBLEM_DIR,
        help=f"Directory containing problem files (default: {DEFAULT_PROBLEM_DIR})",
    )
    parser.add_argument(
        "--add-context",
        action="store_true",
        help="Add context to the problem (i.e., context.py will be run to produce an input to the main program)",
    )

    # Redis configuration
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

    # Evolution configuration
    evolution_group = parser.add_argument_group("Evolution Configuration")
    evolution_group.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Maximum number of generations (default: unlimited)",
    )
    evolution_group.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Initial population size (default: auto-determined)",
    )

    # Redis selection configuration
    redis_selection_group = parser.add_argument_group(
        "Redis Selection Configuration"
    )
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

    # Logging configuration
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

    # Performance configuration
    performance_group = parser.add_argument_group("Performance Configuration")
    performance_group.add_argument(
        "--max-concurrent-dags",
        type=int,
        default=10,
        help="Maximum concurrent DAG executions (default: 10)",
    )

    return parser.parse_args()


# Configuration constants
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = os.getenv("REDIS_DB", "0")

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")


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
        max_size=75,  # tighter, exploit-focused
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


async def setup_llm_wrapper() -> dict[str, MultiModelLLMWrapper]:
    """Setup the LLM wrapper for code generation and insights (post-refactor)."""

    if not LLM_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    # Updated for longer programs and better model alignment
    settings_per_stage = {
        "insights": {
            "temperature": 0.6,
            "max_tokens": 32768,
            "top_p": 0.95,
            "top_k": 20,
        },
        "lineage": {
            "temperature": 0.6,
            "max_tokens": 32768,
            "top_p": 0.95,
            "top_k": 20,
        },
        "mutation": {
            "temperature": 0.6,
            "max_tokens": 32768,
            "top_p": 0.95,
            "top_k": 20,
        },
    }

    def build_wrapper_with_params(
        params: dict[str, float],
    ) -> MultiModelLLMWrapper:

        return MultiModelLLMWrapper(
            models=[
                "Qwen3-235B-A22B-Thinking-2507",
            ],
            probabilities=[1.0],
            api_key=LLM_API_KEY,
            configs=[
                LLMConfig(**params, api_endpoint="http://localhost:8777/v1"),
            ],
        )

    res = {
        stage: build_wrapper_with_params(params)
        for stage, params in settings_per_stage.items()
    }
    return res


def _resolve_redis_url(
    host: str, port: int, db: int, url_override: str | None
) -> str:
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


async def run_evolution_experiment(
    cli_args: argparse.Namespace, log_file_path: str
):
    """Run the complete evolution experiment with provided configuration."""

    start_time = time.time()
    problem_dir = Path(cli_args.problem_dir)

    logger.info("🔄 Starting MetaEvolve Evolution Experiment")
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
        logger.info(
            f"🧹 Clearing Redis database {cli_args.redis_db} for restart..."
        )
        await redis_storage.flushdb()
        logger.info(f"✓ Redis database {cli_args.redis_db} cleared")

        # Build problem context (centralized assets)
        problem_ctx = ProblemContext(problem_dir)
        problem_ctx.validate(add_context=cli_args.add_context)
        metrics_context = problem_ctx.metrics_context

        # Initialize new DB with initial programs
        if cli_args.use_redis_selection:
            logger.info(
                "🔍 Initializing database with selected programs from Redis..."
            )
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
            programs = await DirectoryProgramLoader(problem_dir).load(
                redis_storage
            )

        task_description = problem_ctx.task_description
        task_hints = problem_ctx.task_hints

        logger.info("Setting up LLM wrapper...")
        llm_wrapper = await setup_llm_wrapper()

        logger.info("Creating DAG pipeline...")
        metrics_formatter = MetricsFormatter(
            metrics_context, use_range_normalization=False
        )

        pctx = PipelineContext(
            problem_ctx=problem_ctx,
            metrics_context=metrics_context,
            metrics_formatter=metrics_formatter,
            llm_wrapper=llm_wrapper,
            storage=redis_storage,
            task_description=task_description,
            add_context=cli_args.add_context,
        )
        if cli_args.add_context:
            builder = ContextPipelineBuilder(pctx)
        else:
            builder = DefaultPipelineBuilder(pctx)
        dag_blueprint = builder.set_limits(
            dag_timeout=1800, max_parallel=8
        ).build_blueprint()

        # Create evolution strategy
        logger.info("Creating evolution strategy...")
        evolution_strategy = await create_evolution_strategy(
            redis_storage, metrics_context
        )

        # Create LLM mutation operator
        logger.info("Creating LLM mutation operator...")

        mutation_operator = LLMMutationOperator(
            llm_wrapper=llm_wrapper["mutation"],
            mutation_mode="rewrite",  # Start with rewrite for maximum change
            fetch_insights_fn=lambda x: x.metadata.get(
                "insights", "No insights available."
            ),
            fetch_lineage_insights_fn=lambda x: x.metadata.get(
                "lineage_insights", "No lineage insights available."
            ),
            task_definition=task_description,
            task_hints=task_hints,
            system_prompt_template=problem_ctx.mutation_system_prompt,
            user_prompt_templates=[
                problem_ctx.mutation_user_prompt
            ],  # optionally use a list of templates with weights to be randomly selected
            user_prompt_template_weights_factory=lambda x: [1.0],
            metrics_context=metrics_context,
            metrics_formatter=metrics_formatter,
        )
        required_behavior_keys = set()
        for island in evolution_strategy.islands.values():
            required_behavior_keys |= set(
                island.config.behavior_space.behavior_keys
            )

        # Note: Consider an abstraction for a function filter to drop unsuitable programs
        # (e.g., when metrics are missing).
        logger.info("Creating evolution engine...")

        engine_config = EngineConfig(
            loop_interval=1.0,
            max_elites_per_generation=5,
            max_mutations_per_generation=8,
            max_generations=cli_args.max_generations, 
            program_acceptor=RequiredBehaviorKeysAcceptor(required_behavior_keys=required_behavior_keys),
            parent_selector=AllCombinationsParentSelector(num_parents=2),
        )

        evolution_engine = EvolutionEngine(
            storage=redis_storage,
            strategy=evolution_strategy,
            mutation_operator=mutation_operator,
            config=engine_config,
        )

        # Create runner with optimized concurrency
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
            f"⏱️ Total experiment duration: {duration:.2f} seconds ({duration/3600:.2f} hours)"
        )
        logger.info(f"🕐 End time: {datetime.now(timezone.utc).isoformat()}")


def main() -> int:
    """CLI entrypoint for running MetaEvolve experiment."""
    # Parse command-line arguments
    cli_args = parse_arguments()

    # Reconfigure logging with user preferences
    log_file_path = setup_logger(
        log_dir=cli_args.log_dir,
        level=cli_args.log_level,
        rotation="50 MB",
        retention="30 days",
    )

    # Check prerequisites
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY environment variable must be set")
        raise SystemExit(1)

    cli_problem_dir = Path(cli_args.problem_dir)
    if not cli_problem_dir.exists():
        logger.error(f"❌ Problem directory not found: {cli_problem_dir}")
        raise SystemExit(1)

    # Run the evolution experiment
    asyncio.run(run_evolution_experiment(cli_args, log_file_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
