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
from typing import Any, Dict, List

# Main imports
from loguru import logger

from src.database.program_storage import RedisProgramStorage
from src.evolution.engine import EngineConfig, EvolutionEngine
from src.evolution.mutation.llm import LLMMutationOperator
from src.evolution.strategies.map_elites import (
    BehaviorSpace,
    BinningType,
    FitnessArchiveRemover,
    FitnessProportionalEliteSelector,
    IslandConfig,
    MapElitesMultiIsland,
    ParetoFrontArchiveRemoverDropOldest,
    ParetoFrontMigrantSelector,
    RandomMigrantSelector,
    ParetoFrontSelector,
    ParetoTournamentEliteSelector,
    SumArchiveSelector,
    TopFitnessMigrantSelector,
)
from src.llm.wrapper import LLMConfig, LLMWrapper, MultiModelLLMWrapper
from src.programs.automata import ExecutionOrderDependency
from src.programs.program import Program
from src.programs.stages.base import Stage
from src.programs.stages.complexity import ComputeComplexityStage
from src.programs.stages.execution import RunCodeStage, RunValidationStage, RunPythonCode
from src.programs.stages.insights import (
    GenerateLLMInsightsStage,
    InsightsConfig,
)
from src.programs.stages.insights_lineage import (
    GenerateLineageInsightsStage,
    LineageInsightsConfig,
)
from src.programs.stages.metrics import FactoryMetricsStage
from src.programs.stages.validation import ValidateCodeStage
from src.runner.dag_spec import DAGSpec
from src.runner.manager import RunnerConfig, RunnerManager

# Setup logging first
from src.utils.logger_setup import setup_logger
from src.utils.performance_monitor import (
    auto_optimize,
    get_performance_dashboard,
)

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
  %(prog)s --problem-dir problems/hexagon_pack --min-fitness 0 --max-fitness 100
  %(prog)s --problem-dir problems/hexagon_pack --redis-db 1 --verbose --min-fitness 0 --max-fitness 100
  
  # Use top programs from existing Redis database (by fitness)
  %(prog)s --problem-dir problems/hexagon_pack --use-redis-selection --source-redis-db 0 --top-n 30 --min-fitness 0 --max-fitness 100
  %(prog)s --problem-dir problems/hexagon_pack --use-redis-selection --redis-host remote-host --redis-port 6379 --source-redis-db 2 --top-n 50 --min-fitness 0 --max-fitness 100
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

    # TODO: Will be replaced with config files, command line arguments are temporary
    evolution_group.add_argument(
        "--min-fitness",
        type=float,
        required=True,
        help="Minimum fitness value for program to be considered",
    )
    evolution_group.add_argument(
        "--max-fitness",
        type=float,
        required=True,
        help="Maximum fitness value for program to be considered",
    )

    # Redis selection configuration
    redis_selection_group = parser.add_argument_group("Redis Selection Configuration")
    redis_selection_group.add_argument(
        "--use-redis-selection",
        action="store_true",
        help="Use Redis selection instead of initial programs directory",
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
    logging_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Performance configuration
    performance_group = parser.add_argument_group("Performance Configuration")
    performance_group.add_argument(
        "--max-concurrent-dags",
        type=int,
        default=10,
        help="Maximum concurrent DAG executions (default: 10)",
    )
    performance_group.add_argument(
        "--disable-monitoring",
        action="store_true",
        help="Disable performance monitoring",
    )

    return parser.parse_args()


def validate_problem_directory(problem_dir: Path, add_context: bool = False) -> None:
    """Validate that the problem directory contains required files and directories."""
    required_files = [
        "task_description.txt",
        "task_hints.txt",
        "validate.py",
        "mutation_system_prompt.txt",
        "mutation_user_prompt.txt",
    ]

    if add_context:
        required_files.append("context.py")

    required_directories = ["initial_programs"]

    missing_files = []
    for filename in required_files:
        if not (problem_dir / filename).exists():
            missing_files.append(filename)

    missing_directories = []
    for dirname in required_directories:
        if not (problem_dir / dirname).exists():
            missing_directories.append(dirname)

    if missing_files or missing_directories:
        missing_items = missing_files + [f"{d}/" for d in missing_directories]
        raise FileNotFoundError(
            f"Missing required files/directories in {problem_dir}: {', '.join(missing_items)}"
        )

    # Validate that initial_programs directory has at least one Python file
    initial_programs_dir = problem_dir / "initial_programs"
    python_files = list(initial_programs_dir.glob("*.py"))
    if not python_files:
        raise FileNotFoundError(
            f"No Python files found in {initial_programs_dir}. "
            "At least one initial program is required."
        )


def load_problem_file(problem_dir: Path, filename: str) -> str:
    """Load a text file from the problem directory."""
    file_path = problem_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Problem file not found: {file_path}")

    return file_path.read_text().strip()


log_file_path = setup_logger(
    log_dir="logs", level="DEBUG", rotation="50 MB", retention="30 days"
)


# Configuration constants
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = os.getenv("REDIS_DB", "0")

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")


def load_initial_programs_from_directory(problem_dir: Path) -> List[Program]:
    """Load initial programs from the initial_programs subdirectory."""
    initial_programs_dir = problem_dir / "initial_programs"

    if not initial_programs_dir.exists():
        logger.warning(f"No initial_programs directory found in {problem_dir}")
        return []

    programs = []

    # Load all Python files from the initial_programs directory
    python_files = list(initial_programs_dir.glob("*.py"))

    if not python_files:
        logger.warning(f"No Python files found in {initial_programs_dir}")
        return []

    logger.info(
        f"Loading {len(python_files)} initial programs from {initial_programs_dir}"
    )

    for program_file in python_files:
        try:
            program_code = program_file.read_text()
            program = Program(code=program_code)
            program.metadata = {
                "source": "initial_program",
                "strategy_name": program_file.stem,
                "file_path": str(program_file),
            }
            programs.append(program)
            logger.info(f"  ✅ Loaded initial program: {program_file.stem}")
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to load {program_file}: {e}")

    return programs


async def create_initial_population(
    redis_storage: RedisProgramStorage, problem_dir: Path
) -> List[Program]:
    """
    Create initial population from programs in the initial_programs directory.

    Args:
        redis_storage: Target Redis storage
        problem_dir: Problem directory path

    Returns:
        List of programs added to database
    """
    logger.info("🌱 Creating initial population...")

    # Load all initial programs from the initial_programs directory
    initial_programs = load_initial_programs_from_directory(problem_dir)

    if not initial_programs:
        raise RuntimeError(
            f"No initial programs found in {problem_dir}/initial_programs/. "
            "At least one Python file is required."
        )

    programs = []

    # Add all initial programs to the database
    for program in initial_programs:
        await redis_storage.add(program)
        programs.append(program)
        strategy_name = program.metadata.get("strategy_name", "unknown")
        logger.info(f"  ✅ Added initial program: {strategy_name}")

    logger.info(
        f"🎯 Successfully created initial population of {len(programs)} programs"
    )
    return programs


async def select_top_programs_from_redis(
    redis_storage: RedisProgramStorage,
    source_redis_host: str,
    source_redis_port: int,
    source_redis_db: int,
    problem_dir: Path,
    top_n: int = 50,
) -> List[Program]:
    """
    Select top programs by fitness from an existing Redis database.
    
    Args:
        redis_storage: Target Redis storage where selected programs will be added
        source_redis_host: Host of the source Redis database
        source_redis_port: Port of the source Redis database
        source_redis_db: Database number of the source Redis database
        problem_dir: Problem directory path (used to construct key prefix)
        top_n: Number of top programs to select
    
    Returns:
        List of selected programs added to target database
    """
    logger.info(f"🔍 Selecting top {top_n} programs from Redis {source_redis_host}:{source_redis_port}/{source_redis_db}...")
    
    # Create source Redis storage connection with same key prefix construction
    source_storage = RedisProgramStorage({
        "redis_url": f"redis://{source_redis_host}:{source_redis_port}/{source_redis_db}",
        "key_prefix": f"{problem_dir.name}_evolution",
        "max_connections": 50,
        "connection_pool_timeout": 30.0,
        "health_check_interval": 60,
    })
    
    try:
        # Get all programs from source database
        logger.info("📥 Retrieving all programs from source database...")
        all_programs = await source_storage.get_all()
        logger.info(f"📊 Found {len(all_programs)} total programs in source database")
        
        if not all_programs:
            logger.warning("⚠️ No programs found in source database")
            return []
        
        # Filter programs that have fitness metrics
        programs_with_fitness = []
        for program in all_programs:
            if program.metrics and "fitness" in program.metrics:
                programs_with_fitness.append(program)
        
        logger.info(f"🔄 Found {len(programs_with_fitness)} programs with fitness metrics")
        
        if not programs_with_fitness:
            logger.warning("⚠️ No programs with fitness metrics found")
            return []
        
        # Sort by fitness (descending) and take top N
        programs_with_fitness.sort(
            key=lambda p: p.metrics.get("fitness", -1000.0), 
            reverse=True
        )
        selected_programs = programs_with_fitness[:top_n]
        
        logger.info(f"🎯 Selected {len(selected_programs)} top programs by fitness")
        
        # Add selected programs to target database
        added_programs = []
        for i, program in enumerate(selected_programs):
            program_to_add = Program(code=program.code)
            # Update metadata to indicate source
            program_to_add.metadata = {
                "source": "redis_selection",
                "source_db": source_redis_db,
                "selection_rank": i + 1,
                "original_id": program.id,
            }
            
            # Add to target database
            await redis_storage.add(program_to_add)
            added_programs.append(program_to_add)
            
            fitness = program.metrics.get("fitness", "N/A")
            logger.info(f"  ✅ Added program {i+1}/{len(selected_programs)}: fitness={fitness}")
        
        logger.info(f"🎯 Successfully selected and added {len(added_programs)} top programs")
        return added_programs
        
    except Exception as e:
        logger.error(f"❌ Failed to select top programs: {e}")
        raise
    finally:
        # Clean up source connection
        try:
            await source_storage.close()
        except Exception as e:
            logger.warning(f"⚠️ Error closing source storage: {e}")


def create_behavior_spaces(args: argparse.Namespace) -> List[BehaviorSpace]:
    """Improved behavior spaces for four diverse islands with better fitness coverage and resolution balance."""

    # 🏝️ 1. Fitness–Validity Island
    fitness_validity_space = BehaviorSpace(
        feature_bounds={
            "fitness": (args.min_fitness, args.max_fitness),
            "is_valid": (-0.01, 1.01),
        },
        resolution={"fitness": 30, "is_valid": 2},
        binning_types={
            "fitness": BinningType.LINEAR,
            "is_valid": BinningType.LINEAR,
        },
    )

    # 🏝️ 2. Simplicity Island
    simplicity_space = BehaviorSpace(
        feature_bounds={
            "fitness": (args.min_fitness, args.max_fitness),
            "complexity_score": (0.5, 20),
            "is_valid": (-0.01, 1.01),
        },
        resolution={
            "fitness": 12,
            "complexity_score": 12,
            "is_valid": 2,
        },
        binning_types={
            "fitness": BinningType.LINEAR,
            "complexity_score": BinningType.LOGARITHMIC,
            "is_valid": BinningType.LINEAR,
        },
    )

    # 🏝️ 3. Entropy Island
    entropy_space = BehaviorSpace(
        feature_bounds={
            "fitness": (args.min_fitness, args.max_fitness),
            "ast_entropy": (1.5, 3.5),
            "is_valid": (-0.01, 1.01),
        },
        resolution={
            "fitness": 10,
            "ast_entropy": 6,
            "is_valid": 2,
        },
        binning_types={
            "fitness": BinningType.LINEAR,
            "ast_entropy": BinningType.LOGARITHMIC,
            "is_valid": BinningType.LINEAR,
        },
    )

    return [fitness_validity_space, simplicity_space, entropy_space]


def create_island_configs(
    behavior_spaces: List[BehaviorSpace],
) -> List[IslandConfig]:
    """Create 4 island configurations with improved resolution balance and migration strategies."""

    configs = [
        IslandConfig(
            island_id="fitness_island",
            max_size=50,
            behavior_space=behavior_spaces[0],
            archive_selector=SumArchiveSelector(["fitness"]),
            elite_selector=FitnessProportionalEliteSelector("fitness"),
            archive_remover=FitnessArchiveRemover("fitness"),
            migrant_selector=TopFitnessMigrantSelector("fitness"),
            migration_rate=0.05,
        ),
        IslandConfig(
            island_id="simplicity_island",
            max_size=50,
            behavior_space=behavior_spaces[1],
            archive_selector=ParetoFrontSelector(["fitness", "complexity_score"], fitness_key_higher_is_better={"fitness": True, "complexity_score": False}),
            elite_selector=ParetoTournamentEliteSelector(["fitness", "complexity_score"], fitness_key_higher_is_better={"fitness": True, "complexity_score": False}),
            archive_remover=ParetoFrontArchiveRemoverDropOldest(["fitness", "complexity_score"], fitness_key_higher_is_better={"fitness": True, "complexity_score": False}),
            migrant_selector=ParetoFrontMigrantSelector(["fitness", "complexity_score"], fitness_key_higher_is_better={"fitness": True, "complexity_score": False}),
            migration_rate=0.15,
        ),
        IslandConfig(
            island_id="entropy_island",
            max_size=50,
            behavior_space=behavior_spaces[2],
            archive_selector=ParetoFrontSelector(["fitness", "ast_entropy"]),
            elite_selector=ParetoTournamentEliteSelector(["fitness", "ast_entropy"]),
            archive_remover=ParetoFrontArchiveRemoverDropOldest(["fitness", "ast_entropy"]),
            migrant_selector=RandomMigrantSelector(),
            migration_rate=0.1,
        ),
    ]

    return configs


def create_dag_stages(
    llm_wrapper: dict[str, LLMWrapper],
    redis_storage: RedisProgramStorage,
    task_description: str,
    problem_dir: Path,
    add_context: bool = False,
) -> Dict[str, Stage]:
    """Create optimized DAG stages with stage-specific timeouts and resource allocation."""

    stages = {}

    # Stage 1: Validate that code compiles (FAST - 30s timeout)
    stages["ValidateCompiles"] = lambda: ValidateCodeStage(
        stage_name="ValidateCompiles",
        max_code_length=13000,
        timeout=30.0,  # OPTIMIZED: Reduced timeout for fast operations
        safe_mode=True,
    )

    if add_context:
        stages["AddContext"] = lambda: RunPythonCode(
            code=load_problem_file(problem_dir, "context.py"),
            stage_name="AddContext",
            function_name="build_context",
            python_path=[problem_dir.resolve()],
            timeout=120.0,
            max_memory_mb=1024,
        )


    # Stage 2: Execute the code to get results
    stages["ExecuteCode"] = lambda: RunCodeStage(
        stage_name="ExecuteCode",
        function_name="entrypoint",
        context_stage="AddContext" if add_context else None,
        python_path=[problem_dir.resolve()],
        timeout=1200.0,
        max_memory_mb=512,
    )

    # Stage 2.5: Compute complexity metrics (FAST - 30s timeout)
    stages["ComputeComplexity"] = lambda: ComputeComplexityStage(
        stage_name="ComputeComplexity",
        timeout=60.0,  # OPTIMIZED: Reduced timeout for AST analysis
    )

    # Stage 3: Run validation against the output (MEDIUM - 60s timeout)
    validator_path = problem_dir / "validate.py"
    stages["RunValidation"] = lambda: RunValidationStage(
        stage_name="RunValidation",
        validator_path=validator_path,
        data_to_validate_stage="ExecuteCode",
        context_stage="AddContext" if add_context else None,
        function_name="validate",
        timeout=60.0,  # OPTIMIZED: Reduced from 120s for faster validation
    )

    def create_insights_stage():
        """Create insights stage with optimized timeout and caching."""
        return GenerateLLMInsightsStage(
            config=InsightsConfig(
                llm_wrapper=llm_wrapper["insights"],
                evolutionary_task_description=task_description,
                max_insights=8,
                output_format="text",
                metrics_to_display={
                    "fitness": "main objective, higher is better",
                },
                metadata_key="insights",
                excluded_error_stages=[
                    "FactoryMetricUpdate",
                    "ComplexityMetricUpdate",
                ],
            ),
            timeout=240.0,  # OPTIMIZED: Reduced from 120s
        )

    def create_lineage_insights_stage():
        return GenerateLineageInsightsStage(
            config=LineageInsightsConfig(
                llm_wrapper=llm_wrapper["lineage"],
                metric_key="fitness",
                metric_description="main objective",
                higher_is_better=True,
                parent_selection_strategy="best_fitness",
                fitness_selector_metric="fitness",
                fitness_selector_higher_is_better=True,
            ),
            storage=redis_storage,
            timeout=240,  # OPTIMIZED: Reduced from 120s
        )

    stages["LLMInsights"] = create_insights_stage
    stages["LineageInsights"] = create_lineage_insights_stage

    # Stage 5a: Validation metrics factory (FAST - 15s timeout)
    def validation_metrics_factory() -> Dict[str, Any]:
        """Factory function that creates default validation metrics when validation fails."""
        return {
            "fitness": -1000.0,  # Very bad fitness for failed programs
            "is_valid": 0,
        }

    stages["ValidationMetricUpdate"] = lambda: FactoryMetricsStage(
        stage_name="ValidationMetricUpdate",
        prev_stage_name="RunValidation",
        metrics_factory=validation_metrics_factory,
        required_keys=["fitness", "is_valid"],
        timeout=15.0,  # OPTIMIZED: Reduced for fast metric updates
    )

    # Stage 5b: Complexity metrics factory (FAST - 15s timeout)
    def complexity_metrics_factory() -> Dict[str, Any]:
        """Factory function that creates default complexity metrics when complexity computation fails."""
        return {
            "complexity_score": 0.0,
            "negative_complexity_score": 0.0,
            "total_nodes": 0,
            "max_depth": 0,
            "loop_count": 0,
            "call_count": 0,
            "unique_identifiers": 0,
            "binop_count": 0,
            "subscript_count": 0,
            "condition_count": 0,
            "function_def_count": 0,
            "class_def_count": 0,
            "ast_entropy": 0,
        }

    stages["ComplexityMetricUpdate"] = lambda: FactoryMetricsStage(
        stage_name="ComplexityMetricUpdate",
        prev_stage_name="ComputeComplexity",
        metrics_factory=complexity_metrics_factory,
        required_keys=[
            "complexity_score",
            "negative_complexity_score",
            "ast_entropy",
            "loop_count",
            "total_nodes",
            "max_depth",
            "call_count",
            "unique_identifiers",
            "binop_count",
            "subscript_count",
            "condition_count",
            "function_def_count",
            "class_def_count",
        ],
        timeout=15.0,
    )

    return stages


def create_dag_edges(add_context: bool = False) -> Dict[str, List[str]]:
    """Define the DAG execution dependencies."""
    base_dag =  {
        "ValidateCompiles": [
            "ExecuteCode",
            "ComputeComplexity",
        ],  # Both can run after validation
        "ExecuteCode": ["RunValidation"],
        "ComputeComplexity": [],  # Complexity feeds into insights
        "RunValidation": [],
        "LLMInsights": [],  # Terminal stage
        "LineageInsights": [],  # Terminal stage
        "ValidationMetricUpdate": [],  # Independent terminal stage
        "ComplexityMetricUpdate": [],  # Independent terminal stage - NO dependency on ValidationMetricUpdate
    }
    if add_context:
        # first create code, then everything else
        base_dag["AddContext"] = ["ValidateCompiles",]
    return base_dag


def create_execution_order_deps() -> Dict[str, List[ExecutionOrderDependency]]:
    """Define execution order dependencies for stages that should run under special conditions."""
    return {
        # ValidationMetricUpdate should ALWAYS run after validation pipeline, regardless of success/failure
        "ValidationMetricUpdate": [
            ExecutionOrderDependency.always_after("ValidateCompiles"),
            ExecutionOrderDependency.always_after("ExecuteCode"),
            ExecutionOrderDependency.always_after("RunValidation"),
        ],
        # ComplexityMetricUpdate should ALWAYS run after complexity computation, regardless of success/failure
        "ComplexityMetricUpdate": [
            ExecutionOrderDependency.always_after("ValidateCompiles"),
            ExecutionOrderDependency.always_after("ComputeComplexity"),
        ],
        "LLMInsights": [
            ExecutionOrderDependency.always_after("ValidateCompiles"),
            ExecutionOrderDependency.always_after("ComputeComplexity"),
            ExecutionOrderDependency.always_after("ExecuteCode"),
            ExecutionOrderDependency.always_after("RunValidation"),
            ExecutionOrderDependency.always_after("ValidationMetricUpdate"),
        ],
        "LineageInsights": [
            ExecutionOrderDependency.always_after("ValidateCompiles"),
            ExecutionOrderDependency.always_after("ComputeComplexity"),
            ExecutionOrderDependency.always_after("ExecuteCode"),
            ExecutionOrderDependency.always_after("RunValidation"),
            ExecutionOrderDependency.always_after("ValidationMetricUpdate"),
        ],
    }


async def create_evolution_strategy(
    redis_storage: RedisProgramStorage,
    args: argparse.Namespace,
) -> MapElitesMultiIsland:
    """Create the multi-island MAP-Elites evolution strategy with standard behavior spaces."""

    behavior_spaces = create_behavior_spaces(args)
    island_configs = create_island_configs(behavior_spaces)

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
            "temperature": 0.65,
            "max_tokens": 4096,
            "top_p": 0.9,
        },
        "lineage": {
            "temperature": 0.6,
            "max_tokens": 8192,
            "top_p": 0.85,
        },
        "mutation": {
            "temperature": 0.75,
            "max_tokens": 10000,
            "top_p": 0.85,
        },
    }

    def build_wrapper_with_params(
        params: dict[str, float],
    ) -> MultiModelLLMWrapper:
        return MultiModelLLMWrapper(
            models=[
                "google/gemini-2.5-flash:nitro",       # fast, good for insights
                "anthropic/claude-sonnet-4:nitro",         # deep context for lineage/mutation
            ],
            probabilities=[0.8, 0.2],  # favor Gemini unless structure is needed
            api_key=LLM_API_KEY,
            configs=[
                LLMConfig(
                    **params,
                    api_endpoint="https://openrouter.ai/api/v1",
                ),
                LLMConfig(
                    **params,
                    api_endpoint="https://openrouter.ai/api/v1",
                ),
            ],
        )

    res = {
        stage: build_wrapper_with_params(params)
        for stage, params in settings_per_stage.items()
    }
    return res


async def run_evolution_experiment(args: argparse.Namespace):
    """Run the complete evolution experiment with provided configuration."""

    start_time = time.time()
    problem_dir = Path(args.problem_dir)

    logger.info("🔄 Starting MetaEvolve Evolution Experiment")
    logger.info(f"📁 Problem directory: {problem_dir}")
    logger.info(f"📁 Log file: {log_file_path}")
    logger.info(f"🕐 Start time: {datetime.now(timezone.utc).isoformat()}")

    # Validate problem directory
    try:
        validate_problem_directory(problem_dir, args.add_context)
        logger.info("✅ Problem directory validated")
    except Exception as e:
        logger.error(f"❌ Problem directory validation failed: {e}")
        return

    # Setup Redis storage
    redis_storage = RedisProgramStorage(
        {
            "redis_url": f"redis://{args.redis_host}:{args.redis_port}/{args.redis_db}",
            "key_prefix": f"{problem_dir.name}_evolution",
            "max_connections": 150,
            "connection_pool_timeout": 45.0,
            "health_check_interval": 120,
            "max_retries": 5,
            "retry_delay": 0.5,
        }
    )

    try:
        # Clear the target database to start fresh
        logger.info(
            f"🧹 Clearing Redis database {args.redis_db} for restart..."
        )
        redis_conn = await redis_storage._conn()
        await redis_conn.flushdb()
        logger.info(f"✓ Redis database {args.redis_db} cleared")

        # Initialize new DB with initial programs
        if args.use_redis_selection:
            logger.info("🔍 Initializing database with selected programs from Redis...")
            programs = await select_top_programs_from_redis(
                redis_storage=redis_storage,
                source_redis_host=args.redis_host,
                source_redis_port=args.redis_port,
                source_redis_db=args.source_redis_db,
                problem_dir=problem_dir,
                top_n=args.top_n,
            )
        else:
            logger.info("🌱 Initializing database with initial programs...")
            programs = await create_initial_population(redis_storage, problem_dir)

        task_description = load_problem_file(
            problem_dir, "task_description.txt"
        )
        task_hints = load_problem_file(problem_dir, "task_hints.txt")

        # Setup LLM
        logger.info("Setting up LLM wrapper...")
        llm_wrapper = await setup_llm_wrapper()

        # Create DAG pipeline
        logger.info("Creating DAG pipeline...")
        dag_stages = create_dag_stages(
            llm_wrapper, redis_storage, task_description, problem_dir, args.add_context
        )
        dag_edges = create_dag_edges(args.add_context)
        execution_order_deps = create_execution_order_deps()
        entry_points = ["ValidateCompiles"]

        # Create evolution strategy
        logger.info("Creating evolution strategy...")
        evolution_strategy = await create_evolution_strategy(redis_storage, args)

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
            max_parents=1,
            task_definition=task_description,
            task_hints=task_hints,
            system_prompt_template=load_problem_file(
                problem_dir, "mutation_system_prompt.txt"
            ),
            user_prompt_template_rewrite=load_problem_file(
                problem_dir, "mutation_user_prompt.txt"
            ),
            metric_descriptions={
                "fitness": "main objective, higher is better",
                "is_valid": "whether the program is valid, 1 if valid, 0 if invalid",
            },
        )
        required_behavior_keys = set()
        for island in evolution_strategy.islands.values():
            required_behavior_keys |= set(island.config.behavior_space.behavior_keys)

        # TODO refactor? Create an abstraction for `function filter` which drops unsuitable programs; e.g. when metrics are missing
        logger.info("Creating evolution engine...")

        engine_config = EngineConfig(
            loop_interval=1.0,
            max_elites_per_generation=6,  # INCREASED: More elites for better diversity preservation
            max_mutations_per_generation=8,  # INCREASED: More mutations per generation for faster exploration
            required_behavior_keys=required_behavior_keys,
            log_validation_failures=True,
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
            max_concurrent_dags=args.max_concurrent_dags,
            log_interval=15,
            dag_timeout=2400,
        )

        runner = RunnerManager(
            engine=evolution_engine,
            dag_spec=DAGSpec(
                nodes=dag_stages,
                edges=dag_edges,
                entry_points=entry_points,
                exec_order_deps=execution_order_deps,
                dag_timeout=2400,
                max_parallel_stages=3,
            ),
            storage=redis_storage,
            config=runner_config,
        )

        logger.info("🎯 Starting evolution run...")
        logger.info("Configuration:")
        logger.info(f"  - Problem directory: {problem_dir}")
        logger.info(f"  - Target DB: {args.redis_db}")
        logger.info(f"  - Initial population: {len(programs)} programs")
        logger.info(f"  - DAG stages: {list(dag_stages.keys())}")

        # Start performance monitoring
        if not args.disable_monitoring:
            logger.info("🔍 Performance monitoring enabled")
        else:
            logger.info("🔍 Performance monitoring disabled")

        runner._dag_spec.visualize(save_path=Path(args.log_dir) / "dag_visualization.png")

        # Run the complete system with performance monitoring
        monitor_task = None

        if not args.disable_monitoring:

            async def monitor_and_optimize():
                """Background task for performance monitoring and auto-optimization."""
                while True:
                    try:
                        await asyncio.sleep(60)  # Check every minute

                        # Auto-optimize if needed
                        optimizations_applied = await auto_optimize()
                        if optimizations_applied > 0:
                            logger.info(
                                f"🔧 Applied {optimizations_applied} automatic optimizations"
                            )

                        # Log performance dashboard every 5 minutes
                        if int(time.time()) % 300 == 0:
                            dashboard = await get_performance_dashboard()
                            logger.info(
                                f"📊 Performance: {dashboard['health']['status']} "
                                f"(score: {dashboard['health']['performance_score']:.1f}/100)"
                            )

                            if dashboard["alerts"]:
                                logger.warning(
                                    f"⚠️ System alerts: {len(dashboard['alerts'])} active"
                                )
                                for alert in dashboard["alerts"]:
                                    logger.warning(
                                        f"  - {alert['level']}: {alert['message']}"
                                    )

                    except Exception as e:
                        logger.error(f"Performance monitoring error: {e}")

            # Start monitoring task
            monitor_task = asyncio.create_task(monitor_and_optimize())

        try:
            await runner.run()
        finally:
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

    except KeyboardInterrupt:
        logger.info("🛑 Evolution experiment interrupted by user")
    except Exception as e:
        logger.error(f"❌ Evolution experiment failed: {e}")
        raise
    finally:
        # Improved cleanup with connection pool closure
        logger.info("🧹 Starting cleanup...")
        try:
            redis_conn = await redis_storage._conn()
            if redis_conn:
                # Close the connection pool properly
                if hasattr(redis_conn, "connection_pool"):
                    await redis_conn.connection_pool.disconnect()
                    logger.info("✓ Redis connection pool closed")
                if hasattr(redis_conn, "close"):
                    await redis_conn.close()
                    logger.info("✓ Redis connection closed")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Redis cleanup warning: {cleanup_error}")
        logger.info("🧹 Cleanup completed")

        # Log experiment completion
        duration = time.time() - start_time
        logger.info(
            f"⏱️ Total experiment duration: {duration:.2f} seconds ({duration/3600:.2f} hours)"
        )
        logger.info(f"🕐 End time: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Setup logging based on arguments
    if args.verbose:
        log_level = "DEBUG"
    else:
        log_level = args.log_level

    # Reconfigure logging with user preferences
    log_file_path = setup_logger(
        log_dir=args.log_dir,
        level=log_level,
        rotation="50 MB",
        retention="30 days",
    )

    # Check prerequisites
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY environment variable must be set")
        exit(1)

    problem_dir = Path(args.problem_dir)
    if not problem_dir.exists():
        logger.error(f"❌ Problem directory not found: {problem_dir}")
        exit(1)

    # Validate problem directory structure
    try:
        validate_problem_directory(problem_dir)
    except Exception as e:
        logger.error(f"❌ Problem directory validation failed: {e}")
        exit(1)

    # Run the evolution experiment
    asyncio.run(run_evolution_experiment(args))
