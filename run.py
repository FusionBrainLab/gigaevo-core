import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
import time
from typing import Any

# Ensure NO_PROXY covers all internal LLM servers before any imports
# or subprocess spawns. The system Squid proxy intercepts traffic to
# IPs not listed in NO_PROXY.
from tools.no_proxy import ensure_no_proxy

ensure_no_proxy()

from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from gigaevo.config.resolvers import register_resolvers
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.engine import EvolutionEngine
from gigaevo.problems.initial_loaders import InitialProgramLoader
from gigaevo.programs.stages.python_executors.wrapper import default_exec_runner_pool
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
    logger.info(f"Start time: {datetime.now(UTC).isoformat()}")
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

        # Log resolved config for debugging
        logger.info("--- Resolved configuration ---")
        logger.info(
            f"  Pipeline builder: {cfg.get('pipeline_builder', {}).get('_target_', '(default)')}"
        )
        logger.info(f"  Redis DB: {cfg.redis.db} at {cfg.redis.host}:{cfg.redis.port}")
        _prompts_dir = cfg.get("evolution_context", {}).get("prompts_dir", None)
        logger.info(f"  Prompts dir: {_prompts_dir or '(package defaults)'}")
        logger.info(f"  Stage timeout: {cfg.get('stage_timeout', '(not set)')}s")
        logger.info(f"  DAG timeout: {cfg.get('dag_timeout', '(not set)')}s")
        logger.info(
            f"  Max mutations/gen: {cfg.get('max_mutations_per_generation', '(not set)')}"
        )
        logger.info(
            f"  Max elites/gen: {cfg.get('max_elites_per_generation', '(not set)')}"
        )
        logger.info(f"  Mutation mode: {cfg.get('mutation_mode', '(not set)')}")
        _fetcher = cfg.get("prompt_fetcher", {}).get("_target_", "(default)")
        logger.info(f"  Prompt fetcher: {_fetcher}")
        logger.info("--- End configuration ---")
        logger.info("")

        logger.info("Step 2/5: Checking Redis database and acquiring instance lock...")

        try:
            await redis_storage.acquire_instance_lock()
        except Exception as e:
            logger.error(f"Failed to acquire instance lock: {e}")
            raise RuntimeError(
                "Another instance is already running on this Redis prefix, "
                "or failed to acquire lock. See error above for details."
            ) from e

        # Safety check: prevent accidental data loss
        has_data = await redis_storage.has_data()
        resume = cfg.redis.get("resume", False)

        # If data exists and we are NOT resuming, this is an error.
        if has_data and not resume:
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

Or set resume=true to continue with existing data:
  python run.py redis.resume=true ...
"""
            logger.error(error_msg)
            raise RuntimeError(
                f"Redis database {db_num} is not empty. Flush manually to proceed."
            )

        if has_data and resume:
            logger.info(
                f"Resuming experiment on database {cfg.redis.db} (found existing data)"
            )
        elif resume:
            logger.info(
                f"Resume requested but database {cfg.redis.db} is empty. Starting fresh."
            )

        logger.info("Step 2/5: Database check complete and instance lock acquired")
        logger.info("")

        logger.info("Step 3/5: Loading programs...")
        # Determine whether to load from existing Redis data or run the initial loader
        should_resume = has_data and resume

        if should_resume:
            # Recover programs stuck in RUNNING state from prior kill/crash
            recovered = await redis_storage.recover_stranded_programs()
            if recovered:
                logger.info(
                    f"Step 3/5: Recovered {recovered} stranded RUNNING program(s) → QUEUED"
                )
            # Restore in-memory counters (generation, migration schedule)
            await evolution_engine.restore_state()
            await evolution_engine.strategy.restore_state()

            program_count = await redis_storage.size()
            logger.info(
                f"Step 3/5: Resumed with {program_count} existing programs from Redis"
            )
        else:
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
        await default_exec_runner_pool().shutdown()
        if redis_storage is not None:
            await redis_storage.close()
        if writer is not None:
            writer.close()
        duration = time.time() - start_time
        logger.info(
            f"Total experiment duration: {duration:.2f} seconds ({duration / 3600:.2f} hours)"
        )
        logger.info(f"End time: {datetime.now(UTC).isoformat()}")
        logger.info("=" * 80)


def _load_memory_config(memory_config_path: Path) -> dict[str, Any]:
    if not memory_config_path.is_file():
        return {}
    payload = OmegaConf.to_container(OmegaConf.load(memory_config_path), resolve=False)
    if not isinstance(payload, dict):
        return {}
    return payload


def _ensure_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    payload[key] = {}
    return payload[key]


def _write_memory_config(memory_config_path: Path, payload: dict[str, Any]) -> None:
    OmegaConf.save(config=OmegaConf.create(payload), f=memory_config_path)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _resolve_checkpoint_dir_arg(cfg: DictConfig, runtime_cwd: Path) -> Path | None:
    raw_checkpoint_dir = cfg.get("checkpoint_dir")
    if raw_checkpoint_dir is None:
        return None
    text = str(raw_checkpoint_dir).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = runtime_cwd / candidate
    return candidate.resolve()


def _build_runtime_memory_config(
    cfg: DictConfig,
    output_dir: Path,
    requested_checkpoint_dir: Path | None,
) -> tuple[Path, bool, Path | None]:
    project_root = Path(__file__).resolve().parent
    default_memory_config_path = project_root / "config" / "memory.yaml"
    runtime_memory_config_path = output_dir / "memory.runtime.yaml"

    payload = _load_memory_config(default_memory_config_path)

    ideas_tracker_cfg = _ensure_mapping(payload, "ideas_tracker")
    memory_write_cfg = ideas_tracker_cfg.get("memory_write_pipeline", False)
    if isinstance(memory_write_cfg, dict):
        memory_write_enabled = _to_bool(memory_write_cfg.get("enabled"))
    else:
        memory_write_enabled = _to_bool(memory_write_cfg)

    redis_cfg = _ensure_mapping(ideas_tracker_cfg, "redis")
    redis_cfg["redis_host"] = str(cfg.redis.host)
    redis_cfg["redis_port"] = int(cfg.redis.port)
    redis_cfg["redis_db"] = int(cfg.redis.db)
    redis_cfg["redis_prefix"] = str(cfg.problem.name)

    applied_checkpoint_dir: Path | None = None
    if memory_write_enabled and requested_checkpoint_dir is not None:
        requested_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        paths_cfg = _ensure_mapping(payload, "paths")
        paths_cfg["checkpoint_dir"] = str(requested_checkpoint_dir)
        applied_checkpoint_dir = requested_checkpoint_dir

    _write_memory_config(runtime_memory_config_path, payload)
    return runtime_memory_config_path, memory_write_enabled, applied_checkpoint_dir


def run_ideas_tracker(cfg: DictConfig, output_dir: Path, runtime_cwd: Path) -> None:
    requested_checkpoint_dir = _resolve_checkpoint_dir_arg(cfg, runtime_cwd)
    runtime_memory_config_path, memory_write_enabled, applied_checkpoint_dir = (
        _build_runtime_memory_config(
            cfg,
            output_dir,
            requested_checkpoint_dir,
        )
    )
    previous_config_path = os.environ.get("EVO_MEMORY_CONFIG_PATH")
    os.environ["EVO_MEMORY_CONFIG_PATH"] = str(runtime_memory_config_path)

    logger.info("Ideas tracker enabled. Config: {}", runtime_memory_config_path)
    if memory_write_enabled and applied_checkpoint_dir is not None:
        logger.info(
            "Memory write checkpoint directory override: {}",
            applied_checkpoint_dir,
        )
    elif memory_write_enabled:
        logger.info(
            "Memory write is enabled. Checkpoint directory is taken from config/memory.yaml."
        )
    elif requested_checkpoint_dir is not None:
        logger.info(
            "checkpoint_dir was provided but ignored because "
            "ideas_tracker.memory_write_pipeline.enabled=false."
        )

    try:
        from gigaevo.llm.ideas_tracker.ideas_tracker import IdeaTracker

        tracker = IdeaTracker(config_path=runtime_memory_config_path)
        tracker.run()
    finally:
        if previous_config_path is None:
            os.environ.pop("EVO_MEMORY_CONFIG_PATH", None)
        else:
            os.environ["EVO_MEMORY_CONFIG_PATH"] = previous_config_path


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entrypoint with Hydra configuration management."""
    load_dotenv()
    hydra_config = hydra.core.hydra_config.HydraConfig.get().runtime
    hydra_output_dir = Path(hydra_config.output_dir)
    hydra_runtime_cwd = Path(getattr(hydra_config, "cwd", os.getcwd()))

    log_file_path = setup_logger(
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
    )
    logger.info(
        "Experiment working directory: {}.",
        hydra_output_dir,
    )
    logger.info(f"Log file: {log_file_path}")
    asyncio.run(run_experiment(cfg))

    if bool(cfg.get("ideas_tracker", False)):
        run_ideas_tracker(cfg, hydra_output_dir, hydra_runtime_cwd)


if __name__ == "__main__":
    register_resolvers()
    main()
