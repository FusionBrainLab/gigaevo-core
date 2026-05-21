import asyncio
from pathlib import Path
import time

from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gigaevo.config.resolvers import register_resolvers
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.engine import EvolutionEngine
from gigaevo.monitoring.emit import (
    configure_event_counters_from_cfg,
    reset_event_counters,
)
from gigaevo.monitoring.live_frontier_compare import start_live_frontier_compare
from gigaevo.monitoring.live_profiler import start_live_profiler
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
            "Redis DB {} at {}:{}", cfg.redis.db, cfg.redis.host, cfg.redis.port
        )
        configure_event_counters_from_cfg(cfg)

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
        logger.info("Evolution running (max_mutants={})", cfg.max_mutants)

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
        reset_event_counters()
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
    output_dir = Path(hydra_config.output_dir)
    logger.info("Output dir: {} | Log: {}", output_dir, log_file_path)
    last_n = int(cfg.live_profiler.last_n)
    start_live_profiler(
        log_file_path,
        output_dir,
        interval_s=float(cfg.live_profiler.interval_s),
        last_n=last_n if last_n > 0 else None,
    )
    _maybe_start_live_frontier_compare(cfg, output_dir)
    asyncio.run(run_experiment(cfg))


def _maybe_start_live_frontier_compare(cfg: DictConfig, output_dir: Path) -> None:
    """Wire ``cfg.live_frontier_compare`` to the daemon entry point.

    The cfg group is optional — older configs may not declare it. We
    silently skip when missing so this never breaks an existing run.
    """
    lfc = cfg.get("live_frontier_compare") if hasattr(cfg, "get") else None
    if lfc is None:
        return
    if not bool(lfc.get("enabled", True)):
        return

    # Resolve higher_is_better per metric from problems/<name>/metrics.yaml.
    import yaml

    metrics_yaml = Path(cfg.problem.dir) / "metrics.yaml"
    higher_is_better: dict[str, bool] = {}
    if metrics_yaml.exists():
        with open(metrics_yaml) as f:
            data = yaml.safe_load(f) or {}
        for name, spec in (data.get("specs") or {}).items():
            if isinstance(spec, dict):
                higher_is_better[name] = bool(spec.get("higher_is_better", True))

    frontier_source = str(lfc.get("frontier_source", "hof"))
    if frontier_source != "hof":
        logger.warning(
            "[live_frontier_compare] frontier_source={!r} not yet "
            "implemented; falling back to 'hof'.",
            frontier_source,
        )

    redis_url = f"redis://{cfg.redis.host}:{cfg.redis.port}/{cfg.redis.db}"
    # The metrics tracker writes under "${problem.name}:metrics" — same as
    # the RedisMetricsConfig.key_prefix in config/logging/{tensorboard,wandb}.yaml.
    key_prefix = f"{cfg.problem.name}:metrics"
    metrics = [str(m) for m in lfc.get("metrics", ["fitness"])]
    emit_targets = [str(t) for t in lfc.get("emit_targets", ["log"])]

    start_live_frontier_compare(
        redis_url=redis_url,
        key_prefix=key_prefix,
        metrics=metrics,
        higher_is_better=higher_is_better,
        interval_s=float(lfc.get("interval_s", 60.0)),
        emit_targets=emit_targets,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    register_resolvers()
    main()
