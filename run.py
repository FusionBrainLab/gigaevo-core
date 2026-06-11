import asyncio
from pathlib import Path
import time

from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gigaevo.config.resolvers import register_resolvers
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.evolution.engine import EvolutionEngine
from gigaevo.monitoring.emit import (
    configure_event_counters_from_cfg,
    reset_event_counters,
)
from gigaevo.monitoring.eta_ticker import start_eta_ticker
from gigaevo.monitoring.live_frontier_compare import (
    _fetch_histories,
    _render_frontier_plot,
    start_live_frontier_compare,
)
from gigaevo.monitoring.live_profiler import _render_once, start_live_profiler
from gigaevo.problems.initial_loaders import InitialProgramLoader
from gigaevo.programs.stages.python_executors.wrapper import default_exec_runner_pool
from gigaevo.runner.dag_runner import DagRunner
from gigaevo.utils.experiment import check_storage_resume
from gigaevo.utils.logger_setup import setup_logger
from gigaevo.utils.serve import serve_until_signal
from gigaevo.utils.trackers.base import LogWriter


async def run_experiment(cfg: DictConfig) -> None:
    start_time = time.time()
    logger.info("GigaEvo — Problem: {}", cfg.problem.name)

    storage: ProgramStorage | None = None
    writer: LogWriter | None = None
    try:
        config_with_instances = instantiate(cfg, recursive=True)
        storage: ProgramStorage = config_with_instances.program_storage
        program_loader: InitialProgramLoader = config_with_instances.program_loader
        dag_runner: DagRunner = config_with_instances.dag_runner
        evolution_engine: EvolutionEngine = config_with_instances.evolution_engine
        writer: LogWriter = config_with_instances.writer

        logger.info(
            "Redis DB {} at {}:{}", cfg.redis.db, cfg.redis.host, cfg.redis.port
        )
        configure_event_counters_from_cfg(cfg)

        await storage.acquire_instance_lock()

        resume = await check_storage_resume(
            storage,
            resume=bool(cfg.redis.get("resume", False)),
            location=f"Redis DB {cfg.redis.db} at {cfg.redis.host}:{cfg.redis.port}",
            flush_hint=f"gigaevo flush --db {cfg.redis.db} --confirm",
        )
        if resume:
            recovered = await storage.recover_stranded_programs()
            if recovered:
                logger.info("Recovered {} stranded RUNNING program(s)", recovered)
            await evolution_engine.restore_state()
            await evolution_engine.strategy.restore_state()
            logger.info(
                "Resumed with {} existing programs",
                await storage.size(),
            )
        else:
            programs = await program_loader.load(storage)
            # Seeds occupy ordinals 0..N-1 (set by the loader); the engine's
            # next ordinal to hand out to the first mutant is therefore N.
            # Without this bootstrap the first mutant collides with seed 0.
            evolution_engine.metrics.iteration = len(programs)
            logger.info(
                "Loaded {} initial programs (next_iteration={})",
                len(programs),
                evolution_engine.metrics.iteration,
            )

        dag_runner.start()
        evolution_engine.start()
        logger.info("Evolution running (max_mutants={})", cfg.max_mutants)
        start_eta_ticker(
            evolution_engine,
            interval_s=float(cfg.live_profiler.interval_s),
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
        reset_event_counters()
        await default_exec_runner_pool().shutdown()
        if storage is not None:
            await storage.close()
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
    last_n_raw = int(cfg.live_profiler.last_n)
    last_n: int | None = last_n_raw if last_n_raw > 0 else None
    start_live_profiler(
        log_file_path,
        output_dir,
        interval_s=float(cfg.live_profiler.interval_s),
        last_n=last_n,
    )
    frontier_ctx = _maybe_start_live_frontier_compare(cfg, output_dir)
    try:
        asyncio.run(run_experiment(cfg))
    finally:
        _finalize_live_artifacts(
            log_file_path=log_file_path,
            output_dir=output_dir,
            last_n=last_n,
            frontier_ctx=frontier_ctx,
        )


def _maybe_start_live_frontier_compare(
    cfg: DictConfig, output_dir: Path
) -> dict | None:
    """Wire ``cfg.live_frontier_compare`` to the daemon entry point.

    Returns a context dict ``{redis_url, key_prefix, metrics,
    higher_is_better}`` describing what the periodic thread is rendering,
    so the end-of-run finalizer can issue one more synchronous render
    against the same Redis state. Returns ``None`` when the group is
    missing or disabled.
    """
    lfc = cfg.get("live_frontier_compare") if hasattr(cfg, "get") else None
    if lfc is None:
        return None
    if not bool(lfc.get("enabled", True)):
        return None

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
    return {
        "redis_url": redis_url,
        "key_prefix": key_prefix,
        "metrics": metrics,
        "higher_is_better": higher_is_better,
    }


def _finalize_live_artifacts(
    *,
    log_file_path: Path,
    output_dir: Path,
    last_n: int | None,
    frontier_ctx: dict | None,
) -> None:
    """Render profiler HTML and frontier PNGs once more before exit.

    The periodic daemons re-render every ~60s, so artifacts on disk can
    lag the run's final iteration by up to one interval. Calling the
    same renderers here guarantees the run's last state lands on disk.
    Each artifact is best-effort: a failure is logged and shutdown
    continues.
    """
    try:
        html_path = output_dir / "profile_live.html"
        n_prog, n_llm = _render_once(
            Path(log_file_path), html_path, "final", last_n=last_n
        )
        logger.info(
            "[finalize] {} rendered ({} programs, {} LLM events)",
            html_path,
            n_prog,
            n_llm,
        )
    except Exception:
        logger.opt(exception=True).warning("[finalize] profile_live.html render failed")

    if frontier_ctx is None:
        return
    try:
        import redis as redis_lib

        client = redis_lib.Redis.from_url(
            frontier_ctx["redis_url"], decode_responses=True
        )
        frontier, iter_mean, _ = _fetch_histories(
            client, frontier_ctx["key_prefix"], frontier_ctx["metrics"]
        )
    except Exception:
        logger.opt(exception=True).warning(
            "[finalize] frontier Redis fetch failed; skipping plot finalize"
        )
        return
    for m in frontier_ctx["metrics"]:
        try:
            out = _render_frontier_plot(
                output_dir=output_dir,
                metric=m,
                frontier_history=frontier.get(m, []),
                iter_mean_history=iter_mean.get(m, []),
                higher_is_better=frontier_ctx["higher_is_better"].get(m, True),
            )
            if out is not None:
                logger.info("[finalize] {} rendered", out)
        except Exception:
            logger.opt(exception=True).warning(
                "[finalize] frontier_{} render failed", m
            )


if __name__ == "__main__":
    register_resolvers()
    main()
