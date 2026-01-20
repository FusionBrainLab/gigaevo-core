from gigaevo.utils.trackers.backends.redis import RedisMetricsBackend
from gigaevo.utils.trackers.backends.tensorboard import TBBackend
from gigaevo.utils.trackers.backends.wandb import WandBBackend
from gigaevo.utils.trackers.composite import CompositeLogger
from gigaevo.utils.trackers.configs import RedisMetricsConfig, TBConfig, WBConfig
from gigaevo.utils.trackers.core import GenericLogger

_tb_default: GenericLogger | None = None
_wb_default: GenericLogger | None = None
_redis_default: GenericLogger | None = None


def init_tb(
    cfg: TBConfig, *, queue_size: int = 8192, flush_secs: float = 3.0
) -> GenericLogger:
    global _tb_default
    if _tb_default is not None:
        return _tb_default
    backend = TBBackend(cfg)
    _tb_default = GenericLogger(backend, queue_size=queue_size, flush_secs=flush_secs)
    return _tb_default


def get_tb() -> GenericLogger:
    if _tb_default is None:
        raise ValueError("TBLogger not initialized. Call init_tb() first.")
    return _tb_default


def init_wandb(
    cfg: WBConfig, *, queue_size: int = 8192, flush_secs: float = 3.0
) -> GenericLogger:
    global _wb_default
    if _wb_default is not None:
        return _wb_default
    backend = WandBBackend(cfg)
    _wb_default = GenericLogger(backend, queue_size=queue_size, flush_secs=flush_secs)
    return _wb_default


def get_wandb() -> GenericLogger:
    if _wb_default is None:
        raise ValueError("WandBLogger not initialized. Call init_wandb() first.")
    return _wb_default


def init_redis(
    cfg: RedisMetricsConfig, *, queue_size: int = 8192, flush_secs: float = 3.0
) -> GenericLogger:
    """Initialize Redis metrics logger.

    Returns a GenericLogger wrapping RedisMetricsBackend.
    Access the backend directly via logger.backend for query methods.
    """
    global _redis_default
    if _redis_default is not None:
        return _redis_default
    backend = RedisMetricsBackend(cfg)
    _redis_default = GenericLogger(
        backend, queue_size=queue_size, flush_secs=flush_secs
    )
    return _redis_default


def get_redis() -> GenericLogger:
    if _redis_default is None:
        raise ValueError("RedisLogger not initialized. Call init_redis() first.")
    return _redis_default


def get_redis_backend() -> RedisMetricsBackend:
    """Get the Redis backend directly for query methods."""
    logger = get_redis()
    return logger.backend  # type: ignore


def init_composite(*loggers: GenericLogger) -> CompositeLogger:
    """Create a composite logger that writes to multiple backends.

    Example:
        >>> tb = init_tb(tb_config)
        >>> redis = init_redis(redis_config)
        >>> writer = init_composite(tb, redis)
        >>> writer.scalar("loss", 0.5)  # writes to both
    """
    return CompositeLogger(list(loggers))


def init_tb_redis(
    tb_cfg: TBConfig,
    redis_cfg: RedisMetricsConfig,
    *,
    queue_size: int = 8192,
    flush_secs: float = 3.0,
) -> CompositeLogger:
    """Initialize composite logger with TensorBoard + Redis backends.

    Hydra usage:
        writer:
          _target_: gigaevo.utils.trackers.init_tb_redis
          tb_cfg: ${tb_config}
          redis_cfg: ${redis_metrics_config}
    """
    tb = init_tb(tb_cfg, queue_size=queue_size, flush_secs=flush_secs)
    redis_logger = init_redis(redis_cfg, queue_size=queue_size, flush_secs=flush_secs)
    return CompositeLogger([tb, redis_logger])


def init_wandb_redis(
    wandb_cfg: WBConfig,
    redis_cfg: RedisMetricsConfig,
    *,
    queue_size: int = 8192,
    flush_secs: float = 3.0,
) -> CompositeLogger:
    """Initialize composite logger with WandB + Redis backends.

    Hydra usage:
        writer:
          _target_: gigaevo.utils.trackers.init_wandb_redis
          wandb_cfg: ${wandb_config}
          redis_cfg: ${redis_metrics_config}
    """
    wb = init_wandb(wandb_cfg, queue_size=queue_size, flush_secs=flush_secs)
    redis_logger = init_redis(redis_cfg, queue_size=queue_size, flush_secs=flush_secs)
    return CompositeLogger([wb, redis_logger])
