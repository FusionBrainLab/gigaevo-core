from gigaevo.utils.trackers.backends.redis import RedisMetricsBackend
from gigaevo.utils.trackers.backends.tensorboard import TBBackend
from gigaevo.utils.trackers.backends.wandb import WandBBackend
from gigaevo.utils.trackers.composite import CompositeLogger
from gigaevo.utils.trackers.configs import RedisMetricsConfig, TBConfig, WBConfig
from gigaevo.utils.trackers.core import GenericLogger


def init_tb(
    cfg: TBConfig, *, queue_size: int = 8192, flush_secs: float = 3.0
) -> GenericLogger:
    backend = TBBackend(cfg)
    return GenericLogger(backend, queue_size=queue_size, flush_secs=flush_secs)


def init_wandb(
    cfg: WBConfig, *, queue_size: int = 8192, flush_secs: float = 3.0
) -> GenericLogger:
    backend = WandBBackend(cfg)
    return GenericLogger(backend, queue_size=queue_size, flush_secs=flush_secs)


def init_redis(
    cfg: RedisMetricsConfig, *, queue_size: int = 8192, flush_secs: float = 3.0
) -> GenericLogger:
    """Build a Redis metrics logger.

    Returns a GenericLogger wrapping RedisMetricsBackend. The backend is
    available via ``logger.backend`` for direct query methods.
    """
    backend = RedisMetricsBackend(cfg)
    return GenericLogger(backend, queue_size=queue_size, flush_secs=flush_secs)


def init_composite(*loggers: GenericLogger) -> CompositeLogger:
    """Create a composite logger that fans out writes to every backend.

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
    """Composite logger that fans writes to TensorBoard and Redis."""
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
    """Composite logger that fans writes to WandB and Redis."""
    wb = init_wandb(wandb_cfg, queue_size=queue_size, flush_secs=flush_secs)
    redis_logger = init_redis(redis_cfg, queue_size=queue_size, flush_secs=flush_secs)
    return CompositeLogger([wb, redis_logger])
