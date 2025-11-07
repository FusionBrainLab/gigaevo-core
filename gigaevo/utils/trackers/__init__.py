from gigaevo.utils.trackers.backends.tensorboard import TBBackend
from gigaevo.utils.trackers.backends.wandb import WandBBackend
from gigaevo.utils.trackers.configs import TBConfig, WBConfig
from gigaevo.utils.trackers.core import GenericLogger

_tb_default: GenericLogger | None = None
_wb_default: GenericLogger | None = None


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
