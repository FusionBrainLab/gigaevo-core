"""Regression tests for the tracker factory functions.

Each ``init_*`` factory builds a fresh ``GenericLogger`` / ``CompositeLogger``
per call. There is no module-level memoization, so sequential calls in the
same process (sweeps, notebook re-runs, multirun plugins) get isolated
instances pointing at the configs they were handed.

The factories instantiate ``GenericLogger`` directly, which opens the
underlying backend (filesystem / Redis / W&B) on construction and spins up a
flusher thread. To exercise the factory plumbing in isolation, the test
suite monkeypatches each backend's ``open``/``close``/``flush`` to no-ops,
keeping the ``cfg`` plumbing intact so identity and config-routing claims
can still be asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import gigaevo.utils.trackers as trackers_module
from gigaevo.utils.trackers import (
    init_redis,
    init_tb,
    init_tb_redis,
    init_wandb,
    init_wandb_redis,
)
from gigaevo.utils.trackers.composite import CompositeLogger
from gigaevo.utils.trackers.configs import RedisMetricsConfig, TBConfig, WBConfig
from gigaevo.utils.trackers.core import GenericLogger


@pytest.fixture(autouse=True)
def _inert_backends(monkeypatch):
    """Replace backend network/filesystem side effects with no-ops."""

    def _noop(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    for backend_cls in (
        trackers_module.TBBackend,
        trackers_module.WandBBackend,
        trackers_module.RedisMetricsBackend,
    ):
        monkeypatch.setattr(backend_cls, "open", _noop)
        monkeypatch.setattr(backend_cls, "close", _noop)
        monkeypatch.setattr(backend_cls, "flush", _noop)

    # Suppress the background flusher thread; it would otherwise call
    # backend.flush() while the test is finishing.
    monkeypatch.setattr(GenericLogger, "_loop", lambda self: None)


@pytest.fixture
def tb_cfg(tmp_path: Path) -> TBConfig:
    return TBConfig(logdir=tmp_path / "tb_a")


@pytest.fixture
def tb_cfg_alt(tmp_path: Path) -> TBConfig:
    return TBConfig(logdir=tmp_path / "tb_b")


@pytest.fixture
def wb_cfg() -> WBConfig:
    return WBConfig(project="proj_a", name="run_a")


@pytest.fixture
def wb_cfg_alt() -> WBConfig:
    return WBConfig(project="proj_b", name="run_b")


@pytest.fixture
def redis_cfg() -> RedisMetricsConfig:
    return RedisMetricsConfig(redis_url="redis://x:6379/0", key_prefix="prefix_a")


@pytest.fixture
def redis_cfg_alt() -> RedisMetricsConfig:
    return RedisMetricsConfig(redis_url="redis://x:6379/0", key_prefix="prefix_b")


class TestFreshPerCall:
    """Each factory call returns a distinct instance."""

    def test_init_tb_returns_fresh_per_call(self, tb_cfg: TBConfig):
        a = init_tb(tb_cfg)
        b = init_tb(tb_cfg)
        assert a is not b
        assert isinstance(a, GenericLogger)
        assert isinstance(b, GenericLogger)
        assert a.backend is not b.backend

    def test_init_wandb_returns_fresh_per_call(self, wb_cfg: WBConfig):
        a = init_wandb(wb_cfg)
        b = init_wandb(wb_cfg)
        assert a is not b
        assert a.backend is not b.backend

    def test_init_redis_returns_fresh_per_call(self, redis_cfg: RedisMetricsConfig):
        a = init_redis(redis_cfg)
        b = init_redis(redis_cfg)
        assert a is not b
        assert a.backend is not b.backend


class TestMultirunSafety:
    """A second call with a different config does not leak the first's state."""

    def test_init_redis_second_call_honors_new_prefix(
        self,
        redis_cfg: RedisMetricsConfig,
        redis_cfg_alt: RedisMetricsConfig,
    ):
        first = init_redis(redis_cfg)
        second = init_redis(redis_cfg_alt)
        assert first is not second
        assert first.backend.cfg.key_prefix == "prefix_a"
        assert second.backend.cfg.key_prefix == "prefix_b"

    def test_init_tb_second_call_honors_new_logdir(
        self,
        tb_cfg: TBConfig,
        tb_cfg_alt: TBConfig,
    ):
        first = init_tb(tb_cfg)
        second = init_tb(tb_cfg_alt)
        assert first is not second
        assert first.backend.cfg.logdir == tb_cfg.logdir
        assert second.backend.cfg.logdir == tb_cfg_alt.logdir

    def test_init_wandb_second_call_honors_new_project(
        self,
        wb_cfg: WBConfig,
        wb_cfg_alt: WBConfig,
    ):
        first = init_wandb(wb_cfg)
        second = init_wandb(wb_cfg_alt)
        assert first is not second
        assert first.backend.cfg.project == "proj_a"
        assert second.backend.cfg.project == "proj_b"


class TestCompositeFactories:
    """Composite factories rebuild both backends every call."""

    def test_init_tb_redis_multirun_safety(
        self,
        tb_cfg: TBConfig,
        tb_cfg_alt: TBConfig,
        redis_cfg: RedisMetricsConfig,
        redis_cfg_alt: RedisMetricsConfig,
    ):
        first = init_tb_redis(tb_cfg, redis_cfg)
        second = init_tb_redis(tb_cfg_alt, redis_cfg_alt)
        assert isinstance(first, CompositeLogger)
        assert isinstance(second, CompositeLogger)
        assert first is not second
        first_loggers = list(first._loggers)
        second_loggers = list(second._loggers)
        assert all(a is not b for a, b in zip(first_loggers, second_loggers))
        tb_first, redis_first = first_loggers
        tb_second, redis_second = second_loggers
        assert tb_first.backend.cfg.logdir == tb_cfg.logdir
        assert tb_second.backend.cfg.logdir == tb_cfg_alt.logdir
        assert redis_first.backend.cfg.key_prefix == "prefix_a"
        assert redis_second.backend.cfg.key_prefix == "prefix_b"

    def test_init_wandb_redis_multirun_safety(
        self,
        wb_cfg: WBConfig,
        wb_cfg_alt: WBConfig,
        redis_cfg: RedisMetricsConfig,
        redis_cfg_alt: RedisMetricsConfig,
    ):
        first = init_wandb_redis(wb_cfg, redis_cfg)
        second = init_wandb_redis(wb_cfg_alt, redis_cfg_alt)
        assert first is not second
        wb_first, redis_first = list(first._loggers)
        wb_second, redis_second = list(second._loggers)
        assert wb_first.backend.cfg.project == "proj_a"
        assert wb_second.backend.cfg.project == "proj_b"
        assert redis_first.backend.cfg.key_prefix == "prefix_a"
        assert redis_second.backend.cfg.key_prefix == "prefix_b"
