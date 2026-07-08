"""Compatibility guards for memory read/write modes and pipeline DAGs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest

from gigaevo.config.validation import validate_memory_pipeline_compat

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_BASE = ["problem.name=_test_"]


def _compose(*overrides: str) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(config_name="config", overrides=[*_BASE, *overrides])


def test_reader_on_non_reading_pipeline_is_rejected() -> None:
    cfg = _compose("pipeline=guided", "memory=reader", "checkpoint_dir=/tmp/bank")

    with pytest.raises(ValueError, match="pipeline=guided|memory_guided"):
        validate_memory_pipeline_compat(cfg)


def test_memory_guided_requires_read_enabled_memory() -> None:
    cfg = _compose("pipeline=memory_guided", "memory=none")

    with pytest.raises(ValueError, match="reads external memory cards"):
        validate_memory_pipeline_compat(cfg)


def test_memory_guided_reader_is_allowed_with_explicit_bank() -> None:
    cfg = _compose("pipeline=memory_guided", "memory=reader", "checkpoint_dir=/tmp/bank")

    validate_memory_pipeline_compat(cfg)


def test_reader_requires_explicit_checkpoint_dir() -> None:
    cfg = _compose("pipeline=memory_guided", "memory=reader")

    with pytest.raises(ValueError, match="checkpoint_dir"):
        validate_memory_pipeline_compat(cfg)


def test_static_reader_does_not_need_writer_or_checkpoint_bank() -> None:
    cfg = _compose(
        "pipeline=memory_guided",
        "memory=static",
        "memory.provider.levers_file=/tmp/levers.md",
    )

    validate_memory_pipeline_compat(cfg)


def test_live_write_requires_writer_enabled_memory() -> None:
    cfg = _compose("pipeline=guided", "memory=none", "memory/write=live")

    with pytest.raises(ValueError, match="writer-enabled"):
        validate_memory_pipeline_compat(cfg)


def test_full_live_read_write_is_allowed() -> None:
    cfg = _compose("pipeline=memory_guided", "memory=full", "memory/write=live")

    validate_memory_pipeline_compat(cfg)


def test_live_write_installs_live_refresh_hook() -> None:
    cfg = _compose("pipeline=memory_guided", "memory=full", "memory/write=live")

    assert (
        cfg.post_step_hook._target_
        == "gigaevo.memory.live_memory_hook.LiveMemoryRefreshHook"
    )
    assert cfg.memory.write.mode == "live"
    assert cfg.engine_config.post_step_hook_timeout_s == 900.0


def test_pipeline_configs_expose_read_capability_metadata() -> None:
    guided = _compose("pipeline=guided")
    memory_guided = _compose("pipeline=memory_guided")

    assert guided.pipeline.id == "guided"
    assert guided.pipeline.reads_external_memory is False
    assert memory_guided.pipeline.id == "memory_guided"
    assert memory_guided.pipeline.reads_external_memory is True


def test_no_pipeline_config_defines_write_hook() -> None:
    """Write hooks belong to memory/write, not pipeline YAML."""

    for path in (_CONFIG_DIR / "pipeline").glob("*.yaml"):
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        assert isinstance(raw, dict)
        assert "post_step_hook" not in raw, path.name
        assert "post_step_hook_timeout_s" not in raw, path.name

