"""Hydra composition test for the post_step_hook wall-clock budget knob.

``post_step_hook_timeout_s`` bounds a single ``post_step_hook`` invocation.
Live memory writes need a larger budget than memory-free runs because they add
network-bound LLM enrichment. These tests pin configuration wiring and that
relative recipe contract without duplicating the tunable timeout values.

Resolution only — no instantiation (that needs a real Redis).
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

_BASE_OVERRIDES = [
    "problem.name=_test_",
    "algorithm=multi_island",
    "evolution=steady_state",
]


def _compose(*overrides: str):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(
            config_name="config", overrides=_BASE_OVERRIDES + list(overrides)
        )


def test_default_timeout_propagates_to_engine_config():
    cfg = _compose()
    assert cfg.engine_config.post_step_hook_timeout_s == cfg.post_step_hook_timeout_s


def test_memory_free_timeout_propagates_to_engine_config():
    cfg = _compose("memory=none")
    assert cfg.engine_config.post_step_hook_timeout_s == cfg.post_step_hook_timeout_s


def test_cli_override_propagates():
    configured = _compose("memory=none")
    override_timeout = float(configured.post_step_hook_timeout_s) * 2
    cfg = _compose(f"post_step_hook_timeout_s={override_timeout}")
    assert cfg.engine_config.post_step_hook_timeout_s == override_timeout


def test_live_memory_write_raises_budget_over_memory_free_run():
    memory_free = _compose("memory=none")
    live_memory = _compose("pipeline=memory_guided", "memory=v2", "memory/write=live")
    assert (
        live_memory.engine_config.post_step_hook_timeout_s
        > memory_free.engine_config.post_step_hook_timeout_s
    )
