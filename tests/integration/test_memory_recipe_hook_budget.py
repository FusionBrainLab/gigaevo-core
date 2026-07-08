"""Hydra composition test for the post_step_hook wall-clock budget knob.

``post_step_hook_timeout_s`` bounds a single ``post_step_hook`` invocation.
The 300 s default was sized for the CPU-bound production
``CompositionInjectionHook`` archive walk; ``memory/write=live`` instead wires
the network-bound ``LiveMemoryRefreshHook`` (LLM enrichment), whose
per-increment latency balloons under shared-endpoint load and gets cancelled at
300 s. This test pins the contract:

* the global default resolves to 300 s on a plain steady-state run,
* a short-name Hydra CLI override propagates to ``engine_config``,
* the live memory write recipe raises the budget to 900 s.

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


def test_default_post_step_hook_timeout_is_300():
    cfg = _compose()
    assert cfg.engine_config.post_step_hook_timeout_s == 300.0


def test_cli_override_propagates():
    cfg = _compose("post_step_hook_timeout_s=1200")
    assert cfg.engine_config.post_step_hook_timeout_s == 1200.0


def test_live_memory_write_raises_budget():
    cfg = _compose("pipeline=memory_guided", "memory=full", "memory/write=live")
    assert cfg.engine_config.post_step_hook_timeout_s == 900.0
