"""Ships-off pin for the novelty-admission gate.

The A/B isolating the gate's causal effect has no verdict yet, so the
``MemoryWriter`` code default must stay false; flipping it is a deliberate
decision, not a drive-by.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gigaevo.memory.selection_leases import SharedSelectionRegistry
from gigaevo.memory.write.writer import MemoryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_memory_writer_defaults_novelty_gate_off():
    parameters = inspect.signature(MemoryWriter.__init__).parameters
    assert parameters["novelty_admission_gate"].default is False


def test_memory_writer_task_key_defaults_to_empty():
    parameters = inspect.signature(MemoryWriter.__init__).parameters
    assert parameters["task_key"].default == ""


def test_composed_config_instantiates_shared_selection_registry(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str((_REPO_ROOT / "config").absolute()), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=["problem.name=_test_", f"checkpoint_dir={tmp_path}"],
        )

    registry = instantiate(cfg.selection_leases)

    assert isinstance(registry, SharedSelectionRegistry)
    assert registry._path == tmp_path / "selection_leases.json"


def test_no_memory_provider_and_writer_targets_stay_legacy_noops():
    memory = OmegaConf.load(_REPO_ROOT / "config" / "memory" / "none.yaml")

    assert memory.provider._target_ == "gigaevo.memory.provider.NullMemoryProvider"
    assert memory.writer._target_ == "gigaevo.evolution.engine.hooks.NullPostRunHook"
