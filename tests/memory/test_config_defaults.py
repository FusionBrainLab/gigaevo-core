"""Ships-off pin for the novelty-admission gate.

The A/B isolating the gate's causal effect has no verdict yet, so both the
shipped ``config/memory/full.yaml`` and the ``MemoryWriter`` code default must
stay false; flipping either is a deliberate decision, not a drive-by.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gigaevo.memory.read.reputation import BetaBinomialReputation
from gigaevo.memory.selection_leases import SharedSelectionRegistry
from gigaevo.memory.write.eviction import (
    BirthFailureEvictor,
    CompositeEvictor,
    CrossTaskRetentionGuard,
    HarmEvictor,
    PolicyNonViableEvictor,
)
from gigaevo.memory.write.writer import MemoryWriter
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose_full():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str((_REPO_ROOT / "config").absolute()), version_base=None
    ):
        return compose(
            config_name="config", overrides=["problem.name=_test_", "memory=full"]
        )


def test_full_yaml_ships_novelty_gate_off():
    cfg = OmegaConf.load(_REPO_ROOT / "config" / "memory" / "full.yaml")
    assert cfg.writer.novelty_admission_gate is False


def test_memory_writer_defaults_novelty_gate_off():
    parameters = inspect.signature(MemoryWriter.__init__).parameters
    assert parameters["novelty_admission_gate"].default is False


def test_memory_writer_task_key_defaults_to_empty():
    parameters = inspect.signature(MemoryWriter.__init__).parameters
    assert parameters["task_key"].default == ""


def test_full_memory_wires_one_shared_selection_registry():
    root = OmegaConf.to_container(
        OmegaConf.load(_REPO_ROOT / "config" / "config.yaml"), resolve=False
    )
    memory = OmegaConf.to_container(
        OmegaConf.load(_REPO_ROOT / "config" / "memory" / "full.yaml"),
        resolve=False,
    )
    default_engine = OmegaConf.to_container(
        OmegaConf.load(_REPO_ROOT / "config" / "evolution" / "default.yaml"),
        resolve=False,
    )
    steady_engine = OmegaConf.to_container(
        OmegaConf.load(_REPO_ROOT / "config" / "evolution" / "steady_state.yaml"),
        resolve=False,
    )

    assert (
        root["selection_leases"]["_target_"]
        == "gigaevo.memory.selection_leases.SharedSelectionRegistry"
    )
    assert root["selection_leases"]["path"] == (
        "${checkpoint_dir}/selection_leases.json"
    )
    assert (
        memory["provider"]["_target_"] == "gigaevo.memory.provider.LeasedMemoryProvider"
    )
    assert memory["provider"]["registry"] == "${ref:selection_leases}"
    assert memory["writer"]["selection_leases"] == "${ref:selection_leases}"
    assert (
        default_engine["evolution_engine"]["selection_leases"]
        == "${ref:selection_leases}"
    )
    assert (
        steady_engine["evolution_engine"]["selection_leases"]
        == "${ref:selection_leases}"
    )


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


def test_memory_presets_thread_problem_name_as_task_key():
    for preset in ("full", "reader", "writer"):
        memory = OmegaConf.to_container(
            OmegaConf.load(_REPO_ROOT / "config" / "memory" / f"{preset}.yaml"),
            resolve=False,
        )
        assert memory["context_model"]["task_key"] == "${problem.name}"

    full = OmegaConf.to_container(
        OmegaConf.load(_REPO_ROOT / "config" / "memory" / "full.yaml"), resolve=False
    )
    writer = OmegaConf.to_container(
        OmegaConf.load(_REPO_ROOT / "config" / "memory" / "writer.yaml"),
        resolve=False,
    )
    assert full["writer"]["task_key"] == "${problem.name}"
    assert writer["writer"]["task_key"] == "${problem.name}"

    for preset in ("full", "reader"):
        memory = OmegaConf.to_container(
            OmegaConf.load(_REPO_ROOT / "config" / "memory" / f"{preset}.yaml"),
            resolve=False,
        )
        assert memory["reader"]["renderer"]["task_key"] == "${problem.name}"

    recommended = OmegaConf.to_container(
        OmegaConf.load(
            _REPO_ROOT / "config" / "memory" / "evictor" / "recommended.yaml"
        ),
        resolve=False,
    )
    assert all(
        evictor["task_key"] == "${problem.name}" for evictor in recommended["evictors"]
    )
    assert all(
        evictor["inner"]["task_key"] == "${problem.name}"
        for evictor in recommended["evictors"]
    )
    harm = OmegaConf.to_container(
        OmegaConf.load(_REPO_ROOT / "config" / "memory" / "evictor" / "harm.yaml"),
        resolve=False,
    )
    assert harm["task_key"] == "${problem.name}"
    assert harm["inner"]["task_key"] == "${problem.name}"


def test_full_memory_instantiates_cross_task_guarded_evictors():
    cfg = _compose_full()
    raw = OmegaConf.to_container(cfg.memory.evictor, resolve=False)
    floor = int(cfg.memory.evidence.min_effective_events)
    metrics = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness", higher_is_better=True, is_primary=True
            )
        }
    )
    for guarded in raw["evictors"]:
        guarded["task_key"] = cfg.problem.name
        guarded["min_effective_events"] = floor
        inner = guarded["inner"]
        inner["task_key"] = cfg.problem.name
        if "scorer" in inner:
            inner["scorer"] = BetaBinomialReputation(harm_min_events=floor)
        if "metrics_context" in inner:
            inner["metrics_context"] = metrics
        if "rescue_min_events" in inner:
            inner["rescue_min_events"] = floor
        if "neutral_gain" in inner:
            inner["neutral_gain"] = float(cfg.memory.neutral_gain)
        if "min_effective_events" in inner:
            inner["min_effective_events"] = floor
        if "skip_contextual_without_context" in inner:
            inner["skip_contextual_without_context"] = True
    evictor = instantiate(OmegaConf.create(raw, flags={"allow_objects": True}))

    assert isinstance(evictor, CompositeEvictor)
    assert all(
        isinstance(guarded, CrossTaskRetentionGuard) for guarded in evictor._evictors
    )
    assert [type(guarded.inner) for guarded in evictor._evictors] == [
        BirthFailureEvictor,
        HarmEvictor,
        PolicyNonViableEvictor,
    ]
    assert all(guarded.task_key == "_test_" for guarded in evictor._evictors)
    assert all(guarded.min_effective_events == floor for guarded in evictor._evictors)
    assert all(
        guarded.inner._task_key == guarded.task_key for guarded in evictor._evictors
    )


def test_no_memory_provider_and_writer_targets_stay_legacy_noops():
    memory = OmegaConf.load(_REPO_ROOT / "config" / "memory" / "none.yaml")

    assert memory.provider._target_ == "gigaevo.memory.provider.NullMemoryProvider"
    assert memory.writer._target_ == "gigaevo.evolution.engine.hooks.NullPostRunHook"


def _cold_priors(node: Any) -> list[Any]:
    if isinstance(node, dict):
        values = []
        if "cold_prior" in node:
            values.append(node["cold_prior"])
        for value in node.values():
            values.extend(_cold_priors(value))
        return values
    if isinstance(node, list):
        values = []
        for value in node:
            values.extend(_cold_priors(value))
        return values
    return []


def test_reputation_cold_priors_match_auction_baseline():
    auction = OmegaConf.load(
        _REPO_ROOT / "config" / "memory" / "auction" / "thompson_bootstrap.yaml"
    )
    raw_auction = OmegaConf.to_container(auction, resolve=False)
    assert raw_auction["baseline_prior"] == "${memory.baseline_prior}"
    for path in (_REPO_ROOT / "config" / "memory" / "reputation").glob("*.yaml"):
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        for cold_prior in _cold_priors(raw):
            assert cold_prior == raw_auction["baseline_prior"]
