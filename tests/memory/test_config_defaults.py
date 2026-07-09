"""Ships-off pin for the novelty-admission gate.

The A/B isolating the gate's causal effect has no verdict yet, so both the
shipped ``config/memory/full.yaml`` and the ``MemoryWriter`` code default must
stay false; flipping either is a deliberate decision, not a drive-by.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from gigaevo.memory.write.writer import MemoryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_full_yaml_ships_novelty_gate_off():
    cfg = OmegaConf.load(_REPO_ROOT / "config" / "memory" / "full.yaml")
    assert cfg.writer.novelty_admission_gate is False


def test_memory_writer_defaults_novelty_gate_off():
    parameters = inspect.signature(MemoryWriter.__init__).parameters
    assert parameters["novelty_admission_gate"].default is False


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
