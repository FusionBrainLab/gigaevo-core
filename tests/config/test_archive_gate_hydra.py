"""Hydra composition smoke test for archive-gate wiring metadata.

Verifies the contract documented in
``docs/superpowers/specs/2026-05-14-archive-potential-gate-design.md``:

* ``archive_gate_enabled`` defaults to ``true`` for builder-backed guided
  pipelines and resolves at the top level.
* ``pipeline_builder.archive_gate_enabled`` and
  ``archive_gate_provider.enabled`` both interpolate from it.
* The CLI override ``archive_gate_enabled=false`` flips both downstream.

Instantiation is NOT exercised here — full instantiation requires a real
Redis. The behavioral test for the gate node itself lives in
``tests/entrypoint/test_archive_gate_wiring.py`` and
``tests/stages/test_archive_potential_gate.py``.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from gigaevo.config.validation import validate_archive_gate_pipeline_compat

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

_BASE_OVERRIDES = [
    "problem.name=_test_",
    "algorithm=multi_island",
]


def _compose(*overrides: str):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(
            config_name="config", overrides=_BASE_OVERRIDES + list(overrides)
        )


def test_default_archive_gate_enabled_is_true():
    cfg = _compose()
    assert cfg.archive_gate_enabled is True


def test_archive_gate_provider_block_present_in_guided():
    """guided.yaml must declare the provider so evolution_context can ref it."""
    cfg = _compose()
    assert cfg.archive_gate_provider._target_ == (
        "gigaevo.config.helpers.build_archive_gate_provider"
    )


def test_pipeline_builder_flag_interpolation_off():
    cfg = _compose("archive_gate_enabled=false")
    assert cfg.pipeline_builder._target_ == (
        "gigaevo.entrypoint.noise_aware_pipeline.NoiseAwareMemoryGuidedPipelineBuilder"
    )
    assert cfg.pipeline_builder.archive_gate_enabled is False
    assert cfg.archive_gate_provider.enabled is False


def test_pipeline_builder_flag_on_by_default():
    cfg = _compose()
    assert cfg.pipeline.archive_gate_mode == "builder"
    assert cfg.pipeline_builder.archive_gate_enabled is True
    assert cfg.archive_gate_provider.enabled is True
    validate_archive_gate_pipeline_compat(cfg)


def test_explicit_guided_pipeline_wires_archive_gate():
    cfg = _compose("pipeline=guided")
    assert cfg.pipeline.archive_gate_mode == "builder"
    assert cfg.pipeline_builder.archive_gate_enabled is True
    assert cfg.archive_gate_provider.enabled is True
    raw_evolution_context = OmegaConf.to_container(cfg.evolution_context, resolve=False)
    assert isinstance(raw_evolution_context, dict)
    assert raw_evolution_context["archive_gate_provider"] == (
        "${ref:archive_gate_provider}"
    )


def test_explicit_memory_guided_pipeline_wires_archive_gate():
    cfg = _compose("pipeline=memory_guided", "memory=v2", "checkpoint_dir=/tmp/bank")
    assert cfg.pipeline.archive_gate_mode == "builder"
    assert cfg.pipeline_builder.archive_gate_enabled is True
    assert cfg.archive_gate_provider.enabled is True
    raw_evolution_context = OmegaConf.to_container(cfg.evolution_context, resolve=False)
    assert isinstance(raw_evolution_context, dict)
    assert raw_evolution_context["archive_gate_provider"] == (
        "${ref:archive_gate_provider}"
    )


def test_custom_pipeline_opts_out_of_archive_gate_by_default():
    cfg = _compose("pipeline=custom")
    assert cfg.pipeline.archive_gate_mode == "none"
    assert cfg.archive_gate_enabled is False
    validate_archive_gate_pipeline_compat(cfg)


def test_custom_pipeline_rejects_forced_archive_gate():
    cfg = _compose("pipeline=custom", "archive_gate_enabled=true")
    with pytest.raises(ValueError, match="archive_gate_mode=none"):
        validate_archive_gate_pipeline_compat(cfg)
