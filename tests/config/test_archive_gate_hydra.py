"""Hydra composition smoke test for the archive-gate wiring in auto.yaml.

Verifies the contract documented in
``docs/superpowers/specs/2026-05-14-archive-potential-gate-design.md``:

* ``archive_gate_enabled`` defaults to ``true`` (lazy insights ON) and
  resolves at the top level.
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

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

_BASE_OVERRIDES = [
    "problem.name=_test_",
    "algorithm=multi_island",
    "pipeline=auto",
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


def test_archive_gate_provider_block_present_in_auto():
    """auto.yaml must declare the provider so evolution_context can ref it."""
    cfg = _compose()
    assert cfg.archive_gate_provider._target_ == (
        "gigaevo.config.helpers.build_archive_gate_provider"
    )


def test_pipeline_builder_flag_interpolation_off():
    cfg = _compose("archive_gate_enabled=false")
    # The pipeline_builder target is select_pipeline_builder and must receive
    # the resolved boolean.
    assert cfg.pipeline_builder._target_ == (
        "gigaevo.config.helpers.select_pipeline_builder"
    )
    assert cfg.pipeline_builder.archive_gate_enabled is False
    assert cfg.archive_gate_provider.enabled is False


def test_pipeline_builder_flag_on_by_default():
    cfg = _compose()
    assert cfg.pipeline_builder.archive_gate_enabled is True
    assert cfg.archive_gate_provider.enabled is True
