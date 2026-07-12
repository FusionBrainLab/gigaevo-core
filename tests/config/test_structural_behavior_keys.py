"""Algorithm presets declare config prerequisites; the validator enforces them.

Presets whose behavior keys are produced by opt-in wiring (e.g. chain
structural metrics) declare ``algorithm_requires`` in their yaml; a mismatch
otherwise surfaces as KeyError at first archive insert or as a silently
degenerate one-cell behavior space (the inert-treatment trap). The validator
is generic — all domain knowledge lives in the preset configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import pytest

from gigaevo.config.validation import validate_algorithm_requirements

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_BASE = ["problem.name=_test_"]


def _compose(*overrides: str) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(config_name="config", overrides=[*_BASE, *overrides])


def test_preset_without_requirements_passes() -> None:
    validate_algorithm_requirements(_compose())


def test_topology_keys_without_stage_flag_are_rejected() -> None:
    cfg = _compose("algorithm=topology_3d_ret")

    with pytest.raises(ValueError, match="enable_chain_structural_metrics"):
        validate_algorithm_requirements(cfg)


def test_topology_keys_with_stage_flag_pass() -> None:
    cfg = _compose("algorithm=topology_3d_ret", "enable_chain_structural_metrics=true")

    validate_algorithm_requirements(cfg)


def test_semantic_keys_without_stage_flag_are_rejected() -> None:
    cfg = _compose("algorithm=chains_bd3d", "program_format=json_document")

    with pytest.raises(ValueError, match="enable_chain_structural_metrics"):
        validate_algorithm_requirements(cfg)


def test_semantic_keys_under_python_source_are_rejected() -> None:
    cfg = _compose("algorithm=chains_bd3d", "enable_chain_structural_metrics=true")

    with pytest.raises(ValueError, match="json_document"):
        validate_algorithm_requirements(cfg)


def test_semantic_keys_fully_wired_pass() -> None:
    cfg = _compose(
        "algorithm=chains_bd3d",
        "enable_chain_structural_metrics=true",
        "program_format=json_document",
    )

    validate_algorithm_requirements(cfg)
