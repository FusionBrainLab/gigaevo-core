"""Validate Hydra-composable stopper configs for the steady-state engine."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

_BASE_OVERRIDES = ["problem.name=_test_"]


def _compose(*overrides: str):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(
            config_name="config", overrides=_BASE_OVERRIDES + list(overrides)
        )


def test_max_mutants_stopper_config_resolves():
    """`stopper=max_mutants` should yield a MaxMutantsStopper instance."""
    from gigaevo.evolution.engine.stopper import MaxMutantsStopper

    cfg = _compose("stopper=max_mutants", "max_mutants=42")
    stopper = instantiate(cfg.stopper)
    assert isinstance(stopper, MaxMutantsStopper)
    assert stopper.max_mutants == 42


def test_max_mutants_default_value_is_800():
    """Default ``max_mutants`` preserves prior ~800-mutant effective run length."""
    cfg = _compose()
    assert cfg.max_mutants == 800
    assert cfg.stopper._target_.endswith("MaxMutantsStopper")


def test_max_generations_stopper_is_gone():
    """No back-compat alias — referencing the old stopper must fail loudly."""
    from hydra.errors import MissingConfigException

    with pytest.raises(MissingConfigException):
        _compose("stopper=max_generations")


def test_combined_stopper_config_resolves():
    """Combined max_mutants + plateau stopper instantiates as CompositeStopper."""
    from gigaevo.evolution.engine.stopper import CompositeStopper

    cfg = _compose("stopper=max_mutants_or_fitness_plateau", "max_mutants=10")
    stopper = instantiate(cfg.stopper)
    assert isinstance(stopper, CompositeStopper)
