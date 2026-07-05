"""Validate Hydra-composable mutation configs (the /mutation group)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
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


def test_default_mutation_is_llm_rewrite():
    cfg = _compose()
    assert get_class(cfg.mutation_operator._target_).__name__ == "LLMMutationOperator"


def test_structured_diff_override_selects_diff_operator():
    cfg = _compose("mutation=structured_diff_chains")
    operator = get_class(cfg.mutation_operator._target_)
    assert operator.__name__ == "StructuredDiffMutationOperator"
    changes_target = cfg.mutation_operator.allowed_changes._target_
    if importlib.util.find_spec("mmar_carl"):
        assert get_class(changes_target).__name__ == "AllowedDagChanges"
    else:
        assert changes_target.endswith("AllowedDagChanges")


@pytest.mark.parametrize("experiment", ["base", "full_featured", "prompt_coevolution"])
def test_experiment_configs_carry_a_mutation_operator(experiment):
    """Every experiment defaults list must select a /mutation group entry.

    config/evolution/*.yaml interpolates ``${mutation_operator}``; an experiment
    that omits ``/mutation`` composes fine but crashes at engine instantiation.
    """
    cfg = _compose(f"experiment={experiment}")
    operator = get_class(cfg.evolution_engine.mutation_operator._target_)
    assert operator.__name__.endswith("MutationOperator")
