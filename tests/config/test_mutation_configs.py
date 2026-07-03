"""Validate Hydra-composable mutation configs (the /mutation group)."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
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
    assert cfg.mutation_operator._target_.endswith("LLMMutationOperator")


def test_structured_diff_override_selects_diff_operator():
    cfg = _compose("mutation=structured_diff_chains")
    assert cfg.mutation_operator._target_.endswith("StructuredDiffMutationOperator")
    assert cfg.mutation_operator.allowed_changes._target_.endswith("AllowedDagChanges")


@pytest.mark.parametrize("experiment", ["base", "full_featured", "prompt_coevolution"])
def test_experiment_configs_carry_a_mutation_operator(experiment):
    """Every experiment defaults list must select a /mutation group entry.

    config/evolution/*.yaml interpolates ``${mutation_operator}``; an experiment
    that omits ``/mutation`` composes fine but crashes at engine instantiation.
    """
    cfg = _compose(f"experiment={experiment}")
    assert cfg.evolution_engine.mutation_operator._target_.endswith("MutationOperator")
