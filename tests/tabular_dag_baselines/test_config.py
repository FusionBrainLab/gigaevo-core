from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest

CONFIG_DIR = Path(__file__).parents[2] / "config"
MODELS = (
    "catboost",
    "tabm",
    "realmlp",
    "tabicl",
    "tabpfn",
    "lightgbm",
    "xgboost",
)


@pytest.mark.parametrize("model", MODELS)
def test_tabular_dag_experiment_presets(model):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment=tabular_dag/{model}"],
        )

    expected_problem = (
        "dag_tab" if model == "catboost" else f"tabular_dag_baselines/{model}"
    )
    assert cfg.problem.name == expected_problem
    assert cfg.problem.dataset == "california"
    assert cfg.program_format.id == "json_document"
    assert cfg.loader.pattern == "*.json"
    assert cfg.program_loader._target_.endswith("DagTabSeedLoader")
    assert cfg.problem_context._target_.endswith("DagTabProblemContext")
    assert cfg.mutation_operator.allowed_changes._target_.endswith(
        "AllowedDagTabChanges"
    )
    # Group replacement is important here. Overlaying these mappings onto the
    # default groups used to leak ``pattern`` into DagTabSeedLoader and
    # ``mutation_mode`` into StructuredDiffMutationOperator.
    assert set(cfg.program_loader) == {"_target_", "dataset", "problem_dir"}
    assert set(cfg.mutation_operator) == {
        "_target_",
        "llm_wrapper",
        "allowed_changes",
        "problem_context",
        "prompts_dir",
        "prompt_fetcher",
    }


@pytest.mark.parametrize("model", MODELS)
def test_tabular_dag_experiment_runtime_objects_instantiate(model):
    if model == "catboost":
        problem_dir = CONFIG_DIR.parent / "problems" / "dag_tab"
    else:
        problem_dir = CONFIG_DIR.parent / "problems" / "tabular_dag_baselines" / model
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment=tabular_dag/{model}",
                f"problem.dir={problem_dir}",
            ],
        )

    loader = instantiate(cfg.program_loader)
    mutation = instantiate(
        cfg.mutation_operator,
        llm_wrapper=object(),
        prompt_fetcher=None,
    )

    assert type(loader).__name__ == "DagTabSeedLoader"
    assert type(mutation).__name__ == "StructuredDiffMutationOperator"
    assert type(mutation.allowed_changes).__name__ == "AllowedDagTabChanges"


def test_dataset_and_node_budget_remain_short_overrides():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=tabular_dag/xgboost",
                "problem.dataset=adult",
                "mutation_operator.allowed_changes.max_nodes=10",
            ],
        )

    assert cfg.problem.dataset == "adult"
    assert cfg.mutation_operator.allowed_changes.max_nodes == 10


def test_baselines_are_scoped_under_experiment_configs():
    assert (CONFIG_DIR / "experiment" / "tabular_dag" / "_base.yaml").is_file()
    assert not (CONFIG_DIR / "tabular_dag_baseline").exists()

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["problem.name=prompt_optimization"],
        )

    assert "tabular_dag_baseline" not in cfg
    assert "tabular_dag_model" not in cfg
    assert cfg.program_format.id == "python_source"
