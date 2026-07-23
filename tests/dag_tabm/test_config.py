from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
import yaml

CONFIG_DIR = Path(__file__).parents[2] / "config"


def test_tabm_preset_uses_nested_problem_and_dynamic_seed():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=tabular_dag/tabm"],
        )

    assert cfg.problem.name == "tabular_dag_baselines/tabm"
    assert cfg.program_format.id == "json_document"
    assert get_class(cfg.problem_context._target_).__name__ == "DagTabProblemContext"
    assert get_class(cfg.program_loader._target_).__name__ == "DagTabSeedLoader"
    assert cfg.problem_context.dataset == "california"
    assert cfg.program_loader.dataset == "california"


def test_metric_contract_matches_dag_tab_except_estimator_descriptions():
    dag_tab = yaml.safe_load(
        (CONFIG_DIR.parent / "problems/dag_tab/metrics.yaml").read_text()
    )
    dag_tabm = yaml.safe_load(
        (
            CONFIG_DIR.parent / "problems/tabular_dag_baselines/tabm/metrics.yaml"
        ).read_text()
    )

    def without_descriptions(specs):
        return {
            name: {key: value for key, value in spec.items() if key != "description"}
            for name, spec in specs["specs"].items()
        }

    assert without_descriptions(dag_tabm) == without_descriptions(dag_tab)
