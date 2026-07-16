from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class

CONFIG_DIR = Path(__file__).parents[2] / "config"


def test_dag_tab_config_uses_generic_structured_diff_and_json_loader():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "problem.name=dag_tab",
                "program_format=json_document",
                "mutation=structured_diff_dag_tab",
            ],
        )

    assert cfg.program_loader.pattern == "*.json"
    assert get_class(cfg.mutation_operator._target_).__name__ == (
        "StructuredDiffMutationOperator"
    )
    assert get_class(cfg.mutation_operator.allowed_changes._target_).__name__ == (
        "AllowedDagTabChanges"
    )


def test_qwen_thinking_config_separates_reasoning_and_output_budgets():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "problem.name=dag_tab",
                "program_format=json_document",
                "mutation=structured_diff_dag_tab",
                "llm=qwen_thinking",
                "thinking_token_budget=64000",
                "max_tokens=72000",
            ],
        )

    model = cfg.llm.models[0]
    assert model.max_tokens == 72000
    assert model.extra_body.thinking_token_budget == 64000
    assert model.max_tokens - model.extra_body.thinking_token_budget == 8000


def test_gemini3_flash_uses_openrouter_function_calling():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "problem.name=dag_tab",
                "program_format=json_document",
                "mutation=structured_diff_dag_tab",
                "llm=gemini3_flash",
            ],
        )

    assert cfg.llm.structured_output_method == "function_calling"
    assert cfg.llm.models[0].model == "google/gemini-3-flash-preview"
    assert cfg.llm.models[0].base_url == "https://openrouter.ai/api/v1"
    assert cfg.llm.probabilities == [1.0]


def test_dag_tab_can_extend_final_ingestion_to_dag_timeout():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "problem.name=dag_tab",
                "program_format=json_document",
                "mutation=structured_diff_dag_tab",
                "dag_timeout=7200",
                "final_ingestion_timeout_s=7200",
                "parent_refresh_timeout_s=7920",
            ],
        )

    assert cfg.engine_config.final_ingestion_timeout_s == 7200
    assert cfg.engine_config.parent_refresh_timeout_s == 7920
    assert cfg.runner_config.dag_timeout == 7200
