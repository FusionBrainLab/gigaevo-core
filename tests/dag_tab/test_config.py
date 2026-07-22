from pathlib import Path
from unittest.mock import MagicMock

from hydra import compose, initialize_config_dir
from hydra.utils import get_class

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.program_formats import JsonDocumentEvaluationFeature
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.provider import NullMemoryProvider
from problems.dag_tab.problem_context import DagTabProblemContext

CONFIG_DIR = Path(__file__).parents[2] / "config"


def test_dag_tab_config_uses_generic_structured_diff_and_json_loader():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "problem.name=dag_tab",
                "program_format=json_document",
                "mutation=structured_diff_dag_tab",
                "loader=dag_tab_seed",
            ],
        )

    assert get_class(cfg.program_loader._target_).__name__ == "DagTabSeedLoader"
    assert cfg.program_loader.dataset == "california"
    assert get_class(cfg.problem_context._target_).__name__ == "DagTabProblemContext"
    assert cfg.problem_context.dataset == "california"
    assert get_class(cfg.mutation_operator._target_).__name__ == (
        "StructuredDiffMutationOperator"
    )
    assert get_class(cfg.mutation_operator.allowed_changes._target_).__name__ == (
        "AllowedDagTabChanges"
    )


def test_dag_tab_dataset_override_drives_context_and_seed():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "problem.name=dag_tab",
                "program_format=json_document",
                "mutation=structured_diff_dag_tab",
                "loader=dag_tab_seed",
                "problem.dataset=adult",
            ],
        )

    assert cfg.problem_context.dataset == "adult"
    assert cfg.program_loader.dataset == "adult"


def test_dag_tab_problem_context_does_not_enable_runtime_add_context_stage():
    problem_dir = Path(__file__).parents[2] / "problems" / "dag_tab"
    problem_context = DagTabProblemContext(problem_dir, dataset="adult")
    context = EvolutionContext(
        problem_ctx=problem_context,
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
        memory_provider=NullMemoryProvider(),
    )

    blueprint = DefaultPipelineBuilder(
        context,
        program_format_feature=JsonDocumentEvaluationFeature(),
    ).build_blueprint()

    assert problem_context.is_contextual is False
    assert "AddContext" not in blueprint.nodes


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
    nodes = cfg.mutation_operator.allowed_changes.max_nodes
    worst_case_chars = nodes * (6000 + 500)
    assert cfg.llm.models[0].max_tokens >= worst_case_chars / 4
    assert cfg.llm.probabilities == [1.0]
