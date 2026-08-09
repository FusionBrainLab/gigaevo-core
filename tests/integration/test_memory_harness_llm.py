"""Hydra composition tests for the harness-backed memory LLM presets.

``memory/llm=harness`` and ``memory/llm=codex`` mirror the evolution-side
backends in ``config/llm/``. The mirror-sync tests pin the resolved
``HarnessChat`` block to be identical between the twin files, so a wire-contract
change in one cannot silently drift from the other. The schema sweep pins every
memory-path response model to survive ``strict_json_schema`` — the codex arm
refuses map-shaped objects and non-nullable optionals under unions at call
time, and a schema edit must fail here first, not mid-run.

Resolution only — no instantiation (``HarnessChat`` preflights the real CLI).
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from gigaevo.llm.agents.admission_novelty import NoveltyVerdict
from gigaevo.llm.agents.card_author import CardAuthorResponse
from gigaevo.llm.agents.equivalence import EquivalenceResponse
from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.llm.agents.task_summary import TaskSummaryResponse
from gigaevo.llm.schema_compat import portable_json_schema, strict_json_schema
from gigaevo.memory.storage.research import SearchPlan, ShortlistDecision

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

_BASE_OVERRIDES = [
    "problem.name=_test_",
    "algorithm=multi_island",
    "evolution=steady_state",
    "memory=v2",
]

MEMORY_PATH_MODELS = [
    CardAuthorResponse,
    EquivalenceResponse,
    NoveltyVerdict,
    ProgramAuthorResponse,
    SearchPlan,
    ShortlistDecision,
    TaskSummaryResponse,
]


def _compose(*overrides: str):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(
            config_name="config", overrides=_BASE_OVERRIDES + list(overrides)
        )


def _resolved_model(node):
    return OmegaConf.to_container(node.models[0], resolve=True)


def test_memory_harness_preset_targets_harness_chat():
    cfg = _compose("memory/llm=harness")
    model = cfg.memory.llm.models[0]
    assert model._target_ == "gigaevo.llm.harness.HarnessChat"
    assert model.schema_flag == "--json-schema"
    assert model.answer_key == "structured_output"
    assert cfg.memory.llm.structured_output_method == "json_schema"


def test_memory_codex_preset_targets_harness_chat():
    cfg = _compose("memory/llm=codex")
    model = cfg.memory.llm.models[0]
    assert model._target_ == "gigaevo.llm.harness.HarnessChat"
    assert model.strict_schema is True
    assert model.stdin_prompts is True
    assert model.schema_as_path is True
    assert model.answer_file_flag == "--output-last-message"


def test_memory_harness_mirrors_evolution_harness():
    cfg = _compose("llm=harness", "memory/llm=harness")
    assert _resolved_model(cfg.memory.llm) == _resolved_model(cfg.llm)


def test_memory_codex_mirrors_evolution_codex():
    cfg = _compose("llm=codex", "memory/llm=codex")
    assert _resolved_model(cfg.memory.llm) == _resolved_model(cfg.llm)


def test_cross_arm_composition_keeps_backends_independent():
    cfg = _compose("llm=harness", "memory/llm=codex")
    assert cfg.llm.models[0].model_name == "claude-code/sonnet"
    assert cfg.memory.llm.models[0].model_name == "codex/gpt-5.6-luna"


def test_harness_memory_router_caps_subprocess_concurrency():
    for preset in ("harness", "codex"):
        cfg = _compose(f"memory/llm={preset}")
        assert cfg.memory.llm.max_concurrent <= 8


@pytest.mark.parametrize("model", MEMORY_PATH_MODELS, ids=lambda m: m.__name__)
def test_memory_path_schema_survives_the_codex_strict_wire(model):
    strict_json_schema(portable_json_schema(model.model_json_schema()))
