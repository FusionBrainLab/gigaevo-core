"""The ``config/llms/`` group: memory LLM routers are Hydra-instantiated.

The router composes ONCE at the root-registered ``memory_llm`` entry and is
shared via ``${ref:memory_llm}`` (sharing itself is covered behaviorally in
``test_memory_groups_hydra.py``).
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from gigaevo.llm.models import MultiModelRouter

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(overrides=overrides)


class TestLlmsGroup:
    def test_gemini_flash_openrouter_instantiates(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-llms-group")
        monkeypatch.setattr(MultiModelRouter, "_verify_models", lambda self: None)
        cfg = _compose(["+llms@memory_llm=gemini_flash_openrouter", "+writer=null"])
        router = instantiate(cfg.memory_llm)
        assert isinstance(router, MultiModelRouter)
        assert router.model_names == ["google/gemini-3-flash-preview"]

    def test_backend_llm_switchable_off(self):
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "problem.name=toy_example",
                    f"problem.dir={REPO_ROOT}/problems/toy_example",
                    "algorithm=multi_island",
                    "pipeline=auto",
                    "memory=local",
                    "memory.backend.llm=null",
                ],
            )
        assert cfg.memory.backend.llm is None
