"""AmemGamMemory LLM wiring: the injected llm_service is the only LLM source.

No env-var fallback path exists — when ``llm_service`` is None the agentic
features are simply off; when it is injected without a generator, the
runtime's ``generator_cls`` wraps it so GAM stays live.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.memory_config import MemoryConfig
from tests.fakes.agentic_memory import FakeAMemGenerator, _get_fake_runtime


def _config(tmp_path) -> MemoryConfig:
    return MemoryConfig(
        checkpoint_path=tmp_path / "mem",
        enable_llm_synthesis=False,
        enable_memory_evolution=False,
        enable_llm_card_enrichment=False,
    )


class TestLlmServiceWiring:
    def test_injected_llm_without_generator_builds_generator(self, tmp_path):
        llm = MagicMock()
        mem = AmemGamMemory(
            config=_config(tmp_path),
            runtime=_get_fake_runtime(),
            llm_service=llm,
        )
        assert isinstance(mem.generator, FakeAMemGenerator)
        assert mem.generator._llm_service is llm

    def test_no_llm_service_means_agentic_off_without_env(self, tmp_path, monkeypatch):
        import gigaevo.memory.shared_memory.memory as memory_mod

        def _boom(**kwargs):
            raise AssertionError("env-based LLM init must not be consulted")

        monkeypatch.setattr(memory_mod, "init_llm_and_generator", _boom, raising=False)
        mem = AmemGamMemory(config=_config(tmp_path), runtime=_get_fake_runtime())
        assert mem.llm_service is None
        assert mem.generator is None
        assert mem.memory_system is None

    def test_injected_generator_used_as_is(self, tmp_path):
        llm = MagicMock()
        generator = FakeAMemGenerator({"llm_service": llm})
        mem = AmemGamMemory(
            config=_config(tmp_path),
            runtime=_get_fake_runtime(),
            llm_service=llm,
            generator=generator,
        )
        assert mem.generator is generator
