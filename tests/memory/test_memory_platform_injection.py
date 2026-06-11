"""Legacy platform ``AmemGamMemory`` receives its LLM service and embedding
model by injection (Hydra wires them via ``LegacyApiMemoryBackendFactory``)
instead of self-building from env vars."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("gigaevo_memory")

from gigaevo.memory_platform.shared_memory import memory as legacy_memory  # noqa: E402


class _FakeGenerator:
    def __init__(self, cfg):
        self.llm_service = cfg["llm_service"]


class _FakeAgenticSystem:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _make_legacy(tmp_path, load_classes=None, **kwargs):
    with (
        patch.object(legacy_memory, "build_memory_client", return_value=MagicMock()),
        patch.object(
            legacy_memory.AmemGamMemory,
            "_load_agentic_classes",
            load_classes or (lambda self: None),
        ),
    ):
        return legacy_memory.AmemGamMemory(
            checkpoint_path=str(tmp_path),
            sync_on_init=False,
            **kwargs,
        )


class TestLlmServiceInjection:
    def test_injected_llm_service_stored(self, tmp_path):
        mock_llm = MagicMock()
        mem = _make_legacy(tmp_path, llm_service=mock_llm)
        assert mem.llm_service is mock_llm

    def test_no_llm_service_means_no_generator(self, tmp_path):
        mem = _make_legacy(tmp_path)
        assert mem.llm_service is None
        assert mem.generator is None

    def test_generator_built_from_injected_llm(self, tmp_path):
        mock_llm = MagicMock()

        def load(self):
            self._AMemGeneratorCls = _FakeGenerator

        mem = _make_legacy(tmp_path, load_classes=load, llm_service=mock_llm)
        assert isinstance(mem.generator, _FakeGenerator)
        assert mem.generator.llm_service is mock_llm


class TestEmbeddingModelInjection:
    def test_embedding_model_name_stored(self, tmp_path):
        mem = _make_legacy(tmp_path, embedding_model_name="custom-embedder")
        assert mem.embedding_model_name == "custom-embedder"

    def test_storage_uses_injected_embedding_model(self, tmp_path):
        def load(self):
            self._AgenticMemorySystemCls = _FakeAgenticSystem

        mem = _make_legacy(
            tmp_path,
            load_classes=load,
            llm_service=MagicMock(),
            embedding_model_name="custom-embedder",
        )
        assert mem.memory_system.kwargs["model_name"] == "custom-embedder"


class TestFactoryForwarding:
    def test_legacy_factory_forwards_llm_and_embedding_model(self, tmp_path):
        import pytest

        from gigaevo.memory.backend_factory import LegacyApiMemoryBackendFactory

        captured = {}

        class _Recorder:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mock_llm = MagicMock()
        with pytest.warns(DeprecationWarning):
            factory = LegacyApiMemoryBackendFactory(
                checkpoint_dir=tmp_path,
                llm=mock_llm,
                embedding_model_name="custom-embedder",
                sync_on_init=False,
                namespace="exp9",
            )
        with patch.object(
            LegacyApiMemoryBackendFactory, "backend_class", return_value=_Recorder
        ):
            factory.build()
        assert captured["llm_service"] is mock_llm
        assert captured["embedding_model_name"] == "custom-embedder"
        assert captured["sync_on_init"] is False
