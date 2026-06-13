"""Backend factories: Hydra-composable, pure (no yaml/dotenv), fail-fast."""

from __future__ import annotations

import warnings

from loguru import logger
import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.backend_factory import (
    LegacyApiMemoryBackendFactory,
    LocalMemoryBackendFactory,
    MemoryBackendFactory,
)
from gigaevo.memory.shared_memory.memory_config import GamConfig, MemoryConfig


class _CaptureBackend:
    def __init__(self, config=None, **kwargs):
        self.config = config
        self.kwargs = kwargs


class _ExplodingBackend:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("boom")


def _capture(monkeypatch, factory_cls):
    monkeypatch.setattr(factory_cls, "backend_class", lambda self: _CaptureBackend)


class TestLocalFactory:
    def test_is_a_memory_backend_factory(self):
        assert issubclass(LocalMemoryBackendFactory, MemoryBackendFactory)

    def test_builds_local_backend_with_memory_config(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        factory = LocalMemoryBackendFactory(checkpoint_dir=tmp_path)
        memory = factory.build()
        assert memory.config.checkpoint_path == tmp_path
        assert memory.config.api is None

    def test_runtime_knobs_flow_into_memory_config(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        factory = LocalMemoryBackendFactory(
            checkpoint_dir=tmp_path,
            search_limit=9,
            rebuild_interval=7,
            enable_llm_synthesis=True,
            enable_memory_evolution=True,
            enable_llm_card_enrichment=True,
        )
        cfg = factory.build().config
        assert cfg.search_limit == 9
        assert cfg.rebuild_interval == 7
        assert cfg.enable_llm_synthesis is True
        assert cfg.enable_memory_evolution is True
        assert cfg.enable_llm_card_enrichment is True

    def test_production_defaults(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        cfg = LocalMemoryBackendFactory(checkpoint_dir=tmp_path).build().config
        assert cfg.search_limit == 5
        assert cfg.rebuild_interval == 30
        assert cfg.enable_llm_synthesis is False
        assert cfg.enable_memory_evolution is False
        assert cfg.enable_llm_card_enrichment is False

    def test_memory_config_defaults_match_factory_defaults(self, tmp_path):
        """A MemoryConfig built directly must behave like one built via the
        factory — drifting defaults make direct constructions (tests, tools)
        exercise a different backend than production runs."""
        cfg = MemoryConfig(checkpoint_path=tmp_path)
        assert cfg.rebuild_interval == 30
        assert cfg.enable_llm_synthesis is False
        assert cfg.enable_memory_evolution is False
        assert cfg.enable_llm_card_enrichment is False

    def test_build_checkpoint_dir_overrides_configured(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        factory = LocalMemoryBackendFactory(checkpoint_dir=tmp_path / "configured")
        memory = factory.build(checkpoint_dir=tmp_path / "runtime")
        assert memory.config.checkpoint_path == tmp_path / "runtime"

    def test_missing_checkpoint_dir_raises(self):
        with pytest.raises(MemoryStorageError):
            LocalMemoryBackendFactory().build()

    def test_gam_config_passed_through(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        gam = GamConfig(
            enable_bm25=True,
            allowed_tools=["vector"],
            top_k_by_tool={"vector": 7},
            pipeline_mode="default",
            max_iters=5,
            max_cards=1,
        )
        cfg = LocalMemoryBackendFactory(checkpoint_dir=tmp_path).build(gam=gam).config
        assert cfg.gam == gam

    def test_default_gam_when_not_provided(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        cfg = LocalMemoryBackendFactory(checkpoint_dir=tmp_path).build().config
        assert cfg.gam == GamConfig()

    def test_write_side_components_passed_through(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        evictor = object()
        dedup = object()
        memory = LocalMemoryBackendFactory(checkpoint_dir=tmp_path).build(
            evictor=evictor, deduplicator=dedup
        )
        assert memory.kwargs == {"evictor": evictor, "deduplicator": dedup}

    def test_write_side_components_omitted_by_default(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        memory = LocalMemoryBackendFactory(checkpoint_dir=tmp_path).build()
        assert memory.kwargs == {}

    def test_backend_init_failure_raises_storage_error(self, tmp_path, monkeypatch):
        # A misconfigured backend must abort the run loudly, never silently
        # degrade to a no-memory run.
        monkeypatch.setattr(
            LocalMemoryBackendFactory, "backend_class", lambda self: _ExplodingBackend
        )
        with pytest.raises(MemoryStorageError) as exc_info:
            LocalMemoryBackendFactory(checkpoint_dir=tmp_path).build()
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestLlmWiring:
    def test_llm_default_is_none(self):
        assert MemoryBackendFactory.model_fields["llm"].default is None

    def test_injected_llm_forwarded_as_llm_service(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        llm = object()
        memory = LocalMemoryBackendFactory(checkpoint_dir=tmp_path, llm=llm).build()
        assert memory.kwargs["llm_service"] is llm

    def test_no_llm_means_no_llm_service_kwarg(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        memory = LocalMemoryBackendFactory(checkpoint_dir=tmp_path).build()
        assert "llm_service" not in memory.kwargs

    def test_embedding_model_name_flows_into_memory_config(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        factory = LocalMemoryBackendFactory(
            checkpoint_dir=tmp_path, embedding_model_name="custom-embedder"
        )
        assert factory.build().config.embedding_model_name == "custom-embedder"

    def test_embedding_model_name_default(self, tmp_path, monkeypatch):
        _capture(monkeypatch, LocalMemoryBackendFactory)
        cfg = LocalMemoryBackendFactory(checkpoint_dir=tmp_path).build().config
        assert cfg.embedding_model_name == "all-MiniLM-L6-v2"

    def test_legacy_factory_forwards_llm(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            LegacyApiMemoryBackendFactory,
            "backend_class",
            lambda self: _CaptureBackend,
        )
        llm = object()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            factory = LegacyApiMemoryBackendFactory(
                checkpoint_dir=tmp_path, llm=llm, namespace="exp9"
            )
        assert factory.build().kwargs["llm_service"] is llm


class TestLegacyApiFactory:
    def test_is_a_memory_backend_factory(self):
        assert issubclass(LegacyApiMemoryBackendFactory, MemoryBackendFactory)

    def test_instantiation_warns_deprecated(self, tmp_path):
        with pytest.warns(DeprecationWarning):
            LegacyApiMemoryBackendFactory(checkpoint_dir=tmp_path)

    def test_builds_api_backend_with_connection_kwargs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            LegacyApiMemoryBackendFactory,
            "backend_class",
            lambda self: _CaptureBackend,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            factory = LegacyApiMemoryBackendFactory(
                checkpoint_dir=tmp_path,
                base_url="http://memory:9000",
                namespace="exp9",
                channel="latest",
            )
        memory = factory.build()
        assert memory.kwargs["base_url"] == "http://memory:9000"
        assert memory.kwargs["namespace"] == "exp9"
        assert memory.kwargs["use_api"] is True
        assert memory.kwargs["checkpoint_path"] == str(tmp_path)

    def test_backend_init_failure_raises_storage_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            LegacyApiMemoryBackendFactory,
            "backend_class",
            lambda self: _ExplodingBackend,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            factory = LegacyApiMemoryBackendFactory(
                checkpoint_dir=tmp_path, namespace="exp9"
            )
        with pytest.raises(MemoryStorageError):
            factory.build()

    def test_namespace_accepts_none_at_instantiation(self, tmp_path):
        # config.yaml has namespace: null, so the composed node arrives as None;
        # the factory must instantiate and defer the failure to build().
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            factory = LegacyApiMemoryBackendFactory(
                checkpoint_dir=tmp_path, namespace=None
            )
        assert factory.namespace is None

    def test_build_without_namespace_raises_storage_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            LegacyApiMemoryBackendFactory,
            "backend_class",
            lambda self: _CaptureBackend,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            factory = LegacyApiMemoryBackendFactory(checkpoint_dir=tmp_path)
        with pytest.raises(MemoryStorageError, match="namespace"):
            factory.build()

    def _legacy_factory(self, tmp_path, monkeypatch) -> LegacyApiMemoryBackendFactory:
        monkeypatch.setattr(
            LegacyApiMemoryBackendFactory,
            "backend_class",
            lambda self: _CaptureBackend,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return LegacyApiMemoryBackendFactory(
                checkpoint_dir=tmp_path, namespace="exp9"
            )

    def test_build_warns_when_injected_components_are_dropped(
        self, tmp_path, monkeypatch
    ):
        factory = self._legacy_factory(tmp_path, monkeypatch)
        messages: list[str] = []
        handler_id = logger.add(messages.append, level="WARNING")
        try:
            factory.build(evictor=object(), deduplicator=object())
        finally:
            logger.remove(handler_id)
        assert any("evictor" in m and "deduplicator" in m for m in messages)

    def test_build_silent_when_no_components_injected(self, tmp_path, monkeypatch):
        factory = self._legacy_factory(tmp_path, monkeypatch)
        messages: list[str] = []
        handler_id = logger.add(messages.append, level="WARNING")
        try:
            factory.build()
        finally:
            logger.remove(handler_id)
        assert not any("evictor" in m for m in messages)
