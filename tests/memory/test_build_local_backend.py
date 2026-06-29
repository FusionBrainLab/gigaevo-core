"""build_local_backend: thin Hydra-partial constructor for the local card bank.

Replaces the old MemoryBackendFactory. A module-level function the
``memory/common/backend=local`` group binds via ``_partial_``; MemorySystem completes
it with the shared llm. We monkeypatch the AmemGamMemory symbol so the tests
never load the embedding stack.
"""

from __future__ import annotations

import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.shared_memory.backend import build_local_backend
from gigaevo.memory.shared_memory.memory_config import GamConfig, MemoryConfig


class _CaptureBackend:
    def __init__(self, *, config, **kwargs):
        self.config = config
        self.kwargs = kwargs


class _ExplodingBackend:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("boom")


def _capture(monkeypatch, cls=_CaptureBackend):
    monkeypatch.setattr(
        "gigaevo.memory.shared_memory.memory.AmemGamMemory", cls, raising=False
    )


def _cfg(tmp_path, **over):
    return MemoryConfig(checkpoint_path=tmp_path, **over)


class TestBuildLocalBackend:
    def test_builds_backend_from_config(self, tmp_path, monkeypatch):
        _capture(monkeypatch)
        memory = build_local_backend(config=_cfg(tmp_path))
        assert memory.config.checkpoint_path == tmp_path
        assert memory.config.api is None

    def test_checkpoint_dir_override_wins(self, tmp_path, monkeypatch):
        _capture(monkeypatch)
        memory = build_local_backend(
            config=_cfg(tmp_path / "configured"), checkpoint_dir=tmp_path / "runtime"
        )
        assert memory.config.checkpoint_path == tmp_path / "runtime"

    def test_gam_override_applied(self, tmp_path, monkeypatch):
        _capture(monkeypatch)
        gam = GamConfig(allowed_tools=["vector"], top_k_by_tool={"vector": 7})
        memory = build_local_backend(config=_cfg(tmp_path), gam=gam)
        assert memory.config.gam == gam

    def test_default_gam_left_intact_when_not_overridden(self, tmp_path, monkeypatch):
        _capture(monkeypatch)
        memory = build_local_backend(config=_cfg(tmp_path))
        assert memory.config.gam == GamConfig()

    def test_llm_and_evictor_threaded_through(self, tmp_path, monkeypatch):
        _capture(monkeypatch)
        llm, evictor = object(), object()
        memory = build_local_backend(
            config=_cfg(tmp_path), llm_service=llm, evictor=evictor
        )
        assert memory.kwargs["llm_service"] is llm
        assert memory.kwargs["evictor"] is evictor

    def test_config_is_not_mutated_in_place(self, tmp_path, monkeypatch):
        _capture(monkeypatch)
        cfg = _cfg(tmp_path / "configured")
        build_local_backend(config=cfg, checkpoint_dir=tmp_path / "runtime")
        assert cfg.checkpoint_path == tmp_path / "configured"

    def test_backend_gets_a_fresh_config_even_without_overrides(
        self, tmp_path, monkeypatch
    ):
        # Both backends are built off ONE shared Hydra config node; each build
        # must own a distinct copy so a future in-place config mutation on one
        # side cannot bleed into the other (the old factory built fresh too).
        _capture(monkeypatch)
        cfg = _cfg(tmp_path)
        memory = build_local_backend(config=cfg)
        assert memory.config is not cfg
        assert memory.config == cfg

    def test_init_failure_raises_storage_error(self, tmp_path, monkeypatch):
        _capture(monkeypatch, _ExplodingBackend)
        with pytest.raises(MemoryStorageError) as exc_info:
            build_local_backend(config=_cfg(tmp_path))
        assert isinstance(exc_info.value.__cause__, RuntimeError)
