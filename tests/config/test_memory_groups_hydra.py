"""Memory config groups must instantiate working objects that share the
root-registered singletons (``memory_llm`` router, ``memory.backend`` factory).

Behavioral tests only — composition is exercised end-to-end through
``hydra.utils.instantiate`` against a real problem (``toy_example``) so
``higher_is_better`` and friends resolve from the metrics context.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_BASE_OVERRIDES = [
    "problem.name=toy_example",
    f"problem.dir={REPO_ROOT}/problems/toy_example",
    "algorithm=multi_island",
    "pipeline=auto",
]


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        return compose(
            config_name="config", overrides=_BASE_OVERRIDES + list(overrides)
        )


@pytest.fixture
def llm_env(tmp_path, monkeypatch):
    from gigaevo.llm.models import MultiModelRouter

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-memory-groups")
    monkeypatch.setattr(MultiModelRouter, "_verify_models", lambda self: None)
    return tmp_path


class TestMemoryLocal:
    def test_full_stack_shares_singletons(self, llm_env, monkeypatch):
        """memory=local + ideas_tracker=default: ONE router, ONE backend
        factory, ONE reputation — shared across provider and tracker."""
        from gigaevo.llm.models import MultiModelRouter
        from gigaevo.memory.provider import SelectorMemoryProvider

        init_count = 0
        original_init = MultiModelRouter.__init__

        def counting_init(self, *args, **kwargs):
            nonlocal init_count
            init_count += 1
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(MultiModelRouter, "__init__", counting_init)
        cfg = _compose(
            "memory=local",
            "ideas_tracker=default",
            f"checkpoint_dir={llm_env}",
            "writer=null",
        )
        provider = instantiate(cfg.memory.provider)
        tracker = instantiate(cfg.ideas_tracker)

        assert isinstance(provider, SelectorMemoryProvider)
        assert init_count == 1
        assert tracker._backend is provider._backend_factory
        assert tracker._backend.llm is cfg.memory_llm
        assert provider._reputation is cfg.memory.reputation
        assert tracker._log._admitter is cfg.memory.admitter
        assert tracker._evictor is cfg.memory.evictor
        assert tracker._deduplicator is cfg.memory.dedup

    def test_component_knob_override_reaches_provider(self, llm_env):
        cfg = _compose(
            "memory=local",
            "memory.auction.baseline_prior=[5.0,2.0]",
            f"checkpoint_dir={llm_env}",
            "writer=null",
        )
        provider = instantiate(cfg.memory.provider)
        assert provider._auctioneer.baseline_prior == (5.0, 2.0)

    def test_admitter_group_swap(self):
        from gigaevo.memory.core.admitter import TieredAdmitter

        cfg = _compose("memory=local", "memory/admitter=tiered")
        assert isinstance(instantiate(cfg.memory.admitter), TieredAdmitter)


class TestMemoryNone:
    def test_null_provider_and_no_router(self):
        from gigaevo.memory.provider import NullMemoryProvider

        cfg = _compose()
        assert isinstance(instantiate(cfg.memory.provider), NullMemoryProvider)
        assert "memory_llm" not in cfg

    def test_ideas_tracker_admitter_null_without_memory(self):
        cfg = _compose("ideas_tracker=default")
        assert cfg.ideas_tracker.admitter is None
        assert cfg.ideas_tracker.evictor is None
        assert cfg.ideas_tracker.deduplicator is None


class TestLegacyApi:
    def test_backend_instantiates_with_warning_and_namespace(self, llm_env):
        from gigaevo.memory.backend_factory import LegacyApiMemoryBackendFactory

        cfg = _compose(
            "memory=legacy_api",
            "namespace=test-ns",
            f"checkpoint_dir={llm_env}",
            "writer=null",
        )
        with pytest.warns(DeprecationWarning):
            factory = instantiate(cfg.memory.backend)
        assert isinstance(factory, LegacyApiMemoryBackendFactory)
        assert factory.namespace == "test-ns"
