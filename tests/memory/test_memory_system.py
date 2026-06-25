"""Unit tests for the one assembled MemorySystem node.

These pin the shared-singleton invariants the old ${ref:memory.*} web faked:
ONE reputation reaches provider + evictor; ONE backend (model_copy'd
once with the shared llm + the single dedup config) reaches provider + tracker;
the two enable flags select real components vs Null variants.
"""

from __future__ import annotations

import functools
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest

from gigaevo.evolution.engine.hooks import NullPostRunHook
from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.provider import NullMemoryProvider
from gigaevo.memory.shared_memory.card_update_dedup import CardUpdateDedupConfig
from gigaevo.memory.system import MemorySystem


class _FakeDedup:
    """Stand-in for LLMDeduplicator: carries a `.config` (NullDeduplicator does not)."""

    def __init__(self, config: CardUpdateDedupConfig) -> None:
        self.config = config


def _capturing():
    seen: dict[str, dict] = {}

    def provider(**kw):
        seen["provider"] = kw
        return ("PROVIDER", kw)

    def tracker(**kw):
        seen["tracker"] = kw
        return ("TRACKER", kw)

    return seen, provider, tracker


def _full(tmp_path, **over):
    rep = over.pop("reputation", BetaBinomialReputation())
    cfg = over.pop("dedup_config", CardUpdateDedupConfig())
    dedup = over.pop("dedup", _FakeDedup(cfg))
    backend = over.pop("backend", LocalMemoryBackendFactory(checkpoint_dir=tmp_path))
    seen, provider, tracker = _capturing()
    sys = MemorySystem(
        reader_enabled=over.pop("reader_enabled", True),
        writer_enabled=over.pop("writer_enabled", True),
        reputation=rep,
        dedup=dedup,
        backend=backend,
        llm=object(),
        retriever=object(),
        selector=object(),
        auction=object(),
        budget=object(),
        evictor=functools.partial(HarmEvictor),
        provider=provider,
        tracker=tracker,
        **over,
    )
    return sys, seen, rep, cfg


def test_full_shares_one_reputation(tmp_path):
    _sys, seen, rep, _ = _full(tmp_path)
    assert seen["provider"]["reputation"] is rep
    assert seen["tracker"]["reputation"] is rep
    assert seen["tracker"]["evictor"].reputation is rep


def test_full_shares_one_backend_and_dedup_config(tmp_path):
    _sys, seen, _, cfg = _full(tmp_path)
    prov_backend = seen["provider"]["backend"]
    assert seen["tracker"]["backend"] is prov_backend
    assert prov_backend.dedup is cfg
    assert prov_backend.llm is not None


def test_dedup_config_threaded_into_a_fresh_backend_copy(tmp_path):
    backend = LocalMemoryBackendFactory(checkpoint_dir=tmp_path)
    _sys, seen, _, _ = _full(tmp_path, backend=backend)
    # original factory untouched; provider/tracker see the model_copy
    assert seen["provider"]["backend"] is not backend


def test_backend_threading_skipped_when_dedup_has_no_config(tmp_path):
    class _Null:  # NullDeduplicator has no `.config`
        pass

    _sys, seen, _, _ = _full(tmp_path, dedup=_Null())
    assert seen["provider"]["backend"].llm is not None  # llm threaded, no crash


def test_none_yields_null_provider_and_tracker():
    sys = MemorySystem(reader_enabled=False, writer_enabled=False)
    assert isinstance(sys.provider, NullMemoryProvider)
    assert isinstance(sys.tracker, NullPostRunHook)


def test_reader_only_has_real_provider_null_tracker(tmp_path):
    sys, seen, _, _ = _full(tmp_path, writer_enabled=False)
    assert seen["provider"]
    assert isinstance(sys.tracker, NullPostRunHook)
    assert "tracker" not in seen


def test_writer_only_has_real_tracker_null_provider(tmp_path):
    sys, seen, _, _ = _full(tmp_path, reader_enabled=False)
    assert seen["tracker"]
    assert isinstance(sys.provider, NullMemoryProvider)
    assert "provider" not in seen


# ---------------------------------------------------------------------------
# Compose round-trip: the real Hydra config assembles ONE MemorySystem node.
# `memory={none,reader,writer,full}` is the only knob; `memory/llm=` swaps the
# writer model. These exercise the `_partial_` completion + shared-singleton
# threading end-to-end through hydra.utils.instantiate.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

_BASE = [
    "problem.name=toy_example",
    f"problem.dir={_REPO_ROOT}/problems/toy_example",
    "algorithm=multi_island",
    "pipeline=auto",
    "writer=null",
]


def _compose(*overrides):
    with initialize_config_dir(
        config_dir=str(_REPO_ROOT / "config"), version_base=None
    ):
        return compose(config_name="config", overrides=_BASE + list(overrides))


@pytest.fixture
def _llm_env(monkeypatch):
    from gigaevo.llm.models import MultiModelRouter

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-gigaevo-test")
    monkeypatch.setattr(MultiModelRouter, "_verify_models", lambda self: None)


def test_compose_none_builds_nulls():
    sys = instantiate(_compose("memory=none").memory)
    assert isinstance(sys, MemorySystem)
    assert isinstance(sys.provider, NullMemoryProvider)
    assert isinstance(sys.tracker, NullPostRunHook)


def test_compose_full_shares_singletons(_llm_env, tmp_path):
    from gigaevo.memory.provider import SelectorMemoryProvider

    sys = instantiate(_compose("memory=full", f"checkpoint_dir={tmp_path}").memory)
    assert isinstance(sys.provider, SelectorMemoryProvider)
    # ONE reputation reaches provider + the IdeaTracker itself + evictor
    assert sys.provider._reputation is sys.tracker._reputation
    assert sys.provider._reputation is sys.tracker._evictor.reputation
    # ONE backend (model_copy'd once, carrying the shared llm) reaches both sides
    assert sys.tracker._backend is sys.provider._backend_factory
    assert sys.tracker._backend.llm is not None


def test_compose_reader_only(_llm_env, tmp_path):
    from gigaevo.memory.provider import SelectorMemoryProvider

    sys = instantiate(_compose("memory=reader", f"checkpoint_dir={tmp_path}").memory)
    assert isinstance(sys.provider, SelectorMemoryProvider)
    assert isinstance(sys.tracker, NullPostRunHook)


def test_compose_writer_only(_llm_env, tmp_path):
    sys = instantiate(_compose("memory=writer", f"checkpoint_dir={tmp_path}").memory)
    assert isinstance(sys.provider, NullMemoryProvider)
    assert sys.tracker._backend is not None
    assert sys.tracker._backend.llm is not None


def test_compose_llm_swap_qwen(_llm_env, tmp_path):
    sys = instantiate(
        _compose(
            "memory=full",
            "memory/llm=qwen_instruct",
            f"checkpoint_dir={tmp_path}",
        ).memory
    )
    assert "Qwen" in sys.tracker._backend.llm.model_names[0]


def test_live_pipeline_provider_and_tracker_share_one_system(_llm_env, tmp_path):
    """`${ref:memory::provider}` (pipeline) and `${ref:memory::tracker}`
    (post_step_hook) must resolve to ONE MemorySystem build — the resolver
    write-back — so the IdeaTracker that writes cards and the provider that
    reads them share the same backend factory."""
    from omegaconf import OmegaConf

    from gigaevo.memory.provider import SelectorMemoryProvider

    cfg = _compose(
        "pipeline=intra_extra_memory", "memory=full", f"checkpoint_dir={tmp_path}"
    )
    provider = OmegaConf.select(cfg, "evolution_context.memory_provider")
    tracker = OmegaConf.select(cfg, "post_step_hook.tracker")
    assert isinstance(provider, SelectorMemoryProvider)
    assert provider._backend_factory is tracker._backend
