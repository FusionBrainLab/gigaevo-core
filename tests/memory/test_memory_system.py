"""Unit tests for the one assembled MemorySystem node.

These pin the shared-singleton invariants the old ${ref:memory.*} web faked:
ONE reputation reaches provider + evictor; ONE backend partial (llm-bound once)
reaches provider + tracker; the two enable flags select real components vs Null
variants.
"""

from __future__ import annotations

import functools
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest

from gigaevo.evolution.engine.hooks import NullPostRunHook
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.provider import NullMemoryProvider
from gigaevo.memory.system import MemorySystem


def _capturing():
    seen: dict[str, dict] = {}

    def provider(**kw):
        seen["provider"] = kw
        return ("PROVIDER", kw)

    def tracker(**kw):
        seen["tracker"] = kw
        return ("TRACKER", kw)

    return seen, provider, tracker


def _fake_backend(**kw):
    return ("BACKEND", kw)


def _full(tmp_path, **over):
    rep = over.pop("reputation", BetaBinomialReputation())
    backend = over.pop("backend", _fake_backend)
    seen, provider, tracker = _capturing()
    sys = MemorySystem(
        reader_enabled=over.pop("reader_enabled", True),
        writer_enabled=over.pop("writer_enabled", True),
        reputation=rep,
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
    return sys, seen, rep


def test_full_shares_one_reputation(tmp_path):
    _sys, seen, rep = _full(tmp_path)
    assert seen["provider"]["reputation"] is rep
    assert seen["tracker"]["reputation"] is rep
    assert seen["tracker"]["evictor"].reputation is rep


def test_full_shares_one_backend(tmp_path):
    _sys, seen, _ = _full(tmp_path)
    prov_backend = seen["provider"]["backend"]
    assert seen["tracker"]["backend"] is prov_backend
    assert prov_backend.keywords["llm_service"] is not None


def test_backend_bound_with_shared_llm(tmp_path):
    _sys, seen, _ = _full(tmp_path, backend=_fake_backend)
    # provider/tracker see the llm-bound partial wrapping the raw backend
    bound = seen["provider"]["backend"]
    assert bound is not _fake_backend
    assert bound.func is _fake_backend
    assert bound.keywords["llm_service"] is not None


def test_none_yields_null_provider_and_tracker():
    sys = MemorySystem(reader_enabled=False, writer_enabled=False)
    assert isinstance(sys.provider, NullMemoryProvider)
    assert isinstance(sys.tracker, NullPostRunHook)


def test_reader_only_has_real_provider_null_tracker(tmp_path):
    sys, seen, _ = _full(tmp_path, writer_enabled=False)
    assert seen["provider"]
    assert isinstance(sys.tracker, NullPostRunHook)
    assert "tracker" not in seen


def test_writer_only_has_real_tracker_null_provider(tmp_path):
    sys, seen, _ = _full(tmp_path, reader_enabled=False)
    assert seen["tracker"]
    assert isinstance(sys.provider, NullMemoryProvider)
    assert "provider" not in seen


# ---------------------------------------------------------------------------
# Compose round-trip: the real Hydra config assembles ONE MemorySystem node.
# `memory={none,reader,writer,full}` is the only knob; `memory/common/llm=` swaps
# the writer model. These exercise the `_partial_` completion + shared-singleton
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
    # ONE reputation reaches provider + the IdeaTracker's write stack + evictor
    assert sys.provider._reputation is sys.tracker._stack._reputation
    assert sys.provider._reputation is sys.tracker._stack._evictor.reputation
    # ONE backend partial (llm-bound once) reaches both sides
    assert sys.tracker._stack._backend is sys.provider._backend
    assert sys.tracker._stack._backend.keywords["llm_service"] is not None


def test_compose_reader_only(_llm_env, tmp_path):
    from gigaevo.memory.provider import SelectorMemoryProvider

    sys = instantiate(_compose("memory=reader", f"checkpoint_dir={tmp_path}").memory)
    assert isinstance(sys.provider, SelectorMemoryProvider)
    assert isinstance(sys.tracker, NullPostRunHook)


def test_compose_writer_only(_llm_env, tmp_path):
    sys = instantiate(_compose("memory=writer", f"checkpoint_dir={tmp_path}").memory)
    assert isinstance(sys.provider, NullMemoryProvider)
    assert sys.tracker._stack._backend is not None
    assert sys.tracker._stack._backend.keywords["llm_service"] is not None


def test_compose_llm_swap_qwen(_llm_env, tmp_path):
    sys = instantiate(
        _compose(
            "memory=full",
            "memory/common/llm=qwen_instruct",
            f"checkpoint_dir={tmp_path}",
        ).memory
    )
    assert "Qwen" in sys.tracker._stack._backend.keywords["llm_service"].model_names[0]


def test_live_pipeline_provider_and_tracker_share_one_system(_llm_env, tmp_path):
    """`${ref:memory::provider}` (pipeline) and `${ref:memory::tracker}`
    (post_step_hook) must resolve to ONE MemorySystem build — the resolver
    write-back — so the IdeaTracker that writes cards and the provider that
    reads them share the same backend partial."""
    from omegaconf import OmegaConf

    from gigaevo.memory.provider import SelectorMemoryProvider

    cfg = _compose(
        "pipeline=intra_extra_memory", "memory=full", f"checkpoint_dir={tmp_path}"
    )
    provider = OmegaConf.select(cfg, "evolution_context.memory_provider")
    tracker = OmegaConf.select(cfg, "post_step_hook.tracker")
    assert isinstance(provider, SelectorMemoryProvider)
    assert provider._backend is tracker._stack._backend
