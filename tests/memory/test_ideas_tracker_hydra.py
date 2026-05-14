"""RED regression tests: ``hydra.utils.instantiate`` on the shipped
``config/ideas_tracker/*.yaml`` must produce a working ``IdeaTracker``.

Before the d4abf550 port, ``IdeaTracker.__init__`` rejects the 9+ flat kwargs
the YAML passes (``analyzer_type``, ``analyzer_model``, ..., ``record_conversion_type``,
``memory_write_best_programs_percent``) → Hydra raises ``InstantiationException``
wrapping ``TypeError: unexpected keyword argument``.

After the port, the factory auto-builds the analyzer and ``**extras`` absorbs
any other legacy keys.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from gigaevo.memory.ideas_tracker.analyzers import (
    ClassifyingAnalyzer,
    ClusteringAnalyzer,
)
from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker


@pytest.fixture(autouse=True)
def _stub_llm_clients(monkeypatch):
    """Stub the OpenAI client builder so Hydra instantiation tests don't need a real key."""
    import gigaevo.memory.ideas_tracker.llm as _llm_mod

    def _fake_init_clients(base_url):
        return MagicMock(), MagicMock(), False

    monkeypatch.setattr(_llm_mod, "_init_clients", _fake_init_clients)


# Yaml files ship with ``# @package _global_`` — loading them with OmegaConf.load
# returns a DictConfig whose sole key is ``ideas_tracker``.  We pick that out and
# supply the three interpolated vars (checkpoint_dir, namespace, problem.name)
# from a tempdir-backed dummy context.

REPO_ROOT = Path(__file__).resolve().parents[2]
IDEAS_TRACKER_CONFIG_DIR = REPO_ROOT / "config" / "ideas_tracker"


def _load_ideas_tracker_cfg(name: str, tmp_path: Path):
    """Load ``config/ideas_tracker/<name>.yaml`` and resolve the 3 interpolations."""
    raw = OmegaConf.load(IDEAS_TRACKER_CONFIG_DIR / f"{name}.yaml")
    # Supply the three top-level interpolations that the yamls reference.
    ctx = OmegaConf.create(
        {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "namespace": "pytest",
            "problem": {"name": "pytest_problem"},
        }
    )
    merged = OmegaConf.merge(ctx, raw)
    # Return the ``ideas_tracker`` subsection with interpolations resolved.
    return OmegaConf.create(OmegaConf.to_container(merged.ideas_tracker, resolve=True))


@pytest.fixture
def ideas_cfg_tmpdir(tmp_path):
    return tmp_path


class TestHydraInstantiateDefault:
    def test_instantiate_default_yaml(self, ideas_cfg_tmpdir):
        cfg = _load_ideas_tracker_cfg("default", ideas_cfg_tmpdir)
        tracker = instantiate(cfg)
        assert isinstance(tracker, IdeaTracker)
        # Factory must have materialized a ClassifyingAnalyzer since analyzer_type=default
        assert tracker._analyzer is not None
        assert isinstance(tracker._analyzer, ClassifyingAnalyzer)


class TestHydraInstantiateFast:
    @pytest.mark.xfail(
        reason="CI-only: ClusteringAnalyzer instantiates a real httpx client. "
        "Passes locally. See #234.",
        strict=False,
    )
    def test_instantiate_fast_yaml(self, ideas_cfg_tmpdir):
        cfg = _load_ideas_tracker_cfg("fast", ideas_cfg_tmpdir)
        tracker = instantiate(cfg)
        assert isinstance(tracker, IdeaTracker)
        assert tracker._analyzer is not None
        assert isinstance(tracker._analyzer, ClusteringAnalyzer)


class TestHydraInstantiateTrue:
    """`ideas_tracker=true` is the back-compat alias — must still instantiate."""

    def test_instantiate_true_yaml(self, ideas_cfg_tmpdir):
        cfg = _load_ideas_tracker_cfg("true", ideas_cfg_tmpdir)
        tracker = instantiate(cfg)
        assert isinstance(tracker, IdeaTracker)
        assert isinstance(tracker._analyzer, ClassifyingAnalyzer)


class TestHydraExtrasCatchAll:
    """Unknown YAML keys must land in ``**extras`` and not raise TypeError.

    This pins the forward-compat contract: adding a new key to the YAML must
    not require a matching ``__init__`` argument on main.
    """

    def test_unknown_top_level_key_absorbed(self, ideas_cfg_tmpdir):
        cfg = _load_ideas_tracker_cfg("default", ideas_cfg_tmpdir)
        # Mutate the resolved cfg to inject a bogus key
        cfg_mut = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_mut, dict)
        cfg_mut["some_future_only_key_xyz"] = "ignored"
        cfg_mut["another_bogus"] = 42
        cfg_final = OmegaConf.create(cfg_mut)
        # Should NOT raise TypeError(got an unexpected keyword argument)
        tracker = instantiate(cfg_final)
        assert isinstance(tracker, IdeaTracker)


class TestHydraLegacyKeysAccepted:
    """Every legacy flat kwarg currently in the YAML must be accepted by __init__.

    If any of these raise, the YAML config ships dead on main.
    """

    LEGACY_KEYS = [
        "analyzer_type",
        "analyzer_model",
        "analyzer_base_url",
        "analyzer_reasoning",
        "list_max_ideas",
        "postprocessing_type",
        "description_rewriting",
        "record_conversion_type",
        "memory_write_enabled",
        "memory_write_best_programs_percent",
        "memory_usage_tracking_enabled",
        "checkpoint_dir",
        "namespace",
        "redis_prefix",
    ]

    def test_all_legacy_keys_present_in_default_yaml(self):
        """Sanity: the yaml we rely on in the other test actually carries these."""
        raw = OmegaConf.load(IDEAS_TRACKER_CONFIG_DIR / "default.yaml")
        it = raw.ideas_tracker  # type: ignore[union-attr]
        present = set(OmegaConf.to_container(it, resolve=False).keys())  # type: ignore[arg-type]
        missing = set(self.LEGACY_KEYS) - present
        # checkpoint_dir/namespace are interpolated but still present as keys
        assert not missing, f"default.yaml missing expected legacy keys: {missing}"
