"""``hydra.utils.instantiate`` on the shipped ``config/ideas_tracker/*.yaml``
must produce a working ``IdeaTracker``.

The group files carry ``override`` defaults entries against the root-registered
``memory_llm`` / ``memory.backend`` singletons, so they must be composed
through the primary ``config`` (raw OmegaConf.load or group-only compose would
have no base entry to override). ``${ref:...}`` fields resolve against the full
tree at instantiation time, so the tracker subtree is instantiated in place.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest

from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
from gigaevo.memory.ideas_tracker.analyzers import ClassifyingAnalyzer
from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker


@pytest.fixture(autouse=True)
def _stub_llm_clients(monkeypatch):
    """The composed config carries a real llms-group router node; skip its
    startup probe and satisfy the ``${oc.env:OPENROUTER_API_KEY}`` interpolation."""
    from gigaevo.llm.models import MultiModelRouter

    monkeypatch.setattr(MultiModelRouter, "_verify_models", lambda self: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-ideas-tracker")


REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose_with_tracker(name: str, tmp_path: Path, *extra: str):
    """Compose the primary config with ``ideas_tracker=<name>`` against a real
    problem so ``fitness_higher_is_better`` resolves from the metrics context."""
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        # Quoted so config names like ``true`` are not parsed as booleans.
        return compose(
            config_name="config",
            overrides=[
                "problem.name=toy_example",
                f"problem.dir={REPO_ROOT}/problems/toy_example",
                "algorithm=multi_island",
                "pipeline=auto",
                f"ideas_tracker='{name}'",
                f"checkpoint_dir={tmp_path / 'ckpt'}",
                "writer=null",
                *extra,
            ],
        )


class TestHydraInstantiateDefault:
    def test_instantiate_default_yaml(self, tmp_path):
        cfg = _compose_with_tracker("default", tmp_path)
        tracker = instantiate(cfg.ideas_tracker)
        assert isinstance(tracker, IdeaTracker)
        # Factory must have materialized a ClassifyingAnalyzer since analyzer_type=default
        assert tracker._analyzer is not None
        assert isinstance(tracker._analyzer, ClassifyingAnalyzer)
        # The group's override must have composed the shared local backend factory
        assert isinstance(tracker._backend, LocalMemoryBackendFactory)


class TestHydraInstantiateTrue:
    """`ideas_tracker=true` is the back-compat alias — must still instantiate."""

    def test_instantiate_true_yaml(self, tmp_path):
        cfg = _compose_with_tracker("true", tmp_path)
        tracker = instantiate(cfg.ideas_tracker)
        assert isinstance(tracker, IdeaTracker)
        assert isinstance(tracker._analyzer, ClassifyingAnalyzer)


class TestHydraExtrasCatchAll:
    """Unknown YAML keys must land in ``**extras`` and not raise TypeError.

    This pins the forward-compat contract: adding a new key to the YAML must
    not require a matching ``__init__`` argument on main.
    """

    def test_unknown_top_level_key_absorbed(self, tmp_path):
        cfg = _compose_with_tracker(
            "default",
            tmp_path,
            "+ideas_tracker.some_future_only_key_xyz=ignored",
            "+ideas_tracker.another_bogus=42",
        )
        # Should NOT raise TypeError(got an unexpected keyword argument)
        tracker = instantiate(cfg.ideas_tracker)
        assert isinstance(tracker, IdeaTracker)
