"""Tests for OmegaConf custom resolvers (config/resolvers.py).

Covers _ref_resolver edge cases: nested paths, method chains, invalid syntax,
and the register_resolvers function.
"""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from gigaevo.config.resolvers import _ref_resolver, register_resolvers


class TestRefResolver:
    def test_simple_scalar_access(self):
        cfg = OmegaConf.create({"a": {"b": 42}})
        result = _ref_resolver("a.b", _root_=cfg)
        assert result == 42

    def test_nested_dict_access(self):
        """DictConfig nodes resolve correctly."""
        cfg = OmegaConf.create({"a": {"x": 1, "y": 2}})
        result = _ref_resolver("a.x", _root_=cfg)
        assert result == 1

    def test_top_level_key(self):
        """Single key without dots accesses root-level config."""
        cfg = OmegaConf.create({"simple": 99})
        result = _ref_resolver("simple", _root_=cfg)
        assert result == 99

    def test_invalid_syntax_raises(self):
        """Invalid characters in method chain should raise ValueError."""
        cfg = OmegaConf.create({"a": {"b": 42}})
        with pytest.raises(ValueError, match="Invalid syntax"):
            _ref_resolver("a::invalid-name", _root_=cfg)

    def test_recursive_instantiation_preserves_shared_object_identity(self):
        register_resolvers()
        cfg = OmegaConf.create(
            {
                "strategy": {
                    "_target_": "types.SimpleNamespace",
                    "name": "live",
                },
                "engine": {
                    "_target_": "types.SimpleNamespace",
                    "strategy": "${ref:strategy}",
                },
                "context": {
                    "_target_": "types.SimpleNamespace",
                    "strategy": "${ref:strategy}",
                },
            }
        )

        instances = instantiate(cfg, _recursive_=True)

        assert instances.strategy is instances.engine.strategy
        assert instances.strategy is instances.context.strategy
