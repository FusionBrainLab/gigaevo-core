"""Tests for OmegaConf custom resolvers (config/resolvers.py).

Covers _ref_resolver edge cases plus the public registration surface
exposed by ``register_resolvers``.
"""

from __future__ import annotations

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


class TestRegisterResolvers:
    """The public ``register_resolvers`` surface."""

    def test_dangerous_resolvers_are_absent(self):
        """``eval``, ``merge``, ``len`` must not be registered."""
        register_resolvers()
        assert OmegaConf.has_resolver("eval") is False
        assert OmegaConf.has_resolver("merge") is False
        assert OmegaConf.has_resolver("len") is False

    def test_required_resolvers_are_present(self):
        register_resolvers()
        assert OmegaConf.has_resolver("ref") is True
        assert OmegaConf.has_resolver("get_object") is True

    def test_double_registration_is_idempotent(self):
        register_resolvers()
        register_resolvers()

    def test_ref_resolver_resolves_via_interpolation(self):
        register_resolvers()
        cfg = OmegaConf.create(
            {"target": {"value": 7}, "pointer": "${ref:target.value}"}
        )
        assert cfg.pointer == 7
