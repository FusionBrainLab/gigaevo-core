"""Tests for gigaevo.memory.runtime_config path-resolution utilities."""

from pathlib import Path

from gigaevo.memory.runtime_config import (
    resolve_local_path,
    resolve_settings_path,
)


def test_dead_helpers_removed():
    """Ensure deprecated helpers are no longer exported from runtime_config."""
    import gigaevo.memory.runtime_config as rc

    for dead in ("to_bool", "to_int", "to_str", "to_list", "deep_get", "load_settings"):
        assert not hasattr(rc, dead), f"{dead!r} should have been deleted"


# ===========================================================================
# resolve_settings_path
# ===========================================================================


class TestResolveSettingsPath:
    def test_explicit_path(self):
        p = resolve_settings_path("/some/path.yaml")
        assert p == Path("/some/path.yaml")

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("EVO_MEMORY_CONFIG_PATH", "/from/env.yaml")
        p = resolve_settings_path()
        assert p == Path("/from/env.yaml")

    def test_fallback_env_var(self, monkeypatch):
        monkeypatch.delenv("EVO_MEMORY_CONFIG_PATH", raising=False)
        monkeypatch.setenv("EVO_MEMORY_SETTINGS_PATH", "/fallback.yaml")
        p = resolve_settings_path()
        assert p == Path("/fallback.yaml")


# ===========================================================================
# resolve_local_path
# ===========================================================================


class TestResolveLocalPath:
    def test_absolute_path_returned(self):
        p = resolve_local_path(Path("/base"), "/abs/path", "default")
        assert p == Path("/abs/path")

    def test_relative_resolved_against_base(self):
        p = resolve_local_path(Path("/base"), "rel/path", "default")
        assert p == Path("/base/rel/path")

    def test_none_uses_default(self):
        p = resolve_local_path(Path("/base"), None, "default/dir")
        assert p == Path("/base/default/dir")

    def test_empty_string_uses_default(self):
        p = resolve_local_path(Path("/base"), "", "default/dir")
        assert p == Path("/base/default/dir")
