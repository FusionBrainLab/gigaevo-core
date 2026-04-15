"""Tests for WatchdogPlugin ABC, @register decorator, and resolve_plugin()."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gigaevo.monitoring.watchdog_plugin import (
    _REGISTRY,
    WatchdogPlugin,
    get_registry,
    register,
    resolve_plugin,
)

# ── ABC enforcement ──────────────────────────────────────────────────────────


class TestWatchdogPluginABC:
    """WatchdogPlugin cannot be instantiated; subclasses must implement abstract methods."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            WatchdogPlugin()  # type: ignore[abstract]

    def test_missing_generate_plots_raises(self):
        class BadPlugin(WatchdogPlugin):
            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        with pytest.raises(TypeError, match="abstract"):
            BadPlugin()  # type: ignore[abstract]

    def test_missing_format_status_body_raises(self):
        class BadPlugin(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

        with pytest.raises(TypeError, match="abstract"):
            BadPlugin()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        class GoodPlugin(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return "ok"

        plugin = GoodPlugin()
        assert isinstance(plugin, WatchdogPlugin)


class TestDefaultMethods:
    """extra_telegram_content(), extra_redis_queries(), and format_telegram_body() have defaults."""

    def _make_plugin(self):
        class MinimalPlugin(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        return MinimalPlugin()

    def test_extra_telegram_content_returns_none(self):
        plugin = self._make_plugin()
        assert plugin.extra_telegram_content([]) is None

    def test_extra_redis_queries_returns_empty_dict(self):
        plugin = self._make_plugin()
        assert plugin.extra_redis_queries() == {}

    def test_format_telegram_body_returns_none_by_default(self):
        plugin = self._make_plugin()
        result = plugin.format_telegram_body(
            snapshots=[],
            experiment_name="test/exp",
            cycle=1,
            max_generations=50,
            baseline=0.65,
        )
        assert result is None

    def test_format_telegram_body_is_overridable(self):
        class CustomPlugin(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

            def format_telegram_body(
                self, snapshots, experiment_name, cycle, max_generations, baseline=None
            ):
                return f"Custom body for {experiment_name}"

        plugin = CustomPlugin()
        result = plugin.format_telegram_body([], "test/exp", 1, 50)
        assert result == "Custom body for test/exp"


# ── Registry ─────────────────────────────────────────────────────────────────


class TestPluginRegistry:
    """@register decorator and get_registry()."""

    def test_register_adds_to_registry(self):
        @register("test_plugin_a")
        class TestPluginA(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        assert "test_plugin_a" in get_registry()
        assert get_registry()["test_plugin_a"] is TestPluginA
        # Cleanup
        del _REGISTRY["test_plugin_a"]

    def test_register_duplicate_raises(self):
        @register("test_dup")
        class TestDup1(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        with pytest.raises(ValueError, match="already registered"):

            @register("test_dup")
            class TestDup2(WatchdogPlugin):
                def generate_plots(self, snapshots, output_dir, cycle):
                    return []

                def format_status_body(
                    self, snapshots, experiment_name, cycle, max_gen
                ):
                    return ""

        # Cleanup
        del _REGISTRY["test_dup"]

    def test_get_registry_returns_copy(self):
        reg = get_registry()
        reg["mutated"] = object  # type: ignore[assignment]
        assert "mutated" not in get_registry()


# ── resolve_plugin ───────────────────────────────────────────────────────────


class TestResolvePlugin:
    """resolve_plugin() priority: manifest field > task heuristic > solo fallback."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self):
        """Save and restore plugin registry around each test."""
        saved = dict(_REGISTRY)
        # Remove real plugins so tests can register their own
        for name in ("adversarial", "heilbron", "solo", "prompt_coevo", "my_explicit"):
            _REGISTRY.pop(name, None)
        yield
        _REGISTRY.clear()
        _REGISTRY.update(saved)

    def _make_manifest(
        self,
        *,
        task: str = "hover",
        name: str = "hover/test",
        watchdog_plugin: str | None = None,
    ):
        """Build a mock manifest with minimal fields for resolve_plugin."""
        manifest = MagicMock()
        manifest.contract.identity.task = task
        manifest.contract.identity.name = name
        manifest.control_plane.watchdog.plugin = watchdog_plugin
        return manifest

    def test_explicit_plugin_field(self):
        @register("my_explicit")
        class ExplicitPlugin(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        manifest = self._make_manifest(watchdog_plugin="my_explicit")
        cls = resolve_plugin(manifest)
        assert cls is ExplicitPlugin

    def test_explicit_plugin_not_found_raises(self):
        manifest = self._make_manifest(watchdog_plugin="nonexistent_plugin_xyz")
        with pytest.raises(KeyError, match="nonexistent_plugin_xyz"):
            resolve_plugin(manifest)

    def test_task_heuristic_adversarial(self):
        @register("adversarial")
        class AdvPlugin(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        manifest = self._make_manifest(task="adversarial", watchdog_plugin=None)
        cls = resolve_plugin(manifest)
        assert cls is AdvPlugin

    def test_task_heuristic_heilbron(self):
        @register("heilbron")
        class HeilbronPlugin(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        manifest = self._make_manifest(task="heilbron", watchdog_plugin=None)
        cls = resolve_plugin(manifest)
        assert cls is HeilbronPlugin

    def test_fallback_to_solo(self):
        """When no explicit plugin and no heuristic match, resolve to 'solo'."""

        @register("solo")
        class SoloFallback(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        manifest = self._make_manifest(task="unknown_task_xyz", watchdog_plugin=None)
        cls = resolve_plugin(manifest)
        assert cls is SoloFallback
        # cleanup handled by _isolate_registry fixture

    def test_hover_resolves_to_solo_heuristic(self):
        """HoVer task uses the solo plugin via heuristic."""

        @register("solo")
        class SoloHover(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        manifest = self._make_manifest(task="hover", watchdog_plugin=None)
        cls = resolve_plugin(manifest)
        assert cls is SoloHover
        # cleanup handled by _isolate_registry fixture

    def test_none_manifest_returns_solo(self):
        """resolve_plugin(None) returns solo plugin."""

        @register("solo")
        class SoloNone(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        cls = resolve_plugin(None)
        assert cls is SoloNone
        # cleanup handled by _isolate_registry fixture
