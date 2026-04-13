"""Tests for WatchdogPlugin ABC, @register decorator, resolve_plugin(), and WatchdogPluginOptions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gigaevo.monitoring.manifest_schema import WatchdogPluginOptions
from gigaevo.monitoring.watchdog_plugin import (
    _REGISTRY,
    WatchdogPlugin,
    get_registry,
    register,
    resolve_plugin,
)

# -- ABC enforcement ----------------------------------------------------------


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
    """extra_telegram_content() and extra_redis_queries() have defaults."""

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


# -- Registry -----------------------------------------------------------------


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


# -- resolve_plugin -----------------------------------------------------------


class TestResolvePlugin:
    """resolve_plugin() priority: manifest.watchdog_plugin field > solo fallback."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self):
        """Save and restore plugin registry around each test."""
        saved = dict(_REGISTRY)
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
        manifest.experiment.task = task
        manifest.experiment.name = name
        manifest.watchdog_plugin = watchdog_plugin
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

    def test_fallback_to_solo_when_no_explicit_plugin(self):
        """When manifest has no watchdog_plugin field, fallback to solo."""

        @register("solo")
        class SoloFallback(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        manifest = self._make_manifest(task="adversarial", watchdog_plugin=None)
        cls = resolve_plugin(manifest)
        assert cls is SoloFallback

    def test_fallback_to_solo_for_unknown_task(self):
        """Unknown task (no heuristic) falls back to solo."""

        @register("solo")
        class SoloFallback(WatchdogPlugin):
            def generate_plots(self, snapshots, output_dir, cycle):
                return []

            def format_status_body(self, snapshots, experiment_name, cycle, max_gen):
                return ""

        manifest = self._make_manifest(task="unknown_task_xyz", watchdog_plugin=None)
        cls = resolve_plugin(manifest)
        assert cls is SoloFallback

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

    def test_no_solo_registered_raises(self):
        """If solo plugin is not registered and no explicit match, KeyError."""
        manifest = self._make_manifest(task="hover", watchdog_plugin=None)
        with pytest.raises(KeyError, match="solo"):
            resolve_plugin(manifest)


# -- WatchdogPluginOptions ----------------------------------------------------


class TestWatchdogPluginOptions:
    """WatchdogPluginOptions round-trip and validate_plot_metrics."""

    def test_default_plot_metrics_empty(self):
        opts = WatchdogPluginOptions()
        assert opts.plot_metrics == []

    def test_round_trip_with_metrics(self):
        opts = WatchdogPluginOptions(plot_metrics=["fitness", "actual_fitness"])
        dumped = opts.model_dump()
        restored = WatchdogPluginOptions.model_validate(dumped)
        assert restored.plot_metrics == ["fitness", "actual_fitness"]

    def test_validate_empty_metrics_returns_empty(self):
        opts = WatchdogPluginOptions(plot_metrics=[])
        result = opts.validate_plot_metrics("nonexistent_problem")
        assert result == []

    def test_validate_known_metrics_passes(self, tmp_path, monkeypatch):
        """Known metrics pass validation without warnings."""
        monkeypatch.setattr(
            "gigaevo.cli.run_resolver._load_metric_names",
            lambda pn: ["fitness", "actual_fitness", "soft_fitness"],
        )

        opts = WatchdogPluginOptions(plot_metrics=["fitness", "actual_fitness"])
        result = opts.validate_plot_metrics("test_problem")
        assert result == ["fitness", "actual_fitness"]

    def test_validate_unknown_metrics_warns(self, monkeypatch):
        """Unknown metrics emit a warning but still return the list."""
        monkeypatch.setattr(
            "gigaevo.cli.run_resolver._load_metric_names",
            lambda pn: ["fitness"],
        )

        opts = WatchdogPluginOptions(plot_metrics=["fitness", "bogus_metric"])
        with pytest.warns(UserWarning, match="bogus_metric"):
            result = opts.validate_plot_metrics("test_problem")
        assert result == ["fitness", "bogus_metric"]
