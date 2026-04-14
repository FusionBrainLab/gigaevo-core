"""Tests verifying heilbron plugin has been removed from the registry.

The HeilbronPlugin was absorbed into AdversarialPlugin (multi-metric panels).
These tests ensure the heilbron module and registration are gone.
"""

from __future__ import annotations

import importlib

import pytest

from gigaevo.monitoring.watchdog_plugin import get_registry


class TestHeilbronPluginRemoved:
    """HeilbronPlugin no longer exists -- absorbed into AdversarialPlugin."""

    def test_heilbron_not_in_registry(self):
        """The 'heilbron' name must not appear in the plugin registry."""
        assert "heilbron" not in get_registry()

    def test_heilbron_module_does_not_exist(self):
        """gigaevo.monitoring.plugins.heilbron module must not be importable."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("gigaevo.monitoring.plugins.heilbron")

    def test_adversarial_still_registered(self):
        """AdversarialPlugin (which absorbed heilbron) is still registered."""
        assert "adversarial" in get_registry()
