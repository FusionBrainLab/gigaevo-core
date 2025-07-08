"""Prometheus exporter for EvolutionEngine metrics.

This adapter isolates the EvolutionEngine package from the `MetricsService`
located in `src.runner` to avoid circular or hard imports.  If
`prometheus_client` (and therefore `MetricsService`) is not installed, all
methods degrade to no-ops.
"""
from __future__ import annotations


class EnginePrometheusExporter:
    """Static helpers that proxy Evolution metrics to the central MetricsService."""

    @staticmethod
    def inc_generation(n: int = 1) -> None:
        try:
            from src.runner.metrics import MetricsService  # imported lazily

            MetricsService.inc_generation(n)
        except ImportError:  # pragma: no cover – optional dependency
            pass

    @staticmethod
    def inc_error(n: int = 1) -> None:
        try:
            from src.runner.metrics import MetricsService

            MetricsService.inc_evolution_error(n)
        except ImportError:  # pragma: no cover
            pass 