from __future__ import annotations

"""Centralised Prometheus metrics helper used by Runner modules.

The service initialises Prometheus counters/gauges only if ``prometheus_client``
is available *and* a port is provided; otherwise the helper falls back to cheap
no-op stubs so the library remains an optional dependency.
"""

from datetime import datetime, timezone
from typing import Optional

try:
    from prometheus_client import (  # type: ignore
        Counter,
        Gauge,
        start_http_server,
    )

    _PROM_AVAILABLE = True
except (
    ImportError
):  # pragma: no cover – allow running without prometheus_client

    class _NoOp:  # pylint: disable=too-few-public-methods
        def inc(self, *_a, **_kw):
            pass

        def dec(self, *_a, **_kw):
            pass

        def set(self, *_a, **_kw):
            pass

    def Counter(*_a, **_kw):  # type: ignore
        return _NoOp()

    def Gauge(*_a, **_kw):  # type: ignore
        return _NoOp()

    def start_http_server(*_a, **_kw):  # type: ignore
        pass

    _PROM_AVAILABLE = False

__all__ = ["MetricsService"]


class MetricsService:
    """Singleton façade for Prometheus counters used across Runner components."""

    _inited = False

    # Metric objects (populated by init)
    dag_runs_started: Counter
    dag_runs_completed: Counter
    dag_errors: Counter
    dag_runs_active: Gauge

    evolution_generations: Counter
    evolution_errors: Counter

    uptime_seconds: Gauge

    _started_at: datetime

    # Lazily created gauges by name
    _gauges: dict[str, Gauge] = {}

    # ------------------------------------------------------------------
    @classmethod
    def init(cls, port: Optional[int] = None) -> None:
        """Initialise metrics once; safe to call repeatedly."""
        if cls._inited:
            return

        if port is not None and _PROM_AVAILABLE:
            start_http_server(port)

        # Create counters/gauges (NoOp when prom not available)
        cls.dag_runs_started = Counter(
            "dag_runs_started_total", "Total DAG runs started"
        )
        cls.dag_runs_completed = Counter(
            "dag_runs_completed_total", "Total DAG runs finished successfully"
        )
        cls.dag_errors = Counter(
            "dag_errors_total", "Total DAG runs that errored"
        )
        cls.dag_runs_active = Gauge("dag_runs_active", "Currently running DAGs")

        cls.evolution_generations = Counter(
            "evolution_generations_total",
            "Evolution engine generations completed",
        )
        cls.evolution_errors = Counter(
            "evolution_errors_total", "Errors encountered by evolution engine"
        )

        cls.uptime_seconds = Gauge(
            "runner_uptime_seconds", "Runner uptime in seconds"
        )
        cls._started_at = datetime.now(timezone.utc)

        cls._inited = True

    # ------------------------------------------------------------------
    # Convenience helpers used by other modules
    # ------------------------------------------------------------------

    @classmethod
    def inc_dag_started(cls, n: int = 1):
        if cls._inited:
            cls.dag_runs_started.inc(n)
            cls.dag_runs_active.inc(n)

    @classmethod
    def inc_dag_completed(cls, n: int = 1):
        if cls._inited:
            cls.dag_runs_completed.inc(n)
            cls.dag_runs_active.dec(n)

    @classmethod
    def inc_dag_error(cls, n: int = 1):
        if cls._inited:
            cls.dag_errors.inc(n)
            cls.dag_runs_active.dec(n)

    @classmethod
    def inc_generation(cls, n: int = 1):
        if cls._inited:
            cls.evolution_generations.inc(n)

    @classmethod
    def inc_evolution_error(cls, n: int = 1):
        if cls._inited:
            cls.evolution_errors.inc(n)

    @classmethod
    def tick_uptime(cls):
        if cls._inited:
            delta = int(
                (datetime.now(timezone.utc) - cls._started_at).total_seconds()
            )
            cls.uptime_seconds.set(delta)

    # ------------------------------------------------------------------
    # Generic exporter
    # ------------------------------------------------------------------

    @classmethod
    def export_dict(
        cls, prefix: str, data: "dict[str, object]"
    ) -> None:  # noqa: D401
        """Publish numeric values from *data* as Gauge metrics.

        Non-numeric entries are skipped. Gauge names are formed as
        ``f"{prefix}_{key}"`` and are cached so labels/descriptions are
        created only once.
        """
        if not cls._inited:
            return

        for key, value in data.items():
            if not isinstance(value, (int, float)):
                continue

            metric_name = f"{prefix}_{key}"
            gauge = cls._gauges.get(metric_name)
            if gauge is None:
                # Sanitize metric_name: prometheus allows [a-zA-Z0-9_:]
                safe_name = metric_name.replace(".", "_").replace("-", "_")
                gauge = Gauge(safe_name, f"{prefix} metric {key}")
                cls._gauges[metric_name] = gauge

            gauge.set(value)
