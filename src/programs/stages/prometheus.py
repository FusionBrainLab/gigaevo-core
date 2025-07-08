from __future__ import annotations

"""Prometheus export helpers for individual Stage executions.

Designed to avoid a hard dependency on Prometheus/runner code.  If
``prometheus_client`` is not available the helpers degrade gracefully to no-ops.
"""

from typing import Dict

try:
    from prometheus_client import Counter, Histogram  # type: ignore

    _PROM = True
except ImportError:  # pragma: no cover – optional dependency

    class _NoOp:  # pylint: disable=too-few-public-methods
        def inc(self, *_a, **_kw):
            pass

        def observe(self, *_a, **_kw):
            pass

    def Counter(*_a, **_kw):  # type: ignore
        return _NoOp()

    def Histogram(*_a, **_kw):  # type: ignore
        return _NoOp()

    _PROM = False

__all__ = ["StagePrometheusExporter"]


class StagePrometheusExporter:  # pylint: disable=too-few-public-methods
    """Utility for recording per-stage execution statistics in Prometheus.

    The Prometheus Python client enforces *unique* metric names within a
    process-wide :pyclass:`~prometheus_client.registry.CollectorRegistry`.
    Our earlier implementation created **one Counter/Histogram *per stage***
    (all sharing the same metric name).  When a second stage was exercised
    the client tried to re-register a metric with an already-used name and
    raised ``ValueError: Duplicated timeseries in CollectorRegistry`` –
    exactly what the failing tests revealed.

    The fix is to register each metric *once* and leverage the built-in
    ``labels(...)`` mechanism to obtain child objects for individual stage
    labels.  This keeps the exporter cheap while remaining thread-safe (the
    Prometheus client handles locking internally).
    """

    # --- class-level, lazily-initialised metric definitions ------------

    _SUCCESS_COUNTER: "Counter" | None = None
    _FAILURE_COUNTER: "Counter" | None = None
    _LATENCY_HISTOGRAM: "Histogram" | None = None

    # Caches for the *labelled* metric children, keyed by stage name ------
    _success_children: Dict[str, Counter] = {}
    _failure_children: Dict[str, Counter] = {}
    _latency_children: Dict[str, Histogram] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_metrics(cls) -> None:
        """Create the base Prometheus metrics if they don't exist."""
        if not _PROM or cls._SUCCESS_COUNTER is not None:
            return  # Already initialised or Prometheus unavailable

        cls._SUCCESS_COUNTER = Counter(
            "stage_success_total",
            "Successful executions of a Stage",
            labelnames=("stage",),
        )
        cls._FAILURE_COUNTER = Counter(
            "stage_failure_total",
            "Failed executions of a Stage",
            labelnames=("stage",),
        )
        cls._LATENCY_HISTOGRAM = Histogram(
            "stage_latency_seconds",
            "Stage execution latency in seconds",
            labelnames=("stage",),
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
        )

    @classmethod
    def _children(cls, stage: str):
        """Return (success, failure, latency) metric children for *stage*."""
        cls._ensure_metrics()

        if stage not in cls._success_children:
            # The mypy ignore is required because we guard initialisation with
            # `_ensure_metrics` but mypy can't prove it.
            cls._success_children[stage] = cls._SUCCESS_COUNTER.labels(stage)  # type: ignore[union-attr]
            cls._failure_children[stage] = cls._FAILURE_COUNTER.labels(stage)  # type: ignore[union-attr]
            cls._latency_children[stage] = cls._LATENCY_HISTOGRAM.labels(stage)  # type: ignore[union-attr]

        return (
            cls._success_children[stage],
            cls._failure_children[stage],
            cls._latency_children[stage],
        )

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    @classmethod
    def record(cls, stage_name: str, duration_s: float, success: bool) -> None:
        """Record single execution stats for *stage_name*."""
        if not _PROM:
            return
        succ, fail, lat = cls._children(stage_name)
        (succ if success else fail).inc()
        lat.observe(duration_s) 