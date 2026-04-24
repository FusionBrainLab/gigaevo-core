"""Declarative metrics aggregators.

The SBF-LineageStage produces a parent/child metrics dict restricted to the
subset of G opponents both programs have faced. That dict must match the
program.metrics schema so `MetricsFormatter` can render it into a prompt.

This module supplies ONE general aggregator class, `ConfigurableAggregator`,
composed of small `OutputSpec` primitives (`ConstantSpec`, `IntrinsicSpec`,
`ReduceSpec`, `LinearSpec`). Per-population / per-task semantics live
entirely in YAML — no Python subclasses per task.

Validity is delegated to :meth:`MetricsContext.is_valid`, which is the
single source of truth for what "valid evaluation" means.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from gigaevo.programs.metrics.context import MetricsContext

__all__ = [
    "ConfigurableAggregator",
    "ConstantSpec",
    "IntrinsicSpec",
    "LinearSpec",
    "MetricsAggregator",
    "NullAggregator",
    "OutputSpec",
    "ReduceSpec",
]


# ---------------------------------------------------------------------------
# OutputSpec primitives
# ---------------------------------------------------------------------------


class OutputSpec(ABC):
    """One output metric's computation rule. Task-agnostic."""

    @abstractmethod
    def compute(
        self,
        per_opp: Sequence[Mapping[str, float]],
        intrinsic: Mapping[str, float],
        computed: Mapping[str, float],
    ) -> float:
        """Compute this metric's value.

        Args:
            per_opp: per-opponent records already filtered through the
                aggregator's validity gate.
            intrinsic: the program's own ``metrics`` dict — values that do
                not depend on which opponents were faced.
            computed: outputs produced earlier in this aggregate() call.
                A spec may only read outputs declared before it in the
                aggregator's ``outputs`` mapping.
        """


class ConstantSpec(OutputSpec):
    """Always return a fixed value."""

    def __init__(self, value: float):
        self._value = float(value)

    def compute(self, per_opp, intrinsic, computed):  # noqa: ARG002
        return self._value


class IntrinsicSpec(OutputSpec):
    """Pass through ``intrinsic[key]`` unchanged."""

    def __init__(self, key: str, default: float = 0.0):
        self._key = key
        self._default = float(default)

    def compute(self, per_opp, intrinsic, computed):  # noqa: ARG002
        return float(intrinsic.get(self._key, self._default))


class ReduceSpec(OutputSpec):
    """Reduce per-opponent values with ``op`` over ``field``.

    ``op`` in {``mean``, ``max``, ``min``, ``sum``, ``count``}. For
    ``count``, ``field`` is ignored.
    """

    _OPS: dict[str, Callable[[Sequence[float]], float]] = {
        "mean": lambda xs: sum(xs) / len(xs),
        "max": lambda xs: float(max(xs)),
        "min": lambda xs: float(min(xs)),
        "sum": lambda xs: float(sum(xs)),
        "count": lambda xs: float(len(xs)),
    }

    def __init__(self, op: str, field: str | None = None):
        if op not in self._OPS:
            raise ValueError(f"Unknown reduce op: {op!r}")
        if op != "count" and field is None:
            raise ValueError(f"Op {op!r} requires a `field`")
        self._op = op
        self._field = field

    def compute(self, per_opp, intrinsic, computed):  # noqa: ARG002
        if self._op == "count":
            return float(len(per_opp))
        values = [float(r[self._field]) for r in per_opp]
        return float(self._OPS[self._op](values))


class LinearSpec(OutputSpec):
    """Linear combination of intrinsic values and earlier outputs.

    Each term: ``{"coeff": float, "source": "intrinsic"|"output", "key": str}``.
    """

    def __init__(self, terms: Sequence[Mapping[str, object]]):
        parsed: list[tuple[float, str, str]] = []
        for term in terms:
            source = str(term["source"])
            if source not in ("intrinsic", "output"):
                raise ValueError(
                    f"Bad source: {source!r} (expected 'intrinsic' or 'output')"
                )
            parsed.append(
                (
                    float(cast(float, term["coeff"])),
                    source,
                    str(term["key"]),
                )
            )
        self._terms = parsed

    def compute(self, per_opp, intrinsic, computed):  # noqa: ARG002
        total = 0.0
        for coeff, source, key in self._terms:
            src = intrinsic if source == "intrinsic" else computed
            total += coeff * float(src[key])
        return total


# ---------------------------------------------------------------------------
# The aggregator
# ---------------------------------------------------------------------------


class MetricsAggregator(ABC):
    """Re-compute program-level metrics from per-opponent records.

    Callers pass the per-opponent primitives produced by evaluate.py
    (from any subset of opponents) and the program's intrinsic metrics;
    `aggregate` returns a dict whose keys match ``program.metrics``.
    """

    @abstractmethod
    def aggregate(
        self,
        per_opp: Sequence[Mapping[str, float]],
        intrinsic: Mapping[str, float],
    ) -> dict[str, float]:
        """Return a metrics dict matching the program.metrics schema."""

    @property
    @abstractmethod
    def output_keys(self) -> frozenset[str]:
        """Set of metric keys this aggregator produces."""


class ConfigurableAggregator(MetricsAggregator):
    """Task-agnostic aggregator, entirely driven by config.

    Args:
        outputs: ordered mapping from metric key to :class:`OutputSpec`.
            Order matters — :class:`LinearSpec` terms with
            ``source="output"`` may only reference outputs declared
            earlier in this mapping. Referencing a later output raises
            ``KeyError`` at aggregate time.
        invalid_defaults: dict returned verbatim when zero per-opponent
            records pass the validity gate. Must supply a value for
            every key in ``outputs``.
        metrics_context: the population's
            :class:`MetricsContext`. Its ``is_valid(record)`` method
            gates each per-opponent record. Wire via
            ``metrics_context: ${ref:metrics_context}`` in the aggregator
            YAML — Hydra resolves the top-level singleton at instantiate
            time.
    """

    def __init__(
        self,
        outputs: Mapping[str, OutputSpec],
        invalid_defaults: Mapping[str, float],
        metrics_context: MetricsContext,
    ):
        self._outputs: dict[str, OutputSpec] = dict(outputs)
        self._invalid_defaults: dict[str, float] = {
            k: float(v) for k, v in invalid_defaults.items()
        }
        self._metrics_context = metrics_context
        missing = set(self._outputs) - set(self._invalid_defaults)
        if missing:
            raise ValueError(f"invalid_defaults missing keys: {sorted(missing)}")

    @property
    def output_keys(self) -> frozenset[str]:
        return frozenset(self._outputs)

    def aggregate(
        self,
        per_opp: Sequence[Mapping[str, float]],
        intrinsic: Mapping[str, float],
    ) -> dict[str, float]:
        valid = [r for r in per_opp if self._metrics_context.is_valid(r)]
        if not valid:
            return dict(self._invalid_defaults)
        computed: dict[str, float] = {}
        for key, spec in self._outputs.items():
            computed[key] = float(spec.compute(valid, intrinsic, computed))
        return computed


class NullAggregator(MetricsAggregator):
    """Sentinel 'no aggregator configured' marker.

    The pipeline builder checks ``isinstance(aggregator, NullAggregator)``
    and skips installing ParseMetricsStage when true — the DAG keeps the
    legacy ``CallValidatorFunction → FetchMetrics`` edge, and evaluate.py's
    old-contract ``metrics`` dict flows through unchanged.

    This lets Hydra's ``aggregator=none`` default resolve to a real object
    (no null footgun in ``${ref:aggregator}``) while preserving the
    "non-Heilbron pipelines untouched" scope constraint.
    """

    @property
    def output_keys(self) -> frozenset[str]:
        return frozenset()

    def aggregate(self, per_opp, intrinsic):
        return {}
