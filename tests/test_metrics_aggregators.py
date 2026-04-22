"""Tests for gigaevo/programs/metrics/aggregators.py.

Declarative, task-agnostic aggregator built from composable OutputSpec
primitives. No hardcoded task knowledge in Python.
"""

from __future__ import annotations

import pytest

from gigaevo.programs.metrics.aggregators import (
    ConfigurableAggregator,
    ConstantSpec,
    IntrinsicSpec,
    LinearSpec,
    MetricsAggregator,
    OutputSpec,
    ReduceSpec,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec


def _ctx() -> MetricsContext:
    """Build a minimal MetricsContext whose .is_valid() gates records."""
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness", higher_is_better=True, is_primary=True
            ),
            "is_valid": MetricSpec(description="validity flag", higher_is_better=True),
        }
    )


# ---------------------------------------------------------------------------
# OutputSpec primitives
# ---------------------------------------------------------------------------


class TestConstantSpec:
    def test_returns_value_regardless_of_inputs(self) -> None:
        spec = ConstantSpec(value=0.5)
        assert spec.compute([], {}, {}) == 0.5
        assert spec.compute([{"x": 1.0}], {"x": 99.0}, {"x": -1.0}) == 0.5

    def test_value_coerced_to_float(self) -> None:
        assert ConstantSpec(value=1).compute([], {}, {}) == 1.0


class TestIntrinsicSpec:
    def test_reads_key_from_intrinsic(self) -> None:
        assert IntrinsicSpec(key="quality").compute([], {"quality": 0.7}, {}) == 0.7

    def test_missing_key_uses_default(self) -> None:
        assert IntrinsicSpec(key="nope", default=-1.0).compute([], {}, {}) == -1.0

    def test_default_default_is_zero(self) -> None:
        assert IntrinsicSpec(key="nope").compute([], {}, {}) == 0.0


class TestReduceSpec:
    @pytest.mark.parametrize(
        "op,field,records,expected",
        [
            ("mean", "x", [{"x": 1.0}, {"x": 3.0}], 2.0),
            ("max", "x", [{"x": 1.0}, {"x": 3.0}, {"x": 2.0}], 3.0),
            ("min", "x", [{"x": 1.0}, {"x": 3.0}, {"x": -4.0}], -4.0),
            ("sum", "x", [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}], 6.0),
        ],
    )
    def test_reduce_ops(self, op, field, records, expected) -> None:
        assert ReduceSpec(op=op, field=field).compute(records, {}, {}) == expected

    def test_count_ignores_field(self) -> None:
        spec = ReduceSpec(op="count")
        assert spec.compute([{"x": 1.0}, {"y": 2.0}], {}, {}) == 2.0

    def test_unknown_op_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reduce op"):
            ReduceSpec(op="median", field="x")

    def test_non_count_without_field_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a `field`"):
            ReduceSpec(op="mean")


class TestLinearSpec:
    def test_intrinsic_plus_output(self) -> None:
        spec = LinearSpec(
            terms=[
                {"coeff": 0.5, "source": "intrinsic", "key": "quality"},
                {"coeff": 0.5, "source": "output", "key": "resistance"},
            ]
        )
        result = spec.compute([], {"quality": 0.4}, {"resistance": 0.8})
        assert result == pytest.approx(0.5 * 0.4 + 0.5 * 0.8)

    def test_bad_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Bad source"):
            LinearSpec(terms=[{"coeff": 1.0, "source": "whatever", "key": "x"}])

    def test_coeffs_coerced_to_float(self) -> None:
        spec = LinearSpec(terms=[{"coeff": 2, "source": "intrinsic", "key": "x"}])
        assert spec.compute([], {"x": 3.0}, {}) == 6.0


# ---------------------------------------------------------------------------
# MetricsAggregator ABC
# ---------------------------------------------------------------------------


class TestAbstractBase:
    def test_base_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            MetricsAggregator()  # type: ignore[abstract]

    def test_output_spec_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            OutputSpec()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# ConfigurableAggregator
# ---------------------------------------------------------------------------


def _basic_aggregator(ctx: MetricsContext | None = None) -> ConfigurableAggregator:
    """Constructor-role-shaped aggregator, small enough to test mechanics."""
    return ConfigurableAggregator(
        outputs={
            "is_valid": ConstantSpec(value=1.0),
            "n_opponents": ReduceSpec(op="count"),
            "quality": IntrinsicSpec(key="quality"),
            "resistance": ReduceSpec(op="mean", field="resistance_score"),
            "fitness": LinearSpec(
                terms=[
                    {"coeff": 0.5, "source": "intrinsic", "key": "quality"},
                    {"coeff": 0.5, "source": "output", "key": "resistance"},
                ]
            ),
        },
        invalid_defaults={
            "is_valid": 0.0,
            "n_opponents": 0.0,
            "quality": 0.0,
            "resistance": 1.0,
            "fitness": 0.0,
        },
        metrics_context=ctx or _ctx(),
    )


class TestConfigurableAggregator:
    def test_happy_path_all_valid(self) -> None:
        agg = _basic_aggregator()
        per_opp = [
            {"resistance_score": 1.0, "is_valid": 1.0},
            {"resistance_score": 0.0, "is_valid": 1.0},
        ]
        result = agg.aggregate(per_opp, intrinsic={"quality": 0.4})
        assert result["is_valid"] == 1.0
        assert result["n_opponents"] == 2.0
        assert result["quality"] == 0.4
        assert result["resistance"] == pytest.approx(0.5)
        assert result["fitness"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.5)

    def test_all_invalid_returns_defaults_verbatim(self) -> None:
        agg = _basic_aggregator()
        per_opp = [{"resistance_score": 1.0, "is_valid": 0.0}]
        result = agg.aggregate(per_opp, intrinsic={"quality": 0.4})
        assert result == {
            "is_valid": 0.0,
            "n_opponents": 0.0,
            "quality": 0.0,
            "resistance": 1.0,
            "fitness": 0.0,
        }

    def test_empty_per_opp_returns_defaults(self) -> None:
        agg = _basic_aggregator()
        result = agg.aggregate([], intrinsic={"quality": 0.4})
        assert result["is_valid"] == 0.0
        assert result["fitness"] == 0.0

    def test_uses_metrics_context_is_valid_gate(self) -> None:
        """Records with is_valid < 1.0 are rejected — context semantics."""
        agg = _basic_aggregator()
        per_opp = [
            {"resistance_score": 1.0, "is_valid": 1.0},
            {"resistance_score": 0.0, "is_valid": 0.5},  # rejected by ctx.is_valid
        ]
        result = agg.aggregate(per_opp, intrinsic={"quality": 0.4})
        assert result["n_opponents"] == 1.0
        assert result["resistance"] == 1.0

    def test_missing_validity_key_defaults_to_valid(self) -> None:
        """MetricsContext.is_valid defaults missing VALIDITY_KEY to 1.0."""
        agg = _basic_aggregator()
        per_opp = [{"resistance_score": 1.0}, {"resistance_score": 0.0}]
        result = agg.aggregate(per_opp, intrinsic={"quality": 0.4})
        assert result["n_opponents"] == 2.0

    def test_invalid_defaults_must_cover_all_outputs(self) -> None:
        with pytest.raises(ValueError, match="invalid_defaults missing keys"):
            ConfigurableAggregator(
                outputs={"a": ConstantSpec(value=1.0), "b": ConstantSpec(value=2.0)},
                invalid_defaults={"a": 0.0},
                metrics_context=_ctx(),
            )

    def test_output_keys_frozenset(self) -> None:
        agg = _basic_aggregator()
        assert isinstance(agg.output_keys, frozenset)
        assert "fitness" in agg.output_keys
        assert "resistance" in agg.output_keys

    def test_linear_spec_references_earlier_output(self) -> None:
        """LinearSpec can read outputs declared earlier in the dict."""
        agg = ConfigurableAggregator(
            outputs={
                "a": ConstantSpec(value=2.0),
                "b": LinearSpec(terms=[{"coeff": 3.0, "source": "output", "key": "a"}]),
            },
            invalid_defaults={"a": 0.0, "b": 0.0},
            metrics_context=_ctx(),
        )
        result = agg.aggregate([{"is_valid": 1.0}], intrinsic={})
        assert result["a"] == 2.0
        assert result["b"] == 6.0

    def test_linear_spec_referencing_later_output_raises(self) -> None:
        """A LinearSpec that reads an output declared AFTER it raises KeyError."""
        agg = ConfigurableAggregator(
            outputs={
                "b_first": LinearSpec(
                    terms=[{"coeff": 1.0, "source": "output", "key": "a_later"}]
                ),
                "a_later": ConstantSpec(value=1.0),
            },
            invalid_defaults={"b_first": 0.0, "a_later": 0.0},
            metrics_context=_ctx(),
        )
        with pytest.raises(KeyError):
            agg.aggregate([{"is_valid": 1.0}], intrinsic={})

    def test_preserves_output_declaration_order(self) -> None:
        agg = _basic_aggregator()
        result = agg.aggregate(
            [{"resistance_score": 0.0, "is_valid": 1.0}], intrinsic={"quality": 0.4}
        )
        assert list(result.keys()) == [
            "is_valid",
            "n_opponents",
            "quality",
            "resistance",
            "fitness",
        ]

    def test_heilbron_improver_shape_via_config(self) -> None:
        """Smoke-check: a D-shaped aggregator built from primitives produces
        the exact metrics schema evaluate.py emits on a full opponent set.
        """
        agg = ConfigurableAggregator(
            outputs={
                "is_valid": ConstantSpec(value=1.0),
                "n_opponents": ReduceSpec(op="count"),
                "fitness": ReduceSpec(op="mean", field="score"),
                "actual_fitness": ReduceSpec(op="max", field="post_q"),
                "mean_pre_quality": ReduceSpec(op="mean", field="pre_q"),
                "mean_post_quality": ReduceSpec(op="mean", field="post_q"),
                "max_post_quality": ReduceSpec(op="max", field="post_q"),
                "mean_improvement_raw": ReduceSpec(op="mean", field="delta"),
            },
            invalid_defaults={
                "is_valid": 0.0,
                "n_opponents": 0.0,
                "fitness": -1.0,
                "actual_fitness": -1.0,
                "mean_pre_quality": -1.0,
                "mean_post_quality": -1.0,
                "max_post_quality": -1.0,
                "mean_improvement_raw": -1.0,
            },
            metrics_context=_ctx(),
        )
        per_opp = [
            {
                "pre_q": 0.30,
                "post_q": 0.35,
                "delta": 0.05,
                "score": 0.5,
                "is_valid": 1.0,
            },
            {
                "pre_q": 0.30,
                "post_q": 0.31,
                "delta": 0.01,
                "score": 0.2,
                "is_valid": 1.0,
            },
        ]
        result = agg.aggregate(per_opp, intrinsic={})
        assert result["fitness"] == pytest.approx((0.5 + 0.2) / 2)
        assert result["actual_fitness"] == 0.35
        assert result["max_post_quality"] == 0.35
        assert result["mean_pre_quality"] == pytest.approx(0.30)
        assert result["mean_post_quality"] == pytest.approx(0.33)
        assert result["mean_improvement_raw"] == pytest.approx(0.03)
        assert result["n_opponents"] == 2.0


class TestNullAggregator:
    def test_is_metrics_aggregator_subclass(self):
        from gigaevo.programs.metrics.aggregators import (
            MetricsAggregator,
            NullAggregator,
        )

        assert issubclass(NullAggregator, MetricsAggregator)

    def test_output_keys_is_empty(self):
        from gigaevo.programs.metrics.aggregators import NullAggregator

        assert NullAggregator().output_keys == frozenset()

    def test_aggregate_is_a_noop_returning_empty(self):
        """NullAggregator is a sentinel — the builder gates on isinstance and
        never actually calls it. But if something does call it, return {}."""
        from gigaevo.programs.metrics.aggregators import NullAggregator

        assert NullAggregator().aggregate([], {}) == {}
        assert NullAggregator().aggregate([{"x": 1.0}], {"y": 2.0}) == {}
