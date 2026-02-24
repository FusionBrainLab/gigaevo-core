"""Tests for gigaevo/programs/metrics/context.py — MetricSpec and MetricsContext."""

from __future__ import annotations

import pytest

from gigaevo.programs.metrics.context import (
    EPSILON,
    MAX_VALUE_DEFAULT,
    MIN_VALUE_DEFAULT,
    MetricsContext,
    MetricSpec,
)

# ---------------------------------------------------------------------------
# MetricSpec
# ---------------------------------------------------------------------------


class TestMetricSpec:
    def test_default_sentinel_higher_is_better(self) -> None:
        spec = MetricSpec(description="score", higher_is_better=True)
        assert spec.sentinel_value == MIN_VALUE_DEFAULT

    def test_default_sentinel_lower_is_better(self) -> None:
        spec = MetricSpec(description="cost", higher_is_better=False)
        assert spec.sentinel_value == MAX_VALUE_DEFAULT

    def test_explicit_sentinel_preserved(self) -> None:
        spec = MetricSpec(description="x", higher_is_better=True, sentinel_value=-999.0)
        assert spec.sentinel_value == -999.0

    def test_sentinel_inside_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="Sentinel value.*must be outside bounds"):
            MetricSpec(
                description="x",
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=100.0,
                sentinel_value=50.0,
            )

    def test_sentinel_outside_bounds_ok(self) -> None:
        spec = MetricSpec(
            description="x",
            higher_is_better=True,
            lower_bound=0.0,
            upper_bound=100.0,
            sentinel_value=-1.0,
        )
        assert spec.sentinel_value == -1.0

    def test_sentinel_equal_to_lower_bound_ok(self) -> None:
        # Equal to bound (not strictly inside) — should not raise
        spec = MetricSpec(
            description="x",
            higher_is_better=True,
            lower_bound=0.0,
            upper_bound=100.0,
            sentinel_value=0.0,
        )
        assert spec.sentinel_value == 0.0

    def test_is_sentinel_true(self) -> None:
        spec = MetricSpec(description="x", higher_is_better=True, sentinel_value=-1.0)
        assert spec.is_sentinel(-1.0) is True

    def test_is_sentinel_within_epsilon(self) -> None:
        spec = MetricSpec(description="x", higher_is_better=True, sentinel_value=-1.0)
        assert spec.is_sentinel(-1.0 + EPSILON / 2) is True

    def test_is_sentinel_false(self) -> None:
        spec = MetricSpec(description="x", higher_is_better=True, sentinel_value=-1.0)
        assert spec.is_sentinel(42.0) is False

    def test_is_sentinel_none_sentinel(self) -> None:
        # When sentinel_value is None (shouldn't happen with defaults, but test the branch)
        spec = MetricSpec(description="x", higher_is_better=True)
        # Manually set to None to test the branch
        spec.sentinel_value = None
        assert spec.is_sentinel(0.0) is False


# ---------------------------------------------------------------------------
# MetricsContext — validation
# ---------------------------------------------------------------------------


class TestMetricsContextValidation:
    def test_no_primary_raises(self) -> None:
        with pytest.raises(ValueError, match="Exactly one.*is_primary=True"):
            MetricsContext(
                specs={
                    "a": MetricSpec(
                        description="A", higher_is_better=True, is_primary=False
                    )
                }
            )

    def test_two_primaries_raises(self) -> None:
        with pytest.raises(ValueError, match="Exactly one.*is_primary=True"):
            MetricsContext(
                specs={
                    "a": MetricSpec(
                        description="A", higher_is_better=True, is_primary=True
                    ),
                    "b": MetricSpec(
                        description="B", higher_is_better=True, is_primary=True
                    ),
                }
            )


# ---------------------------------------------------------------------------
# MetricsContext — accessors
# ---------------------------------------------------------------------------


def _make_ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="Main score",
                higher_is_better=True,
                is_primary=True,
                lower_bound=0.0,
                upper_bound=100.0,
                decimals=3,
                unit="pts",
            ),
            "cost": MetricSpec(
                description="Computation cost",
                higher_is_better=False,
                lower_bound=0.0,
                upper_bound=50.0,
            ),
            "hidden": MetricSpec(
                description="Not in prompts",
                higher_is_better=True,
                include_in_prompts=False,
            ),
        }
    )


class TestMetricsContextAccessors:
    def test_get_primary_spec(self) -> None:
        ctx = _make_ctx()
        spec = ctx.get_primary_spec()
        assert spec.description == "Main score"
        assert spec.is_primary is True

    def test_get_primary_key(self) -> None:
        ctx = _make_ctx()
        assert ctx.get_primary_key() == "score"

    def test_get_description(self) -> None:
        ctx = _make_ctx()
        assert ctx.get_description("cost") == "Computation cost"

    def test_get_decimals(self) -> None:
        ctx = _make_ctx()
        assert ctx.get_decimals("score") == 3

    def test_metrics_descriptions(self) -> None:
        ctx = _make_ctx()
        desc = ctx.metrics_descriptions()
        assert desc["score"] == "Main score"
        assert desc["cost"] == "Computation cost"
        assert desc["hidden"] == "Not in prompts"

    def test_prompt_keys_primary_first_hidden_excluded(self) -> None:
        ctx = _make_ctx()
        keys = ctx.prompt_keys()
        assert keys[0] == "score"  # primary first
        assert "cost" in keys
        assert "hidden" not in keys  # include_in_prompts=False

    def test_additional_metrics_excludes_primary(self) -> None:
        ctx = _make_ctx()
        additional = ctx.additional_metrics()
        assert "score" not in additional
        assert "cost" in additional
        assert "hidden" in additional

    def test_get_sentinels(self) -> None:
        ctx = _make_ctx()
        sentinels = ctx.get_sentinels()
        assert "score" in sentinels
        assert "cost" in sentinels
        assert sentinels["score"] == MIN_VALUE_DEFAULT  # higher_is_better=True
        assert sentinels["cost"] == MAX_VALUE_DEFAULT  # higher_is_better=False

    def test_get_bounds_both_defined(self) -> None:
        ctx = _make_ctx()
        bounds = ctx.get_bounds("score")
        assert bounds == (0.0, 100.0)

    def test_get_bounds_not_defined(self) -> None:
        ctx = _make_ctx()
        bounds = ctx.get_bounds("hidden")  # no bounds set
        assert bounds is None

    def test_get_bounds_missing_key_raises(self) -> None:
        ctx = _make_ctx()
        with pytest.raises(KeyError):
            ctx.get_bounds("nonexistent")

    def test_is_higher_better(self) -> None:
        ctx = _make_ctx()
        assert ctx.is_higher_better("score") is True
        assert ctx.is_higher_better("cost") is False

    def test_is_higher_better_missing_key_raises(self) -> None:
        ctx = _make_ctx()
        with pytest.raises(KeyError):
            ctx.is_higher_better("nonexistent")


# ---------------------------------------------------------------------------
# MetricsContext — add_metric
# ---------------------------------------------------------------------------


class TestMetricsContextAddMetric:
    def test_add_non_primary(self) -> None:
        ctx = _make_ctx()
        new_spec = MetricSpec(description="Latency", higher_is_better=False)
        result = ctx.add_metric("latency", new_spec)
        assert result is ctx  # returns self
        assert "latency" in ctx.specs

    def test_add_duplicate_key_raises(self) -> None:
        ctx = _make_ctx()
        with pytest.raises(ValueError, match="already exists"):
            ctx.add_metric(
                "score",
                MetricSpec(description="dup", higher_is_better=True),
            )

    def test_add_primary_raises(self) -> None:
        ctx = _make_ctx()
        with pytest.raises(ValueError, match="Cannot add primary"):
            ctx.add_metric(
                "new_primary",
                MetricSpec(
                    description="conflict", higher_is_better=True, is_primary=True
                ),
            )


# ---------------------------------------------------------------------------
# MetricsContext — from_descriptions
# ---------------------------------------------------------------------------


class TestMetricsContextFromDescriptions:
    def test_minimal(self) -> None:
        ctx = MetricsContext.from_descriptions(
            primary_key="fitness",
            primary_description="Main fitness",
        )
        assert ctx.get_primary_key() == "fitness"
        assert ctx.get_primary_spec().higher_is_better is True

    def test_with_additional(self) -> None:
        ctx = MetricsContext.from_descriptions(
            primary_key="fitness",
            primary_description="Main fitness",
            additional_metrics={"cost": "Compute cost", "complexity": "Code size"},
        )
        assert len(ctx.specs) == 3
        assert ctx.specs["cost"].description == "Compute cost"

    def test_with_additional_higher_is_better(self) -> None:
        ctx = MetricsContext.from_descriptions(
            primary_key="fitness",
            primary_description="Main fitness",
            additional_metrics={"cost": "Compute cost"},
            additional_metrics_higher_is_better={"cost": False},
        )
        assert ctx.specs["cost"].higher_is_better is False

    def test_per_metric_decimals(self) -> None:
        ctx = MetricsContext.from_descriptions(
            primary_key="fitness",
            primary_description="Main fitness",
            additional_metrics={"cost": "Compute cost"},
            decimals=3,
            per_metric_decimals={"fitness": 2, "cost": 4},
        )
        assert ctx.specs["fitness"].decimals == 2
        assert ctx.specs["cost"].decimals == 4


# ---------------------------------------------------------------------------
# MetricsContext — from_dict
# ---------------------------------------------------------------------------


class TestMetricsContextFromDict:
    def test_roundtrip(self) -> None:
        ctx = MetricsContext.from_dict(
            specs={
                "score": {
                    "description": "Score",
                    "higher_is_better": True,
                    "is_primary": True,
                },
                "time": {
                    "description": "Time",
                    "higher_is_better": False,
                },
            }
        )
        assert ctx.get_primary_key() == "score"
        assert ctx.specs["time"].higher_is_better is False
