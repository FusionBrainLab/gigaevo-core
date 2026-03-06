"""Tests for gigaevo/programs/metrics/formatter.py — MetricsFormatter."""

from __future__ import annotations

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.formatter import MetricsFormatter


def _make_ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Main objective",
                higher_is_better=True,
                is_primary=True,
                lower_bound=0.0,
                upper_bound=1.0,
                decimals=5,
                unit="",
            ),
            "is_valid": MetricSpec(
                description="Validity flag",
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
                decimals=0,
                sentinel_value=0.0,
            ),
        }
    )


# ---------------------------------------------------------------------------
# format_metrics_block
# ---------------------------------------------------------------------------


class TestFormatMetricsBlock:
    def test_basic_output(self) -> None:
        fmt = MetricsFormatter(_make_ctx())
        block = fmt.format_metrics_block({"fitness": 0.12345, "is_valid": 1.0})
        assert "fitness" in block
        assert "0.12345" in block
        assert "↑ better" in block

    def test_sentinel_marked(self) -> None:
        fmt = MetricsFormatter(_make_ctx())
        # is_valid sentinel is 0.0
        block = fmt.format_metrics_block({"fitness": 0.5, "is_valid": 0.0})
        # fitness should NOT have [sentinel], but is_valid with value=sentinel
        # is_valid is the VALIDITY_KEY, so sentinel marker is suppressed
        assert "[sentinel]" not in block

    def test_sentinel_on_non_validity_key(self) -> None:
        ctx = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="Score",
                    higher_is_better=True,
                    is_primary=True,
                    sentinel_value=-1.0,
                ),
            }
        )
        fmt = MetricsFormatter(ctx)
        block = fmt.format_metrics_block({"score": -1.0})
        assert "[sentinel]" in block

    def test_unit_displayed(self) -> None:
        ctx = MetricsContext(
            specs={
                "speed": MetricSpec(
                    description="Speed",
                    higher_is_better=True,
                    is_primary=True,
                    unit="m/s",
                    decimals=2,
                ),
            }
        )
        fmt = MetricsFormatter(ctx)
        block = fmt.format_metrics_block({"speed": 3.14})
        assert "m/s" in block

    def test_lower_is_better_arrow(self) -> None:
        ctx = MetricsContext(
            specs={
                "cost": MetricSpec(
                    description="Cost",
                    higher_is_better=False,
                    is_primary=True,
                ),
            }
        )
        fmt = MetricsFormatter(ctx)
        block = fmt.format_metrics_block({"cost": 42.0})
        assert "↓ better" in block


# ---------------------------------------------------------------------------
# format_delta_block — table style
# ---------------------------------------------------------------------------


class TestFormatDeltaBlockTable:
    def _make_fmt(self) -> MetricsFormatter:
        ctx = MetricsContext(
            specs={
                "fitness": MetricSpec(
                    description="Main",
                    higher_is_better=True,
                    is_primary=True,
                    decimals=2,
                    significant_change=0.05,
                ),
                "cost": MetricSpec(
                    description="Cost",
                    higher_is_better=False,
                    decimals=2,
                ),
            }
        )
        return MetricsFormatter(ctx)

    def test_table_style_default(self) -> None:
        fmt = self._make_fmt()
        result = fmt.format_delta_block(
            parent={"fitness": 0.5, "cost": 10.0},
            child={"fitness": 0.6, "cost": 8.0},
            include_primary=True,
        )
        assert "| metric |" in result
        assert "fitness" in result
        assert "cost" in result

    def test_exclude_primary(self) -> None:
        fmt = self._make_fmt()
        result = fmt.format_delta_block(
            parent={"fitness": 0.5, "cost": 10.0},
            child={"fitness": 0.6, "cost": 8.0},
            include_primary=False,
        )
        assert "fitness" not in result
        assert "cost" in result

    def test_improved_impact(self) -> None:
        fmt = self._make_fmt()
        result = fmt.format_delta_block(
            parent={"fitness": 0.5, "cost": 10.0},
            child={"fitness": 0.6, "cost": 8.0},
            include_primary=True,
        )
        assert "improved" in result

    def test_worsened_impact(self) -> None:
        fmt = self._make_fmt()
        result = fmt.format_delta_block(
            parent={"fitness": 0.5, "cost": 10.0},
            child={"fitness": 0.3, "cost": 15.0},
            include_primary=True,
        )
        assert "worsened" in result

    def test_no_change_impact(self) -> None:
        fmt = self._make_fmt()
        result = fmt.format_delta_block(
            parent={"fitness": 0.5, "cost": 10.0},
            child={"fitness": 0.5, "cost": 10.0},
            include_primary=True,
        )
        assert "no change" in result

    def test_significant_change_star(self) -> None:
        fmt = self._make_fmt()
        # fitness significant_change=0.05, delta=0.1 → starred
        result = fmt.format_delta_block(
            parent={"fitness": 0.5, "cost": 10.0},
            child={"fitness": 0.6, "cost": 10.0},
            include_primary=True,
        )
        assert "★" in result

    def test_zero_parent_shows_dash(self) -> None:
        fmt = self._make_fmt()
        result = fmt.format_delta_block(
            parent={"fitness": 0.0, "cost": 0.0},
            child={"fitness": 0.5, "cost": 5.0},
            include_primary=True,
        )
        # Parent is zero → pct is None → shows "—"
        assert "—" in result

    def test_pct_clamped_gt_100(self) -> None:
        fmt = self._make_fmt()
        # parent=0.01, child=1.0 → delta=0.99, pct=9900% → clamped to >+100.0%
        result = fmt.format_delta_block(
            parent={"fitness": 0.01, "cost": 10.0},
            child={"fitness": 1.0, "cost": 10.0},
            include_primary=True,
        )
        assert ">+100.0%" in result

    def test_pct_clamped_lt_minus_100(self) -> None:
        ctx = MetricsContext(
            specs={
                "x": MetricSpec(
                    description="X",
                    higher_is_better=True,
                    is_primary=True,
                    decimals=2,
                ),
            }
        )
        fmt2 = MetricsFormatter(ctx)
        result = fmt2.format_delta_block(
            parent={"x": 0.5},
            child={"x": -1.0},
            include_primary=True,
        )
        assert "<-100.0%" in result

    def test_empty_rows_returns_na(self) -> None:
        ctx = MetricsContext(
            specs={
                "x": MetricSpec(
                    description="X",
                    higher_is_better=True,
                    is_primary=True,
                ),
            }
        )
        fmt = MetricsFormatter(ctx)
        # exclude primary, no other keys → empty rows
        result = fmt.format_delta_block(
            parent={"x": 1.0},
            child={"x": 2.0},
            include_primary=False,
        )
        assert result == "N/A"


# ---------------------------------------------------------------------------
# format_delta_block — bullets style
# ---------------------------------------------------------------------------


class TestFormatDeltaBlockBullets:
    def test_bullets_style(self) -> None:
        ctx = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="Score",
                    higher_is_better=True,
                    is_primary=True,
                    decimals=2,
                ),
                "time": MetricSpec(
                    description="Time",
                    higher_is_better=False,
                    decimals=2,
                ),
            }
        )
        fmt = MetricsFormatter(ctx)
        result = fmt.format_delta_block(
            parent={"score": 1.0, "time": 5.0},
            child={"score": 2.0, "time": 3.0},
            include_primary=True,
            style="bullets",
        )
        assert result.startswith("- ")
        assert "→" in result


# ---------------------------------------------------------------------------
# format_metrics_description
# ---------------------------------------------------------------------------


class TestFormatMetricsDescription:
    def test_basic_description(self) -> None:
        ctx = MetricsContext(
            specs={
                "fitness": MetricSpec(
                    description="Main fitness",
                    higher_is_better=True,
                    is_primary=True,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    unit="pts",
                ),
                "cost": MetricSpec(
                    description="Compute cost",
                    higher_is_better=False,
                    lower_bound=0.0,
                    upper_bound=100.0,
                ),
            }
        )
        fmt = MetricsFormatter(ctx)
        desc = fmt.format_metrics_description()
        lines = desc.split("\n")
        # Primary first
        assert lines[0].startswith("- fitness:")
        assert "↑ better" in lines[0]
        assert "[0.0, 1.0] range" in lines[0]
        assert 'unit="pts"' in lines[0]

    def test_no_bounds_omitted(self) -> None:
        ctx = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="Score",
                    higher_is_better=True,
                    is_primary=True,
                ),
            }
        )
        fmt = MetricsFormatter(ctx)
        desc = fmt.format_metrics_description()
        assert "range" not in desc

    def test_lower_is_better_arrow(self) -> None:
        ctx = MetricsContext(
            specs={
                "cost": MetricSpec(
                    description="Cost",
                    higher_is_better=False,
                    is_primary=True,
                ),
            }
        )
        fmt = MetricsFormatter(ctx)
        desc = fmt.format_metrics_description()
        assert "↓ better" in desc
