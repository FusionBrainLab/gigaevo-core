"""Hydra-instantiation tests for config/aggregator/*.yaml.

Tests the declarative Heilbron aggregator YAMLs used by
SharedBenchmarkFilteredLineageStage. These YAMLs MUST:

1. Resolve under Hydra to :class:`ConfigurableAggregator` instances.
2. Produce the same program-level metrics schema that the frozen
   ``problems/heilbron_repro_v1/pop_{a,b}/evaluate.py`` emits — this
   is the parity gate.
3. Be composable through the ``heilbron_repro_v1`` pipeline config's
   ``defaults:`` list at ``pipeline_builder.lineage_filter.aggregator``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
import pytest

from gigaevo.programs.metrics.aggregators import ConfigurableAggregator
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGG_DIR = PROJECT_ROOT / "config" / "aggregator"
PIPELINE_DIR = PROJECT_ROOT / "config" / "pipeline"
IMPROVER_YAML = AGG_DIR / "heilbron_improver.yaml"
CONSTRUCTOR_YAML = AGG_DIR / "heilbron_constructor.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> MetricsContext:
    """Minimal MetricsContext whose .is_valid() gates records on is_valid==1.0."""
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness", higher_is_better=True, is_primary=True
            ),
            "is_valid": MetricSpec(description="validity", higher_is_better=True),
        }
    )


def _load_module(name: str, file: Path, extra_path: Path):
    sys.path.insert(0, str(extra_path))
    try:
        spec = importlib.util.spec_from_file_location(name, file)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(extra_path))


def _load_pop(problem: str, pop: str):
    pop_dir = PROJECT_ROOT / "problems" / problem / pop
    helper = _load_module(f"{problem}_{pop}_helper", pop_dir / "helper.py", pop_dir)
    ev = _load_module(f"{problem}_{pop}_eval", pop_dir / "evaluate.py", pop_dir)
    return ev, helper


def _seed_grid(helper_mod, n: int = 11, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A, B, C = helper_mod.get_unit_triangle()
    pts = []
    rows = 5
    count = 0
    for row in range(rows):
        num = rows - row
        v = (row + 0.5) / rows
        for i in range(num):
            if count >= n:
                break
            u = (i + 0.5) / num * (1.0 - v)
            P = (1 - u - v) * A + u * B + v * C
            P = P + rng.uniform(-0.001, 0.001, size=2)
            pts.append(P)
            count += 1
        if count >= n:
            break
    return np.array(pts)


def _noop_improver(p):
    return p.copy()


def _centroid_improver(p):
    p = p.copy()
    centroid = p.mean(axis=0)
    p[0] = 0.7 * p[0] + 0.3 * centroid
    return p


# ===========================================================================
# heilbron_improver.yaml (pop_b / D-side)
# ===========================================================================


class TestHeilbronImproverYAML:
    IMPROVER_KEYS = frozenset(
        {
            "is_valid",
            "n_opponents",
            "fitness",
            "actual_fitness",
            "mean_pre_quality",
            "mean_post_quality",
            "max_post_quality",
            "mean_improvement_raw",
        }
    )

    def test_yaml_resolves_to_configurable_aggregator(self):
        cfg = OmegaConf.load(IMPROVER_YAML)
        agg = hydra.utils.instantiate(cfg, metrics_context=_ctx())
        assert isinstance(agg, ConfigurableAggregator)
        assert agg.output_keys == self.IMPROVER_KEYS

    def test_yaml_reduces_per_opp_records_correctly(self):
        cfg = OmegaConf.load(IMPROVER_YAML)
        agg = hydra.utils.instantiate(cfg, metrics_context=_ctx())
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
        assert result["is_valid"] == 1.0
        assert result["n_opponents"] == 2.0
        assert result["fitness"] == pytest.approx((0.5 + 0.2) / 2)
        assert result["actual_fitness"] == 0.35
        assert result["mean_pre_quality"] == pytest.approx(0.30)
        assert result["mean_post_quality"] == pytest.approx(0.33)
        assert result["max_post_quality"] == 0.35
        assert result["mean_improvement_raw"] == pytest.approx(0.03)

    def test_yaml_invalid_defaults_returned_on_empty_per_opp(self):
        cfg = OmegaConf.load(IMPROVER_YAML)
        agg = hydra.utils.instantiate(cfg, metrics_context=_ctx())
        result = agg.aggregate([], intrinsic={})
        assert result["is_valid"] == 0.0
        assert result["n_opponents"] == 0.0
        assert result["fitness"] == -1.0
        assert result["actual_fitness"] == -1.0
        assert result["mean_pre_quality"] == -1.0
        assert result["mean_post_quality"] == -1.0
        assert result["max_post_quality"] == -1.0
        assert result["mean_improvement_raw"] == -1.0


# ===========================================================================
# Pipeline composition — heilbron_repro_v1 wires aggregator via defaults:
# ===========================================================================


class TestHeilbronReproV1PipelineComposition:
    """The pipeline YAML must install heilbron_improver.yaml at
    ``pipeline_builder.lineage_filter.aggregator`` via its ``defaults:`` list.
    """

    def test_pipeline_yaml_references_aggregator(self):
        """Composition smoke-test: check that heilbron_repro_v1.yaml wires
        aggregators via ${ref:aggregator} (unresolved string, not instantiated).

        We compose with aggregator=heilbron_improver to get the aggregator
        config into the top-level namespace, then verify both
        pipeline_builder.aggregator and lineage_filter.aggregator point to
        the shared singleton via the ref resolver.
        """
        from hydra import compose, initialize_config_dir

        config_dir = str(PROJECT_ROOT / "config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "pipeline=heilbron_repro_v1",
                    "aggregator=heilbron_improver",
                    "problem.name=heilbron_repro_v1/pop_b",
                    "redis.db=0",
                ],
            )
        # Both aggregator slots reference the shared top-level singleton.
        raw_cfg = OmegaConf.to_container(cfg, resolve=False)
        assert raw_cfg["pipeline_builder"]["aggregator"] == "${ref:aggregator}"
        assert (
            raw_cfg["pipeline_builder"]["lineage_filter"]["aggregator"]
            == "${ref:aggregator}"
        )
        # The top-level aggregator is the improver config.
        assert cfg.aggregator._target_.endswith("ConfigurableAggregator")
        # And the outputs block matches the improver schema.
        assert (
            set(cfg.aggregator.outputs.keys()) == TestHeilbronImproverYAML.IMPROVER_KEYS
        )
