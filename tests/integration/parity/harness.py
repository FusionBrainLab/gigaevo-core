"""Parity harness comparing the typed object graph against the Hydra path.

Plan §4.10 envisions one parity test per shipped experiment that runs
both entry points on the same logical inputs and asserts the produced
object graphs are observably equivalent. This module provides the
machinery; concrete tests under tests/integration/parity/test_*.py
parameterise it across the migrated experiments.

The harness is deliberately structural: it compares CLASS IDENTITY,
KEY FIELD VALUES, and BLUEPRINT SHAPES rather than byte-identical
``__dict__`` equality. Two paths may construct objects with different
internal caches, lazy fields, or socket handles; observable behaviour
is what matters.

Comparisons fall into three buckets:

- ``passes``: actively compared and matched.
- ``failures``: actively compared and divergent (test fails).
- ``deferred``: not compared because the typed path does not yet
  construct the object (mutation_operator, dag_runner, writer,
  metrics_tracker, program_loader, evolution_engine). The Phase 2
  schemas close these gaps; this list shrinks as they land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParityReport:
    """Structured diff between the typed and Hydra object graphs."""

    passes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)

    def add_pass(self, name: str) -> None:
        self.passes.append(name)

    def add_failure(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")

    def add_deferred(self, name: str, reason: str) -> None:
        self.deferred.append(f"{name}: {reason}")

    @property
    def is_green(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [f"Parity report: {len(self.passes)} pass, "
                 f"{len(self.failures)} fail, {len(self.deferred)} deferred"]
        for label, group in (
            ("PASS", self.passes),
            ("FAIL", self.failures),
            ("DEFER", self.deferred),
        ):
            for entry in group:
                lines.append(f"  [{label}] {entry}")
        return "\n".join(lines)


def load_typed_path(experiment_py: Path) -> dict[str, Any]:
    """Load the typed path: experiment_loader -> build_object_graph."""
    from gigaevo.config.experiment_loader import build_experiment
    from gigaevo.config.object_graph import build_object_graph

    cfg = build_experiment(experiment_py)
    graph = build_object_graph(cfg)
    graph["_config"] = cfg
    return graph


def load_hydra_path(
    experiment_name: str, overrides: list[str]
) -> dict[str, Any]:
    """Retired in hydra-3.2.

    The OmegaConf custom resolvers (``get_object``, ``ref``) that the
    shipped YAMLs interpolated through were deleted along with
    ``gigaevo/config/resolvers.py``. Composing any experiment YAML
    now fails at the interpolation step. Typed-vs-typed parity
    continues to flow through :func:`load_typed_path`; the full
    parity harness removal lands in hydra-3.8 once every test that
    consumed the cross-path comparison migrates.

    The signature is preserved so existing tests still pass the
    function reference around — calling it always raises a typed
    error captured by :func:`run_parity`'s exception handler."""
    del experiment_name, overrides
    raise RuntimeError(
        "Hydra-side load retired in hydra-3.2 (resolvers.py deleted). "
        "Typed-vs-typed parity continues via load_typed_path; full "
        "harness removal lands in hydra-3.8."
    )


def compare_redis_storage(
    typed: Any, hydra: Any, report: ParityReport
) -> None:
    """Compare RedisProgramStorage configurations."""
    typed_url = typed.config.redis_url
    hydra_url = hydra.config.redis_url
    if typed_url == hydra_url:
        report.add_pass(f"redis_storage.redis_url == {typed_url}")
    else:
        report.add_failure(
            "redis_storage.redis_url",
            f"typed={typed_url!r} hydra={hydra_url!r}",
        )

    if typed.config.key_prefix == hydra.config.key_prefix:
        report.add_pass(
            f"redis_storage.key_prefix == {typed.config.key_prefix}"
        )
    else:
        report.add_failure(
            "redis_storage.key_prefix",
            f"typed={typed.config.key_prefix!r} "
            f"hydra={hydra.config.key_prefix!r}",
        )


def compare_problem_context(
    typed: Any, hydra: Any, report: ParityReport
) -> None:
    if typed.problem_dir == hydra.problem_dir:
        report.add_pass(f"problem_context.problem_dir == {typed.problem_dir}")
    else:
        report.add_failure(
            "problem_context.problem_dir",
            f"typed={typed.problem_dir!r} hydra={hydra.problem_dir!r}",
        )


def compare_strategy(typed: Any, hydra: Any, report: ParityReport) -> None:
    """Compare MapElitesMultiIsland strategy island layout."""
    if type(typed) is type(hydra):
        report.add_pass(f"strategy.class == {type(typed).__name__}")
    else:
        report.add_failure(
            "strategy.class",
            f"typed={type(typed).__name__} hydra={type(hydra).__name__}",
        )

    if set(typed.islands.keys()) == set(hydra.islands.keys()):
        report.add_pass(f"strategy.island_ids == {sorted(typed.islands)}")
    else:
        report.add_failure(
            "strategy.island_ids",
            f"typed={sorted(typed.islands)} hydra={sorted(hydra.islands)}",
        )


def compare_blueprint(typed: Any, hydra: Any, report: ParityReport) -> None:
    """Compare DAG blueprint stage names + edge counts."""
    typed_stages = set(typed.nodes.keys()) if hasattr(typed, "nodes") else set()
    hydra_stages = set(hydra.nodes.keys()) if hasattr(hydra, "nodes") else set()
    if typed_stages == hydra_stages:
        report.add_pass(f"dag_blueprint.stages ({len(typed_stages)} stages)")
    else:
        only_typed = typed_stages - hydra_stages
        only_hydra = hydra_stages - typed_stages
        report.add_failure(
            "dag_blueprint.stages",
            f"only_typed={sorted(only_typed)} only_hydra={sorted(only_hydra)}",
        )


def add_deferred_comparisons(report: ParityReport) -> None:
    """Record the comparisons the harness intends to make once the
    Phase 2 schemas close the wiring gaps."""
    deferred_objects = (
        ("evolution_engine", "engine wiring depends on hydra-2.4 logging schema"),
        ("dag_runner", "depends on hydra-2.11 runner + hydra-2.2 scheduling"),
        ("writer", "depends on hydra-2.4 logging schema"),
        ("metrics_tracker", "depends on hydra-2.4 logging schema"),
        ("mutation_operator", "depends on hydra-2.3 prompt_fetcher schema"),
        ("program_loader", "depends on loader schema (Phase 2)"),
        ("smoke_evolution_10_iter", "requires evolution_engine + dag_runner"),
    )
    for name, reason in deferred_objects:
        report.add_deferred(name, reason)


def run_parity(
    experiment_py: Path,
    hydra_experiment: str,
    hydra_overrides: list[str],
) -> ParityReport:
    """Top-level harness driver.

    Loads both paths, runs every available comparison, populates the
    deferred slots, and returns the structured report. The caller
    asserts ``report.is_green`` to fail the test on real divergences.
    """
    report = ParityReport()
    typed_graph = load_typed_path(experiment_py)

    try:
        hydra_graph = load_hydra_path(hydra_experiment, hydra_overrides)
    except Exception as exc:  # noqa: BLE001
        report.add_failure(
            "load_hydra_path",
            f"hydra composition raised {type(exc).__name__}: {exc}",
        )
        add_deferred_comparisons(report)
        return report

    if "redis_storage" in typed_graph and "redis_storage" in hydra_graph:
        compare_redis_storage(
            typed_graph["redis_storage"], hydra_graph["redis_storage"], report
        )
    else:
        report.add_failure(
            "redis_storage", "missing from one or both graphs"
        )

    if "problem_context" in typed_graph and "problem_context" in hydra_graph:
        compare_problem_context(
            typed_graph["problem_context"],
            hydra_graph["problem_context"],
            report,
        )

    if "strategy" in typed_graph and "evolution_strategy" in hydra_graph:
        compare_strategy(
            typed_graph["strategy"],
            hydra_graph["evolution_strategy"],
            report,
        )

    if "dag_blueprint" in typed_graph and "dag_blueprint" in hydra_graph:
        compare_blueprint(
            typed_graph["dag_blueprint"],
            hydra_graph["dag_blueprint"],
            report,
        )

    add_deferred_comparisons(report)
    return report
