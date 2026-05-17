"""Tests for the parity harness machinery itself.

Verifies ParityReport bookkeeping and each comparison helper against
synthetic stand-ins, without requiring a full Hydra composition. The
real cross-path comparison runs in test_steady_state_hotpotqa_parity.py.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.integration.parity.harness import (
    ParityReport,
    add_deferred_comparisons,
    compare_blueprint,
    compare_problem_context,
    compare_redis_storage,
    compare_strategy,
)


class TestParityReport:
    def test_empty_report_is_green(self) -> None:
        r = ParityReport()
        assert r.is_green
        assert "0 pass, 0 fail" in r.render()

    def test_failure_makes_report_red(self) -> None:
        r = ParityReport()
        r.add_pass("x")
        r.add_failure("y", "detail")
        assert not r.is_green
        rendered = r.render()
        assert "[PASS] x" in rendered
        assert "[FAIL] y: detail" in rendered

    def test_deferred_does_not_affect_green(self) -> None:
        r = ParityReport()
        r.add_deferred("evolution_engine", "missing schema")
        assert r.is_green
        assert "[DEFER] evolution_engine: missing schema" in r.render()


class TestCompareRedisStorage:
    def _stub_storage(self, url: str, key_prefix: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(redis_url=url, key_prefix=key_prefix)
        )

    def test_matching_storage_passes(self) -> None:
        a = self._stub_storage("redis://localhost:6379/0", "gigaevo:x")
        b = self._stub_storage("redis://localhost:6379/0", "gigaevo:x")
        r = ParityReport()
        compare_redis_storage(a, b, r)
        assert r.is_green
        assert len(r.passes) == 2

    def test_url_mismatch_fails(self) -> None:
        a = self._stub_storage("redis://A:1/0", "p")
        b = self._stub_storage("redis://B:2/0", "p")
        r = ParityReport()
        compare_redis_storage(a, b, r)
        assert not r.is_green
        assert any("redis_url" in f for f in r.failures)

    def test_key_prefix_mismatch_fails(self) -> None:
        a = self._stub_storage("u", "alpha")
        b = self._stub_storage("u", "beta")
        r = ParityReport()
        compare_redis_storage(a, b, r)
        assert not r.is_green
        assert any("key_prefix" in f for f in r.failures)


class TestCompareProblemContext:
    def test_matching_problem_dir_passes(self) -> None:
        a = SimpleNamespace(problem_dir="problems/x")
        b = SimpleNamespace(problem_dir="problems/x")
        r = ParityReport()
        compare_problem_context(a, b, r)
        assert r.is_green

    def test_divergent_problem_dir_fails(self) -> None:
        a = SimpleNamespace(problem_dir="problems/x")
        b = SimpleNamespace(problem_dir="problems/y")
        r = ParityReport()
        compare_problem_context(a, b, r)
        assert not r.is_green


class _StrategyStub:
    """Stand-in for MapElitesMultiIsland that exposes the .islands
    dict the harness reads against."""

    def __init__(self, island_ids: list[str]) -> None:
        self.islands = {island_id: object() for island_id in island_ids}


class TestCompareStrategy:
    def test_matching_strategy_passes(self) -> None:
        a = _StrategyStub(["main"])
        b = _StrategyStub(["main"])
        r = ParityReport()
        compare_strategy(a, b, r)
        assert r.is_green
        assert len(r.passes) == 2  # class + island_ids

    def test_island_id_divergence_fails(self) -> None:
        a = _StrategyStub(["main"])
        b = _StrategyStub(["alpha", "beta"])
        r = ParityReport()
        compare_strategy(a, b, r)
        assert not r.is_green

    def test_class_divergence_fails(self) -> None:
        class OtherStrategy:
            def __init__(self) -> None:
                self.islands: dict[str, object] = {}

        r = ParityReport()
        compare_strategy(_StrategyStub([]), OtherStrategy(), r)
        assert not r.is_green
        assert any("strategy.class" in f for f in r.failures)


class TestCompareBlueprint:
    def test_matching_stage_set_passes(self) -> None:
        a = SimpleNamespace(nodes={"A": 1, "B": 2})
        b = SimpleNamespace(nodes={"A": 1, "B": 2})
        r = ParityReport()
        compare_blueprint(a, b, r)
        assert r.is_green

    def test_extra_stage_on_one_side_fails(self) -> None:
        a = SimpleNamespace(nodes={"A": 1, "B": 2})
        b = SimpleNamespace(nodes={"A": 1, "B": 2, "C": 3})
        r = ParityReport()
        compare_blueprint(a, b, r)
        assert not r.is_green
        assert any("only_hydra=['C']" in f for f in r.failures)


class TestAddDeferredComparisons:
    def test_records_each_phase_2_dependent_object(self) -> None:
        r = ParityReport()
        add_deferred_comparisons(r)
        deferred_names = {entry.split(":")[0].strip() for entry in r.deferred}
        for name in (
            "evolution_engine",
            "dag_runner",
            "writer",
            "metrics_tracker",
            "mutation_operator",
            "program_loader",
            "smoke_evolution_10_iter",
        ):
            assert name in deferred_names
        # All deferred — none counted as failures.
        assert r.is_green
