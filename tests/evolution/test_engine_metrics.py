"""Tests for gigaevo.evolution.engine.metrics — EngineMetrics accumulation logic."""

from __future__ import annotations

from gigaevo.evolution.engine.metrics import EngineMetrics


def test_iteration_is_the_canonical_progress_counter():
    """Engine progress counter is named ``iteration`` (canonical, per #232)."""
    m = EngineMetrics()
    assert m.iteration == 0
    assert "total_generations" not in EngineMetrics.model_fields
    assert "total_mutants" not in EngineMetrics.model_fields


class TestEngineMetricsDefaults:
    def test_all_counters_start_at_zero(self):
        m = EngineMetrics()
        assert m.iteration == 0
        assert m.programs_processed == 0
        assert m.mutations_created == 0
        assert m.added == 0
        assert m.rejected_validation == 0
        assert m.rejected_strategy == 0
        assert m.elites_selected == 0
        assert m.submitted_for_refresh == 0


class TestRecordIngestionMetrics:
    def test_single_ingestion(self):
        m = EngineMetrics()
        m.record_ingestion_metrics(added=5, rejected_validation=2, rejected_strategy=1)

        assert m.added == 5
        assert m.rejected_validation == 2
        assert m.rejected_strategy == 1

    def test_accumulates_across_calls(self):
        m = EngineMetrics()
        m.record_ingestion_metrics(added=3, rejected_validation=1, rejected_strategy=0)
        m.record_ingestion_metrics(added=2, rejected_validation=0, rejected_strategy=4)

        assert m.added == 5
        assert m.rejected_validation == 1
        assert m.rejected_strategy == 4

    def test_zero_ingestion(self):
        m = EngineMetrics()
        m.record_ingestion_metrics(added=0, rejected_validation=0, rejected_strategy=0)

        assert m.added == 0
        assert m.rejected_validation == 0
        assert m.rejected_strategy == 0


class TestDirectFieldAccumulation:
    """elites_selected / mutations_created / submitted_for_refresh are
    incremented directly by the engine (no helper methods)."""

    def test_elites_selected_accumulates(self):
        m = EngineMetrics()
        m.elites_selected += 5
        m.elites_selected += 3
        assert m.elites_selected == 8

    def test_mutations_created_accumulates(self):
        m = EngineMetrics()
        m.mutations_created += 8
        m.mutations_created += 4
        assert m.mutations_created == 12

    def test_submitted_for_refresh_accumulates(self):
        m = EngineMetrics()
        m.submitted_for_refresh += 3
        m.submitted_for_refresh += 4
        assert m.submitted_for_refresh == 7


class TestEngineMetricsCombined:
    def test_independent_counters(self):
        """Verify that different counters update independently."""
        m = EngineMetrics()
        m.record_ingestion_metrics(added=10, rejected_validation=2, rejected_strategy=3)
        m.elites_selected += 5
        m.submitted_for_refresh += 7
        m.mutations_created += 4

        assert m.added == 10
        assert m.rejected_validation == 2
        assert m.rejected_strategy == 3
        assert m.elites_selected == 5
        assert m.submitted_for_refresh == 7
        assert m.mutations_created == 4
        # These aren't touched
        assert m.iteration == 0
        assert m.programs_processed == 0

    def test_direct_field_mutation(self):
        """iteration and programs_processed are set directly, not via record methods."""
        m = EngineMetrics()
        m.iteration = 42
        m.programs_processed = 100

        assert m.iteration == 42
        assert m.programs_processed == 100

    def test_realistic_multi_generation_scenario(self):
        """Simulate 3 generations of typical engine activity."""
        m = EngineMetrics()

        for _gen in range(3):
            m.iteration += 1
            m.elites_selected += 2
            m.mutations_created += 2
            m.record_ingestion_metrics(
                added=2, rejected_validation=1, rejected_strategy=0
            )
            m.submitted_for_refresh += 2
            m.programs_processed += 2

        assert m.iteration == 3
        assert m.elites_selected == 6
        assert m.mutations_created == 6
        assert m.added == 6
        assert m.rejected_validation == 3
        assert m.programs_processed == 6
        assert m.submitted_for_refresh == 6
