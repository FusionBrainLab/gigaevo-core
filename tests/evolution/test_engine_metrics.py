"""Tests for gigaevo.evolution.engine.metrics — EngineMetrics accumulation logic."""

from __future__ import annotations

from gigaevo.evolution.engine.metrics import EngineMetrics


class TestEngineMetricsDefaults:
    def test_all_counters_start_at_zero(self):
        m = EngineMetrics()
        assert m.total_generations == 0
        assert m.programs_processed == 0
        assert m.mutations_created == 0
        assert m.errors_encountered == 0
        assert m.added == 0
        assert m.rejected_validation == 0
        assert m.rejected_strategy == 0
        assert m.elites_selected == 0
        assert m.elites_selection_errors == 0
        assert m.submitted_for_refresh == 0
        assert m.mutations_creation_errors == 0


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


class TestRecordEliteSelectionMetrics:
    def test_single_selection(self):
        m = EngineMetrics()
        m.record_elite_selection_metrics(elites_selected=10, elites_selection_errors=1)

        assert m.elites_selected == 10
        assert m.elites_selection_errors == 1

    def test_accumulates_across_calls(self):
        m = EngineMetrics()
        m.record_elite_selection_metrics(elites_selected=5, elites_selection_errors=0)
        m.record_elite_selection_metrics(elites_selected=3, elites_selection_errors=2)

        assert m.elites_selected == 8
        assert m.elites_selection_errors == 2


class TestRecordReprocessMetrics:
    def test_single_reprocess(self):
        m = EngineMetrics()
        m.record_reprocess_metrics(submitted_for_refresh=7)

        assert m.submitted_for_refresh == 7

    def test_accumulates_across_calls(self):
        m = EngineMetrics()
        m.record_reprocess_metrics(submitted_for_refresh=3)
        m.record_reprocess_metrics(submitted_for_refresh=4)

        assert m.submitted_for_refresh == 7


class TestRecordMutationMetrics:
    def test_single_mutation(self):
        m = EngineMetrics()
        m.record_mutation_metrics(mutations_created=8, mutations_creation_errors=2)

        assert m.mutations_created == 8
        assert m.mutations_creation_errors == 2

    def test_accumulates_across_calls(self):
        m = EngineMetrics()
        m.record_mutation_metrics(mutations_created=5, mutations_creation_errors=1)
        m.record_mutation_metrics(mutations_created=3, mutations_creation_errors=0)

        assert m.mutations_created == 8
        assert m.mutations_creation_errors == 1


class TestEngineMetricsCombined:
    def test_all_record_methods_independent(self):
        """Verify that different record methods update independent counters."""
        m = EngineMetrics()
        m.record_ingestion_metrics(added=10, rejected_validation=2, rejected_strategy=3)
        m.record_elite_selection_metrics(elites_selected=5, elites_selection_errors=1)
        m.record_reprocess_metrics(submitted_for_refresh=7)
        m.record_mutation_metrics(mutations_created=4, mutations_creation_errors=0)

        assert m.added == 10
        assert m.rejected_validation == 2
        assert m.rejected_strategy == 3
        assert m.elites_selected == 5
        assert m.elites_selection_errors == 1
        assert m.submitted_for_refresh == 7
        assert m.mutations_created == 4
        assert m.mutations_creation_errors == 0
        # These aren't touched by record methods
        assert m.total_generations == 0
        assert m.programs_processed == 0
        assert m.errors_encountered == 0

    def test_direct_field_mutation(self):
        """total_generations and programs_processed are set directly, not via record methods."""
        m = EngineMetrics()
        m.total_generations = 42
        m.programs_processed = 100
        m.errors_encountered = 3

        assert m.total_generations == 42
        assert m.programs_processed == 100
        assert m.errors_encountered == 3

    def test_realistic_multi_generation_scenario(self):
        """Simulate 3 generations of typical engine activity."""
        m = EngineMetrics()

        for gen in range(3):
            m.total_generations += 1
            m.record_elite_selection_metrics(
                elites_selected=2, elites_selection_errors=0
            )
            m.record_mutation_metrics(mutations_created=2, mutations_creation_errors=0)
            m.record_ingestion_metrics(
                added=2, rejected_validation=1, rejected_strategy=0
            )
            m.record_reprocess_metrics(submitted_for_refresh=2)
            m.programs_processed += 2

        assert m.total_generations == 3
        assert m.elites_selected == 6
        assert m.mutations_created == 6
        assert m.added == 6
        assert m.rejected_validation == 3
        assert m.programs_processed == 6
        assert m.submitted_for_refresh == 6
