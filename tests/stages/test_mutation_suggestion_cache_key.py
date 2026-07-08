"""Cache-key quantization for MutationSuggestionStage.

The suggester's InputHashCache key must NOT churn on every collector
refresh: noise-level changes in EvolutionaryStatistics (fitness jitter,
archive snapshot growth, window-rank reshuffles) keep the hash stable,
while meaningful momentum changes (rounded fitness moves, trend flips,
plateau-bucket crossings) invalidate it. The agent itself still receives
the full, unquantized stats object.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.stages.collector import EvolutionaryStatistics
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.mutation_suggestions import (
    MutationSuggestionInputs,
    MutationSuggestionStage,
    offered_memory_card_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stats(**overrides) -> EvolutionaryStatistics:
    base = EvolutionaryStatistics(
        generation=1,
        iteration=42,
        current_program_metrics={"score": 50.0},
        best_fitness={"score": 90.0},
        worst_fitness={"score": 10.0},
        average_fitness={"score": 50.0},
        valid_rate=1.0,
        total_program_count=5,
        avg_num_children=1.0,
        max_num_children=3,
        ancestor_count=1,
        best_fitness_in_ancestors={"score": 70.0},
        worst_fitness_in_ancestors={"score": 70.0},
        average_fitness_in_ancestors={"score": 70.0},
        valid_rate_in_ancestors=1.0,
        descendant_count=0,
        best_fitness_in_descendants={},
        worst_fitness_in_descendants={},
        average_fitness_in_descendants={},
        valid_rate_in_descendants=0.0,
    )
    return base.model_copy(update=overrides) if overrides else base


def _hash(stats: EvolutionaryStatistics | None, **inputs) -> str | None:
    params = MutationSuggestionInputs(
        intra_card=inputs.get("intra_card", StringContainer(data="intra")),
        memory_cards=inputs.get("memory_cards", StringContainer(data="cards")),
        evolutionary_statistics=stats,
    )
    return MutationSuggestionStage.compute_hash(params)


# ---------------------------------------------------------------------------
# Noise-level stat changes keep the hash stable
# ---------------------------------------------------------------------------


class TestNoiseInvariance:
    def test_fitness_jitter_below_rounding_same_hash(self):
        a = _hash(_make_stats(average_fitness={"score": 50.0}))
        b = _hash(_make_stats(average_fitness={"score": 50.0012}))
        assert a == b

    def test_population_count_within_bucket_same_hash(self):
        a = _hash(_make_stats(total_program_count=51))
        b = _hash(_make_stats(total_program_count=58))
        assert a == b

    def test_archive_snapshot_growth_same_hash(self):
        a = _hash(_make_stats(archive_valid_fitnesses=(0.1, 0.5)))
        b = _hash(_make_stats(archive_valid_fitnesses=(0.1, 0.4, 0.5, 0.7)))
        assert a == b

    def test_window_rank_reshuffle_same_hash(self):
        a = _hash(_make_stats(iter_window_rank=2, iter_window_programs=8))
        b = _hash(_make_stats(iter_window_rank=5, iter_window_programs=11))
        assert a == b

    def test_plateau_within_bucket_same_hash(self):
        a = _hash(_make_stats(iters_since_last_new_best=1))
        b = _hash(_make_stats(iters_since_last_new_best=4))
        assert a == b


# ---------------------------------------------------------------------------
# Meaningful momentum changes invalidate the hash
# ---------------------------------------------------------------------------


class TestMeaningfulInvalidation:
    def test_best_fitness_move_changes_hash(self):
        a = _hash(_make_stats(best_fitness={"score": 90.0}))
        b = _hash(_make_stats(best_fitness={"score": 90.5}))
        assert a != b

    def test_trend_flip_changes_hash(self):
        a = _hash(_make_stats(iter_window_trend="flat"))
        b = _hash(_make_stats(iter_window_trend="up"))
        assert a != b

    def test_plateau_bucket_crossing_changes_hash(self):
        a = _hash(_make_stats(iters_since_last_new_best=4))
        b = _hash(_make_stats(iters_since_last_new_best=5))
        assert a != b

    def test_valid_rate_drop_changes_hash(self):
        a = _hash(_make_stats(valid_rate=1.0))
        b = _hash(_make_stats(valid_rate=0.5))
        assert a != b

    def test_stats_none_vs_present_differ(self):
        assert _hash(None) != _hash(_make_stats())


# ---------------------------------------------------------------------------
# Task-dependent quantum: MetricSpec.significant_change drives fitness buckets
# ---------------------------------------------------------------------------


class TestTaskDependentQuantum:
    def test_move_within_significant_change_keeps_hash(self):
        a = _hash(
            _make_stats(significant_change={"score": 0.5}, best_fitness={"score": 90.0})
        )
        b = _hash(
            _make_stats(significant_change={"score": 0.5}, best_fitness={"score": 90.2})
        )
        assert a == b

    def test_move_beyond_significant_change_invalidates(self):
        a = _hash(
            _make_stats(significant_change={"score": 0.5}, best_fitness={"score": 90.0})
        )
        b = _hash(
            _make_stats(significant_change={"score": 0.5}, best_fitness={"score": 90.3})
        )
        assert a != b

    def test_coarse_quantum_absorbs_decimal_level_moves(self):
        a = _hash(
            _make_stats(significant_change={"score": 5.0}, best_fitness={"score": 90.0})
        )
        b = _hash(
            _make_stats(significant_change={"score": 5.0}, best_fitness={"score": 92.0})
        )
        assert a == b

    def test_metric_without_quantum_falls_back_to_decimal_rounding(self):
        a = _hash(_make_stats(best_fitness={"score": 90.0}))
        b = _hash(_make_stats(best_fitness={"score": 90.01}))
        assert a != b


# ---------------------------------------------------------------------------
# Card inputs still drive invalidation as before
# ---------------------------------------------------------------------------


class TestCardInputsStillInvalidate:
    def test_intra_card_change_changes_hash(self):
        a = _hash(_make_stats(), intra_card=StringContainer(data="v1"))
        b = _hash(_make_stats(), intra_card=StringContainer(data="v2"))
        assert a != b

    def test_memory_cards_change_changes_hash(self):
        a = _hash(_make_stats(), memory_cards=StringContainer(data="v1"))
        b = _hash(_make_stats(), memory_cards=StringContainer(data="v2"))
        assert a != b


# ---------------------------------------------------------------------------
# The agent still receives the full, unquantized stats
# ---------------------------------------------------------------------------


class TestAgentSeesFullStats:
    async def test_preprocess_passes_original_stats_object(self):
        llm = MagicMock()
        llm.with_structured_output.return_value = llm
        ctx = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="primary",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=100.0,
                    sentinel_value=-1.0,
                )
            }
        )
        stage = MutationSuggestionStage(
            llm=llm,
            storage=MagicMock(),
            metrics_context=ctx,
            task_description="task",
            timeout=5.0,
        )
        stats = _make_stats()
        params = MutationSuggestionInputs(
            intra_card=None,
            memory_cards=None,
            evolutionary_statistics=stats,
        )
        prep = await stage.preprocess(MagicMock(), params)
        assert prep["evolutionary_statistics"] is stats


class TestOfferedMemoryCardIds:
    def test_parses_only_memory_card_headers(self):
        rendered = (
            "free text card-ghost\n\n"
            "[card 1] id=card-abc\n"
            "description\n\n"
            "[card 2] id=program-123\n"
            "description"
        )

        assert offered_memory_card_ids(rendered) == frozenset(
            {"card-abc", "program-123"}
        )
