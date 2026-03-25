from __future__ import annotations

from gigaevo.memory.shared_memory.card_update_dedup import (
    QUERY_DESCRIPTION,
    QUERY_DESCRIPTION_EXPLANATION_SUMMARY,
    QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY,
    QUERY_EXPLANATION_SUMMARY,
    RetrievalWeights,
    build_dedup_queries,
    compute_weighted_candidates,
    merge_updated_card,
    merge_usage_payloads,
    parse_llm_card_decision,
)


def test_compute_weighted_candidates_uses_all_query_scores() -> None:
    scores_by_query = {
        QUERY_DESCRIPTION: {"a": 0.9, "b": 0.2},
        QUERY_EXPLANATION_SUMMARY: {"a": 0.1, "b": 0.8},
        QUERY_DESCRIPTION_EXPLANATION_SUMMARY: {"a": 0.6, "c": 0.7},
        QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY: {"b": 0.4, "c": 0.6},
    }
    ranked = compute_weighted_candidates(
        scores_by_query,
        weights=RetrievalWeights(
            description=0.25,
            explanation_summary=0.25,
            description_explanation_summary=0.25,
            description_task_description_summary=0.25,
        ),
        final_top_n=2,
        min_final_score=0.0,
    )

    assert [item["card_id"] for item in ranked] == ["a", "b"]
    assert ranked[0]["final_score"] > ranked[1]["final_score"]


def test_parse_llm_card_decision_supports_fenced_json() -> None:
    raw = """```json
{
  "action": "update",
  "reason": "same core idea, new task and extra explanation",
  "duplicate_of": "",
  "updates": [
    {
      "card_id": "card-2",
      "update_task_description": true,
      "task_description_append": "Use in noisy optimization tasks",
      "task_description_summary": "also works on noisy tasks",
      "update_explanation": true,
      "explanation_append": "Added robust clipping before normalization.",
      "explanation_summary": "robust clipping for noisy cases"
    }
  ]
}
```"""
    decision = parse_llm_card_decision(raw, candidate_ids={"card-1", "card-2"})

    assert decision["action"] == "update"
    assert len(decision["updates"]) == 1
    assert decision["updates"][0]["card_id"] == "card-2"
    assert decision["updates"][0]["update_task_description"] is True
    assert decision["updates"][0]["update_explanation"] is True


def test_merge_updated_card_appends_context_and_explanations() -> None:
    existing = {
        "id": "card-1",
        "task_description": "Task A",
        "task_description_summary": "summary A",
        "description": "Base idea",
        "programs": ["prog-a"],
        "last_generation": 2,
        "explanation": {
            "explanations": ["initial rationale"],
            "summary": "initial summary",
        },
    }
    incoming = {
        "id": "incoming",
        "task_description": "Task B",
        "task_description_summary": "summary B",
        "description": "Base idea with noise handling",
        "programs": ["prog-b"],
        "last_generation": 5,
        "explanation": {
            "explanations": ["extra rationale for noisy tasks"],
            "summary": "extra summary",
        },
    }
    update = {
        "card_id": "card-1",
        "update_task_description": True,
        "task_description_append": "",
        "task_description_summary": "merged summary",
        "update_explanation": True,
        "explanation_append": "",
        "explanation_summary": "merged explanation summary",
    }

    merged = merge_updated_card(existing, incoming, update)

    assert merged["task_description_summary"] == "merged summary"
    assert "Task A" in merged["task_description"]
    assert "Task B" in merged["task_description"]
    assert merged["explanation"]["summary"] == "merged explanation summary"
    assert merged["explanation"]["explanations"] == [
        "initial rationale",
        "extra rationale for noisy tasks",
    ]
    assert merged["programs"] == ["prog-a", "prog-b"]
    assert merged["last_generation"] == 5


def test_build_dedup_queries_include_combined_fields() -> None:
    card = {
        "description": "Apply adaptive clipping",
        "task_description_summary": "improve robustness on noisy objectives",
        "explanation": {
            "explanations": ["clip outliers before scaling"],
            "summary": "outlier clipping",
        },
    }

    queries = build_dedup_queries(card)

    assert queries[QUERY_DESCRIPTION] == "Apply adaptive clipping"
    assert queries[QUERY_EXPLANATION_SUMMARY] == "outlier clipping"
    assert "IDEA_DESCRIPTION" in queries[QUERY_DESCRIPTION_EXPLANATION_SUMMARY]
    assert "TASK_DESCRIPTION_SUMMARY" in queries[
        QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY
    ]


def test_build_dedup_queries_skip_full_task_description_fallback() -> None:
    card = {
        "description": "Apply adaptive clipping",
        "task_description": "Very long shared task boilerplate that should not enter dedup queries",
        "task_description_summary": "",
        "explanation": {
            "summary": "outlier clipping",
        },
    }

    queries = build_dedup_queries(card)

    assert queries[QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY] == "IDEA_DESCRIPTION: Apply adaptive clipping"


def test_merge_usage_payloads_accumulates_per_task_and_total() -> None:
    existing_usage = {
        "used": {
            "entries": [
                {
                    "task_description_summary": "task A",
                    "used_count": 1,
                    "fitness_delta_per_use": [0.1],
                    "median_delta_fitness": 0.1,
                }
            ],
            "total": {"total_used": 1, "median_delta_fitness": 0.1},
        }
    }
    incoming_usage = {
        "used": {
            "entries": [
                {
                    "task_description_summary": "task A",
                    "used_count": 1,
                    "fitness_delta_per_use": [0.3],
                    "median_delta_fitness": 0.3,
                },
                {
                    "task_description_summary": "task B",
                    "used_count": 1,
                    "fitness_delta_per_use": [-0.2],
                    "median_delta_fitness": -0.2,
                },
            ],
            "total": {"total_used": 2, "median_delta_fitness": 0.05},
        }
    }

    merged = merge_usage_payloads(existing_usage, incoming_usage)
    used = merged["used"]
    assert used["total"]["total_used"] == 3
    assert used["total"]["median_delta_fitness"] == 0.1

    entries = {entry["task_description_summary"]: entry for entry in used["entries"]}
    assert entries["task A"]["used_count"] == 2
    assert entries["task A"]["fitness_delta_per_use"] == [0.1, 0.3]
    assert entries["task A"]["median_delta_fitness"] == 0.2
    assert entries["task B"]["used_count"] == 1
    assert entries["task B"]["fitness_delta_per_use"] == [-0.2]
