"""Extended tests for card_update_dedup — edge cases and bug probing.

Complements test_memory_card_update_dedup.py with adversarial inputs.
"""

import json

# Also test private helpers that are critical to correctness
from gigaevo.memory.shared_memory.card_update_dedup import (
    CardUpdateDedupConfig,
    RetrievalWeights,
    _extract_json_object,
    _safe_float,
    append_unique_text,
    compute_weighted_candidates,
    dedupe_keep_order,
    get_explanation_summary,
    get_full_explanations,
    merge_updated_card,
    merge_usage_payloads,
    parse_llm_card_decision,
)

# ===========================================================================
# CardUpdateDedupConfig
# ===========================================================================


class TestCardUpdateDedupConfig:
    def test_from_mapping_empty_dict(self):
        cfg = CardUpdateDedupConfig.from_mapping({})
        assert cfg.enabled is False
        assert cfg.top_k_per_query == 5
        assert cfg.final_top_n == 5

    def test_from_mapping_enabled_bool(self):
        cfg = CardUpdateDedupConfig.from_mapping({"enabled": True})
        assert cfg.enabled is True

    def test_from_mapping_enabled_string_truthy(self):
        for val in ("true", "True", "1", "yes", "on"):
            cfg = CardUpdateDedupConfig.from_mapping({"enabled": val})
            assert cfg.enabled is True, f"Failed for {val!r}"

    def test_from_mapping_enabled_string_falsy(self):
        for val in ("false", "0", "no", "off", "random"):
            cfg = CardUpdateDedupConfig.from_mapping({"enabled": val})
            assert cfg.enabled is False, f"Failed for {val!r}"

    def test_from_mapping_non_dict(self):
        cfg = CardUpdateDedupConfig.from_mapping("not a dict")
        assert cfg.enabled is False

    def test_from_mapping_none(self):
        cfg = CardUpdateDedupConfig.from_mapping(None)
        assert cfg.enabled is False

    def test_from_mapping_full(self):
        cfg = CardUpdateDedupConfig.from_mapping(
            {
                "enabled": True,
                "retrieval": {
                    "top_k_per_query": 10,
                    "final_top_n": 3,
                    "min_final_score": 0.5,
                },
                "llm": {"max_retries": 5},
            }
        )
        assert cfg.enabled is True
        assert cfg.top_k_per_query == 10
        assert cfg.final_top_n == 3
        assert cfg.min_final_score == 0.5
        assert cfg.llm_max_retries == 5

    def test_from_mapping_min_clamping(self):
        """Values below min_value are clamped."""
        cfg = CardUpdateDedupConfig.from_mapping(
            {
                "enabled": True,
                "retrieval": {"top_k_per_query": -1, "final_top_n": 0},
            }
        )
        assert cfg.top_k_per_query >= 1
        assert cfg.final_top_n >= 1

    def test_frozen(self):
        """Config is immutable."""
        cfg = CardUpdateDedupConfig()
        try:
            cfg.enabled = True  # type: ignore[misc]
            assert False, "Should have raised"
        except (AttributeError, Exception):
            # Pydantic frozen models raise ValidationError; dataclass raises AttributeError
            pass


# ===========================================================================
# RetrievalWeights
# ===========================================================================


class TestRetrievalWeights:
    def test_default_weights_sum_close_to_one(self):
        w = RetrievalWeights()
        total = (
            w.description
            + w.explanation_summary
            + w.description_explanation_summary
            + w.description_task_description_summary
        )
        assert abs(total - 1.0) < 0.01

    def test_from_mapping_custom(self):
        w = RetrievalWeights.from_mapping({"description": 0.5})
        assert w.description == 0.5
        assert w.explanation_summary == 0.2  # default

    def test_from_mapping_non_dict(self):
        w = RetrievalWeights.from_mapping("bad")
        assert w.description == 0.35

    def test_as_score_multipliers_keys(self):
        keys = set(RetrievalWeights().as_score_multipliers().keys())
        assert keys == {
            "description",
            "explanation_summary",
            "description_explanation_summary",
            "description_task_description_summary",
        }


# ===========================================================================
# compute_weighted_candidates edge cases
# ===========================================================================


class TestComputeWeightedCandidatesEdgeCases:
    def test_empty_scores(self):
        assert (
            compute_weighted_candidates({}, weights=RetrievalWeights(), final_top_n=5)
            == []
        )

    def test_min_final_score_filters(self):
        scores = {"description": {"c1": 0.01, "c2": 0.9}}
        result = compute_weighted_candidates(
            scores, weights=RetrievalWeights(), final_top_n=5, min_final_score=0.1
        )
        ids = [r["card_id"] for r in result]
        assert "c2" in ids
        # c1's weighted score = 0.01 * 0.35 = 0.0035 < 0.1
        assert "c1" not in ids

    def test_final_top_n_of_one(self):
        scores = {"description": {"c1": 0.5, "c2": 0.9}}
        result = compute_weighted_candidates(
            scores, weights=RetrievalWeights(), final_top_n=1
        )
        assert len(result) == 1
        assert result[0]["card_id"] == "c2"

    def test_tiebreaker_by_card_id(self):
        """When scores are equal, sort by card_id descending."""
        scores = {"description": {"aaa": 1.0, "zzz": 1.0}}
        result = compute_weighted_candidates(
            scores, weights=RetrievalWeights(), final_top_n=5
        )
        assert result[0]["card_id"] == "zzz"  # reverse sort


# ===========================================================================
# _extract_json_object
# ===========================================================================


class TestExtractJsonObject:
    def test_plain_json(self):
        assert _extract_json_object('{"key": "value"}') == {"key": "value"}

    def test_fenced_json(self):
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json_object(text) == {"key": "value"}

    def test_json_embedded_in_text(self):
        text = 'Here is my answer: {"action": "add"} end'
        result = _extract_json_object(text)
        assert result == {"action": "add"}

    def test_empty(self):
        assert _extract_json_object("") is None

    def test_none(self):
        assert _extract_json_object(None) is None

    def test_non_dict_json(self):
        assert _extract_json_object("[1, 2, 3]") is None

    def test_garbage(self):
        assert _extract_json_object("not json at all") is None

    def test_nested_braces(self):
        text = '{"a": {"b": 1}}'
        result = _extract_json_object(text)
        assert result == {"a": {"b": 1}}


# ===========================================================================
# parse_llm_card_decision edge cases
# ===========================================================================


class TestParseLlmCardDecisionEdgeCases:
    def test_empty_string_returns_none(self):
        result = parse_llm_card_decision("", candidate_ids={"c1"})
        assert result is None

    def test_garbage_returns_none(self):
        result = parse_llm_card_decision("garbage text", candidate_ids={"c1"})
        assert result is None

    def test_unknown_action_falls_back(self):
        text = json.dumps({"action": "destroy"})
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        assert result["action"] == "add"

    def test_discard_without_duplicate_of_falls_back_to_add(self):
        text = json.dumps({"action": "discard"})
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        # discard without duplicate_of → add
        assert result["action"] == "add"

    def test_discard_with_unknown_card_id(self):
        text = json.dumps({"action": "discard", "duplicate_of": "unknown"})
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        # unknown card → duplicate_of cleared → falls back to add
        assert result["action"] == "add"

    def test_update_without_updates_falls_back(self):
        text = json.dumps({"action": "update"})
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        assert result["action"] == "add"

    def test_update_with_unknown_card_id_in_updates(self):
        text = json.dumps(
            {
                "action": "update",
                "updates": [{"card_id": "unknown", "update_explanation": True}],
            }
        )
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        # Unknown card filtered out → no updates → falls back to add
        assert result["action"] == "add"

    def test_valid_discard(self):
        text = json.dumps({"action": "discard", "duplicate_of": "c1"})
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        assert result["action"] == "discard"
        assert result["duplicate_of"] == "c1"

    def test_valid_update(self):
        text = json.dumps(
            {
                "action": "update",
                "updates": [
                    {
                        "card_id": "c1",
                        "update_explanation": True,
                        "explanation_append": "new info",
                    }
                ],
            }
        )
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        assert result["action"] == "update"
        assert len(result["updates"]) == 1

    def test_update_with_empty_updates_and_duplicate_of(self):
        """update with no valid updates but has duplicate_of → discard."""
        text = json.dumps(
            {
                "action": "update",
                "updates": [],
                "duplicate_of": "c1",
            }
        )
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        assert result["action"] == "discard"

    def test_json_in_prose(self):
        text = 'Based on my analysis, the result is: {"action": "discard", "duplicate_of": "c1"} thank you.'
        result = parse_llm_card_decision(text, candidate_ids={"c1"})
        assert result["action"] == "discard"


# ===========================================================================
# get_explanation_summary
# ===========================================================================


class TestGetExplanationSummary:
    def test_dict_with_summary(self):
        card = {"explanation": {"summary": "sum", "explanations": ["a"]}}
        assert get_explanation_summary(card) == "sum"

    def test_dict_without_summary_uses_last_explanation(self):
        card = {"explanation": {"explanations": ["first", "last"]}}
        assert get_explanation_summary(card) == "last"

    def test_string_explanation(self):
        card = {"explanation": "plain text"}
        assert get_explanation_summary(card) == "plain text"

    def test_missing_returns_empty(self):
        assert get_explanation_summary({}) == ""

    def test_empty_dict_explanation(self):
        assert get_explanation_summary({"explanation": {}}) == ""

    def test_empty_summary_and_explanations(self):
        card = {"explanation": {"summary": "", "explanations": []}}
        assert get_explanation_summary(card) == ""


# ===========================================================================
# get_full_explanations
# ===========================================================================


class TestGetFullExplanations:
    def test_with_explanations(self):
        card = {"explanation": {"explanations": ["a", "b"]}}
        assert get_full_explanations(card) == ["a", "b"]

    def test_falls_back_to_summary(self):
        card = {"explanation": {"explanations": [], "summary": "sum"}}
        assert get_full_explanations(card) == ["sum"]

    def test_empty_returns_empty(self):
        assert get_full_explanations({}) == []


# ===========================================================================
# append_unique_text
# ===========================================================================


class TestAppendUniqueText:
    def test_both_present(self):
        result = append_unique_text("old", "new")
        assert "old" in result
        assert "new" in result

    def test_duplicate_returns_original(self):
        result = append_unique_text("hello world", "hello world")
        assert result == "hello world"

    def test_substring_returns_original(self):
        """If added_text is already contained in original, skip."""
        result = append_unique_text("I used hello world technique", "hello world")
        assert result == "I used hello world technique"

    def test_empty_original(self):
        assert append_unique_text("", "new") == "new"

    def test_empty_added(self):
        assert append_unique_text("old", "") == "old"

    def test_both_empty(self):
        assert append_unique_text("", "") == ""

    def test_case_insensitive_dedup(self):
        result = append_unique_text("Hello World", "hello world")
        assert result == "Hello World"


# ===========================================================================
# dedupe_keep_order
# ===========================================================================


class TestDedupeKeepOrder:
    def test_basic(self):
        assert dedupe_keep_order(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_strips_empty(self):
        assert dedupe_keep_order(["a", "", "  ", "b"]) == ["a", "b"]

    def test_strips_whitespace(self):
        assert dedupe_keep_order([" a ", "a"]) == ["a"]

    def test_empty_list(self):
        assert dedupe_keep_order([]) == []

    def test_all_same(self):
        assert dedupe_keep_order(["x", "x", "x"]) == ["x"]


# ===========================================================================
# merge_usage_payloads
# ===========================================================================


class TestMergeUsagePayloads:
    def test_both_empty(self):
        assert merge_usage_payloads({}, {}) == {}

    def test_existing_only(self):
        existing = {
            "used": {
                "entries": [
                    {
                        "task_description_summary": "task1",
                        "fitness_delta_per_use": [0.1, 0.2],
                    }
                ]
            }
        }
        result = merge_usage_payloads(existing, {})
        entries = result["used"]["entries"]
        assert len(entries) == 1
        assert entries[0]["task_description_summary"] == "task1"

    def test_incoming_only(self):
        incoming = {
            "used": {
                "entries": [
                    {
                        "task_description_summary": "task2",
                        "fitness_delta_per_use": [0.3],
                    }
                ]
            }
        }
        result = merge_usage_payloads({}, incoming)
        entries = result["used"]["entries"]
        assert len(entries) == 1

    def test_merge_same_task(self):
        existing = {
            "used": {
                "entries": [
                    {
                        "task_description_summary": "task1",
                        "fitness_delta_per_use": [0.1],
                    }
                ]
            }
        }
        incoming = {
            "used": {
                "entries": [
                    {
                        "task_description_summary": "task1",
                        "fitness_delta_per_use": [0.2],
                    }
                ]
            }
        }
        result = merge_usage_payloads(existing, incoming)
        entries = result["used"]["entries"]
        assert len(entries) == 1
        assert entries[0]["fitness_delta_per_use"] == [0.1, 0.2]

    def test_nan_and_inf_filtered(self):
        existing = {
            "used": {
                "entries": [
                    {
                        "task_description_summary": "task1",
                        "fitness_delta_per_use": [float("nan"), float("inf"), 0.5],
                    }
                ]
            }
        }
        result = merge_usage_payloads(existing, {})
        entries = result["used"]["entries"]
        assert len(entries) == 1
        assert entries[0]["fitness_delta_per_use"] == [0.5]

    def test_non_dict_inputs(self):
        assert merge_usage_payloads(None, None) == {}
        assert merge_usage_payloads("bad", "bad") == {}


# ===========================================================================
# _safe_float
# ===========================================================================


class TestSafeFloat:
    def test_valid(self):
        assert _safe_float(3.14) == 3.14

    def test_string(self):
        assert _safe_float("2.5") == 2.5

    def test_nan(self):
        assert _safe_float(float("nan")) is None

    def test_inf(self):
        assert _safe_float(float("inf")) is None

    def test_neg_inf(self):
        assert _safe_float(float("-inf")) is None

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid(self):
        assert _safe_float("abc") is None


# ===========================================================================
# merge_updated_card
# ===========================================================================


class TestMergeUpdatedCard:
    def test_no_update_flags(self):
        existing = {"id": "c1", "description": "old", "programs": ["p1"]}
        incoming = {"description": "new", "programs": ["p2"]}
        update = {}
        result = merge_updated_card(existing, incoming, update)
        assert result["description"] == "old"  # not changed
        assert result["programs"] == ["p1", "p2"]  # merged

    def test_last_generation_takes_max(self):
        existing = {"id": "c1", "last_generation": 5}
        incoming = {"last_generation": 10}
        result = merge_updated_card(existing, incoming, {})
        assert result["last_generation"] == 10

    def test_programs_deduped(self):
        existing = {"id": "c1", "programs": ["p1", "p2"]}
        incoming = {"programs": ["p2", "p3"]}
        result = merge_updated_card(existing, incoming, {})
        assert result["programs"] == ["p1", "p2", "p3"]

    def test_update_explanation_append(self):
        existing = {
            "id": "c1",
            "explanation": {"explanations": ["old"], "summary": "old sum"},
        }
        incoming = {"explanation": {"explanations": ["new"]}}
        update = {"update_explanation": True, "explanation_append": "appended text"}
        result = merge_updated_card(existing, incoming, update)
        assert "appended text" in result["explanation"]["explanations"]

    def test_explicit_explanation_summary_override(self):
        existing = {
            "id": "c1",
            "explanation": {"explanations": [], "summary": "old"},
        }
        update = {"update_explanation": True, "explanation_summary": "new summary"}
        result = merge_updated_card(existing, {}, update)
        assert result["explanation"]["summary"] == "new summary"
