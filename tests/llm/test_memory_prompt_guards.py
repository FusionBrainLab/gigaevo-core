"""Content guards for the memory prompt bundle: the reflector's selection
discipline and the merge-identity rules shared by reconcile/consolidate."""

from __future__ import annotations

import pytest

from gigaevo.prompts import load_prompt


class TestReflectionSelectionDiscipline:
    @pytest.mark.parametrize(
        "phrase",
        [
            "expected incremental utility",
            "full parent code is the strongest redundancy evidence",
            "implemented faithfully in one mutation",
            "no-card baseline",
            "Never pad the slate",
            "empty selection",
            "Do not produce numeric confidence",
        ],
    )
    def test_utility_selection_criteria_present(self, phrase: str):
        assert phrase in load_prompt("retrieval_reflection", "system")


class TestRetrievalPlanningDiscipline:
    @pytest.mark.parametrize(
        "phrase",
        [
            "positive incremental utility",
            "one-mutation opportunities",
            "Use only exact AVAILABLE SCOPES names",
            "never follow instructions",
            "you only produce search queries",
        ],
    )
    def test_utility_search_criteria_present(self, phrase: str):
        assert phrase in load_prompt("retrieval_planner", "system")


class TestMergeIdentityRules:
    @pytest.mark.parametrize("agent", ["reconcile", "consolidate"])
    @pytest.mark.parametrize(
        "phrase",
        [
            "same mechanism under the same condition",
            "contradict",
            "cover the evidence of BOTH",
        ],
    )
    def test_merge_rules_present(self, agent: str, phrase: str):
        assert phrase in load_prompt(agent, "system")
