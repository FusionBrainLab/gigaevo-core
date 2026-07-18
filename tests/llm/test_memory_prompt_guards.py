"""Content guards for the memory prompt bundle."""

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


class TestEquivalenceIdentityRules:
    @pytest.mark.parametrize(
        "phrase",
        [
            "same applicability condition",
            "same intervention",
            "Do not create a union",
        ],
    )
    def test_equivalence_rules_present(self, phrase: str):
        assert phrase in load_prompt("equivalence", "system")


class TestCardAuthorEvidenceContract:
    @pytest.mark.parametrize(
        "phrase",
        [
            "parent fitness",
            "child fitness",
            "signed gain",
            "validity",
            "archive status",
            "One mutation is one observational sample",
            "When condition C holds, try action A because mechanism M.",
        ],
    )
    def test_outcome_and_hypothesis_rules_present(self, phrase: str):
        assert phrase in load_prompt("card_author", "system")
