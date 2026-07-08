"""Content guards for the memory prompt bundle: the reflector's selection
discipline and the merge-identity rules shared by reconcile/consolidate."""

from __future__ import annotations

import pytest

from gigaevo.prompts import load_prompt


class TestReflectionSelectionDiscipline:
    @pytest.mark.parametrize(
        "phrase",
        [
            "synonym tolerance",
            "already implements",
            "strongest anti-redundancy signal",
            "empty selection",
        ],
    )
    def test_ported_selection_criteria_present(self, phrase: str):
        assert phrase in load_prompt("retrieval_reflection", "system")


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
