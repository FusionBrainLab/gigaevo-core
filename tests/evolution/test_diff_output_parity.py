"""Guard: the diff evidence base stays in lockstep with MutationStructuredOutput.

The structured-diff operator must emit the same evidence/tracking surface as the
standard mutation operator so archetype, citations, and change hypotheses flow
through identically; only the artifact differs (diff slots replace `code`, and
parents are cited by namespace letter instead of 1-based number).
"""

from __future__ import annotations

from gigaevo.evolution.mutation.allowed_changes import (
    DiffInsightCitation,
    DiffStructuredOutputBase,
)
from gigaevo.llm.agents.mutation import InsightCitation, MutationStructuredOutput

# descriptions read from MutationStructuredOutput via _mirrored — must not drift
_MIRRORED = ("archetype", "justification", "insights_used", "card_ids_used", "changes")


def test_diff_base_carries_every_evidence_field_except_code():
    diff_fields = set(DiffStructuredOutputBase.model_fields)
    mutation_fields = set(MutationStructuredOutput.model_fields)
    assert diff_fields == mutation_fields - {"code"}


def test_shared_field_descriptions_do_not_drift():
    for name in _MIRRORED:
        assert (
            DiffStructuredOutputBase.model_fields[name].description
            == MutationStructuredOutput.model_fields[name].description
        )


def test_change_item_shape_is_identical():
    diff_changes = DiffStructuredOutputBase.model_fields["changes"].annotation
    mutation_changes = MutationStructuredOutput.model_fields["changes"].annotation
    assert diff_changes == mutation_changes


def test_parent_references_are_letter_typed_on_the_diff_side():
    # the two intentionally diverging fields: parents cited by letter, not number
    assert DiffStructuredOutputBase.model_fields["base_parent"].annotation is str
    assert MutationStructuredOutput.model_fields["base_parent"].annotation is int
    assert DiffInsightCitation.model_fields["parent"].annotation is str
    assert InsightCitation.model_fields["parent"].annotation is int


def test_diff_base_forbids_unknown_keys():
    assert DiffStructuredOutputBase.model_config.get("extra") == "forbid"
