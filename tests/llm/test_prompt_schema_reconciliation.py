"""Contract: each operator's prompt and its structured-output schema are a
single source of truth.

A field's MEANING lives in its Pydantic ``Field(description=...)`` — that text is
serialised into the request by ``with_structured_output`` (function_calling /
json_schema), so the model already receives it. The PROMPT therefore carries only
what a per-field description cannot express: cross-field rules, gates, ordering
contracts, the input-format the operator is handed, worked examples, and (for a
union schema shared by two operators) which field subset this operator populates.

These tests guard the two silent regressions of that split:

1. A bound schema field shipping with NO description (the model loses that field's
   meaning entirely).
2. A prompt that re-enumerates or contradicts the schema's field set — e.g.
   claiming a closed key list that omits real fields ("no extra keys"), which
   actively discourages fields the system credits on.
"""

from __future__ import annotations

from pydantic import BaseModel

from gigaevo.llm.agents.insights import ProgramInsight, ProgramInsights
from gigaevo.llm.agents.lineage import TransitionInsight, TransitionInsights
from gigaevo.llm.agents.mutation import MutationChange, MutationStructuredOutput
from gigaevo.programs.stages.lineage_memory import (
    INTRA_SYSTEM_PROMPT_TEMPLATE,
    IntraCardLLMOutput,
    IntraTriedStrategyLLM,
)
from gigaevo.programs.stages.optimization.optuna.models import (
    CodeModification,
    OptunaSearchSpace,
    ParamSpec,
)
from gigaevo.prompts import load_prompt


def _fields_missing_description(model: type[BaseModel]) -> list[str]:
    return [
        name
        for name, field in model.model_fields.items()
        if not (isinstance(field.description, str) and field.description.strip())
    ]


class TestEveryBoundSchemaFieldIsDocumented:
    """No LLM-bound field may rely on the prompt alone for its meaning."""

    def test_mutation_fields_documented(self) -> None:
        assert _fields_missing_description(MutationStructuredOutput) == []
        assert _fields_missing_description(MutationChange) == []

    def test_insight_fields_documented(self) -> None:
        assert _fields_missing_description(ProgramInsight) == []
        assert _fields_missing_description(ProgramInsights) == []

    def test_transition_fields_documented(self) -> None:
        assert _fields_missing_description(TransitionInsight) == []
        assert _fields_missing_description(TransitionInsights) == []

    def test_optuna_fields_documented(self) -> None:
        assert _fields_missing_description(OptunaSearchSpace) == []
        assert _fields_missing_description(ParamSpec) == []
        assert _fields_missing_description(CodeModification) == []

    def test_intra_card_fields_documented(self) -> None:
        assert _fields_missing_description(IntraCardLLMOutput) == []
        assert _fields_missing_description(IntraTriedStrategyLLM) == []


class TestMutationPromptDoesNotContradictSchema:
    def test_no_closed_field_list_claim(self) -> None:
        # "no extra keys" contradicts the 7-field schema — a hand-written list
        # in the prompt drifted to omit base_parent and card_ids_used. The schema
        # is the single source of truth for the field set.
        assert "no extra keys" not in load_prompt("mutation", "system").lower()

    def test_prompt_instructs_card_credit_field(self) -> None:
        # Schema-only proved insufficient for this optional list field (live run:
        # empty in 25/28 children), so the prompt must instruct populating it.
        assert "card_ids_used" in load_prompt("mutation", "system")


class TestMutationPromptRetainsNonSchemaRules:
    """Trimming field-enumeration prose must not drop the cross-field rules."""

    def test_retains_falsifiability_rule(self) -> None:
        assert "Tautology test" in load_prompt("mutation", "system")

    def test_retains_citation_rule(self) -> None:
        assert "Cite, never invent" in load_prompt("mutation", "system")


class TestStructuredDiffPromptMirrorsMutationPrompt:
    """The diff operator's prompt is the mutation prompt with only the output form
    (diff language for Python source) and the parent noun (genome for program)
    swapped. The shared cross-field rules and section structure must match so the
    two operators cite evidence and hypothesise change identically."""

    _SHARED_HEADERS = (
        "## ROLE",
        "## EVIDENCE INPUTS (user message)",
        "## ARCHETYPE — apply gates FIRST, then pick exactly ONE",
        "## EXECUTION PRINCIPLES",
        "## OUTPUT RULES",
    )

    def test_shared_section_headers_are_present_in_both(self) -> None:
        diff = load_prompt("structured_diff", "system")
        mutation = load_prompt("mutation", "system")
        for header in self._SHARED_HEADERS:
            assert header in mutation
            assert header in diff

    def test_shares_falsifiability_and_citation_rules(self) -> None:
        diff = load_prompt("structured_diff", "system")
        assert "Tautology test" in diff
        assert "Cite, never invent" in diff
        assert "card_ids_used" in diff
        assert "no extra keys" not in diff.lower()

    def test_output_form_is_diff_language_not_python(self) -> None:
        diff = load_prompt("structured_diff", "system")
        mutation = load_prompt("mutation", "system")
        # the one deliberate divergence: artifact is a diff, not Python source
        assert "DIFF LANGUAGE" in diff
        assert "Python source" not in diff
        assert "Python source" in mutation


class TestSharedInsightSchemaCarriesFieldMeaning:
    """The structured-suggestion prompt no longer re-describes each field, so the
    shared schema must carry the meaning it used to spell out in prose."""

    def test_type_description_specifies_format(self) -> None:
        assert "snake_case" in ProgramInsight.model_fields["type"].description

    def test_mechanism_description_rejects_vagueness(self) -> None:
        assert (
            "concrete" in ProgramInsight.model_fields["mechanism"].description.lower()
        )

    def test_substitute_description_rejects_bare_direction(self) -> None:
        assert (
            "direction" in ProgramInsight.model_fields["substitute"].description.lower()
        )

    def test_tag_description_specifies_actions(self) -> None:
        desc = ProgramInsight.model_fields["tag"].description.lower()
        assert "preserve" in desc and "remove" in desc

    def test_severity_description_lists_levels(self) -> None:
        assert "high" in ProgramInsight.model_fields["severity"].description.lower()


class TestTransitionSchemaCarriesStrategyMeaning:
    def test_strategy_description_defines_each_value(self) -> None:
        # Per-value semantics moved out of lineage/system.txt into the field.
        desc = TransitionInsight.model_fields["strategy"].description.lower()
        assert "preserved" in desc and "removed" in desc


class TestIntraCardPromptDefersGatesToSchema:
    """IntraTriedStrategyLLM's field descriptions own the failure_signature /
    mechanism_note validity gates, so CARD CONSTRUCTION no longer restates them;
    the code-vs-archetype clustering directive stays (not expressible per-field)."""

    def test_intra_template_drops_redundant_field_gates(self) -> None:
        assert (
            "for clusters with at least one valid child, a brief sentence"
            not in INTRA_SYSTEM_PROMPT_TEMPLATE
        )

    def test_intra_template_retains_clustering_directive(self) -> None:
        assert "not by self-reported archetype" in INTRA_SYSTEM_PROMPT_TEMPLATE


class TestSuggestionsPromptDefersFieldSetToSchema:
    """The suggestion analyst's per-field meaning lives in ProgramInsight; the
    prompt must not re-enumerate the schema's field set, but must keep the two
    things the schema cannot express: the deprecated-field directive and that
    the rules govern HOW to fill the fields across the slate."""

    def test_drops_field_name_enumeration(self) -> None:
        # The parenthetical roll-call of every field name duplicated the schema.
        assert (
            "`evidence_refs`, `relation_to_lineage`, `tag`, `severity`)"
            not in load_prompt("mutation_suggestions", "system")
        )

    def test_retains_deprecated_field_directive(self) -> None:
        # Cross-field: populate the grounding fields, not the legacy `insight`.
        assert "leave the deprecated free-string `insight` empty" in load_prompt(
            "mutation_suggestions", "system"
        )

    def test_retains_schema_meaning_deferral(self) -> None:
        assert "defines every field and its meaning" in load_prompt(
            "mutation_suggestions", "system"
        )

    def test_output_shape_matches_structured_schema_wrapper(self) -> None:
        prompt = load_prompt("mutation_suggestions", "system")
        assert "single JSON array" not in prompt
        assert "single JSON object" not in prompt
        assert "bound structured-output schema" in prompt
        assert "Use the `insights` collection" in prompt
