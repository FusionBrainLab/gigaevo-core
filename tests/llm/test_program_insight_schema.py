"""Schema tests for ProgramInsight enum fields and card attribution."""

from __future__ import annotations

import importlib
import pkgutil

from pydantic import BaseModel, ValidationError
import pytest

import gigaevo.llm.agents as agents_pkg
from gigaevo.llm.agents.insights import (
    UNSOURCED_MECHANISM,
    ProgramInsight,
    ProgramInsights,
)


def _insight(**overrides) -> ProgramInsight:
    payload = {"type": "threshold_tuning", "tag": "rigid", "severity": "medium"}
    payload.update(overrides)
    return ProgramInsight(**payload)


def _response_models() -> list[type[BaseModel]]:
    """Every structured-output model DEFINED by an agent module.

    Declared here, not imported into it: the agent modules also pull in LLM
    clients such as ChatOpenAI, which are pydantic models but never response
    schemas (they carry callables and have no JSON schema at all).
    """
    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.iter_modules(agents_pkg.__path__):
        module = importlib.import_module(f"{agents_pkg.__name__}.{info.name}")
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, BaseModel)
                and value is not BaseModel
                and value.__module__.startswith(agents_pkg.__name__)
            ):
                found[f"{value.__module__}.{value.__qualname__}"] = value
    return list(found.values())


def _enums(node: object) -> list[list]:
    """Every `enum` list anywhere in a JSON schema, including nested $defs."""
    out: list[list] = []
    if isinstance(node, dict):
        if isinstance(node.get("enum"), list):
            out.append(node["enum"])
        for value in node.values():
            out.extend(_enums(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(_enums(value))
    return out


class TestProviderEnumConstraint:
    """No structured-output enum may offer an empty member.

    Gemini validates a function declaration against its OpenAPI subset and
    rejects an empty enum value outright:

        GenerateContentRequest.tools[0].function_declarations[0].parameters
        .properties[insights].items.properties[mechanism_source].enum[0]:
        cannot be empty

    JSON-schema mode accepts `""` happily, so this cannot be caught by
    round-tripping a model through Pydantic -- only by reading the schema the
    way a provider does. `config/llm/gemini3_flash.yaml` pins
    `structured_output_method: function_calling` on measured evidence, so an
    empty member is an outage on that path, not a style question: the agent
    400s on every call and the stage degrades to empty output.
    """

    @pytest.mark.parametrize("model", _response_models(), ids=lambda m: m.__qualname__)
    def test_no_empty_enum_member(self, model: type[BaseModel]) -> None:
        for enum in _enums(model.model_json_schema()):
            assert "" not in enum, (
                f"{model.__qualname__} offers an empty enum member in {enum} -- "
                "Gemini function_calling rejects the whole declaration. "
                "Use an explicit named member instead."
            )


class TestTagSeverityEnums:
    @pytest.mark.parametrize(
        "tag", ["beneficial", "harmful", "fragile", "rigid", "neutral"]
    )
    def test_valid_tags_accepted(self, tag: str) -> None:
        assert _insight(tag=tag).tag == tag

    @pytest.mark.parametrize("tag", ["optimization", "cache", "high", ""])
    def test_invalid_tags_rejected(self, tag: str) -> None:
        with pytest.raises(ValidationError):
            _insight(tag=tag)

    @pytest.mark.parametrize("severity", ["high", "medium", "low"])
    def test_valid_severities_accepted(self, severity: str) -> None:
        assert _insight(severity=severity).severity == severity

    @pytest.mark.parametrize("severity", ["urgent", "beneficial", ""])
    def test_invalid_severities_rejected(self, severity: str) -> None:
        with pytest.raises(ValidationError):
            _insight(severity=severity)


class TestMechanismSource:
    def test_unsourced_mechanism_is_named_not_empty(self) -> None:
        """An unattributed mechanism must still be a legal enum member."""
        ins = _insight()
        assert ins.mechanism_source == "own_synthesis"
        assert ins.card_id == ""

    def test_sentinel_is_offered_in_the_schema(self) -> None:
        """The sentinel must stay inside the enum it is a member of.

        `context.py` suppresses this exact value so it never reaches the
        mutator as a source. Were it to drift out of the enum, that
        suppression would quietly stop matching and the sentinel would render
        as if it were a real evidence source.
        """
        field = ProgramInsight.model_json_schema()["properties"]["mechanism_source"]
        assert UNSOURCED_MECHANISM in field["enum"]

    @pytest.mark.parametrize(
        "source",
        [
            "own_synthesis",
            "program",
            "metrics",
            "intra_memory",
            "memory_cards",
            "ancestral_trail",
            "evolutionary_statistics",
        ],
    )
    def test_valid_sources_accepted(self, source: str) -> None:
        assert _insight(mechanism_source=source).mechanism_source == source

    @pytest.mark.parametrize("source", ["global_bank", ""])
    def test_invalid_source_rejected(self, source: str) -> None:
        with pytest.raises(ValidationError):
            _insight(mechanism_source=source)

    def test_card_attribution_round_trip(self) -> None:
        ins = _insight(mechanism_source="memory_cards", card_id="card-abc-123")
        assert ins.card_id == "card-abc-123"


class TestMemoryCardGrounding:
    def test_keeps_exact_offered_card_refs(self) -> None:
        ins = _insight(
            evidence_source="program",
            mechanism_source="memory_cards",
            card_id="card-abc",
            evidence_refs=["card-abc"],
        )

        grounded = ins.grounded_memory_card_refs({"card-abc"})

        assert grounded.card_id == "card-abc"
        assert grounded.evidence_refs == ["card-abc"]

    def test_clears_unoffered_card_refs(self) -> None:
        ins = _insight(
            evidence_source="program",
            mechanism_source="memory_cards",
            card_id="ghost-card",
            evidence_refs=["ghost-card", "card-abc"],
        )

        grounded = ins.grounded_memory_card_refs({"card-abc"})

        assert grounded.card_id == ""
        assert grounded.evidence_refs == ["card-abc"]

    def test_collection_grounds_each_insight(self) -> None:
        insights = ProgramInsights(
            insights=[
                _insight(mechanism_source="memory_cards", card_id="card-abc"),
                _insight(mechanism_source="memory_cards", card_id="ghost-card"),
            ]
        )

        grounded = insights.grounded_memory_card_refs({"card-abc"})

        assert [ins.card_id for ins in grounded.insights] == ["card-abc", ""]
