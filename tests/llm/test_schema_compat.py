"""Tests for the portable-JSON-schema rewrites."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter
import pytest

from gigaevo.llm.schema_compat import (
    const_to_enum,
    drop_annotations,
    inline_refs,
    nonportable_keys,
    portable_json_schema,
)


def test_inline_refs_resolves_and_drops_defs():
    schema = {
        "$defs": {"Leaf": {"type": "string"}},
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/Leaf"}},
    }
    assert inline_refs(schema) == {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }


def test_inline_refs_merges_ref_siblings_over_target():
    schema = {
        "$defs": {"Leaf": {"type": "string", "description": "from def"}},
        "properties": {"x": {"$ref": "#/$defs/Leaf", "description": "from site"}},
    }
    assert inline_refs(schema)["properties"]["x"] == {
        "type": "string",
        "description": "from site",
    }


def test_inline_refs_resolves_defs_referencing_defs():
    schema = {
        "$defs": {
            "Inner": {"type": "integer"},
            "Outer": {"type": "array", "items": {"$ref": "#/$defs/Inner"}},
        },
        "properties": {"x": {"$ref": "#/$defs/Outer"}},
    }
    assert inline_refs(schema)["properties"]["x"] == {
        "type": "array",
        "items": {"type": "integer"},
    }


def test_inline_refs_raises_on_recursive_ref():
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }
    with pytest.raises(ValueError, match="recursive"):
        inline_refs(schema)


def test_const_to_enum_rewrites_at_any_depth():
    schema = {
        "anyOf": [
            {"properties": {"kind": {"const": "keep"}}},
            {"properties": {"kind": {"const": "new"}}},
        ]
    }
    assert const_to_enum(schema) == {
        "anyOf": [
            {"properties": {"kind": {"enum": ["keep"]}}},
            {"properties": {"kind": {"enum": ["new"]}}},
        ]
    }


def test_drop_annotations_removes_default_and_discriminator():
    schema = {
        "discriminator": {"propertyName": "kind"},
        "properties": {"deps": {"type": "array", "default": []}},
    }
    assert drop_annotations(schema) == {"properties": {"deps": {"type": "array"}}}


def test_field_names_matching_keywords_are_untouched():
    schema = {
        "type": "object",
        "properties": {
            "const": {"type": "string"},
            "default": {"type": "integer", "default": 3},
        },
    }
    cleaned = drop_annotations(const_to_enum(schema))
    assert set(cleaned["properties"]) == {"const", "default"}
    assert cleaned["properties"]["const"] == {"type": "string"}
    assert cleaned["properties"]["default"] == {"type": "integer"}


def test_nonportable_keys_reports_offenders():
    schema = {
        "$defs": {"L": {"type": "string"}},
        "properties": {"x": {"$ref": "#/$defs/L", "const": "a"}},
    }
    assert nonportable_keys(schema) == {"$defs", "$ref", "const"}
    assert nonportable_keys(inline_refs(schema)) == {"const"}


def test_portable_json_schema_cleans_a_pydantic_union():
    class Keep(BaseModel):
        kind: Literal["keep"]
        id: Literal["a1", "a2"]
        dependencies: list[str] = []

    class New(BaseModel):
        kind: Literal["new"]
        title: str = Field(..., min_length=1)

    schema = TypeAdapter(Keep | New).json_schema()
    assert nonportable_keys(schema) != set()
    portable = portable_json_schema(schema)
    assert nonportable_keys(portable) == set()
    assert portable["anyOf"][0]["properties"]["kind"]["enum"] == ["keep"]
    assert portable["anyOf"][1]["properties"]["title"]["minLength"] == 1
