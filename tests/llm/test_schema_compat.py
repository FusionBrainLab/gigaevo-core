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
    strict_json_schema,
    strip_strict_nulls,
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


class TestStrictJsonSchema:
    def test_objects_close_and_require_every_key(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        strict = strict_json_schema(schema)
        assert strict["additionalProperties"] is False
        assert strict["required"] == ["a", "b"]

    def test_an_optional_property_becomes_nullable(self):
        schema = {
            "type": "object",
            "properties": {"opt": {"type": "string"}},
        }
        strict = strict_json_schema(schema)
        assert strict["properties"]["opt"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }

    def test_a_required_property_is_left_alone(self):
        schema = {
            "type": "object",
            "properties": {"req": {"type": "string"}},
            "required": ["req"],
        }
        assert strict_json_schema(schema)["properties"]["req"] == {"type": "string"}

    def test_an_already_nullable_optional_is_not_double_wrapped(self):
        schema = {
            "type": "object",
            "properties": {
                "u": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        }
        strict = strict_json_schema(schema)
        assert strict["properties"]["u"]["anyOf"] == [
            {"type": "string"},
            {"type": "null"},
        ]

    def test_an_optional_union_gains_a_null_branch_in_place(self):
        schema = {
            "type": "object",
            "properties": {
                "u": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
        }
        strict = strict_json_schema(schema)
        assert strict["properties"]["u"]["anyOf"] == [
            {"type": "string"},
            {"type": "integer"},
            {"type": "null"},
        ]

    def test_nested_objects_in_array_items_are_transformed(self):
        schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": [],
                    },
                }
            },
            "required": ["rows"],
        }
        inner = strict_json_schema(schema)["properties"]["rows"]["items"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["x"]

    def test_non_object_nodes_pass_through(self):
        assert strict_json_schema({"type": "string"}) == {"type": "string"}

    def test_a_map_shaped_object_is_refused(self):
        schema = {
            "type": "object",
            "properties": {
                "counts": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
            },
            "required": ["counts"],
        }
        with pytest.raises(ValueError, match="map-shaped"):
            strict_json_schema(schema)

    def test_a_bare_object_with_no_properties_is_refused(self):
        schema = {
            "type": "object",
            "properties": {"blob": {"type": "object"}},
            "required": ["blob"],
        }
        with pytest.raises(ValueError, match="map-shaped"):
            strict_json_schema(schema)

    def test_an_empty_properties_object_still_strictifies(self):
        strict = strict_json_schema({"type": "object", "properties": {}})
        assert strict["additionalProperties"] is False
        assert strict["required"] == []

    def test_optional_properties_under_a_union_are_refused(self):
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": [],
                },
                {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    "required": ["b"],
                },
            ]
        }
        with pytest.raises(ValueError, match="union"):
            strict_json_schema(schema)

    def test_all_required_union_branches_still_strictify(self):
        schema = {
            "type": "object",
            "properties": {
                "card": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"kind": {"type": "string"}},
                            "required": ["kind"],
                        },
                        {"type": "null"},
                    ]
                }
            },
            "required": ["card"],
        }
        branch = strict_json_schema(schema)["properties"]["card"]["anyOf"][0]
        assert branch["additionalProperties"] is False
        assert branch["required"] == ["kind"]

    def test_a_nullable_optional_under_a_union_is_allowed(self):
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "required": [],
                },
            ]
        }
        branch = strict_json_schema(schema)["anyOf"][0]
        assert branch["required"] == ["note"]

    def test_a_field_literally_named_anyof_is_not_a_union(self):
        schema = {
            "type": "object",
            "properties": {
                "anyOf": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": [],
                },
            },
            "required": ["anyOf"],
        }
        inner = strict_json_schema(schema)["properties"]["anyOf"]
        assert inner["required"] == ["x"]


class TestStripStrictNulls:
    SCHEMA = {
        "type": "object",
        "properties": {
            "req": {"type": "string"},
            "opt": {"type": "integer"},
            "truly_nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": [],
                },
            },
        },
        "required": ["req", "truly_nullable"],
    }

    def test_a_null_for_a_strictified_optional_is_dropped(self):
        payload = {"req": "a", "opt": None}
        assert strip_strict_nulls(payload, self.SCHEMA) == {"req": "a"}

    def test_a_null_for_a_genuinely_nullable_field_survives(self):
        payload = {"req": "a", "truly_nullable": None}
        assert strip_strict_nulls(payload, self.SCHEMA) == payload

    def test_a_null_for_a_required_field_survives_to_fail_validation(self):
        payload = {"req": None}
        assert strip_strict_nulls(payload, self.SCHEMA) == payload

    def test_nulls_inside_array_items_are_dropped(self):
        payload = {"req": "a", "truly_nullable": "b", "rows": [{"x": None}, {"x": 2}]}
        stripped = strip_strict_nulls(payload, self.SCHEMA)
        assert stripped["rows"] == [{}, {"x": 2}]

    def test_unknown_keys_and_non_dict_payloads_pass_through(self):
        payload = {"req": "a", "stray": None}
        assert strip_strict_nulls(payload, self.SCHEMA) == payload
        assert strip_strict_nulls([1, None], self.SCHEMA) == [1, None]
        assert strip_strict_nulls("x", self.SCHEMA) == "x"
