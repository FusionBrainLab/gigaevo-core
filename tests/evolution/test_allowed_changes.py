"""Tests for the genome-agnostic AllowedChanges diff contract."""

from __future__ import annotations

import dataclasses

import pytest

from gigaevo.evolution.mutation.allowed_changes import AllowedChanges, DiffSchema


def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AllowedChanges()


def test_diff_schema_is_frozen():
    schema = DiffSchema(json_schema={"type": "object"}, validate=lambda x: x)
    with pytest.raises(dataclasses.FrozenInstanceError):
        schema.json_schema = {}


def test_minimal_subclass_satisfies_contract():
    class Echo(AllowedChanges):
        def build_schema(self, parents):
            return DiffSchema(json_schema={"type": "object"}, validate=lambda x: x)

        def render_parents(self, parents):
            return "\n".join(f"{ns}: {code}" for ns, code in parents.items())

        def apply(self, diff, parents):
            return str(diff)

        def describe(self):
            return "echo"

    changes = Echo()
    assert changes.build_schema({"A": "{}"}).json_schema == {"type": "object"}
    assert changes.apply({"k": 1}, {"A": "{}"}) == "{'k': 1}"
