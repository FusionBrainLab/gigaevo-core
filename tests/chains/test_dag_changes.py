"""Tests for AllowedDagChanges: schema-valid diffs -> valid chains by construction."""

from __future__ import annotations

import json
import random

from mmar_carl.chain import ReasoningChain
from mmar_carl.models.steps import LLMStepDescription
from pydantic import ValidationError
import pytest

from gigaevo.chains.dag_changes import CONTENT_FIELDS, AllowedDagChanges
from gigaevo.exceptions import MutationError
from gigaevo.llm.schema_compat import nonportable_keys


def make_genome(n_steps: int, label: str = "") -> str:
    steps = [
        LLMStepDescription(
            number=i + 1,
            dependencies=[i] if i else [],
            title=f"{label}step {i + 1}",
            aim=f"{label}aim {i + 1}",
            stage_action=f"{label}action {i + 1}",
        )
        for i in range(n_steps)
    ]
    wire = ReasoningChain(
        steps=steps, max_workers=1, enable_progress=False, metadata={}
    ).to_dict()
    wire["task_description"] = "summarize the text"
    wire["steps"][-1]["is_output_step"] = True
    return json.dumps(wire, ensure_ascii=False)


@pytest.fixture
def parents() -> dict[str, str]:
    return {"A": make_genome(2), "B": make_genome(3)}


@pytest.fixture
def changes() -> AllowedDagChanges:
    return AllowedDagChanges()


def n_filled(payload: dict) -> int:
    return sum(1 for k, v in payload.items() if k.startswith("slot_") and v is not None)


ARCHETYPES = {
    "edit": {
        "reasoning": "sharpen the final step",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_2": {
            "kind": "keep",
            "id": "a2",
            "dependencies": ["slot_1"],
            "edits": {"aim": "one crisp final sentence"},
        },
    },
    "insert": {
        "reasoning": "add a verification step",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_2": {
            "kind": "new",
            "title": "Verify facts",
            "aim": "Cross-check the draft against the source",
            "dependencies": ["slot_1"],
        },
        "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_2"]},
    },
    "delete": {
        "reasoning": "single-shot summarizer",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a2"},
    },
    "duplicate": {
        "reasoning": "two drafts then merge",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_2": {"kind": "keep", "id": "a1", "dependencies": []},
        "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_1", "slot_2"]},
    },
    "rewire": {
        "reasoning": "parallel off the raw input",
        "base_parent": "B",
        "slot_1": {"kind": "keep", "id": "b1"},
        "slot_2": {"kind": "keep", "id": "b2", "dependencies": []},
        "slot_3": {"kind": "keep", "id": "b3", "dependencies": ["slot_1", "slot_2"]},
    },
    "full_rewrite": {
        "reasoning": "all-new skeleton",
        "base_parent": "B",
        "slot_1": {
            "kind": "new",
            "title": "Extract entities",
            "aim": "Name actors and event",
        },
        "slot_2": {
            "kind": "new",
            "title": "Compose summary",
            "aim": "One sentence from the entities",
            "dependencies": ["slot_1"],
        },
    },
    "crossover": {
        "reasoning": "graft B's opener onto A's lineage",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "b1"},
        "slot_2": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"]},
    },
    "explicit trailing nulls": {
        "reasoning": "trailing slots may be null",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_2": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"]},
        "slot_3": None,
        "slot_4": None,
    },
}

REJECTS = {
    "dependency on own slot": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_2": {"kind": "keep", "id": "a2", "dependencies": ["slot_2"]},
    },
    "forward dependency": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_2": {"kind": "keep", "id": "a2", "dependencies": ["slot_5"]},
    },
    "dependencies on slot 1": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1", "dependencies": []},
    },
    "dangling keep id": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a9"},
    },
    "empty chain": {"reasoning": "x", "base_parent": "A"},
    "gap fill": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_3": {"kind": "keep", "id": "a2"},
    },
    "empty aim on new step": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "new", "title": "t", "aim": ""},
    },
    "edit aim to empty": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1", "edits": {"aim": ""}},
    },
    "9th slot": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        **{
            f"slot_{k}": {"kind": "new", "title": "t", "aim": "a"} for k in range(2, 10)
        },
    },
    "dependency list over position cap": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_2": {
            "kind": "keep",
            "id": "a2",
            "dependencies": ["slot_1", "slot_1"],
        },
    },
    "degenerate dependency repetition": {
        "reasoning": "x",
        "base_parent": "A",
        "slot_1": {"kind": "keep", "id": "a1"},
        "slot_2": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"] * 71},
    },
}


@pytest.mark.parametrize("label", sorted(ARCHETYPES))
def test_archetype_diffs_yield_valid_child_genomes(changes, parents, label):
    schema = changes.build_schema(parents)
    diff = schema.validate(ARCHETYPES[label])
    child_code = changes.apply(diff, parents)
    child = ReasoningChain.from_dict(json.loads(child_code), use_typed_steps=True)
    assert len(child.steps) == n_filled(ARCHETYPES[label])
    assert json.loads(child_code)["steps"][-1]["is_output_step"] is True


@pytest.mark.parametrize("label", sorted(REJECTS))
def test_illegal_diffs_are_rejected(changes, parents, label):
    schema = changes.build_schema(parents)
    with pytest.raises(ValidationError):
        schema.validate(REJECTS[label])


def _object_branches(prop: dict) -> list[dict]:
    return [b for b in prop.get("anyOf", [prop]) if b.get("type") == "object"]


def test_dependency_enums_are_narrowed_by_position(changes, parents):
    schema = changes.build_schema(parents).json_schema
    for k in range(1, changes.max_steps + 1):
        branches = _object_branches(schema["properties"][f"slot_{k}"])
        assert len(branches) == 2
        for branch in branches:
            if k == 1:
                assert "dependencies" not in branch["properties"]
            else:
                deps = branch["properties"]["dependencies"]
                assert deps["items"]["enum"] == [f"slot_{j}" for j in range(1, k)]
                assert deps["maxItems"] == k - 1


def test_keep_ids_span_all_parents(changes, parents):
    schema = changes.build_schema(parents).json_schema
    keep = _object_branches(schema["properties"]["slot_1"])[0]
    assert set(keep["properties"]["id"]["enum"]) == {"a1", "a2", "b1", "b2", "b3"}
    assert schema["properties"]["base_parent"]["enum"] == ["A", "B"]


def test_crossover_keep_pulls_donor_content(changes, parents):
    schema = changes.build_schema(parents)
    diff = schema.validate(ARCHETYPES["crossover"])
    child = json.loads(changes.apply(diff, parents))
    assert child["steps"][0]["title"] == "step 1"
    assert child["steps"][1]["aim"] == "aim 2"


def test_edit_overrides_kept_field(changes, parents):
    schema = changes.build_schema(parents)
    diff = schema.validate(ARCHETYPES["edit"])
    child = json.loads(changes.apply(diff, parents))
    assert child["steps"][1]["aim"] == "one crisp final sentence"
    assert child["steps"][0]["aim"] == "aim 1"


def test_dependencies_are_renumbered_and_deduped(changes, parents):
    schema = changes.build_schema(parents)
    diff = schema.validate(
        {
            "reasoning": "x",
            "base_parent": "A",
            "slot_1": {"kind": "keep", "id": "a1"},
            "slot_2": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"]},
            "slot_3": {
                "kind": "keep",
                "id": "a2",
                "dependencies": ["slot_1", "slot_1"],
            },
        }
    )
    child = json.loads(changes.apply(diff, parents))
    assert child["steps"][2]["dependencies"] == [1]


def test_emitted_schema_stays_in_portable_subset(changes, parents):
    assert nonportable_keys(changes.build_schema(parents).json_schema) == set()


def test_single_parent_schema_works(changes):
    parents = {"A": make_genome(2)}
    schema = changes.build_schema(parents)
    diff = schema.validate(ARCHETYPES["delete"])
    changes.apply(diff, parents)


def test_min_steps_requires_leading_slots(parents):
    changes = AllowedDagChanges(min_steps=2, max_steps=4)
    schema = changes.build_schema(parents)
    with pytest.raises(ValidationError):
        schema.validate(
            {
                "reasoning": "x",
                "base_parent": "A",
                "slot_1": {"kind": "keep", "id": "a1"},
                "slot_2": None,
            }
        )


def test_unparseable_parent_raises_carl_validation_error(changes):
    with pytest.raises(MutationError, match="carl_validation_error"):
        changes.build_schema({"A": "not json"})


def test_render_parents_lists_ids_and_deps(changes, parents):
    rendered = changes.render_parents(parents)
    for sid in ("a1", "a2", "b1", "b2", "b3"):
        assert sid in rendered
    assert "step 1" in rendered


def test_describe_mentions_slots_and_bounds():
    text = AllowedDagChanges(min_steps=1, max_steps=8).describe()
    assert "slot" in text.lower()
    assert "8" in text


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        AllowedDagChanges(min_steps=0)
    with pytest.raises(ValueError):
        AllowedDagChanges(min_steps=5, max_steps=4)


def test_fuzz_schema_valid_diffs_always_apply(changes, parents):
    schema = changes.build_schema(parents)
    all_ids = ["a1", "a2", "b1", "b2", "b3"]
    rng = random.Random(7)
    words = "draft polish verify extract merge rank filter compress".split()
    for i in range(300):
        n_slots = rng.randint(1, 8)
        payload = {"reasoning": f"fuzz {i}", "base_parent": rng.choice(["A", "B"])}
        for k in range(1, n_slots + 1):
            deps = (
                {}
                if k == 1
                else {
                    "dependencies": rng.sample(
                        [f"slot_{j}" for j in range(1, k)], k=rng.randint(0, k - 1)
                    )
                }
            )
            if rng.random() < 0.6:
                edits = (
                    {"edits": {rng.choice(CONTENT_FIELDS): rng.choice(words)}}
                    if rng.random() < 0.4
                    else {}
                )
                payload[f"slot_{k}"] = {
                    "kind": "keep",
                    "id": rng.choice(all_ids),
                    **edits,
                    **deps,
                }
            else:
                payload[f"slot_{k}"] = {
                    "kind": "new",
                    "title": rng.choice(words),
                    "aim": rng.choice(words),
                    **deps,
                }
        diff = schema.validate(payload)
        child_code = changes.apply(diff, parents)
        ReasoningChain.from_dict(json.loads(child_code), use_typed_steps=True)


def test_transcribe_preserves_chain_and_step_fields(changes):
    doc = json.loads(make_genome(2))
    doc["search_config"] = {"strategy": "vector"}
    doc["steps"][0]["checkpoint"] = True
    parents = {"A": json.dumps(doc)}
    schema = changes.build_schema(parents)
    diff = schema.validate(
        {
            "reasoning": "identity",
            "base_parent": "A",
            "slot_1": {"kind": "keep", "id": "a1"},
            "slot_2": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"]},
        }
    )
    child = json.loads(changes.apply(diff, parents))
    assert child["search_config"]["strategy"] == "vector"
    assert child["steps"][0]["checkpoint"] is True
    assert child["task_description"] == "summarize the text"
    assert child["steps"][-1]["is_output_step"] is True


def test_empty_parents_dict_rejected(changes):
    with pytest.raises(MutationError, match="no parents"):
        changes.build_schema({})


def test_parent_with_zero_steps_rejected(changes):
    wire = {"steps": [], "max_workers": 1, "task_description": "x"}
    with pytest.raises(MutationError, match="parent A.*at least one step"):
        changes.build_schema({"A": json.dumps(wire)})


def test_non_llm_parent_step_rejected(changes):
    from mmar_carl.models.config import MemoryOperation, MemoryStepConfig
    from mmar_carl.models.steps import MemoryStepDescription

    steps = [
        LLMStepDescription(number=1, dependencies=[], title="t", aim="a"),
        MemoryStepDescription(
            number=2,
            dependencies=[1],
            title="m",
            aim="store",
            config=MemoryStepConfig(operation=list(MemoryOperation)[0], memory_key="k"),
        ),
    ]
    wire = ReasoningChain(
        steps=steps, max_workers=1, enable_progress=False, metadata={}
    ).to_dict()
    wire["task_description"] = "x"
    wire["steps"][-1]["is_output_step"] = True
    with pytest.raises(MutationError, match="unsupported type"):
        changes.build_schema({"A": json.dumps(wire)})


def test_render_parents_maps_nonsequential_dep_numbers(changes):
    steps = [
        LLMStepDescription(number=10, dependencies=[], title="first", aim="draft"),
        LLMStepDescription(number=20, dependencies=[10], title="second", aim="polish"),
    ]
    wire = ReasoningChain(
        steps=steps, max_workers=1, enable_progress=False, metadata={}
    ).to_dict()
    wire["task_description"] = "x"
    wire["steps"][-1]["is_output_step"] = True
    rendered = changes.render_parents({"A": json.dumps(wire)})
    assert "a2 | deps=['a1']" in rendered


def test_min_equals_max_pins_slot_count(parents):
    changes = AllowedDagChanges(min_steps=2, max_steps=2)
    schema = changes.build_schema(parents)
    diff = schema.validate(
        {
            "reasoning": "x",
            "base_parent": "A",
            "slot_1": {"kind": "keep", "id": "a1"},
            "slot_2": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"]},
        }
    )
    assert len(json.loads(changes.apply(diff, parents))["steps"]) == 2
    with pytest.raises(ValidationError):
        schema.validate(
            {
                "reasoning": "x",
                "base_parent": "A",
                "slot_1": {"kind": "keep", "id": "a1"},
            }
        )


def test_three_parent_crossover(changes):
    parents = {
        "A": make_genome(2, label="A-"),
        "B": make_genome(3, label="B-"),
        "C": make_genome(2, label="C-"),
    }
    schema = changes.build_schema(parents)
    keep = _object_branches(schema.json_schema["properties"]["slot_1"])[0]
    assert set(keep["properties"]["id"]["enum"]) == {
        "a1",
        "a2",
        "b1",
        "b2",
        "b3",
        "c1",
        "c2",
    }
    diff = schema.validate(
        {
            "reasoning": "graft from all three",
            "base_parent": "B",
            "slot_1": {"kind": "keep", "id": "a1"},
            "slot_2": {"kind": "keep", "id": "b2", "dependencies": ["slot_1"]},
            "slot_3": {"kind": "keep", "id": "c2", "dependencies": ["slot_2"]},
        }
    )
    child = json.loads(changes.apply(diff, parents))
    assert [s["title"] for s in child["steps"]] == ["A-step 1", "B-step 2", "C-step 2"]
    assert [s["aim"] for s in child["steps"]] == ["A-aim 1", "B-aim 2", "C-aim 2"]


def test_apply_wraps_transcription_failures(changes, parents):
    schema = changes.build_schema(parents)
    diff = schema.validate(ARCHETYPES["delete"])
    diff.slot_1.id = "z9"
    with pytest.raises(MutationError, match="diff_apply_assertion"):
        changes.apply(diff, parents)
