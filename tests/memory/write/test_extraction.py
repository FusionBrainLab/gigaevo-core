"""Mutation-output normalisation and eligible-record extraction."""

from __future__ import annotations

import pytest

from gigaevo.evolution.mutation.constants import MUTATION_OUTPUT_METADATA_KEY
from gigaevo.memory.write.extraction import (
    Improvement,
    MutationOutput,
    ProgramRecordExtractor,
    normalize_improvement_item,
    normalize_improvements,
    program_to_record,
    record_note,
)


def test_normalize_item_from_string():
    imp = normalize_improvement_item("  tightened the bound  ")
    assert imp == Improvement(description="tightened the bound")


def test_normalize_item_blank_string_falls_back():
    assert normalize_improvement_item("  ").description == "Unspecified change"


def test_normalize_item_dict_with_description_keys():
    imp = normalize_improvement_item(
        {"summary": "swapped solver", "rationale": "converges faster", "extra": 3}
    )
    assert imp.description == "swapped solver"
    assert imp.explanation == "converges faster"


def test_normalize_item_dict_only_extras_promotes_first():
    imp = normalize_improvement_item({"alpha": "one", "beta": "two"})
    assert imp.description == "alpha: one"
    assert imp.explanation == "beta: two"


def test_normalize_item_non_dict_stringifies():
    assert normalize_improvement_item(42).description == "42"
    assert normalize_improvement_item(None).description == "Unspecified change"


def test_normalize_improvements_shapes():
    assert normalize_improvements(None) == []
    assert [i.description for i in normalize_improvements(["a", "b"])] == ["a", "b"]
    assert [i.description for i in normalize_improvements("solo")] == ["solo"]


def test_mutation_output_coerces_none_fields():
    out = MutationOutput.model_validate(
        {"archetype": None, "base_parent": None, "unknown": "ignored"}
    )
    assert out.archetype == ""
    assert out.base_parent == 1


def test_program_to_record_picks_named_base_parent(make_program):
    prog = make_program(
        parents=["p-first", "p-second"],
        metadata={MUTATION_OUTPUT_METADATA_KEY: {"base_parent": 2}},
    )
    record = program_to_record(
        prog, "task", "summary", parent_codes={"p-second": "y = 2"}
    )
    assert record.base_parent_id == "p-second"
    assert record.parent_code == "y = 2"


def test_program_to_record_out_of_range_base_falls_back_to_first(make_program):
    prog = make_program(
        parents=["p-first", "p-second"],
        metadata={MUTATION_OUTPUT_METADATA_KEY: {"base_parent": 5}},
    )
    record = program_to_record(prog, "task", "summary")
    assert record.base_parent_id == "p-first"
    assert record.parent_code == ""


def test_record_note_joins_descriptions_with_fallback(make_program):
    prog = make_program(
        parents=["p-first"],
        metadata={MUTATION_OUTPUT_METADATA_KEY: {"changes": ["one", "two"]}},
    )
    record = program_to_record(prog, "task", "summary")
    assert record_note(record) == "one; two"
    bare = program_to_record(make_program(parents=["p-first"]), "task", "summary")
    assert record_note(bare) == "Unspecified change"


@pytest.fixture
def extractor(metrics_context):
    return ProgramRecordExtractor(
        task_description="task", fitness_key="fitness", metrics_context=metrics_context
    )


def test_extract_skips_roots_invalid_and_seen(extractor, make_program):
    root = make_program(parents=[])
    invalid = make_program(parents=["p"], valid=0.0)
    missing_fitness = make_program(parents=["p"], fitness=None)
    good = make_program(parents=["p"])

    records = extractor.extract(
        [root, invalid, missing_fitness, good], task_description_summary="s"
    )
    assert [r.id for r in records] == [good.id]
    assert extractor.seen_ids == {good.id}

    again = extractor.extract([good], task_description_summary="s")
    assert again == []


def test_forget_rolls_back_seen_and_records(extractor, make_program):
    prog = make_program(parents=["p"])
    extractor.extract([prog], task_description_summary="s")
    extractor.forget({prog.id})
    assert extractor.seen_ids == set()
    assert extractor.all_records == []
    retried = extractor.extract([prog], task_description_summary="s")
    assert [r.id for r in retried] == [prog.id]


def test_parent_code_resolves_from_posterior_pool(extractor, make_program):
    parent = make_program()
    child = make_program(parents=[parent.id])
    records = extractor.extract(
        [child], task_description_summary="s", posterior_programs=[parent, child]
    )
    assert records[0].parent_code == parent.code
