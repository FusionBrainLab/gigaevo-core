"""Mutation-output validation and eligible-record extraction."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.memory.write.extraction import (
    Improvement,
    MutationOutput,
    ProgramRecord,
    ProgramRecordExtractor,
    program_to_record,
    record_note,
)
from gigaevo.programs.metrics.context import (
    VALIDITY_KEY,
    MetricsContext,
    MetricSpec,
)


def base_meta(*, base_fitness: float = 0.5) -> dict:
    return {
        MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY: ["card-a"],
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: {
            VALIDITY_KEY: 1.0,
            "fitness": base_fitness,
        },
        MUTATION_MEMORY_BASE_ID_METADATA_KEY: "parent-1",
        MUTATION_OUTPUT_METADATA_KEY: {"card_ids_used": ["card-a"]},
    }


def test_mutation_output_validates_typed_changes():
    out = MutationOutput.model_validate(
        {
            "changes": [
                {"description": "swapped solver", "explanation": "converges faster"},
                {"description": "clamped step"},
            ]
        }
    )
    assert out.changes == [
        Improvement(description="swapped solver", explanation="converges faster"),
        Improvement(description="clamped step"),
    ]


def test_mutation_output_coerces_none_fields():
    out = MutationOutput.model_validate(
        {"archetype": None, "base_parent": None, "changes": None, "unknown": "ignored"}
    )
    assert out.archetype == ""
    assert out.base_parent == 1
    assert out.changes == []


def test_mutation_output_accepts_diff_parent_letters():
    out = MutationOutput.model_validate({"base_parent": "B"})
    assert out.base_parent == 2


def test_extraction_models_are_frozen():
    # Records flow through async writer paths shared across sweeps; silent
    # in-place mutation would corrupt gain attribution downstream.
    models = [
        Improvement(description="swapped solver"),
        MutationOutput(),
        ProgramRecord(id="p1", fitness=0.5, generation=1),
    ]
    for model in models:
        field = next(iter(type(model).model_fields))
        with pytest.raises(ValidationError):
            setattr(model, field, "mutated")


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


def test_program_to_record_picks_letter_base_parent(make_program):
    prog = make_program(
        parents=["p-first", "p-second"],
        metadata={MUTATION_OUTPUT_METADATA_KEY: {"base_parent": "B"}},
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
        metadata={
            MUTATION_OUTPUT_METADATA_KEY: {
                "changes": [
                    {"description": "one", "explanation": ""},
                    {"description": "two", "explanation": ""},
                ]
            }
        },
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
    retried = extractor.extract([prog], task_description_summary="s")
    assert [r.id for r in retried] == [prog.id]


def test_parent_code_resolves_from_posterior_pool(extractor, make_program):
    parent = make_program()
    child = make_program(parents=[parent.id])
    records = extractor.extract(
        [child], task_description_summary="s", posterior_programs=[parent, child]
    )
    assert records[0].parent_code == parent.code


def test_program_to_record_founding_gain_defaults_none(make_program):
    record = program_to_record(make_program(parents=["p"]), "task", "summary")
    assert record.founding_gain is None


def test_extract_attaches_signed_founding_gain(extractor, make_program):
    child = make_program(fitness=0.7, parents=["parent-1"], metadata=base_meta())
    (record,) = extractor.extract([child], task_description_summary="s")
    assert record.founding_gain is not None
    assert record.founding_gain.gain == pytest.approx(0.2)
    assert record.founding_gain.founding is True
    assert record.founding_gain.context.parent_id == "parent-1"


def test_extract_no_founding_gain_without_base_snapshot(extractor, make_program):
    child = make_program(fitness=0.7, parents=["p"], metadata={})
    (record,) = extractor.extract([child], task_description_summary="s")
    assert record.founding_gain is None


def test_extract_minimize_direction_flips_founding_sign(metrics_context, make_program):
    minimize = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="loss", higher_is_better=False, is_primary=True
            )
        }
    )
    extractor = ProgramRecordExtractor(
        task_description="task", fitness_key="fitness", metrics_context=minimize
    )
    child = make_program(fitness=0.7, parents=["parent-1"], metadata=base_meta())
    (record,) = extractor.extract([child], task_description_summary="s")
    assert record.founding_gain is not None
    assert record.founding_gain.gain == pytest.approx(-0.2)
