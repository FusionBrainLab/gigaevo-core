from __future__ import annotations

from gigaevo.memory.ideas_tracker.components.data_components import (
    ProgramRecord,
    normalize_improvements,
)
from gigaevo.programs.program import Program


def program_to_record(
    program: Program,
    task_description: str,
    task_description_summary: str,
    fitness_key: str = "fitness",
) -> ProgramRecord:
    mutation_output = program.metadata.get("mutation_output", {})
    if not isinstance(mutation_output, dict):
        mutation_output = {}
    return ProgramRecord(
        id=program.id,
        fitness=program.metrics.get(fitness_key, 0.0),
        generation=program.lineage.generation,
        parents=list(program.lineage.parents),
        insights=mutation_output.get("insights_used", []),
        improvements=normalize_improvements(mutation_output.get("changes")),
        category="",
        strategy=mutation_output.get("archetype", ""),
        task_description=task_description,
        task_description_summary=task_description_summary,
        code=program.code,
    )


def programs_to_records(
    programs: list[Program],
    task_description: str,
    task_description_summary: str,
    fitness_key: str = "fitness",
) -> tuple[list[ProgramRecord], set[str]]:
    records: list[ProgramRecord] = []
    ids: set[str] = set()
    for program in programs:
        records.append(
            program_to_record(
                program,
                task_description,
                task_description_summary,
                fitness_key=fitness_key,
            )
        )
        ids.add(program.id)
    return records, ids
