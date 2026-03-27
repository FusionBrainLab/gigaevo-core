import json

import pandas as pd

from gigaevo.memory.ideas_tracker.components.data_components import ProgramRecord


def convert_programs_to_records(
    programs: pd.DataFrame, task_description: str, task_description_summary: str
) -> tuple[list[ProgramRecord], list[str]]:
    """
    Convert programs DataFrame to list of ProgramRecord instances and return IDs.

    Args:
        programs: DataFrame containing program data with columns: program_id,
            metric_fitness, generation, parent_ids, metadata_mutation_output.

    Returns:
        Tuple of (list of ProgramRecord instances, list of program IDs).
    """
    programs_processed = []
    programs_ids = []
    task_description_summary = task_description_summary
    for _, program in programs.iterrows():
        mutation_metadata = program["metadata_mutation_output"]
        if isinstance(mutation_metadata, str):
            mutation_metadata = json.loads(mutation_metadata)
        parent_ids = program["parent_ids"]
        if isinstance(parent_ids, str):
            try:
                parent_ids = json.loads(parent_ids)
            except (json.JSONDecodeError, TypeError):
                parent_ids = []
        new_program = ProgramRecord(
            id=program["program_id"],
            fitness=program["metric_fitness"],
            generation=program["generation"],
            parents=parent_ids,
            insights=mutation_metadata["insights_used"],
            improvements=mutation_metadata["changes"],
            category="",
            strategy=mutation_metadata["archetype"],
            task_description=task_description,
            task_description_summary=task_description_summary,
            code=str(program.get("code") or ""),
        )
        programs_processed.append(new_program)
        programs_ids.append(program["program_id"])

    return programs_processed, programs_ids
