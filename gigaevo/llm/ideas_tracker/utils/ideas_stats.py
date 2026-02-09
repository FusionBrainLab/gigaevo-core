import json
import os
from pathlib import Path

from gigaevo.llm.ideas_tracker.components.data_components import (
    ProgramRecord,
    RecordCard,
)


def load_data(progs_path, banks_path):
    with open(progs_path, "r") as f:
        programs = json.load(f)[0]["programs"]
    with open(banks_path, "r") as f:
        ideas = json.load(f)[0]["active_bank"]
    programs_c = [ProgramRecord(**v) for v in programs]
    ideas_c = [RecordCard(**v) for v in ideas]
    return programs_c, ideas_c


def get_parents(parent_ids: list[str], all_programs: list[ProgramRecord]):
    parent_cards: list[ProgramRecord] = []
    for parent_id in parent_ids:
        for program in all_programs:
            if program.id == parent_id:
                parent_cards.append(program)
                break
    return parent_cards


def avg_fitness(idea_card: RecordCard, programs: list[ProgramRecord]):
    c = 0
    fitness_c = 0
    for prog in programs:
        if prog.id in idea_card.linked_programs:
            c += 1
            fitness_c += prog.fitness
    return fitness_c / c


def top_delta_ideas(
    programs: list[ProgramRecord], ideas: list[RecordCard], top_k: int = 5
):
    parent_placeholder = ProgramRecord()
    programs_with_deltas = {}
    for program in programs:
        parent_ids = program.parents
        parent_cards = get_parents(parent_ids, programs)
        if len(parent_cards) < 2:
            parent_cards.extend(
                parent_placeholder for _ in range(0, 2 - len(parent_cards))
            )
        delta_1 = program.fitness - parent_cards[0].fitness
        delta_2 = program.fitness - parent_cards[1].fitness
        max_delta = max(delta_1, delta_2)
        programs_with_deltas[program.id] = {"delta": max_delta}

    sorted_programs = {
        k: v
        for k, v in sorted(
            programs_with_deltas.items(),
            reverse=True,
            key=lambda item: item[1]["delta"],
        )
    }
    top_programs_ids: list[str] = [i for i in list(sorted_programs.keys())[0:top_k]]
    selected_ideas = {}
    for idea in ideas:
        if any(
            linked_program_id in top_programs_ids
            for linked_program_id in idea.linked_programs
        ) and (idea.id not in selected_ideas):
            selected_ideas[idea.id] = {
                "description": idea.description,
                "programs": idea.linked_programs,
                "avg_fitness": avg_fitness(idea, programs),
            }

    return selected_ideas


def top_fitness_ideas(
    programs: list[ProgramRecord], ideas: list[RecordCard], top_k: int = 5
):
    top_progs = [
        prog for prog in sorted(programs, reverse=True, key=lambda pr: pr.fitness)
    ][0:top_k]
    top_progs_ids = [prog.id for prog in top_progs]
    selected_ideas = {}
    for idea in ideas:
        if any(
            linked_program_id in top_progs_ids
            for linked_program_id in idea.linked_programs
        ) and (idea.id not in selected_ideas):
            selected_ideas[idea.id] = {
                "description": idea.description,
                "programs": idea.linked_programs,
                "avg_fitness": avg_fitness(idea, programs),
            }
    return selected_ideas


if __name__ == "__main__":
    TOP_K = 5
    p = Path(__file__).resolve().parent
    os.makedirs(p / "st_input", exist_ok=True)
    os.makedirs(p / "st_output", exist_ok=True)
    programs_path = p / "st_input" / "programs.json"
    banks_path = p / "st_input" / "banks.json"
    programs, bank = load_data(programs_path, banks_path)
    ideas_1 = top_delta_ideas(programs, bank, top_k=TOP_K)
    ideas_2 = top_fitness_ideas(programs, bank, top_k=TOP_K)
    f_dict = {"top_delta_ideas": ideas_1, "top_fitness_ideas": ideas_2}
    with open(p / "st_output" / "output.json", "w") as f:
        json.dump(f_dict, f, indent=4)
