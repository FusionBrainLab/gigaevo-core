"""IFBench constraint evaluation for chain outputs."""

from statistics import mean

from problems.chains.ifbench.utils import instructions_registry


def test_instruction_following(sample: dict, response: str) -> float:
    """Test how well a response follows the instruction constraints.

    Generates 8 response variants (removing first/last lines, asterisks)
    and checks each constraint against all variants. Returns the fraction
    of constraints satisfied.

    Args:
        sample: Dict with 'prompt', 'instruction_id_list', 'kwargs'.
        response: The LLM response to evaluate.

    Returns:
        Float in [0, 1] — fraction of constraints satisfied.
    """
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    instruction_list = sample["instruction_id_list"]
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in sample["kwargs"][index].items() if v is not None}
        instruction.build_description(**kwargs)

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=sample["prompt"])

        is_following = False
        for resp in all_responses:
            if resp.strip() and instruction.check_following(resp):
                is_following = True
                break

        is_following_list.append(is_following)

    return mean(is_following_list) if is_following_list else 0.0


def get_violated_constraints(sample: dict, response: str) -> list[str]:
    """Return human-readable descriptions of constraints the response violates.

    Uses the same 8-variant checking logic as ``test_instruction_following``
    so results are consistent with the fitness metric.
    """
    r = response.split("\n")
    all_responses = [
        response,
        response.replace("*", ""),
        "\n".join(r[1:]).strip(),
        "\n".join(r[:-1]).strip(),
        "\n".join(r[1:-1]).strip(),
        "\n".join(r[1:]).strip().replace("*", ""),
        "\n".join(r[:-1]).strip().replace("*", ""),
        "\n".join(r[1:-1]).strip().replace("*", ""),
    ]

    violated: list[str] = []
    for index, instruction_id in enumerate(sample["instruction_id_list"]):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in sample["kwargs"][index].items() if v is not None}
        instruction.build_description(**kwargs)

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=sample["prompt"])

        is_ok = any(
            resp.strip() and instruction.check_following(resp) for resp in all_responses
        )
        if not is_ok:
            desc = getattr(instruction, "_description_pattern", instruction_id)
            violated.append(str(desc) if desc else instruction_id)

    return violated
