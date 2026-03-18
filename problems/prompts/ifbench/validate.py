from statistics import mean

import pandas as pd

from problems.prompts.client import LLMClient
from problems.prompts.ifbench.config import LLM_CONFIG, load_context
from problems.prompts.ifbench.utils import instructions_registry
from problems.prompts.utils import run_prompts, validate_prompt_template


def test_instruction_following(input, response):
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
    instruction_list = input["instruction_id_list"]
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        input["kwargs"][index] = {
            k: v for k, v in input["kwargs"][index].items() if v is not None
        }

        instruction.build_description(**input["kwargs"][index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=input["prompt"])

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return mean(is_following_list)


def calculate_fitness(data: pd.DataFrame, responses: list[str | None]):
    accuracy = []
    for input, response in zip(data.to_dict(orient="records"), responses):
        accuracy.append(test_instruction_following(input, response))
    return mean(accuracy)


def validate(prompt_template: str):
    """Validate prompt template and compute fitness metrics.

    Args:
        prompt_template: The evolved prompt template with {field} placeholders

    Returns:
        dict: Metrics including fitness, avg_cost_utilization, is_valid
    """
    # 1. Load dataset context
    context = load_context(n_samples=300)

    # 2. Validate template structure
    validate_prompt_template(
        prompt_template,
        required_placeholders=context.get("required_placeholders", []),
        available_placeholders=context.get("available_placeholders", []),
    )

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Run on dataset
    results = run_prompts(
        prompt_template,
        client,
        context,
        dataset_key="train_dataset",
    )

    # 5. Extract predictions from raw responses
    raw_responses = results["predictions"]  # Raw LLM response strings

    dataset = context["train_dataset"]

    # 6. Main objective
    fitness = calculate_fitness(dataset, raw_responses)

    return {
        "fitness": fitness,
        "is_valid": 1,
    }
