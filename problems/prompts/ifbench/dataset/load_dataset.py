from pathlib import Path

from datasets import load_dataset
import json
import ast
import random
import urllib.request

from problems.prompts.ifbench.utils.instructions_registry_ifeval import INSTRUCTION_DICT as IFEVAL_CONSTRAINTS


MAX_CHARS = 2048 * 4  # Max tokens * chars per token
SEED = 42

_DIR = Path(__file__).parent


def is_ifeval_only(instruction_id_list):
    return all(inst_id in IFEVAL_CONSTRAINTS for inst_id in instruction_id_list)


def transform_row(row):
    ground_truth = row["ground_truth"]
    if isinstance(ground_truth, str):
        ground_truth = ast.literal_eval(ground_truth)

    return {
        "prompt": row["messages"][0]["content"],
        "kwargs": [{} if k is None else k for k in ground_truth[0]["kwargs"]],
        "instruction_id_list": ground_truth[0]["instruction_id"],
        "constraint": row["constraint"],
        "key": row["key"]
    }


def main():
    random.seed(SEED)

    dataset = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

    all_examples = []
    ifeval_only_examples = []

    for row in dataset:
        transformed = transform_row(row)
        if len(transformed["prompt"]) > MAX_CHARS:
            continue

        all_examples.append(transformed)
        if is_ifeval_only(transformed["instruction_id_list"]):
            ifeval_only_examples.append(transformed)

    sampled_examples = random.sample(all_examples, min(len(ifeval_only_examples), len(all_examples)))

    with open(str(_DIR / "IFEval_train.jsonl"), "w", encoding="utf-8") as f:
        for example in ifeval_only_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    with open(str(_DIR / "IFBench_train.jsonl"), "w", encoding="utf-8") as f:
        for example in sampled_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Dataset ready: IFEval_train.jsonl ({len(ifeval_only_examples):,} samples), "
          f"IFBench_train.jsonl ({len(sampled_examples):,} samples)")

    # Download IFBench_test.jsonl from GitHub
    test_url = "https://raw.githubusercontent.com/allenai/IFBench/main/data/IFBench_test.jsonl"
    urllib.request.urlretrieve(test_url, str(_DIR / "IFBench_test.jsonl"))
    with open(str(_DIR / "IFBench_test.jsonl"), "r") as f:
        test_count = sum(1 for _ in f)
    print(f"Downloaded: IFBench_test.jsonl ({test_count:,} samples)")


if __name__ == "__main__":
    main()
