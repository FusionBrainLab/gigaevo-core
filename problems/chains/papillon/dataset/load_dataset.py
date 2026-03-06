"""Download and prepare the PUPA dataset for PAPILLON chain evolution.

Uses the same Columbia-NLP/PUPA dataset as prompts/pupa, stored locally
as CSV files in this directory.

Usage:
    python -m problems.chains.papillon.dataset.load_dataset
"""

from pathlib import Path

from datasets import load_dataset


SEED = 42

_DIR = Path(__file__).parent


def main():
    pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new", split="train")
    pupa_tnb = load_dataset("Columbia-NLP/PUPA", "pupa_tnb", split="train")

    columns = ["conversation_hash", "user_query", "target_response", "pii_units"]

    train = pupa_new.shuffle(seed=SEED).select_columns(columns)
    test = pupa_tnb.select_columns(columns)

    train.to_csv(str(_DIR / "PUPA_train.csv"))
    test.to_csv(str(_DIR / "PUPA_test.csv"))

    print(
        f"Dataset ready: PUPA_train.csv ({len(train):,} samples), "
        f"PUPA_test.csv ({len(test):,} samples)"
    )


if __name__ == "__main__":
    main()
