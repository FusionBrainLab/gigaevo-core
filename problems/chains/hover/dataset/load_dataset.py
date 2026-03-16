"""Download and prepare HoVer dataset for chain evolution.

Filters to 3-hop examples (claims requiring evidence from 3 Wikipedia articles).
Saves train and validation splits as JSONL files.

Downloads raw JSON directly from the HoVer GitHub repository to avoid
dependency on HuggingFace datasets library script support.

Usage:
    python -m problems.chains.hover.dataset.load_dataset
"""

import json
from pathlib import Path
import random
import urllib.request

SEED = 42
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 300

_DIR = Path(__file__).parent

_TRAIN_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json"
_VALID_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_dev_release_v1.1.json"


def transform_row(row: dict) -> dict:
    """Transform a HoVer dataset row into our format.

    Extracts unique gold document titles from supporting_facts.
    Raw JSON format: supporting_facts is a list of [title, sent_idx] pairs.
    """
    gold_titles = list({sf[0] for sf in row["supporting_facts"]})
    return {
        "claim": row["claim"],
        "label": row["label"],
        "supporting_facts": sorted(gold_titles),
    }


def _download_json(url: str) -> list[dict]:
    """Download and parse a JSON file from URL."""
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    random.seed(SEED)

    print(f"Downloading train split from {_TRAIN_URL}...")
    train_data = _download_json(_TRAIN_URL)
    print(f"Downloading validation split from {_VALID_URL}...")
    val_data = _download_json(_VALID_URL)

    # Filter to 3-hop examples only
    train_3hop = [row for row in train_data if row["num_hops"] == 3]
    val_3hop = [row for row in val_data if row["num_hops"] == 3]

    random.shuffle(train_3hop)
    random.shuffle(val_3hop)

    train_samples = train_3hop[:TRAIN_SAMPLES]
    test_samples = val_3hop[:TEST_SAMPLES]

    train_path = _DIR / "HoVer_train.jsonl"
    test_path = _DIR / "HoVer_test.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_samples:
            f.write(json.dumps(transform_row(row), ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for row in test_samples:
            f.write(json.dumps(transform_row(row), ensure_ascii=False) + "\n")

    print(
        f"Dataset ready: {train_path.name} ({len(train_samples):,} samples), "
        f"{test_path.name} ({len(test_samples):,} samples)"
    )


if __name__ == "__main__":
    main()
