from pathlib import Path

from datasets import load_dataset


SEED = 42
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 300

_DIR = Path(__file__).parent


def main():
    hotpot_qa = load_dataset("hotpot_qa", "distractor", split="train")
    hotpot_qa = hotpot_qa.shuffle(seed=SEED)

    train = hotpot_qa.select(range(TRAIN_SAMPLES))
    test = hotpot_qa.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + TEST_SAMPLES))

    train.to_json(str(_DIR / "HotpotQA_train.jsonl"))
    test.to_json(str(_DIR / "HotpotQA_test.jsonl"))

    print(f"Dataset ready: HotpotQA_train.jsonl ({len(train):,} samples), HotpotQA_test.jsonl ({len(test):,} samples)")


if __name__ == "__main__":
    main()
