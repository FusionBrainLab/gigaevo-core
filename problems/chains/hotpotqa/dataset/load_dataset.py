from pathlib import Path

from datasets import load_dataset


SEED = 42
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 300

_DIR = Path(__file__).parent


def main():
    hotpot_qa = load_dataset("hotpot_qa", "fullwiki", split="train")
    hotpot_qa = hotpot_qa.shuffle(seed=SEED)

    train = hotpot_qa.select(range(TRAIN_SAMPLES))
    test = hotpot_qa.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + TEST_SAMPLES))

    train_path = str(_DIR / "HotpotQA_train.jsonl")
    test_path = str(_DIR / "HotpotQA_test.jsonl")

    train.to_json(train_path)
    test.to_json(test_path)

    print(f"Dataset ready: {train_path} ({len(train):,} samples), {test_path} ({len(test):,} samples)")


if __name__ == "__main__":
    main()
