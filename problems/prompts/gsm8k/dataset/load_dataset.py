from datasets import load_dataset


SEED = 42
NUM_SAMPLES = 1000


def extract_answer(example):
    return {**example, "answer": example["answer"].split("#### ")[-1].strip()}


def main():
    data = load_dataset("openai/gsm8k", name="main")

    train = data["train"].shuffle(seed=SEED).select(range(NUM_SAMPLES)).map(extract_answer)
    test = data["test"].shuffle(seed=SEED).select(range(NUM_SAMPLES)).map(extract_answer)

    train.to_csv("GSM8K_train.csv")
    test.to_csv("GSM8K_test.csv")

    print(f"Dataset ready: GSM8K_train.csv ({len(train):,} samples), GSM8K_test.csv ({len(test):,} samples)")


if __name__ == "__main__":
    main()
