"""GSM8K dataset loader.

Loads the GSM8K (Grade School Math 8K) dataset.

Priority order:
  1. Local JSONL files: dataset/train.jsonl, dataset/test.jsonl
     Each line: {"question": "...", "answer": "... #### <integer>"}
  2. HuggingFace ``datasets`` library (``openai/gsm8k``, split ``main``)

To prepare local files from HuggingFace:

    python -c "
    from datasets import load_dataset
    import json, pathlib
    ds = load_dataset('openai/gsm8k', 'main')
    p = pathlib.Path('problems/chains/neurips_test/gsm8k/dataset')
    for split, name in [('train', 'train'), ('test', 'test')]:
        with open(p / f'{name}.jsonl', 'w') as f:
            for row in ds[split]:
                f.write(json.dumps(row) + '\n')
    print('done')
    "
"""

import json
from pathlib import Path

_DATASET_DIR = Path(__file__).parent


def load_gsm8k(split: str = "train", n_samples: int | None = None) -> list[dict]:
    """Load GSM8K samples.

    Args:
        split: ``"train"`` or ``"test"``.
        n_samples: If given, return only the first ``n_samples`` examples.

    Returns:
        List of dicts with keys ``"question"`` and ``"answer"``.
        The ``"answer"`` field ends with ``"\\n#### <integer>"``.
    """
    local_path = _DATASET_DIR / f"{split}.jsonl"

    if local_path.exists():
        samples = _load_jsonl(local_path)
    else:
        samples = _load_from_huggingface(split)

    if n_samples is not None and n_samples < len(samples):
        samples = samples[:n_samples]

    return samples


def _load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _load_from_huggingface(split: str) -> list[dict]:
    """Download from HuggingFace and return as list of dicts."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "The `datasets` library is required to download GSM8K automatically. "
            "Install it with: pip install datasets\n"
            "Or create local files at: "
            f"{_DATASET_DIR / split}.jsonl"
        ) from e

    hf_split = "train" if split == "train" else "test"
    ds = load_dataset("openai/gsm8k", "main", split=hf_split)
    return [{"question": row["question"], "answer": row["answer"]} for row in ds]
