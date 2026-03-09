"""Build a ColBERT index over wiki17_abstracts for HotpotQA retrieval.

Run once before using RETRIEVER = "colbert" in shared_config.py:
    PYTHONPATH=. ~/envs/evo_fast/bin/python \
        problems/chains/hotpotqa/dataset/build_colbert_index.py [--gpu]

Requirements:
    pip install colbert-ai faiss-cpu   (or faiss-gpu for GPU indexing)

Encodes all ~5.3M passages with colbert-ir/colbertv2.0 and saves the index.
GPU: ~30min, CPU: ~4-8h. Search-time is fast on CPU (~30ms/query).
"""

import argparse
import os
from pathlib import Path

from problems.chains.hotpotqa.utils.retrieval import load_corpus

BASE_DIR = Path(__file__).parent
CORPUS_PATH = BASE_DIR / "wiki17_abstracts.jsonl.gz"
INDEX_DIR = BASE_DIR / "colbert_index"
CHECKPOINT = "colbert-ir/colbertv2.0"


def build_index(passages: list[str], *, use_gpu: bool = False) -> None:
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ColBERT expects a TSV collection: id<TAB>passage
    collection_path = INDEX_DIR / "collection.tsv"
    with open(collection_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(passages):
            clean = p.replace("\t", " ").replace("\n", " ")
            f.write(f"{i}\t{clean}\n")
    print(f"Collection written: {collection_path} ({len(passages):,} passages)")

    # ColBERT resolves index path as {index_root}/{name}/.
    # Set index_root=INDEX_DIR.parent so the index lands in INDEX_DIR.
    with Run().context(RunConfig(nranks=1, experiment="hotpotqa")):
        config = ColBERTConfig(
            nbits=2,
            root=str(INDEX_DIR.parent),
            index_root=str(INDEX_DIR.parent),
        )
        indexer = Indexer(checkpoint=CHECKPOINT, config=config)
        indexer.index(
            name=INDEX_DIR.name,
            collection=str(collection_path),
            overwrite=True,
        )
    print(f"ColBERT index saved to {INDEX_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Build ColBERT index for HotpotQA")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for encoding (~30min). Default: CPU (~4-8h).",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        print(f"Corpus not found at {CORPUS_PATH}. Run download_corpus.py first.")
        return

    print(f"Loading corpus from {CORPUS_PATH}...")
    passages = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(passages):,} passages")
    print(f"Device: {'GPU' if args.gpu else 'CPU'}")

    build_index(passages, use_gpu=args.gpu)


if __name__ == "__main__":
    main()
