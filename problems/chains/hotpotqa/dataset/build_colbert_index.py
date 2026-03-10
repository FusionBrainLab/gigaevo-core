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
_CORPUS_PKL = BASE_DIR / "wiki17_abstracts.jsonl.passages.pkl"
_CORPUS_GZ = BASE_DIR / "wiki17_abstracts.jsonl.gz"
CORPUS_PATH = _CORPUS_PKL if _CORPUS_PKL.exists() else _CORPUS_GZ
# ColBERT saves index to {root}/{experiment}/indexes/{name}.
# With root=REPO/experiments and experiment="hotpotqa", the index lands at
# REPO/experiments/hotpotqa/indexes/colbert_index — matching shared_config.py.
_REPO_ROOT = BASE_DIR.parent.parent.parent.parent  # dataset/->hotpotqa/->chains/->problems/->repo
INDEX_DIR = _REPO_ROOT / "experiments" / "colbert_index"
CHECKPOINT = "colbert-ir/colbertv2.0"


_NUM_PARTITIONS = 32768  # Override ColBERT default (~262K) for tractable CPU k-means.
# Default heuristic: 2^floor(log2(16*sqrt(num_embeddings))) ≈ 262144 for 5.3M passages.
# 32768 = 2^15 gives ~8× faster k-means with acceptable search quality.


def _patch_num_partitions() -> None:
    """Monkey-patch CollectionIndexer.setup to cap num_partitions after computation."""
    import colbert.indexing.collection_indexer as _ci

    _orig_setup = _ci.CollectionIndexer.setup

    def _patched_setup(self: _ci.CollectionIndexer) -> None:  # type: ignore[name-defined]
        _orig_setup(self)
        if self.num_partitions != _NUM_PARTITIONS:
            print(
                f"[build_colbert_index] Overriding num_partitions "
                f"{self.num_partitions:,} → {_NUM_PARTITIONS:,}"
            )
            self.num_partitions = _NUM_PARTITIONS
            self._save_plan()

    _ci.CollectionIndexer.setup = _patched_setup  # type: ignore[method-assign]


def build_index(passages: list[str], *, use_gpu: bool = False) -> None:
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    _patch_num_partitions()

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
            kmeans_niters=4,
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
