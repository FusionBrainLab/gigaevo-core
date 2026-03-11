"""Build a ColBERT index over wiki17_abstracts for HotpotQA retrieval.

Run once before using RETRIEVER = "colbert" in shared_config.py:
    PYTHONPATH=. ~/envs/evo_fast/bin/python \
        problems/chains/hotpotqa/dataset/build_colbert_index.py

Requirements:
    pip install colbert-ai faiss-cpu

Encodes all ~5.3M passages with colbert-ir/colbertv2.0 and saves the index.
GPU k-means runs via PyTorch patched directly into the ColBERT source
(colbert/indexing/collection_indexer.py::compute_faiss_kmeans) to work around
faiss-gpu being compiled against NumPy 1.x (incompatible with NumPy 2.x).
Encoding: ~30min on H100. Search-time is fast on CPU (~30ms/query).
"""

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


def build_index(passages: list[str]) -> None:
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ColBERT expects a TSV collection: id<TAB>passage
    collection_path = INDEX_DIR / "collection.tsv"
    with open(collection_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(passages):
            clean = p.replace("\t", " ").replace("\n", " ")
            f.write(f"{i}\t{clean}\n")
    print(f"Collection written: {collection_path} ({len(passages):,} passages)")

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
    if not CORPUS_PATH.exists():
        print(f"Corpus not found at {CORPUS_PATH}. Run download_corpus.py first.")
        return

    print(f"Loading corpus from {CORPUS_PATH}...")
    passages = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(passages):,} passages")

    build_index(passages)


if __name__ == "__main__":
    main()
