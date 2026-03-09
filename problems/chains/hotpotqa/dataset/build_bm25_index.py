"""Build BM25s index over wiki17_abstracts for HotpotQA retrieval.

Run once after downloading the corpus (download_corpus.py):
    PYTHONPATH=. ~/envs/evo_fast/bin/python \
        problems/chains/hotpotqa/dataset/build_bm25_index.py
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
CORPUS_PATH = BASE_DIR / "wiki17_abstracts.jsonl.gz"
INDEX_DIR = BASE_DIR / "bm25s_index"


def main():
    if INDEX_DIR.exists():
        print(f"BM25s index already exists: {INDEX_DIR}")
        return

    if not CORPUS_PATH.exists():
        print(f"Corpus not found at {CORPUS_PATH}. Run download_corpus.py first.")
        return

    from problems.chains.hotpotqa.utils.retrieval import build_bm25s_index

    build_bm25s_index(CORPUS_PATH, INDEX_DIR)


if __name__ == "__main__":
    main()
