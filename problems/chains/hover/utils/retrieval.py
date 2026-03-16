"""BM25 retrieval over Wikipedia 2017 abstracts for HoVer chain evolution.

Uses bm25s with disk persistence. Index is built once and saved to disk.
Subsequent loads (including from subprocesses) just load the saved index.

Reimplemented from hotpotqa/utils/retrieval.py — not imported because
module-level singletons would conflict across problems.
"""

from collections.abc import Callable
import gzip
import json
from pathlib import Path
import threading

import bm25s
import Stemmer

# Module-level state: lazy-loaded once per process
_retriever: bm25s.BM25 | None = None
_stemmer: Stemmer.Stemmer | None = None
_corpus: list[str] | None = None
_init_lock = threading.Lock()
_initialized = False


def build_bm25s_index(
    corpus_path: str | Path,
    index_dir: str | Path,
    *,
    k1: float = 0.9,
    b: float = 0.4,
) -> None:
    """Build bm25s index from corpus JSONL(.gz) and save to disk."""
    corpus_path = Path(corpus_path)
    index_dir = Path(index_dir)

    passages: list[str] = []

    opener = gzip.open if corpus_path.suffix == ".gz" else open
    with opener(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            title = doc.get("title", "")
            text = doc.get("text", "")
            passages.append(f"{title} | {text}")

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(
        passages, stopwords="en", stemmer=stemmer, show_progress=True
    )

    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus_tokens, show_progress=True)

    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(index_dir))

    print(f"BM25s index saved: {index_dir} ({len(passages):,} passages)")


def _ensure_initialized(
    index_dir: str | Path,
    corpus_path: str | Path | None = None,
) -> None:
    """Lazy-load (or build-then-load) the bm25s index and corpus.

    Thread-safe via double-checked locking.
    """
    global _retriever, _stemmer, _corpus, _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        try:
            index_dir = Path(index_dir)

            if not index_dir.exists():
                if corpus_path is None:
                    raise FileNotFoundError(
                        f"BM25s index not found at {index_dir} and no corpus_path "
                        f"provided for auto-build. Run download_corpus.py first."
                    )
                print(
                    f"BM25s index not found at {index_dir}, building from {corpus_path}..."
                )
                build_bm25s_index(corpus_path, index_dir)

            _retriever = bm25s.BM25.load(str(index_dir))
            _stemmer = Stemmer.Stemmer("english")

            if corpus_path is None:
                raise FileNotFoundError(
                    "corpus_path is required to load formatted passages"
                )
            corpus_path = Path(corpus_path)
            passages: list[str] = []
            opener = gzip.open if corpus_path.suffix == ".gz" else open
            with opener(corpus_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    passages.append(f"{title} | {text}")
            _corpus = passages

            _initialized = True
        except Exception:
            _retriever = None
            _stemmer = None
            _corpus = None
            _initialized = False
            raise


def retrieve(
    query: str,
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> str:
    """Retrieve top-k passages using BM25.

    Returns formatted string with retrieved passages:
    "[1] Title | text\\n[2] Title | text\\n..."
    """
    _ensure_initialized(index_dir, corpus_path)

    tokens = bm25s.tokenize(
        query, stopwords="en", stemmer=_stemmer, show_progress=False
    )
    results, _scores = _retriever.retrieve(
        tokens, k=k, n_threads=1, show_progress=False
    )

    retrieved = [_corpus[int(doc_idx)] for doc_idx in results[0][:k]]
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(retrieved))


def batch_retrieve(
    queries: list[str],
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> list[str]:
    """Batch-retrieve top-k passages for all queries at once.

    Vectorized tokenization + single retriever call — much faster than
    N individual retrieve() calls.
    """
    _ensure_initialized(index_dir, corpus_path)

    tokens = bm25s.tokenize(
        queries, stopwords="en", stemmer=_stemmer, show_progress=False
    )
    results, _scores = _retriever.retrieve(
        tokens, k=k, n_threads=4, show_progress=False
    )

    return [
        "\n".join(f"[{j + 1}] {_corpus[int(idx)]}" for j, idx in enumerate(row[:k]))
        for row in results
    ]


def make_retrieve_fn(
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> Callable[[list[dict]], list[str]]:
    """Create a batched retrieve function for the tool registry.

    Returns:
        Function with signature (items: list[dict]) -> list[str]
        Each dict must have a "query" key.
    """

    def retrieve_fn(items: list[dict]) -> list[str]:
        queries = [item["query"] for item in items]
        return batch_retrieve(queries, index_dir, k=k, corpus_path=corpus_path)

    return retrieve_fn
