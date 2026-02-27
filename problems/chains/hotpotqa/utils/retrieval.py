"""BM25 retrieval over Wikipedia 2017 abstracts for chain evolution.

Uses bm25s with disk persistence. Index is built once and saved to disk.
Subsequent loads (including from subprocesses) just load the saved index.
"""

import gzip
import json
import threading
from collections.abc import Callable
from pathlib import Path

import bm25s
import Stemmer

# Module-level state: lazy-loaded once per process
_retriever: bm25s.BM25 | None = None
_stemmer: Stemmer.Stemmer | None = None
_corpus: list[str] | None = None
_init_lock = threading.Lock()
_initialized = False


# --- Index Building (run once, typically from download_corpus.py) ---


def build_bm25s_index(
    corpus_path: str | Path,
    index_dir: str | Path,
    *,
    k1: float = 0.9,
    b: float = 0.4,
) -> None:
    """Build bm25s index from corpus JSONL(.gz) and save to disk.

    Args:
        corpus_path: Path to wiki17_abstracts.jsonl.gz
        index_dir: Directory where bm25s index will be saved
        k1: BM25 k1 parameter
        b: BM25 b parameter
    """
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


# --- Lazy Initialization (called inside subprocesses) ---


def _ensure_initialized(
    index_dir: str | Path,
    corpus_path: str | Path | None = None,
) -> None:
    """Lazy-load (or build-then-load) the bm25s index and corpus.

    Thread-safe via double-checked locking. If index_dir does not exist,
    builds the index from corpus_path first. On failure, resets all globals
    so the next call can retry cleanly.
    """
    global _retriever, _stemmer, _corpus, _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        try:
            index_dir = Path(index_dir)

            # Auto-build if index directory does not exist
            if not index_dir.exists():
                if corpus_path is None:
                    raise FileNotFoundError(
                        f"BM25s index not found at {index_dir} and no corpus_path "
                        f"provided for auto-build. Run download_corpus.py first."
                    )
                print(f"BM25s index not found at {index_dir}, building from {corpus_path}...")
                build_bm25s_index(corpus_path, index_dir)

            # Load index (without corpus)
            _retriever = bm25s.BM25.load(str(index_dir))
            _stemmer = Stemmer.Stemmer("english")

            # Load corpus separately from JSONL file
            if corpus_path is None:
                raise FileNotFoundError(
                    f"corpus_path is required to load formatted passages"
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
            # Reset to allow retry on next call
            _retriever = None
            _stemmer = None
            _corpus = None
            _initialized = False
            raise


# --- Public API ---


def retrieve(
    query: str,
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> str:
    """Retrieve top-k passages using BM25.

    Lazy-loads the index on first call.

    Args:
        query: Search query string
        index_dir: Path to saved bm25s index directory
        k: Number of passages to retrieve
        corpus_path: Path to corpus JSONL for loading formatted passages

    Returns:
        Formatted string with retrieved passages
    """
    _ensure_initialized(index_dir, corpus_path)

    tokens = bm25s.tokenize(
        query, stopwords="en", stemmer=_stemmer, show_progress=False
    )
    # retrieve() returns doc indices, not strings
    results, _scores = _retriever.retrieve(tokens, k=k, n_threads=4, show_progress=False)

    # Map indices to formatted passages from corpus
    retrieved = [_corpus[int(doc_idx)] for doc_idx in results[0][:k]]
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(retrieved))


def batch_retrieve(
    queries: list[str],
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> list[str]:
    """Batch-retrieve top-k passages for multiple queries at once.

    Much faster than individual retrieve() calls due to vectorized
    tokenization and retrieval.
    """
    _ensure_initialized(index_dir, corpus_path)

    tokens = bm25s.tokenize(
        queries, stopwords="en", stemmer=_stemmer, show_progress=False
    )
    results, _scores = _retriever.retrieve(
        tokens, k=k, n_threads=min(len(queries), 32), show_progress=False
    )

    batch_results = []
    for i in range(len(queries)):
        retrieved = [_corpus[int(doc_idx)] for doc_idx in results[i][:k]]
        batch_results.append(
            "\n".join(f"[{j + 1}] {p}" for j, p in enumerate(retrieved))
        )
    return batch_results


def make_retrieve_fn(
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> Callable[..., str]:
    """Create a retrieve function for the tool registry.

    Args:
        index_dir: Path to saved bm25s index directory
        k: Number of passages to retrieve
        corpus_path: Path to corpus JSONL for auto-build (optional)

    Returns:
        Function with signature (query: str) -> str
    """

    def retrieve_fn(query: str) -> str:
        return retrieve(query, index_dir, k=k, corpus_path=corpus_path)

    return retrieve_fn
