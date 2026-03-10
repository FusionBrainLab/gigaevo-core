"""Passage retrieval for chain evolution (BM25 + ColBERTv2).

Provides a Retriever protocol and two implementations:
  - BM25Retriever: uses bm25s with disk persistence (fast, local).
  - ColBERTRetriever: uses colbert-ai Searcher in-process (higher recall).

Both lazy-load their index on first call and cache as module-level singletons.
"""

from collections.abc import Callable
import gzip
import json
from pathlib import Path
import pickle
import threading
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Corpus loading (shared by BM25 index builder and lazy-init)
# ---------------------------------------------------------------------------


def load_corpus(corpus_path: str | Path) -> list[str]:
    """Load corpus as ``list[str]`` from .pkl, .jsonl, or .jsonl.gz."""
    corpus_path = Path(corpus_path)
    if corpus_path.suffix == ".pkl":
        with open(corpus_path, "rb") as f:
            return pickle.load(f)

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
    return passages


# ---------------------------------------------------------------------------
# BM25 index building (run once via dataset/build_bm25_index.py)
# ---------------------------------------------------------------------------


def build_bm25s_index(
    corpus_path: str | Path,
    index_dir: str | Path,
    *,
    k1: float = 0.9,
    b: float = 0.4,
) -> None:
    """Build bm25s index from corpus and save to disk."""
    import bm25s
    import Stemmer

    passages = load_corpus(corpus_path)

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(
        passages, stopwords="en", stemmer=stemmer, show_progress=True
    )

    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus_tokens, show_progress=True)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(index_dir))

    print(f"BM25s index saved: {index_dir} ({len(passages):,} passages)")


# ---------------------------------------------------------------------------
# Retriever protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Retriever(Protocol):
    """Minimal interface for a passage retriever."""

    def batch_retrieve(self, queries: list[str], k: int = 7) -> list[str]:
        """Return one formatted passage string per query.

        Format per entry: ``"[1] title | text\\n[2] title | text\\n..."``
        """
        ...


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

# Module-level singleton state
_bm25_retriever = None
_bm25_stemmer = None
_bm25_corpus: list[str] | None = None
_bm25_lock = threading.Lock()
_bm25_initialized = False


def _ensure_bm25_initialized(
    index_dir: str | Path,
    corpus_path: str | Path | None = None,
) -> None:
    """Lazy-load (or build-then-load) the bm25s index and corpus.

    Thread-safe via double-checked locking.
    """
    global _bm25_retriever, _bm25_stemmer, _bm25_corpus, _bm25_initialized

    if _bm25_initialized:
        return

    with _bm25_lock:
        if _bm25_initialized:
            return

        try:
            import bm25s
            import Stemmer

            index_dir = Path(index_dir)

            if not index_dir.exists():
                if corpus_path is None:
                    raise FileNotFoundError(
                        f"BM25s index not found at {index_dir} and no corpus_path "
                        f"provided for auto-build. Run build_bm25_index.py first."
                    )
                print(f"BM25s index not found at {index_dir}, building from {corpus_path}...")
                build_bm25s_index(corpus_path, index_dir)

            _bm25_retriever = bm25s.BM25.load(str(index_dir))
            _bm25_stemmer = Stemmer.Stemmer("english")

            if corpus_path is None:
                raise FileNotFoundError("corpus_path is required to load formatted passages")
            _bm25_corpus = load_corpus(corpus_path)

            _bm25_initialized = True
        except Exception:
            _bm25_retriever = None
            _bm25_stemmer = None
            _bm25_corpus = None
            _bm25_initialized = False
            raise


class BM25Retriever:
    """BM25 retriever backed by bm25s. Lazy-loads index on first call."""

    def __init__(self, index_dir: str | Path, corpus_path: str | Path, k: int = 7):
        self._index_dir = index_dir
        self._corpus_path = corpus_path
        self._k = k

    def batch_retrieve(self, queries: list[str], k: int | None = None) -> list[str]:
        import bm25s

        _ensure_bm25_initialized(self._index_dir, self._corpus_path)
        k = k or self._k

        tokens = bm25s.tokenize(
            queries, stopwords="en", stemmer=_bm25_stemmer, show_progress=False
        )
        results, _scores = _bm25_retriever.retrieve(
            tokens, k=k, n_threads=4, show_progress=False
        )
        return [
            "\n".join(
                f"[{j + 1}] {_bm25_corpus[int(idx)]}"
                for j, idx in enumerate(row[:k])
            )
            for row in results
        ]


# ---------------------------------------------------------------------------
# ColBERT retriever
# ---------------------------------------------------------------------------

_colbert_searcher = None
_colbert_lock = threading.Lock()
_colbert_initialized = False


def _ensure_colbert_initialized(index_dir: Path, checkpoint: str) -> None:
    """Lazy-load the ColBERT Searcher. Thread-safe via double-checked locking."""
    global _colbert_searcher, _colbert_initialized

    if _colbert_initialized:
        return

    with _colbert_lock:
        if _colbert_initialized:
            return

        try:
            from colbert import Searcher
            from colbert.infra import ColBERTConfig, Run, RunConfig

            # ColBERT resolves the index to {root}/{experiment}/indexes/{name}
            # where root=index_dir.parent and experiment comes from RunConfig.
            # The virtual index_dir itself may not exist on disk.
            colbert_resolved = index_dir.parent / "hotpotqa" / "indexes" / index_dir.name
            if not colbert_resolved.exists():
                raise FileNotFoundError(
                    f"ColBERT index not found at {colbert_resolved} "
                    f"(virtual index_dir={index_dir}). "
                    "Run dataset/build_colbert_index.py first."
                )

            with Run().context(RunConfig(nranks=1, experiment="hotpotqa")):
                config = ColBERTConfig(
                    root=str(index_dir.parent),
                    index_root=str(index_dir.parent),
                )
                _colbert_searcher = Searcher(
                    index=index_dir.name,
                    config=config,
                    checkpoint=checkpoint,
                )
            _colbert_initialized = True
        except Exception:
            _colbert_searcher = None
            _colbert_initialized = False
            raise


class ColBERTRetriever:
    """ColBERTv2 retriever using colbert-ai Searcher loaded in-process.

    The index must be pre-built via ``dataset/build_colbert_index.py``.
    The Searcher is loaded lazily on first call and cached as a module-level
    singleton (same pattern as BM25).
    """

    def __init__(
        self,
        index_dir: str | Path,
        checkpoint: str = "colbert-ir/colbertv2.0",
        k: int = 7,
    ):
        self._index_dir = Path(index_dir)
        self._checkpoint = checkpoint
        self._k = k

    def batch_retrieve(self, queries: list[str], k: int | None = None) -> list[str]:
        _ensure_colbert_initialized(self._index_dir, self._checkpoint)
        k = k or self._k
        results: list[str] = []
        for query in queries:
            pids, _ranks, _scores = _colbert_searcher.search(query, k=k)
            passages = [_colbert_searcher.collection[pid] for pid in pids[:k]]
            results.append(
                "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
            )
        return results


# ---------------------------------------------------------------------------
# Tool registry helpers
# ---------------------------------------------------------------------------


def make_batch_tool_fn(
    retriever: Retriever,
    k: int = 7,
) -> Callable[[list[dict]], list[str]]:
    """Create a batched retrieve function for the chain runner tool registry.

    Works with any Retriever implementation.

    Args:
        retriever: A Retriever instance (BM25Retriever, ColBERTRetriever, etc.)
        k: Number of passages to retrieve per query

    Returns:
        Function with signature ``(items: list[dict]) -> list[str]``.
        Each dict must have a ``"query"`` key.
    """

    def retrieve_fn(items: list[dict]) -> list[str]:
        queries = [item["query"] for item in items]
        return retriever.batch_retrieve(queries, k=k)

    return retrieve_fn


# ---------------------------------------------------------------------------
# Legacy free functions (backward compatibility for other problem variants)
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> str:
    """Retrieve top-k passages using BM25 (legacy single-query API)."""
    import bm25s

    _ensure_bm25_initialized(index_dir, corpus_path)

    tokens = bm25s.tokenize(
        query, stopwords="en", stemmer=_bm25_stemmer, show_progress=False
    )
    results, _scores = _bm25_retriever.retrieve(
        tokens, k=k, n_threads=4, show_progress=False
    )

    retrieved = [_bm25_corpus[int(doc_idx)] for doc_idx in results[0][:k]]
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(retrieved))


def batch_retrieve(
    queries: list[str],
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> list[str]:
    """Batch-retrieve top-k passages using BM25 (legacy API)."""
    import bm25s

    _ensure_bm25_initialized(index_dir, corpus_path)

    tokens = bm25s.tokenize(
        queries, stopwords="en", stemmer=_bm25_stemmer, show_progress=False
    )
    results, _scores = _bm25_retriever.retrieve(
        tokens, k=k, n_threads=4, show_progress=False
    )

    return [
        "\n".join(f"[{j + 1}] {_bm25_corpus[int(idx)]}" for j, idx in enumerate(row[:k]))
        for row in results
    ]


def make_retrieve_fn(
    index_dir: str | Path,
    k: int = 7,
    corpus_path: str | Path | None = None,
) -> Callable[[list[dict]], list[str]]:
    """Create a batched BM25 retrieve function for the tool registry (legacy API)."""

    def retrieve_fn(items: list[dict]) -> list[str]:
        queries = [item["query"] for item in items]
        return batch_retrieve(queries, index_dir, k=k, corpus_path=corpus_path)

    return retrieve_fn
