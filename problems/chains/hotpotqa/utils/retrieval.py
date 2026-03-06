"""Passage retrieval for chain evolution (BM25 + ColBERTv2).

Provides a Retriever protocol and three implementations:
  - BM25Retriever: uses bm25s with disk persistence (fast, local).
  - ColBERTRetriever: uses colbert-ai Searcher in-process (higher recall).
  - ColBERTServerRetriever: proxies to a dedicated ColBERT server process.

BM25 supports two on-disk index layouts:
- Single index directory (legacy)
- Sharded index directory with a manifest (memory-safe for large corpora)

All retrievers lazy-load their index on first call and cache as module-level singletons.
"""

from collections.abc import Callable, Iterator
import gzip
import json
import pickle
import shutil
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Files required by bm25s.save(...) for one index directory
_BM25S_CORE_FILES = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "vocab.index.json",
    "params.index.json",
)

# Sharded index metadata and per-shard corpus files
_MANIFEST_NAME = "manifest.json"
_PASSAGES_NAME = "passages.txt"
_OFFSETS_NAME = "passage_offsets.npy"

# ---------------------------------------------------------------------------
# Corpus loading (shared by BM25 index builder and lazy-init)
# ---------------------------------------------------------------------------


def _iter_jsonl_passages(corpus_path: Path) -> Iterator[str]:
    opener = gzip.open if corpus_path.suffix == ".gz" else open
    with opener(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            title = doc.get("title", "")
            text = doc.get("text", "")
            yield f"{title} | {text}"


def load_corpus(corpus_path: str | Path) -> list[str]:
    """Load corpus as ``list[str]`` from .pkl, .jsonl, or .jsonl.gz."""
    corpus_path = Path(corpus_path)
    if corpus_path.suffix == ".pkl":
        with open(corpus_path, "rb") as f:
            return pickle.load(f)
    return list(_iter_jsonl_passages(corpus_path))


def _is_single_index_ready(index_dir: Path) -> bool:
    return all((index_dir / name).exists() for name in _BM25S_CORE_FILES)


def is_bm25s_index_ready(index_dir: str | Path) -> bool:
    """Check whether an index directory is complete and loadable."""
    index_dir = Path(index_dir)
    manifest_path = index_dir / _MANIFEST_NAME

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        shards = manifest.get("shards", [])
        if not shards:
            return False

        try:
            for shard in shards:
                shard_dir = index_dir / shard["name"]
                if not _is_single_index_ready(shard_dir):
                    return False
                if not (shard_dir / _PASSAGES_NAME).exists():
                    return False
                if not (shard_dir / _OFFSETS_NAME).exists():
                    return False
        except Exception:
            return False
        return True

    return _is_single_index_ready(index_dir)


def _write_shard_passages(shard_dir: Path, passages: list[str]) -> None:
    import numpy as np

    passages_path = shard_dir / _PASSAGES_NAME
    offsets_path = shard_dir / _OFFSETS_NAME
    offsets = np.zeros(len(passages), dtype=np.int64)

    with open(passages_path, "wb") as out:
        for i, passage in enumerate(passages):
            offsets[i] = out.tell()
            line = passage.replace("\n", " ").strip() + "\n"
            out.write(line.encode("utf-8"))

    np.save(offsets_path, offsets, allow_pickle=False)


def _build_single_shard(
    passages: list[str],
    shard_dir: Path,
    *,
    k1: float,
    b: float,
    dtype: str,
    int_dtype: str,
) -> None:
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(
        passages, stopwords="en", stemmer=stemmer, show_progress=True
    )

    retriever = bm25s.BM25(k1=k1, b=b, dtype=dtype, int_dtype=int_dtype)
    retriever.index(corpus_tokens, show_progress=True)

    shard_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(shard_dir))
    _write_shard_passages(shard_dir, passages)


# ---------------------------------------------------------------------------
# BM25 index building (run once via dataset/build_bm25_index.py)
# ---------------------------------------------------------------------------


def build_bm25s_index(
    corpus_path: str | Path,
    index_dir: str | Path,
    *,
    k1: float = 0.9,
    b: float = 0.4,
    shard_size: int = 250_000,
    dtype: str = "float32",
    int_dtype: str = "int32",
) -> None:
    """Build bm25s index from corpus and save to disk.

    Supports both the legacy single-directory load path and a sharded layout
    for large corpora. JSONL(.gz) inputs stream from disk; pickle inputs fall
    back to loading the full passage list.
    """
    corpus_path = Path(corpus_path)
    index_dir = Path(index_dir)

    if is_bm25s_index_ready(index_dir):
        print(f"BM25s index already exists: {index_dir}")
        return

    shard_size = max(1, int(shard_size))
    temp_dir = index_dir.parent / f"{index_dir.name}.tmp_build"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Building sharded BM25s index (shard_size={shard_size:,}, "
        f"dtype={dtype}, int_dtype={int_dtype})"
    )

    manifest_shards: list[dict[str, Any]] = []
    shard_id = 0
    total_docs = 0
    batch: list[str] = []
    if corpus_path.suffix == ".pkl":
        passage_iter: Iterator[str] = iter(load_corpus(corpus_path))
    else:
        passage_iter = _iter_jsonl_passages(corpus_path)

    for passage in passage_iter:
        batch.append(passage)
        if len(batch) < shard_size:
            continue

        shard_name = f"shard_{shard_id:04d}"
        shard_dir = temp_dir / shard_name
        print(f"Building {shard_name} ({len(batch):,} docs)...")
        _build_single_shard(
            batch, shard_dir, k1=k1, b=b, dtype=dtype, int_dtype=int_dtype
        )
        manifest_shards.append({"name": shard_name, "doc_count": len(batch)})
        total_docs += len(batch)
        print(f"Built {shard_name}; total indexed: {total_docs:,}")
        batch = []
        shard_id += 1

    if batch:
        shard_name = f"shard_{shard_id:04d}"
        shard_dir = temp_dir / shard_name
        print(f"Building {shard_name} ({len(batch):,} docs)...")
        _build_single_shard(
            batch, shard_dir, k1=k1, b=b, dtype=dtype, int_dtype=int_dtype
        )
        manifest_shards.append({"name": shard_name, "doc_count": len(batch)})
        total_docs += len(batch)
        print(f"Built {shard_name}; total indexed: {total_docs:,}")

    manifest = {
        "format_version": 1,
        "index_type": "bm25s_sharded",
        "k1": k1,
        "b": b,
        "dtype": dtype,
        "int_dtype": int_dtype,
        "shard_size": shard_size,
        "num_docs": total_docs,
        "num_shards": len(manifest_shards),
        "shards": manifest_shards,
    }
    (temp_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if index_dir.exists():
        shutil.rmtree(index_dir)
    temp_dir.rename(index_dir)
    print(f"BM25s sharded index saved: {index_dir} ({total_docs:,} passages)")


def _read_passage(shard: dict[str, Any], doc_idx: int) -> str:
    offsets = shard["offsets"]
    if doc_idx < 0 or doc_idx >= len(offsets):
        return ""

    passages_path = shard["passages_path"]
    with open(passages_path, "rb") as f:
        f.seek(int(offsets[doc_idx]))
        return f.readline().decode("utf-8", errors="replace").rstrip("\n")


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
_bm25_shards: list[dict[str, Any]] | None = None
_bm25_is_sharded = False
_bm25_lock = threading.Lock()
_bm25_initialized = False


def _ensure_bm25_initialized(
    index_dir: str | Path,
    corpus_path: str | Path | None = None,
) -> None:
    """Lazy-load (or build-then-load) the bm25s index and corpus."""
    global _bm25_retriever
    global _bm25_stemmer
    global _bm25_corpus
    global _bm25_shards
    global _bm25_is_sharded
    global _bm25_initialized

    if _bm25_initialized:
        return

    with _bm25_lock:
        if _bm25_initialized:
            return

        try:
            import bm25s
            import numpy as np
            import Stemmer

            index_dir = Path(index_dir)

            if not is_bm25s_index_ready(index_dir):
                if corpus_path is None:
                    raise FileNotFoundError(
                        f"BM25s index not ready at {index_dir} and no corpus_path "
                        "provided for auto-build. Run download_corpus.py first."
                    )
                print(
                    f"BM25s index not ready at {index_dir}, building from {corpus_path}..."
                )
                build_bm25s_index(corpus_path, index_dir)

            _bm25_stemmer = Stemmer.Stemmer("english")
            manifest_path = index_dir / _MANIFEST_NAME

            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                shard_states: list[dict[str, Any]] = []
                for shard in manifest["shards"]:
                    shard_dir = index_dir / shard["name"]
                    shard_states.append(
                        {
                            "retriever": bm25s.BM25.load(str(shard_dir), mmap=True),
                            "offsets": np.load(
                                shard_dir / _OFFSETS_NAME, mmap_mode="r"
                            ),
                            "passages_path": shard_dir / _PASSAGES_NAME,
                        }
                    )
                _bm25_shards = shard_states
                _bm25_is_sharded = True
                _bm25_retriever = None
                _bm25_corpus = None
            else:
                _bm25_retriever = bm25s.BM25.load(str(index_dir), mmap=True)
                _bm25_is_sharded = False
                _bm25_shards = None

                if corpus_path is None:
                    raise FileNotFoundError(
                        "corpus_path is required to load formatted passages in "
                        "single-index mode"
                    )
                _bm25_corpus = load_corpus(corpus_path)

            _bm25_initialized = True
        except Exception:
            _bm25_retriever = None
            _bm25_stemmer = None
            _bm25_corpus = None
            _bm25_shards = None
            _bm25_is_sharded = False
            _bm25_initialized = False
            raise


def _batch_retrieve_sharded(tokens, k: int) -> list[str]:
    if not _bm25_shards:
        return ["" for _ in range(len(tokens.ids))]

    candidates: list[list[tuple[float, int, int]]] = [[] for _ in range(len(tokens.ids))]
    for shard_idx, shard in enumerate(_bm25_shards):
        results, scores = shard["retriever"].retrieve(
            tokens, k=k, n_threads=1, show_progress=False
        )
        for query_idx in range(len(tokens.ids)):
            for doc_idx, score in zip(
                results[query_idx][:k], scores[query_idx][:k], strict=False
            ):
                candidates[query_idx].append((float(score), shard_idx, int(doc_idx)))

    outputs: list[str] = []
    for query_candidates in candidates:
        query_candidates.sort(key=lambda item: item[0], reverse=True)
        top = query_candidates[:k]
        lines = [
            f"[{rank + 1}] {_read_passage(_bm25_shards[shard_idx], doc_idx)}"
            for rank, (_score, shard_idx, doc_idx) in enumerate(top)
        ]
        outputs.append("\n".join(lines))
    return outputs


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
        if _bm25_is_sharded:
            return _batch_retrieve_sharded(tokens, k)

        results, _scores = _bm25_retriever.retrieve(
            tokens, k=k, n_threads=4, show_progress=False
        )
        return [
            "\n".join(
                f"[{j + 1}] {_bm25_corpus[int(idx)]}" for j, idx in enumerate(row[:k])
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
            colbert_resolved = (
                index_dir.parent / "hotpotqa" / "indexes" / index_dir.name
            )
            if not colbert_resolved.exists():
                raise FileNotFoundError(
                    f"ColBERT index not found at {colbert_resolved} "
                    f"(virtual index_dir={index_dir}). "
                    "Run dataset/build_colbert_index.py first."
                )

            # Force CPU mode permanently in exec_runner workers — GPUs are
            # reserved for vLLM.  exec_runners are isolated subprocesses so
            # permanent patches are safe and don't affect the main process.
            #
            # Two module-level names must be patched and kept patched for the
            # lifetime of the subprocess:
            #
            # (a) base_colbert.DEVICE: BaseColBERT.__init__ calls
            #     `self.model.to(DEVICE)` using its LOCAL binding of DEVICE
            #     (separate from colbert.parameters.DEVICE due to
            #     `from … import`).  Set to cpu so BERT loads to CPU.
            #     Must remain cpu after init — colbert_score_packed reads
            #     config.total_visible_gpus at CALL TIME (not init time), so
            #     restoring would re-enable GPU scoring and cause a device
            #     mismatch (pids on CPU, scores_sorter.indices on CUDA).
            #
            # (b) ColBERTConfig.total_visible_gpus: plain class attribute
            #     (not a dataclass field).  ColBERT reads it at CALL TIME in
            #     colbert_score_packed / colbert_score / IndexScorer.rank.
            #     Keep it at 0 permanently so all search operations use CPU.
            from colbert.infra.config import ColBERTConfig as _ColBERTConfig
            import colbert.modeling.base_colbert as _base_colbert
            import torch as _torch

            _base_colbert.DEVICE = _torch.device("cpu")
            _ColBERTConfig.total_visible_gpus = 0

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
            results.append("\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)))
        return results


# ---------------------------------------------------------------------------
# ColBERT server retriever (preferred for exec_runner workers)
# ---------------------------------------------------------------------------


class ColBERTServerRetriever:
    """ColBERT retriever that proxies to a running colbert_server.py instance.

    Load the index once in a dedicated server process
    (``experiments/hotpotqa/tools/colbert_server.py``), then point all
    exec_runner workers here via ``HOTPOTQA_COLBERT_SERVER_URL``.
    This avoids loading the 15–20 GB index in every worker process.

    Args:
        server_url: Base URL of the search server, e.g. ``http://127.0.0.1:8888``
        k: Default number of passages to retrieve.
    """

    def __init__(self, server_url: str, k: int = 7):
        self._url = server_url.rstrip("/") + "/search"
        self._k = k

    def batch_retrieve(self, queries: list[str], k: int | None = None) -> list[str]:
        import json
        import urllib.error
        import urllib.request

        k = k or self._k
        payload = json.dumps({"queries": queries, "k": k}).encode()
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["results"]
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"ColBERT server error {exc.code}: {exc.read().decode()}"
            ) from exc


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
    return batch_retrieve([query], index_dir, k=k, corpus_path=corpus_path)[0]


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
    if _bm25_is_sharded:
        return _batch_retrieve_sharded(tokens, k)

    results, _scores = _bm25_retriever.retrieve(
        tokens, k=k, n_threads=4, show_progress=False
    )
    return [
        "\n".join(
            f"[{j + 1}] {_bm25_corpus[int(idx)]}" for j, idx in enumerate(row[:k])
        )
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
