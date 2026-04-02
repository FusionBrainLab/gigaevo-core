"""Per-task BM25 retrieval for MuSiQue chain evolution.

Each task/sample gets its own BM25 index built over that sample's passages.
The retrieval tool then searches only within the current task's index.
"""

from collections import defaultdict
from collections.abc import Callable
from hashlib import sha1
import json
from pathlib import Path
import re
import shutil
import threading

import bm25s
import Stemmer

_BM25S_CORE_FILES = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "vocab.index.json",
    "params.index.json",
)
_PASSAGES_FILE = "passages.json"

_stemmer = Stemmer.Stemmer("english")
_state_lock = threading.Lock()
_task_state: dict[str, tuple[bm25s.BM25, list[str]]] = {}


def _safe_task_dirname(task_id: str) -> str:
    task_id = task_id.strip() or "unknown_task"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", task_id)[:48] or "task"
    digest = sha1(task_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug}_{digest}"


def _task_index_dir(index_root: Path, task_id: str) -> Path:
    return index_root / _safe_task_dirname(task_id)


def _is_task_index_ready(index_dir: Path) -> bool:
    if not index_dir.exists():
        return False
    if not (index_dir / _PASSAGES_FILE).exists():
        return False
    return all((index_dir / name).exists() for name in _BM25S_CORE_FILES)


def _write_passages(index_dir: Path, passages: list[str]) -> None:
    with open(index_dir / _PASSAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)


def _read_passages(index_dir: Path) -> list[str]:
    with open(index_dir / _PASSAGES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [str(p) for p in data]


def _build_task_index(
    index_dir: Path,
    passages: list[str],
    *,
    k1: float = 0.9,
    b: float = 0.4,
    dtype: str = "float32",
    int_dtype: str = "int32",
) -> None:
    if _is_task_index_ready(index_dir):
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_tokens = bm25s.tokenize(
        passages,
        stopwords="en",
        stemmer=_stemmer,
        show_progress=False,
    )
    retriever = bm25s.BM25(k1=k1, b=b, dtype=dtype, int_dtype=int_dtype)
    retriever.index(corpus_tokens, show_progress=False)
    retriever.save(str(index_dir))
    _write_passages(index_dir, passages)


def build_task_indices(
    passages_by_task: dict[str, list[str]],
    index_root: str | Path,
    *,
    force_rebuild: bool = False,
) -> None:
    """Build per-task BM25 indices on disk."""
    index_root = Path(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    for task_id, raw_passages in passages_by_task.items():
        passages = [p for p in raw_passages if isinstance(p, str) and p.strip()]
        if not passages:
            continue
        index_dir = _task_index_dir(index_root, task_id)
        if force_rebuild and index_dir.exists():
            shutil.rmtree(index_dir)
        _build_task_index(index_dir, passages)


def _load_task_state(
    task_id: str,
    index_root: Path,
    passages_by_task: dict[str, list[str]],
) -> tuple[bm25s.BM25, list[str]] | None:
    with _state_lock:
        cached = _task_state.get(task_id)
        if cached is not None:
            return cached

        passages = passages_by_task.get(task_id, [])
        passages = [p for p in passages if isinstance(p, str) and p.strip()]
        if not passages:
            return None

        index_dir = _task_index_dir(index_root, task_id)
        if not _is_task_index_ready(index_dir):
            _build_task_index(index_dir, passages)

        retriever = bm25s.BM25.load(str(index_dir), mmap=True)
        saved_passages = _read_passages(index_dir)
        state = (retriever, saved_passages)
        _task_state[task_id] = state
        return state


def _retrieve_for_task(
    retriever: bm25s.BM25,
    passages: list[str],
    queries: list[str],
    k: int,
) -> list[str]:
    if not passages:
        return ["" for _ in queries]

    effective_k = max(1, min(k, len(passages)))
    tokens = bm25s.tokenize(
        queries,
        stopwords="en",
        stemmer=_stemmer,
        show_progress=False,
    )
    results, _scores = retriever.retrieve(
        tokens,
        k=effective_k,
        n_threads=1,
        show_progress=False,
    )
    return [
        "\n".join(
            f"[{j + 1}] {passages[int(idx)]}" for j, idx in enumerate(row[:effective_k])
        )
        for row in results
    ]


def make_retrieve_fn(
    index_root: str | Path,
    passages_by_task: dict[str, list[str]],
    k: int = 7,
) -> Callable[[list[dict]], list[str]]:
    """Create a batched retrieve function for the tool registry.

    Expected tool input mapping keys:
    - query: search query text
    - task_id: sample/task id identifying which local index to use
    """

    index_root = Path(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    def retrieve_fn(items: list[dict]) -> list[str]:
        outputs = ["" for _ in items]

        grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for i, item in enumerate(items):
            query = str(item.get("query", "")).strip()
            task_id = str(item.get("task_id", "")).strip()
            if not query or not task_id:
                continue
            grouped[task_id].append((i, query))

        for task_id, indexed_queries in grouped.items():
            state = _load_task_state(task_id, index_root, passages_by_task)
            if state is None:
                continue
            retriever, passages = state
            queries = [q for _, q in indexed_queries]
            task_outputs = _retrieve_for_task(retriever, passages, queries, k=k)
            for (orig_idx, _), output in zip(
                indexed_queries, task_outputs, strict=False
            ):
                outputs[orig_idx] = output

        return outputs

    return retrieve_fn
