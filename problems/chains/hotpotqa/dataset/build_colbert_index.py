"""Build a ColBERT index over wiki17_abstracts for HotpotQA retrieval.

Run once before using RETRIEVER = "colbert" in shared_config.py:
    PYTHONPATH=. ~/envs/evo_fast/bin/python \
        problems/chains/hotpotqa/dataset/build_colbert_index.py

Requirements:
    pip install colbert-ai faiss-cpu

Encodes all ~5.3M passages with colbert-ir/colbertv2.0 and saves the index.
GPU k-means runs via PyTorch (bypasses faiss-gpu dependency).
Encoding: ~30min on H100. Search-time is fast on CPU (~30ms/query).
"""

from pathlib import Path
import time

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

# Batch size for GPU k-means distance computation.
# 4096 x 262144 x 4 bytes ~= 4.3 GB per batch -- fits comfortably on H100 80 GB.
_KMEANS_BATCH = 4096


def _torch_gpu_kmeans(
    dim: int,
    num_partitions: int,
    kmeans_niters: int,
    shared_lists,
    return_value_queue=None,
):
    """GPU-accelerated k-means via PyTorch batched matmul.

    Drop-in replacement for colbert.indexing.collection_indexer.compute_faiss_kmeans.
    Uses PyTorch CUDA directly -- no faiss-gpu dependency required.

    Args:
        dim: Embedding dimension (128 for ColBERTv2).
        num_partitions: Number of cluster centroids (k).
        kmeans_niters: Number of Lloyd's iterations.
        shared_lists: ColBERT internal; shared_lists[0][0] is the sample tensor.
        return_value_queue: Optional mp.Queue for forked-process mode.

    Returns:
        torch.Tensor of shape [num_partitions, dim] (float32, on CPU).
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample = shared_lists[0][0]
    if not isinstance(sample, torch.Tensor):
        sample = torch.from_numpy(sample.astype("float32"))
    X = sample.float().to(device)  # [N, D]
    N = X.shape[0]

    print(
        f"[torch_kmeans] N={N:,} D={dim} K={num_partitions:,} "
        f"iters={kmeans_niters} device={device}"
    )

    torch.manual_seed(123)
    perm = torch.randperm(N, device=device)[:num_partitions]
    centroids = X[perm].clone()  # [K, D]

    for it in range(kmeans_niters):
        t0 = time.time()
        new_centroids = torch.zeros(num_partitions, dim, device=device)
        counts = torch.zeros(num_partitions, device=device)

        for start in range(0, N, _KMEANS_BATCH):
            Xb = X[start : start + _KMEANS_BATCH]  # [B, D]
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c.T
            dists = (
                (Xb * Xb).sum(1, keepdim=True)
                + (centroids * centroids).sum(1)
                - 2 * (Xb @ centroids.T)
            )  # [B, K]
            labels = dists.argmin(dim=1)  # [B]
            new_centroids.index_add_(0, labels, Xb)
            counts += torch.bincount(labels, minlength=num_partitions).float()

        # Update centroids; keep old centroid for empty clusters.
        mask = counts > 0
        new_centroids[mask] /= counts[mask, None]
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids

        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[torch_kmeans] iter {it + 1}/{kmeans_niters} done in {time.time() - t0:.1f}s")

    result = centroids.cpu()
    if return_value_queue is not None:
        return_value_queue.put(result)
    return result


def _patch_compute_faiss_kmeans() -> None:
    """Replace ColBERT's faiss k-means with PyTorch GPU k-means."""
    import colbert.indexing.collection_indexer as _ci

    _ci.compute_faiss_kmeans = _torch_gpu_kmeans
    print("[build_colbert_index] Patched compute_faiss_kmeans -> torch GPU k-means")


def build_index(passages: list[str]) -> None:
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    _patch_compute_faiss_kmeans()

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
